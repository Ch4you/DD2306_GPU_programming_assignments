#!/usr/bin/env bash
# Runs every experiment for HW2 Q3 (llama2.c, matmul on GPU); outputs go to q3_llama2/results_q3/.
# Usage:  bash run_q3.sh
# No `set -e`: a failing profiler must not stop the remaining experiments.

cd "$(dirname "$0")/q3_llama2" || exit 1
OUT=results_q3
mkdir -p "$OUT"

CC_=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC_}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), arch $ARCH" | tee "$OUT/env.txt"

# --- model --------------------------------------------------------------
if [ ! -f stories15M.bin ]; then
  wget -q https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O stories15M.bin \
    || { echo "could not download stories15M.bin"; exit 1; }
fi

# --- build --------------------------------------------------------------
make clean >/dev/null
make ARCH=$ARCH || { echo "build failed"; exit 1; }

ARGS="stories15M.bin -t 0 -s 1 -n 256 -i \"Once upon a time\""   # greedy => deterministic

# --- 1. tok/s of every variant (3 repetitions each) ---------------------
# stdout (the generated story) goes to a file, stderr (tok/s, matmul calls) to a log.
for V in cpu naive resident warp; do
  for R in 1 2 3; do
    eval ./run_$V $ARGS > "$OUT/out_${V}_$R.txt" 2> "$OUT/log_${V}_$R.txt"
    echo "$V run $R: $(grep 'achieved' "$OUT/log_${V}_$R.txt")  $(grep 'matmul calls' "$OUT/log_${V}_$R.txt")"
  done
done | tee "$OUT/toks_summary.txt"

# --- 2. correctness: same generated text as the CPU version? -------------
{
  for V in naive resident warp; do
    if diff -q "$OUT/out_cpu_1.txt" "$OUT/out_${V}_1.txt" >/dev/null; then
      echo "$V: generated text IDENTICAL to CPU"
    else
      echo "$V: generated text DIFFERS from CPU (see diff below)"
      diff "$OUT/out_cpu_1.txt" "$OUT/out_${V}_1.txt" | head -5
    fi
  done
  # numerical check of every matmul call against a CPU reference
  for V in naive resident warp; do
    echo "--- $V, VERIFY_MATMUL=1 ---"
    eval VERIFY_MATMUL=1 ./run_$V $ARGS 2>&1 >/dev/null | grep -E 'VERIFY|matmul calls'
  done
} 2>&1 | tee "$OUT/correctness.txt"

# --- 3. Nsight Systems ----------------------------------------------------
# fewer steps keeps the naive profile small
for V in naive resident warp; do
  nsys profile --trace=cuda,osrt --stats=true --force-overwrite=true \
    -o "$OUT/nsys_$V" ./run_$V stories15M.bin -t 0 -s 1 -n 64 -i "Once upon a time" \
    > "$OUT/nsys_${V}_stats.txt" 2>&1 \
    || echo "nsys failed for $V, see $OUT/nsys_${V}_stats.txt"
done

# --- 4. Nsight Compute: the matmul kernel of the optimized build ----------
# If this fails with ERR_NVGPUCTRPERM the container does not allow GPU counters.
ncu --clock-control none --set full -k regex:matmul -s 20 -c 3 --force-overwrite -o "$OUT/ncu_warp" \
    ./run_warp stories15M.bin -t 0 -s 1 -n 8 -i "Once upon a time" \
    > "$OUT/ncu_warp_log.txt" 2>&1
ncu -i "$OUT/ncu_warp.ncu-rep" --page details > "$OUT/ncu_warp_details.txt" 2>&1

echo
echo "Done. Outputs:"
ls -1 "$OUT"
