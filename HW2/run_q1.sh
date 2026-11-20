#!/usr/bin/env bash
# Runs every experiment needed for HW2 Q1 and writes all outputs to ./results_q1/.
# Usage:  bash run_q1.sh
# No `set -e`: a failing profiler (e.g. missing counter permission) must not stop the rest.

SRC=hw2-convolution.cu
OUT=results_q1
mkdir -p "$OUT"

# --- GPU architecture ------------------------------------------------------
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), arch $ARCH" | tee "$OUT/env.txt"
nvcc --version | tail -n 2 | tee -a "$OUT/env.txt"

# --- 1. Build one binary per tile size ------------------------------------
TILES="32 64 128 256 512 1024"
for T in $TILES; do
  nvcc -O3 -arch=$ARCH -lineinfo -DTILE_WIDTH=$T "$SRC" -o "$OUT/conv_$T" || { echo "build failed for T=$T"; exit 1; }
done
DEFT=128   # default tile size used for the required runs and profiling

# --- 2. Required runs (screenshot material), TILE_WIDTH=128 ---------------
{
  for N in 1024 2048 4096 8192 16384; do
    echo "=== N=$N ==="
    "$OUT/conv_$DEFT" $N
  done
} 2>&1 | tee "$OUT/q1_required_runs.txt"

# --- 3. Tile size sweep (cudaEvent, warm-up, 100 reps) ---------------------
SWEEP_N="1024 16384 1048576 16777216 67108864"
echo "N,tile,carveout,reps,basic_us,tiled_us" > "$OUT/tile_sweep.csv"
for T in $TILES; do
  for N in $SWEEP_N; do
    "$OUT/conv_$T" $N 100 | grep '^BENCH' | sed 's/^BENCH,//' >> "$OUT/tile_sweep.csv"
    # also record the correctness line
    "$OUT/conv_$T" $N 1 | grep -E 'PASS|FAIL' | sed "s/^/T=$T N=$N: /" >> "$OUT/tile_sweep_check.txt"
  done
done
cat "$OUT/tile_sweep.csv"

# --- 4. Shared memory carveout sweep (tile=128, large N) -------------------
echo "N,tile,carveout,reps,basic_us,tiled_us" > "$OUT/carveout_sweep.csv"
for C in -1 0 25 50 75 100; do
  "$OUT/conv_$DEFT" 67108864 100 $C | grep '^BENCH' | sed 's/^BENCH,//' >> "$OUT/carveout_sweep.csv"
done
cat "$OUT/carveout_sweep.csv"

# --- 5. Nsight Systems (tile=128, N=16M, 10 reps) --------------------------
nsys profile --trace=cuda,osrt --stats=true --force-overwrite=true \
  -o "$OUT/nsys_tiled" "$OUT/conv_$DEFT" 16777216 10 \
  > "$OUT/nsys_tiled_stats.txt" 2>&1 \
  || echo "nsys failed, see $OUT/nsys_tiled_stats.txt"

# --- 6. Nsight Compute (first launch of each kernel) -----------------------
# If this fails with ERR_NVGPUCTRPERM, the container does not allow GPU counters
# and the ncu part has to be run elsewhere or with elevated permission.
for K in tiled basic; do
  ncu --clock-control none --set full -k "regex:convolution_1D_$K" -c 1 --force-overwrite \
      -o "$OUT/ncu_$K" "$OUT/conv_$DEFT" 16777216 0 \
      > "$OUT/ncu_${K}_log.txt" 2>&1
  ncu -i "$OUT/ncu_$K.ncu-rep" --page details > "$OUT/ncu_${K}_details.txt" 2>&1
done

# ncu for carveout 0 and 100 (shows the resulting shared memory configuration / occupancy)
for C in 0 100; do
  ncu --clock-control none --set full -k regex:convolution_1D_tiled -c 1 --force-overwrite \
      -o "$OUT/ncu_tiled_carveout$C" "$OUT/conv_$DEFT" 16777216 0 $C \
      > "$OUT/ncu_tiled_carveout${C}_log.txt" 2>&1
  ncu -i "$OUT/ncu_tiled_carveout$C.ncu-rep" --page details > "$OUT/ncu_tiled_carveout${C}_details.txt" 2>&1
done

echo
echo "Done. Send back (or inspect) everything in $OUT/ :"
ls -1 "$OUT" | grep -v '^conv_'
