#!/usr/bin/env bash
# Runs every experiment for HW2 Q2 and writes all outputs to ./results_q2/.
# Usage:  bash run_q2.sh
# No `set -e`: a failing profiler must not stop the remaining experiments.

SRC=hw2-gemm.cu
OUT=results_q2
mkdir -p "$OUT"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), arch $ARCH" | tee "$OUT/env.txt"
nvcc --version | tail -n 2 | tee -a "$OUT/env.txt"

nvcc -O3 -arch=$ARCH -lineinfo "$SRC" -o "$OUT/gemm" || { echo "build failed"; exit 1; }
BIN="$OUT/gemm"

# --- 1. The two required inputs (screenshot material), 3 tile sizes ---------
{
  echo "=== A 1024x1024, B 1024x1024 ==="
  "$BIN" 1024 1024 1024 10 8 8 16 16 32 32
  echo
  echo "=== A 513x8192, B 8192x1023 ==="
  "$BIN" 513 8192 1023 10 8 8 16 16 32 32
} 2>&1 | tee "$OUT/q2_required_runs.txt"

# --- 2. Size sweep for the bar chart: CPU, gemm, three tiled kernels --------
echo "m,n,p,kernel,tileX,tileY,time_ms,status" > "$OUT/size_sweep.csv"
for S in 32 64 128 256 512 768 1024 1536 2048; do
  "$BIN" $S $S $S 20 8 8 16 16 32 32 | grep '^CSV,' | sed 's/^CSV,//' >> "$OUT/size_sweep.csv"
done
cat "$OUT/size_sweep.csv"

# --- 3. Nsight Compute: achieved occupancy and shared memory ---------------
# reps=0 -> exactly one launch per kernel, so each kernel is profiled once.
# NOCPU=1 skips the slow CPU reference (not needed for profiling).
# If this fails with ERR_NVGPUCTRPERM the container does not allow GPU counters.
for CASE in "1024 1024 1024" "513 8192 1023"; do
  TAG=$(echo "$CASE" | tr ' ' '_')
  NOCPU=1 ncu --clock-control none --set full -k regex:gemm --force-overwrite \
      -o "$OUT/ncu_$TAG" "$BIN" $CASE 0 8 8 16 16 32 32 \
      > "$OUT/ncu_${TAG}_log.txt" 2>&1
  ncu -i "$OUT/ncu_$TAG.ncu-rep" --page details > "$OUT/ncu_${TAG}_details.txt" 2>&1
  # compact summary of the metrics the assignment asks for
  ncu -i "$OUT/ncu_$TAG.ncu-rep" --page raw --csv \
      --metrics launch__grid_size,launch__block_size,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,gpu__time_duration.sum \
      > "$OUT/ncu_${TAG}_summary.csv" 2>&1
done

echo
echo "Done. Outputs:"
ls -1 "$OUT" | grep -v '^gemm$'
