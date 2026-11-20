#!/usr/bin/env bash
# Runs the HW2 bonus (iPIC3D-mini, GEM_2D.inp) with the CPU and the GPU mover and compares them.
# Outputs go to bonus_ipic3d/results_bonus/.   Usage:  bash run_bonus.sh
# No `set -e`: a failing step must not hide the remaining output.

cd "$(dirname "$0")/bonus_ipic3d" || exit 1
ROOT=$(pwd)
OUT="$ROOT/results_bonus"
mkdir -p "$OUT"

CC_=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC_}"
{
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), arch $ARCH"
  nvcc --version | tail -n 2
  g++ --version | head -1
} | tee "$OUT/env.txt"

# --- build -------------------------------------------------------------
make clean >/dev/null 2>&1
make ARCH=$ARCH 2>&1 | tee "$OUT/build_log.txt"
[ -x ./bin/miniPIC.out ] || { echo "build failed, see $OUT/build_log.txt"; exit 1; }
g++ -O2 -o "$OUT/compare_particles" compare_particles.cpp

# --- one working directory per run: the code writes its VTK files to ./data/ ----
INP="$ROOT/inputfiles/GEM_2D.inp"
for V in cpu gpu; do
  rm -rf "$OUT/run_$V"; mkdir -p "$OUT/run_$V/data"
done

# CPU reference
( cd "$OUT/run_cpu" && MOVER=cpu DUMP_PARTICLES="$OUT/run_cpu/particles" \
    "$ROOT/bin/miniPIC.out" "$INP" > "$OUT/log_cpu.txt" 2>&1 )
# GPU
( cd "$OUT/run_gpu" && MOVER=gpu DUMP_PARTICLES="$OUT/run_gpu/particles" \
    "$ROOT/bin/miniPIC.out" "$INP" > "$OUT/log_gpu.txt" 2>&1 )

# --- timing ------------------------------------------------------------
{
  echo "=== CPU ==="; grep -E "Tot. Simulation|Mover Time|Interp. Time" "$OUT/log_cpu.txt"
  echo "=== GPU ==="; grep -E "Tot. Simulation|Mover Time|Interp. Time" "$OUT/log_gpu.txt"
  echo "=== GPU mover breakdown (per species and cycle) ==="; grep "GPU mover species" "$OUT/log_gpu.txt" | head -8
} | tee "$OUT/timing_summary.txt"

# --- correctness ---------------------------------------------------------
NS=$(grep -c "^CHECKSUM" "$OUT/log_cpu.txt")
{
  echo "=== checksums (sum over all particles) ==="
  echo "CPU:"; grep "^CHECKSUM" "$OUT/log_cpu.txt"
  echo "GPU:"; grep "^CHECKSUM" "$OUT/log_gpu.txt"
  echo
  echo "=== particle-by-particle comparison ==="
  "$OUT/compare_particles" "$OUT/run_cpu/particles" "$OUT/run_gpu/particles" "$NS"
  echo
  echo "=== VTK field / density files written by the two runs ==="
  for f in "$OUT"/run_cpu/data/*.vtk; do
    b=$(basename "$f")
    if cmp -s "$f" "$OUT/run_gpu/data/$b"; then echo "$b: identical";
    else echo "$b: DIFFERS ($(cmp -l "$f" "$OUT/run_gpu/data/$b" 2>/dev/null | wc -l) bytes)"; fi
  done
} 2>&1 | tee "$OUT/correctness.txt"

# the dumps are hundreds of MB: keep them out of the repository
rm -f "$OUT"/run_cpu/particles_s*.bin "$OUT"/run_gpu/particles_s*.bin

# --- Nsight Systems of the GPU run (few cycles) ---------------------------
sed 's/^ncycles *= *[0-9]*/ncycles = 3/' "$INP" > "$OUT/GEM_2D_3cycles.inp"
mkdir -p "$OUT/run_nsys/data"
( cd "$OUT/run_nsys" && MOVER=gpu nsys profile --trace=cuda,osrt --stats=true --force-overwrite=true \
    -o "$OUT/nsys_gpu" "$ROOT/bin/miniPIC.out" "$OUT/GEM_2D_3cycles.inp" \
    > "$OUT/nsys_gpu_stats.txt" 2>&1 ) || echo "nsys failed, see $OUT/nsys_gpu_stats.txt"

echo; echo "Done. Outputs in $OUT :"; ls -1 "$OUT" | grep -v '^run_'
