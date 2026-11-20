#!/usr/bin/env bash
# Re-runs only the Nsight Compute steps of Q1 and Q2. The first run failed because the GPU is a
# MIG instance ("Cannot lock GPU clock frequencies on MIG"); --clock-control none fixes that.
# Usage (from HW2/):  bash rerun_ncu.sh
# Outputs: results_q1/ncu_*_{details,key}.txt and results_q2/ncu_*_{details,key}.txt

cd "$(dirname "$0")" || exit 1
CC_=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC_}"
KEYS='Duration|Achieved Occupancy|Theoretical Occupancy|Block Size|Grid Size|Registers Per Thread|Static Shared Memory Per Block|Dynamic Shared Memory Per Block|Shared Memory Configuration Size|Driver Shared Memory|Memory Throughput|DRAM Throughput|L1/TEX Cache Throughput|L2 Cache Throughput|Compute \(SM\) Throughput|Occupancy Limit|Waves Per SM|Bank Conflicts|convolution_1D|gemm'

summ() {  # $1 = .ncu-rep, $2 = key-metric output file
  ncu -i "$1" --page details > "${1%.ncu-rep}_details.txt" 2>&1
  grep -E "$KEYS" "${1%.ncu-rep}_details.txt" > "$2"
}

# ---------------- Q1 ----------------
Q1=results_q1
[ -x $Q1/conv_128 ] || nvcc -O3 -arch=$ARCH -lineinfo -DTILE_WIDTH=128 hw2-convolution.cu -o $Q1/conv_128
for K in tiled basic; do
  ncu --clock-control none --set full -k regex:convolution_1D_$K -c 1 --force-overwrite \
      -o $Q1/ncu_$K $Q1/conv_128 16777216 0 > $Q1/ncu_${K}_log.txt 2>&1
  summ $Q1/ncu_$K.ncu-rep $Q1/ncu_${K}_key.txt
done
for C in 0 100; do   # shared memory carveout 0 % and 100 %
  ncu --clock-control none --set full -k regex:convolution_1D_tiled -c 1 --force-overwrite \
      -o $Q1/ncu_tiled_carveout$C $Q1/conv_128 16777216 0 $C > $Q1/ncu_tiled_carveout${C}_log.txt 2>&1
  summ $Q1/ncu_tiled_carveout$C.ncu-rep $Q1/ncu_tiled_carveout${C}_key.txt
done

# ---------------- Q2 ----------------
Q2=results_q2
[ -x $Q2/gemm ] || nvcc -O3 -arch=$ARCH -lineinfo hw2-gemm.cu -o $Q2/gemm
for CASE in "1024 1024 1024" "513 8192 1023"; do
  TAG=$(echo "$CASE" | tr ' ' '_')
  # reps=0: one launch per kernel -> gemm, tiled 8x8, tiled 16x16, tiled 32x32
  NOCPU=1 ncu --clock-control none --set full -k regex:gemm --force-overwrite \
      -o $Q2/ncu_$TAG $Q2/gemm $CASE 0 8 8 16 16 32 32 > $Q2/ncu_${TAG}_log.txt 2>&1
  summ $Q2/ncu_$TAG.ncu-rep $Q2/ncu_${TAG}_key.txt
done

echo "Done. Key metrics:"; ls $Q1/*_key.txt $Q2/*_key.txt
