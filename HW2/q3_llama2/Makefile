# HW2 Q3: llama2.c with matmul on the GPU.
#   make          builds all four variants
#   run_cpu       original CPU matmul (gcc -O3, as in the llama2.c README)
#   run_naive     naive GPU port (alloc + copy W and x on every call)
#   run_resident  optimization 1: weights copied to the GPU once
#   run_warp      optimization 1 + warp-per-row kernel
ARCH ?= sm_90
CC   ?= gcc
NVCC ?= nvcc

all: run_cpu run_naive run_resident run_warp

run_cpu: run.c
	$(CC) -O3 -o $@ run.c -lm

run.gpu.o: run.c
	$(CC) -O3 -DUSE_CUDA -c run.c -o $@

matmul_naive.o: matmul_gpu.cu
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -c $< -o $@

matmul_resident.o: matmul_gpu.cu
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -DRESIDENT_WEIGHTS -c $< -o $@

matmul_warp.o: matmul_gpu.cu
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -DRESIDENT_WEIGHTS -DWARP_KERNEL -c $< -o $@

run_naive: run.gpu.o matmul_naive.o
	$(NVCC) -arch=$(ARCH) $^ -o $@ -lm

run_resident: run.gpu.o matmul_resident.o
	$(NVCC) -arch=$(ARCH) $^ -o $@ -lm

run_warp: run.gpu.o matmul_warp.o
	$(NVCC) -arch=$(ARCH) $^ -o $@ -lm

clean:
	rm -f run_cpu run_naive run_resident run_warp *.o
