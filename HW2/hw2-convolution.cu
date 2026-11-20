#include <stdio.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define TILE_WIDTH 128


// Basic version: one thread computes one output element P[i],
// reading everything directly from global memory.
__global__ void convolution_1D_basic(float *N, float *M, float *P, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < MASK_WIDTH; j++) {
        int idx = i + j - MASK_WIDTH / 2;   // MASK_WIDTH/2 = 2
        if (idx >= 0 && idx < n) {          // out-of-boundary elements count as 0
            sum += N[idx] * M[j];
        }
    }
    P[i] = sum;
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int n)
{
  // Shared tile holds TILE_WIDTH inputs plus a halo of MASK_WIDTH/2 on each side
  __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1];

  int base = blockIdx.x * TILE_WIDTH - MASK_WIDTH / 2;

  for (int i = threadIdx.x; i < TILE_WIDTH + MASK_WIDTH - 1; i += blockDim.x) {
      int idx = base + i;
      input_tile[i] = (idx >= 0 && idx < n) ? N[idx] : 0.0f;
  }
  __syncthreads();

  int i = blockIdx.x * TILE_WIDTH + threadIdx.x;
  if (i < n) {
      float sum = 0.0f;
      for (int j = 0; j < MASK_WIDTH; j++) {
          sum += input_tile[threadIdx.x + j] * M[j];
      }
      P[i] = sum;
  }
}

int main(int argc, char *argv[]) {
  
  // Read the arguments from the command line
  int N = atoi(argv[1]);


  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N (basic result)
  float *hostP2; // Output of the tiled kernel

  cputimer_start();
  //@@ Allocate the host memory
  hostN = (float*)malloc(N * sizeof(float));
  hostM = (float*)malloc(MASK_WIDTH * sizeof(float));
  hostP = (float*)malloc(N * sizeof(float));
  hostP2 = (float*)malloc(N * sizeof(float));
  cputimer_stop("Allocated host memory");


  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  gpuCheck(cudaMalloc((void**)&deviceN, N * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceP, N * sizeof(float)));
  cputimer_stop("Allocated device memory");

  cputimer_start();
  //@@ Initialize N with random values
  //@@ Initialize P with 0.0
  for (int i = 0; i < N; i++) {
      hostN[i] = (float)rand() / RAND_MAX;
      hostP[i] = 0.0f;
  }
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  hostM[0] = -0.25f;
  hostM[1] = 0.5f;
  hostM[2] = 1.0f;
  hostM[3] = 0.5f;
  hostM[4] = 0.25f;
  
  cputimer_stop("Initialized data on the host");

  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(deviceN, hostN, N * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceP, hostP, N * sizeof(float), cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");
  

  /* Call the basic kernel */
  cputimer_start();
  //@@  Define the execution configuration
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;
  //@@  Run the 1D convolution kernel (basic)
  convolution_1D_basic<<<gridSize, blockSize>>>(deviceN, deviceM, deviceP, N);
  // Wait for the GPU to finish
  cudaDeviceSynchronize();
  cputimer_stop("Finished 1D convolution(basic)");
  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 5 && i < N; i++) {
      printf("N[%d] = %f, P[%d] = %f\n", i, hostN[i], i, hostP[i]);
  }

  cputimer_stop("Copying output P to the CPU and print out the results");

  /* Call the tiled kernel */
  cputimer_start();
  //@@  Define the execution configuration
  int tiledGrid = (N + TILE_WIDTH - 1) / TILE_WIDTH;
  //@@  Run the 1D convolution kernel (tiled)
  convolution_1D_tiled<<<tiledGrid, TILE_WIDTH>>>(deviceN, deviceM, deviceP, N);
  cudaDeviceSynchronize();
  cputimer_stop("Finished 1D convolution(tiled)");

  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP2, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");


  //@@ Validate the results from the two implementations
  int mismatches = 0;
  for (int i = 0; i < N; i++) {
      if (fabs(hostP[i] - hostP2[i]) > 1e-5f) {
          mismatches++;
          if (mismatches <= 5) {
              printf("Mismatch at %d: basic = %f, tiled = %f\n", i, hostP[i], hostP2[i]);
          }
      }
  }
  if (mismatches == 0) {
      printf("PASS: basic and tiled results match.\n");
  } else {
      printf("FAIL: %d mismatches.\n", mismatches);
  }


  cputimer_start();
  //@@ INSERT CODE HERE
  cudaFree(deviceN);
  cudaFree(deviceM);
  cudaFree(deviceP);
  free(hostN);
  free(hostM);
  free(hostP);
  free(hostP2);
  cputimer_stop("Free memory resources");

  return 0;
}
