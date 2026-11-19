#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

int main(int argc, char* argv[]) {
    int n = 512;
    int repetitions = 1;

    try {
        if (argc > 1) {
            n = std::stoi(argv[1]);
        }
        if (argc > 2) {
            repetitions = std::stoi(argv[2]);
        }
    } catch (const std::exception&) {
        std::cerr << "Usage: vector_add.exe [N] [repetitions]\n";
        return EXIT_FAILURE;
    }

    if (argc > 3 || n <= 0 || repetitions <= 0) {
        std::cerr << "N and repetitions must be positive integers.\n";
        return EXIT_FAILURE;
    }

    // Prepare the inputs and compute the CPU reference.
    std::vector<float> a(n), b(n), cpuResult(n), gpuResult(n);
    for (int i = 0; i < n; ++i) {
        a[i] = 0.25f * i;
        b[i] = 0.5f * (i % 7);
        cpuResult[i] = a[i] + b[i];
    }

    // Allocate device buffers outside the timed regions.
    const auto bytes = a.size() * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    checkCuda(cudaMalloc(&d_a, bytes));
    checkCuda(cudaMalloc(&d_b, bytes));
    checkCuda(cudaMalloc(&d_c, bytes));

    const int threadsPerBlock = 256;
    const int blocks = 1 + (n - 1) / threadsPerBlock;

    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    std::vector<double> h2dTimes, kernelTimes, d2hTimes;
    h2dTimes.reserve(repetitions);
    kernelTimes.reserve(repetitions);
    d2hTimes.reserve(repetitions);

    // In benchmark mode, iteration -1 is an unrecorded warm-up.
    const int firstIteration = (repetitions > 1) ? -1 : 0;
    for (int run = firstIteration; run < repetitions; ++run) {
        const auto start = Clock::now();

        // H2D: copy both inputs and wait for the transfers.
        checkCuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaDeviceSynchronize());
        const auto afterH2D = Clock::now();

        // Kernel: launch and wait for completion.
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        const auto afterKernel = Clock::now();

        // D2H: copy the output back to the CPU.
        checkCuda(cudaMemcpy(gpuResult.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        const auto afterD2H = Clock::now();

        if (run >= 0) {
            h2dTimes.push_back(Milliseconds(afterH2D - start).count());
            kernelTimes.push_back(Milliseconds(afterKernel - afterH2D).count());
            d2hTimes.push_back(Milliseconds(afterD2H - afterKernel).count());
        }
    }

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    // Compare the final GPU result with the CPU reference.
    for (int i = 0; i < n; ++i) {
        if (cpuResult[i] != gpuResult[i]) {
            std::cerr << "FAIL at i=" << i
                      << ", CPU=" << cpuResult[i]
                      << ", GPU=" << gpuResult[i] << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cerr << "PASS: N=" << n << ", blocks=" << blocks
              << ", threads/block=" << threadsPerBlock << '\n';

    // CSV columns: N,H2D_ms,Kernel_ms,D2H_ms
    std::cout << std::fixed << std::setprecision(6)
              << n << ',' << median(h2dTimes) << ','
              << median(kernelTimes) << ',' << median(d2hTimes) << '\n';

    return EXIT_SUCCESS;
}