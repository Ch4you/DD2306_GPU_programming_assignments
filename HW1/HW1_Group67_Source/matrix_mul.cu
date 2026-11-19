#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

template<class T>
__global__ void matrixMultiply(const T* a, const T* b, T* c, int m, int n, int p) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= m || col >= p) return;

    T sum = 0;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[size_t(k) * p + col];
    }
    c[row * p + col] = sum;
}

template<class T>
T input(size_t index, uint32_t seed) {
    uint32_t x = static_cast<uint32_t>(index) + seed;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return T(int((x ^ (x >> 16)) & 7u) - 4) / T(16);
}

double median(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    const auto mid = times.size() / 2;
    return times.size() % 2 ? times[mid] : (times[mid - 1] + times[mid]) / 2;
}

template<class T>
bool validate(const std::vector<T>& a, const std::vector<T>& b,
              const std::vector<T>& c, int n, int p, size_t checks) {
    for (size_t sample = 0; sample < checks; ++sample) {
        // Include the first/last entries and spread the other checks across C.
        const size_t i = sample * (c.size() - 1) / std::max<size_t>(1, checks - 1);
        const size_t row = i / p;
        const size_t col = i % p;
        double expected = 0;
        for (int k = 0; k < n; ++k) {
            expected += double(a[row * n + k]) * double(b[size_t(k) * p + col]);
        }
        if (double(c[i]) != expected) {
            std::cerr << "FAIL at (" << row << ',' << col << "): CPU=" << expected
                      << ", GPU=" << c[i] << '\n';
            return false;
        }
    }
    return true;
}

template<class T>
int run(int m, int n, int p, int repetitions, const std::string& precision) {
    const dim3 block(16, 16);
    const dim3 grid(1 + (p - 1) / block.x, 1 + (m - 1) / block.y);
    std::vector<T> a(size_t(m) * n), b(size_t(n) * p), c(size_t(m) * p);
    for (size_t i = 0; i < a.size(); ++i) a[i] = input<T>(i, 113);
    for (size_t i = 0; i < b.size(); ++i) b[i] = input<T>(i, 271);

    T *da = nullptr, *db = nullptr, *dc = nullptr;
    const size_t aBytes = a.size() * sizeof(T);
    const size_t bBytes = b.size() * sizeof(T);
    const size_t cBytes = c.size() * sizeof(T);
    checkCuda(cudaMalloc(&da, aBytes));
    checkCuda(cudaMalloc(&db, bBytes));
    checkCuda(cudaMalloc(&dc, cBytes));

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    std::vector<double> h2d, kernel, d2h;
    h2d.reserve(repetitions);
    kernel.reserve(repetitions);
    d2h.reserve(repetitions);
    for (int rep = repetitions > 1 ? -1 : 0; rep < repetitions; ++rep) {
        const auto start = Clock::now();
        checkCuda(cudaMemcpy(da, a.data(), aBytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(db, b.data(), bBytes, cudaMemcpyHostToDevice));
        checkCuda(cudaDeviceSynchronize());
        const auto copied = Clock::now();
        matrixMultiply<<<grid, block>>>(da, db, dc, m, n, p);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        const auto computed = Clock::now();
        checkCuda(cudaMemcpy(c.data(), dc, cBytes, cudaMemcpyDeviceToHost));
        const auto done = Clock::now();
        if (rep >= 0) {
            h2d.push_back(Ms(copied - start).count());
            kernel.push_back(Ms(computed - copied).count());
            d2h.push_back(Ms(done - computed).count());
        }
    }
    checkCuda(cudaFree(da));
    checkCuda(cudaFree(db));
    checkCuda(cudaFree(dc));

    const size_t checks = c.size() <= 10000000 / size_t(n)
        ? c.size() : std::min<size_t>(c.size(), 257);
    if (!validate(a, b, c, n, p, checks)) return EXIT_FAILURE;
    std::cerr << "PASS " << precision << ": " << checks << '/' << c.size()
              << " CPU checks, grid=" << grid.x << 'x' << grid.y
              << ", block=" << block.x << 'x' << block.y << '\n';
    std::cout << precision << ',' << m << ',' << n << ',' << p << ','
              << std::fixed << std::setprecision(6)
              << median(h2d) << ',' << median(kernel) << ',' << median(d2h) << '\n';
    return EXIT_SUCCESS;
}

int positive(const char* arg) {
    size_t parsed = 0;
    const int value = std::stoi(arg, &parsed);
    if (value < 1 || arg[parsed] != '\0') throw std::invalid_argument("positive integer required");
    return value;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::invalid_argument("wrong argument count");
        const std::string precision = argv[1];
        const int m = positive(argv[2]), n = positive(argv[3]), p = positive(argv[4]);
        const int repetitions = argc == 6 ? positive(argv[5]) : 1;
        if (precision == "double") return run<double>(m, n, p, repetitions, precision);
        if (precision == "float") return run<float>(m, n, p, repetitions, precision);
        throw std::invalid_argument("precision must be double or float");
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\nUsage: matrix_mul.exe double|float M N P [repetitions]\n";
        return EXIT_FAILURE;
    }
}
