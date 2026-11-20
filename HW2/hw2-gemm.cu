// HW2 Q2: tiled matrix multiplication  C(m x p) = A(m x n) * B(n x p), float.
//
// Usage:   ./gemm m n p [reps] [tileX tileY [tileX tileY ...]]
//   reps   number of timed launches per kernel after one warm-up launch (default 10).
//          reps = 0 launches every kernel exactly once and skips timing (use with ncu).
//   tiles  tile sizes to sweep; default 8x8 16x16 32x32.
// Environment: NOCPU=1 skips the CPU reference (the gemm() result is then the reference).
//
// Output: human readable lines, plus machine readable lines starting with "CSV,":
//   CSV,m,n,p,kernel,tileX,tileY,time_ms,status

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(stmt)                                                           \
    do {                                                                           \
        cudaError_t err_ = (stmt);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt, __FILE__,       \
                    __LINE__, cudaGetErrorString(err_));                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Basic version: one thread per element of C, all operands from global memory.
__global__ void gemm(const float* a, const float* b, float* c, int m, int n, int p) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= p) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[size_t(row) * n + k] * b[size_t(k) * p + col];
    }
    c[size_t(row) * p + col] = sum;
}

// Tiled version. A block has tileX x tileY threads and computes a tileY x tileX tile of C.
// Each step advances along k by bk = tileX and stages
//   an A tile of tileY rows x bk columns and a B tile of bk rows x tileX columns.
// The tiles are loaded cooperatively with strided loops, so any tileX/tileY works
// (the threads need not map 1:1 onto tile elements). Out-of-range elements load as 0.
// Shared memory is dynamic: (tileY*bk + bk*tileX) floats.
__global__ void tiled_gemm(const float* a, const float* b, float* c,
                           int m, int n, int p, int tileX, int tileY) {
    extern __shared__ float smem[];
    const int bk = tileX;
    float* as = smem;                 // tileY x bk
    float* bs = smem + tileY * bk;    // bk x tileX

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * tileY + ty;
    const int col = blockIdx.x * tileX + tx;
    const int tid = ty * tileX + tx;
    const int nthreads = tileX * tileY;

    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += bk) {
        for (int idx = tid; idx < tileY * bk; idx += nthreads) {
            const int r = idx / bk, kk = idx % bk;
            const int gr = blockIdx.y * tileY + r, gk = k0 + kk;
            as[idx] = (gr < m && gk < n) ? a[size_t(gr) * n + gk] : 0.0f;
        }
        for (int idx = tid; idx < bk * tileX; idx += nthreads) {
            const int kk = idx / tileX, cc = idx % tileX;
            const int gk = k0 + kk, gc = blockIdx.x * tileX + cc;
            bs[idx] = (gk < n && gc < p) ? b[size_t(gk) * p + gc] : 0.0f;
        }
        __syncthreads();                       // tiles complete before use

        for (int kk = 0; kk < bk; ++kk) {
            sum += as[ty * bk + kk] * bs[kk * tileX + tx];
        }
        __syncthreads();                       // done reading before the next load
    }
    if (row < m && col < p) c[size_t(row) * p + col] = sum;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Inputs are small multiples of 1/16 (as in HW1), so every product is a multiple of 1/256
// and all partial sums are exactly representable in float for the sizes used here
// (|sum| <= 8192 * 4*4/256 = 512, 512*256 < 2^24). Results therefore do not depend on the
// summation order and CPU and GPU can be compared exactly; a tiny tolerance is kept anyway.
static float input(size_t index, uint32_t seed) {
    uint32_t x = static_cast<uint32_t>(index) + seed;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return float(int((x ^ (x >> 16)) & 7u) - 4) / 16.0f;
}

static void cpuGemm(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c, int m, int n, int p) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (int i = 0; i < m; ++i) {              // i-k-j order: cache friendly on B and C
        for (int k = 0; k < n; ++k) {
            const float aik = a[size_t(i) * n + k];
            for (int j = 0; j < p; ++j) {
                c[size_t(i) * p + j] += aik * b[size_t(k) * p + j];
            }
        }
    }
}

static size_t countMismatches(const std::vector<float>& ref, const std::vector<float>& x) {
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::fabs(ref[i] - x[i]) > 1e-3f) ++bad;
    }
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s m n p [reps] [tileX tileY ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int m = atoi(argv[1]), n = atoi(argv[2]), p = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 10;
    if (m < 1 || n < 1 || p < 1 || reps < 0) {
        fprintf(stderr, "m, n, p must be positive and reps >= 0\n");
        return EXIT_FAILURE;
    }

    std::vector<int> tx, ty;
    for (int i = 5; i + 1 < argc; i += 2) {
        tx.push_back(atoi(argv[i]));
        ty.push_back(atoi(argv[i + 1]));
    }
    if (tx.empty()) {
        for (int t : {8, 16, 32}) { tx.push_back(t); ty.push_back(t); }
    }
    const bool useCpu = getenv("NOCPU") == nullptr;

    std::vector<float> a(size_t(m) * n), b(size_t(n) * p);
    std::vector<float> ref(size_t(m) * p), out(size_t(m) * p);
    for (size_t i = 0; i < a.size(); ++i) a[i] = input(i, 113);
    for (size_t i = 0; i < b.size(); ++i) b[i] = input(i, 271);

    printf("Matrix multiplication: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, p, m, p);

    // CPU reference
    if (useCpu) {
        const auto t0 = std::chrono::steady_clock::now();
        cpuGemm(a, b, ref, m, n, p);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("CPU reference:                 %10.3f ms\n", ms);
        printf("CSV,%d,%d,%d,cpu,0,0,%.6f,REF\n", m, n, p, ms);
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, out.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    int failures = 0;

    // ---- basic gemm() first ----
    {
        const dim3 block(16, 16);
        const dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        CUDA_CHECK(cudaMemset(dc, 0, out.size() * sizeof(float)));
        gemm<<<grid, block>>>(da, db, dc, m, n, p);          // warm-up, also the checked result
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dc, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (!useCpu) ref = out;                              // reference when the CPU is skipped
        const size_t bad = countMismatches(ref, out);

        float ms = -1.0f;
        if (reps > 0) {
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < reps; ++r) gemm<<<grid, block>>>(da, db, dc, m, n, p);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= reps;
        }
        printf("gemm()           block 16x16: %10.3f ms  %s\n", ms, bad ? "FAIL" : "PASS");
        printf("CSV,%d,%d,%d,gemm,16,16,%.6f,%s\n", m, n, p, ms, bad ? "FAIL" : "PASS");
        if (bad) ++failures;
    }

    // ---- tiled_gemm() for each tile size ----
    for (size_t t = 0; t < tx.size(); ++t) {
        const int tileX = tx[t], tileY = ty[t];
        if (tileX < 1 || tileY < 1 || tileX * tileY > 1024) {
            fprintf(stderr, "skipping invalid tile %dx%d\n", tileX, tileY);
            continue;
        }
        const dim3 block(tileX, tileY);
        const dim3 grid((p + tileX - 1) / tileX, (m + tileY - 1) / tileY);
        const size_t shmem = size_t(tileY * tileX + tileX * tileX) * sizeof(float);

        CUDA_CHECK(cudaMemset(dc, 0, out.size() * sizeof(float)));
        tiled_gemm<<<grid, block, shmem>>>(da, db, dc, m, n, p, tileX, tileY);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dc, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const size_t bad = countMismatches(ref, out);

        float ms = -1.0f;
        if (reps > 0) {
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < reps; ++r) {
                tiled_gemm<<<grid, block, shmem>>>(da, db, dc, m, n, p, tileX, tileY);
            }
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= reps;
        }
        printf("tiled_gemm()     tile %2dx%-2d: %10.3f ms  %s\n", tileX, tileY, ms,
               bad ? "FAIL" : "PASS");
        printf("CSV,%d,%d,%d,tiled,%d,%d,%.6f,%s\n", m, n, p, tileX, tileY, ms,
               bad ? "FAIL" : "PASS");
        if (bad) ++failures;
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
