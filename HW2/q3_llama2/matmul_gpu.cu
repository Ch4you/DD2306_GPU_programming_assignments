// GPU port of llama2.c's matmul:  xout (d) = W (d x n) @ x (n)
//
// Compile-time variants (see Makefile):
//   (default)           naive port: every call cudaMallocs, copies W and x to the GPU,
//                       launches a one-thread-per-row kernel, copies the result back, frees.
//   -DRESIDENT_WEIGHTS  optimization 1: the whole checkpoint is copied to the GPU once at
//                       start-up; each call only copies the small vectors x and xout.
//                       Device buffers for x/xout are allocated once as well.
//   -DWARP_KERNEL       optional optimization 2: one warp per output row, coalesced loads
//                       of W and a shuffle reduction (can be combined with the above).
// Runtime option: VERIFY_MATMUL=1 checks every call against a CPU reference and prints
//                 the largest absolute / relative error at exit.

#include <cuda_runtime.h>

#include <cmath>
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

// One thread per output row. Simple, but neighbouring threads read addresses n floats apart.
__global__ void matmul_rows(float* xout, const float* x, const float* w, int n, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[size_t(i) * n + j] * x[j];
    }
    xout[i] = val;
}

// One warp per output row: lanes read consecutive elements of the row (coalesced) and the
// partial sums are combined with warp shuffles.
__global__ void matmul_warp(float* xout, const float* x, const float* w, int n, int d) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= d) return;                      // whole warp exits together
    const float* row = w + size_t(warp) * n;
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        sum += row[j] * x[j];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if (lane == 0) xout[warp] = sum;
}

static void launch(float* d_out, const float* d_x, const float* d_w, int n, int d) {
#ifdef WARP_KERNEL
    const int threads = 256;                    // 8 rows per block
    const int blocks = int((size_t(d) * 32 + threads - 1) / threads);
    matmul_warp<<<blocks, threads>>>(d_out, d_x, d_w, n, d);
#else
    const int threads = 256;
    const int blocks = (d + threads - 1) / threads;
    matmul_rows<<<blocks, threads>>>(d_out, d_x, d_w, n, d);
#endif
    CUDA_CHECK(cudaGetLastError());
}

// --- state for the resident-weights variant and for verification -------------------------
static const float* h_base = nullptr;           // host start of the (mmap'ed) checkpoint
static size_t h_floats = 0;
static float* d_base = nullptr;                 // device copy of the checkpoint
static float* d_x = nullptr;                    // persistent activation buffers
static float* d_out = nullptr;
static int cap_x = 0, cap_out = 0;

static bool verify = false;
static double max_abs_err = 0.0, max_rel_err = 0.0;
static long long verified_calls = 0;

extern "C" void matmul_gpu_init(const float* base, size_t bytes) {
    verify = getenv("VERIFY_MATMUL") != nullptr;
#ifdef RESIDENT_WEIGHTS
    h_base = base;
    h_floats = bytes / sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_base, bytes));
    CUDA_CHECK(cudaMemcpy(d_base, base, bytes, cudaMemcpyHostToDevice));
#else
    (void)base; (void)bytes;
#endif
}

static void check_against_cpu(const float* xout, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        double ref = 0.0;
        for (int j = 0; j < n; j++) ref += double(w[size_t(i) * n + j]) * double(x[j]);
        const double err = std::fabs(double(xout[i]) - ref);
        const double rel = err / (std::fabs(ref) + 1e-6);
        if (err > max_abs_err) max_abs_err = err;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    verified_calls++;
}

extern "C" void matmul_gpu(float* xout, const float* x, const float* w, int n, int d) {
#ifdef RESIDENT_WEIGHTS
    // w points into the checkpoint that was copied at init -> reuse the device copy
    if (d_base && w >= h_base && w + size_t(d) * n <= h_base + h_floats) {
        if (cap_x < n)   { if (d_x)   cudaFree(d_x);   CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));   cap_x = n; }
        if (cap_out < d) { if (d_out) cudaFree(d_out); CUDA_CHECK(cudaMalloc(&d_out, d * sizeof(float))); cap_out = d; }
        CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
        launch(d_out, d_x, d_base + (w - h_base), n, d);
        CUDA_CHECK(cudaMemcpy(xout, d_out, d * sizeof(float), cudaMemcpyDeviceToHost));
        if (verify) check_against_cpu(xout, x, w, n, d);
        return;
    }
#endif
    // naive path: allocate, copy everything, compute, copy back, free -- on every call
    float *dw = nullptr, *dx = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, size_t(d) * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dw, w, size_t(d) * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    launch(dout, dx, dw, n, d);
    CUDA_CHECK(cudaMemcpy(xout, dout, d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dout));
    if (verify) check_against_cpu(xout, x, w, n, d);
}

extern "C" void matmul_gpu_cleanup(void) {
    if (verify) {
        fprintf(stderr, "VERIFY: %lld matmul calls checked, max abs err %.3e, max rel err %.3e\n",
                verified_calls, max_abs_err, max_rel_err);
    }
    if (d_base) cudaFree(d_base);
    if (d_x) cudaFree(d_x);
    if (d_out) cudaFree(d_out);
    d_base = d_x = d_out = nullptr;
}
