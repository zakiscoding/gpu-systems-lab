// Shared helpers for the roadmap step 2 CUDA kernels.
//
// Kept deliberately tiny: an error-check macro, a cudaEvent-based timer,
// a deterministic RNG fill, and table-printing helpers so every program
// reports results in the same compact format.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Wrap every CUDA runtime call. Kernel launches are checked with
// CUDA_CHECK(cudaGetLastError()) after the <<<>>> launch (time_ms does this).
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

// GPU-side timing with cudaEvents. Events are recorded into the stream, so
// the measurement covers device work between the two events.
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Run `fn` a few times to warm up (clocks, caches, lazy module loading), then
// return the average milliseconds per call over `iters` timed calls.
template <typename F>
float time_ms(F fn, int iters = 20, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) fn();
    float ms = t.stop_ms();
    CUDA_CHECK(cudaGetLastError());
    return ms / iters;
}

// Deterministic pseudo-random floats in [-1, 1) (xorshift32), so CPU and GPU
// see identical inputs and runs are reproducible.
inline void fill_random(std::vector<float>& v, unsigned seed = 1234) {
    unsigned s = seed * 2654435761u + 1u;
    for (auto& x : v) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x = (s & 0xFFFFFF) / float(0x800000) - 1.0f;
    }
}

// Max relative error, with a floor of 1 on the denominator so values near
// zero do not blow up the ratio.
inline double max_rel_err(const float* a, const float* b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double denom = std::fmax(1.0, std::fabs((double)b[i]));
        double e = std::fabs((double)a[i] - b[i]) / denom;
        if (std::isnan(e)) return INFINITY;  // fmax would silently drop a NaN
        worst = std::fmax(worst, e);
    }
    return worst;
}

inline void print_device() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)  CUDA runtime %d.%d, driver API %d.%d\n",
                p.name, p.major, p.minor, p.multiProcessorCount,
                p.totalGlobalMem / double(1 << 30), rt / 1000, (rt % 1000) / 10,
                drv / 1000, (drv % 1000) / 10);
}

inline void print_header(const char* metric) {
    std::printf("%-28s %10s %12s %10s %4s\n", "kernel", "ms", metric, "max_err", "ok");
    std::printf("%-28s %10s %12s %10s %4s\n", "----------------------------", "----------",
                "------------", "----------", "----");
}

inline void print_row(const char* name, float ms, double metric, double err, bool ok) {
    std::printf("%-28s %10.4f %12.1f %10.2e %4s\n", name, ms, metric, err, ok ? "yes" : "NO");
}

inline void print_ref_row(const char* name, float ms, double metric) {
    std::printf("%-28s %10.4f %12.1f %10s %4s\n", name, ms, metric, "-", "-");
}
