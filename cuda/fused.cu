// fused.cu - kernel fusion: bias + GELU in one pass vs two separate kernels.
//
// Teaches:
//   * why fusion matters for memory-bound ops. For an M x N activation:
//       unfused: k1 reads X, writes Y = X + bias;  k2 reads Y, writes gelu(Y)
//                -> 4 * M*N*4 bytes of DRAM traffic, plus an extra launch
//       fused:   reads X, writes gelu(X + bias)
//                -> 2 * M*N*4 bytes, the intermediate Y never leaves registers
//     Both versions run near the same GB/s, so the fused one should take
//     roughly half the time. This is what torch.compile / XLA / Triton
//     epilogue fusion buys you.
//   * the extra math (tanh) is free: the kernel is still bandwidth-bound.
//
// GELU uses the tanh approximation (as in GPT-2 / torch "tanh" mode):
//   gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
//
// Usage: build/fused [rows] [cols]
#include "common.cuh"

// Shared by all kernels below (the CPU reference in main uses a separate
// double-precision formula). tanhf costs a few dozen instructions, but each
// element also moves 8+ bytes of DRAM traffic, so the kernels below stay
// memory-bound regardless.
__host__ __device__ inline float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Unfused step 1: y = x + bias, with bias broadcast across rows
// (x is rows x cols row-major, n = rows * cols, bias has `cols` entries).
// Mapping: 1D grid-stride loop over the flattened tensor (384 blocks x 256).
// Memory: x and y accesses are coalesced; bias[i % cols] is also contiguous
//   within a warp and, at 16 KiB, stays cache-resident. Reads X and writes Y:
//   2 * n * 4 bytes. Measured 1.041 ms / 516 GB/s (77% of peak,
//   docs/profiling/README.md).
__global__ void add_bias(const float* __restrict__ x, const float* __restrict__ bias,
                         float* __restrict__ y, size_t n, int cols) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        y[i] = x[i] + bias[i % cols];
}

// Unfused step 2: z = gelu(y), elementwise over n values. Same grid-stride
// mapping and coalesced access as add_bias. It re-reads the intermediate Y
// that add_bias just wrote to DRAM (256 MiB, too big for the L2), which is
// the traffic fusion removes. Measured 1.025 ms / 524 GB/s: the tanh math
// is hidden behind memory latency.
__global__ void gelu(const float* __restrict__ y, float* __restrict__ z, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        z[i] = gelu_tanh(y[i]);
}

// Fused: z = gelu(x + bias) in one pass. Same mapping and access pattern as
// add_bias, but the intermediate lives only in a register, so DRAM traffic is
// 2 * n * 4 bytes instead of 4 * n * 4 and there is one launch instead of two.
// Measured 1.087 ms vs 2.092 ms unfused: 1.93x, matching the 2x cut in bytes
// (cuda/README.md). It runs at the same ~500 GB/s as add_bias alone, so the
// extra math is free.
__global__ void bias_gelu_fused(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ z, size_t n, int cols) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        z[i] = gelu_tanh(x[i] + bias[i % cols]);  // bias is tiny and stays in L1/L2
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 16384;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 4096;
    const size_t n = (size_t)rows * cols, bytes = n * sizeof(float);
    print_device();
    std::printf("fused bias+GELU: %d x %d (%.0f MiB per tensor)\n\n", rows, cols,
                bytes / double(1 << 20));

    std::vector<float> hx(n), hb(cols), hz(n), ref(n);
    fill_random(hx, 6);
    fill_random(hb, 7);
    for (size_t i = 0; i < n; ++i) {
        double v = (double)hx[i] + hb[i % cols];
        ref[i] = (float)(0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))));
    }

    float *x, *bias, *y, *z;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));
    CUDA_CHECK(cudaMalloc(&z, bytes));
    CUDA_CHECK(cudaMalloc(&bias, cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, hb.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256, blocks = sms * 8;
    const double tol = 1e-5;

    bool all_ok = true;
    // Metric column is effective bandwidth = bytes each version must move / time.
    print_header("GB/s");
    // Copy z back, compare with the CPU double reference, print the row using
    // `moved` (bytes that version must transfer), then zero z.
    auto check = [&](const char* name, float ms, double moved) {
        CUDA_CHECK(cudaMemcpy(hz.data(), z, bytes, cudaMemcpyDeviceToHost));
        double err = max_rel_err(hz.data(), ref.data(), n);
        bool ok = err < tol;
        all_ok &= ok;
        print_row(name, ms, moved / (ms * 1e6), err, ok);
        CUDA_CHECK(cudaMemset(z, 0, bytes));
    };

    float ms_k1 = time_ms([&] { add_bias<<<blocks, threads>>>(x, bias, y, n, cols); });
    float ms_k2 = time_ms([&] { gelu<<<blocks, threads>>>(y, z, n); });
    print_ref_row("  add_bias alone", ms_k1, 2.0 * bytes / (ms_k1 * 1e6));
    print_ref_row("  gelu alone", ms_k2, 2.0 * bytes / (ms_k2 * 1e6));

    float ms_unfused = time_ms([&] {
        add_bias<<<blocks, threads>>>(x, bias, y, n, cols);
        gelu<<<blocks, threads>>>(y, z, n);
    });
    check("unfused (add_bias + gelu)", ms_unfused, 4.0 * bytes);

    float ms_fused = time_ms([&] { bias_gelu_fused<<<blocks, threads>>>(x, bias, z, n, cols); });
    check("fused bias_gelu", ms_fused, 2.0 * bytes);

    std::printf("\nDRAM traffic: unfused %.0f MiB, fused %.0f MiB; speedup %.2fx\n",
                4.0 * bytes / (1 << 20), 2.0 * bytes / (1 << 20), ms_unfused / ms_fused);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaFree(bias));
    std::printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
