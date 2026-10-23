// vector_add.cu - the "hello world" of CUDA, used as a bandwidth baseline.
//
// Teaches:
//   * thread indexing: global id = blockIdx.x * blockDim.x + threadIdx.x
//   * the grid-stride loop, which decouples grid size from problem size
//     (any N works with any launch config; we size the grid to a few
//     resident blocks per SM instead of N / blockDim)
//   * effective bandwidth: c = a + b moves 3 * N * 4 bytes (2 reads, 1 write),
//     so GB/s = 12 * N / time. This kernel is purely memory-bound; compare it
//     to the DRAM peak (RTX 5070: 192-bit GDDR7 @ 28 Gbps = 672 GB/s).
//
// Usage: build/vector_add [n_floats]
#include "common.cuh"

// c[i] = a[i] + b[i] for i in [0, n).
// Mapping: 1D grid of 1D blocks; thread g = blockIdx.x * blockDim.x +
//   threadIdx.x handles elements g, g + stride, g + 2*stride, ... where
//   stride = total threads in the grid (grid-stride loop). Launched with
//   48 SMs * 8 blocks * 256 threads, so each thread loops ~680 times at 64M.
// Memory: consecutive lanes touch consecutive floats, so every warp access is
//   one fully coalesced 128-byte transaction per array. No reuse, no shared
//   memory; 1 FLOP per 12 bytes makes it purely DRAM-bound.
// Measured (cuda/README.md): 569 GB/s at 64M floats, ~85% of the 672 GB/s
//   peak and on par with a D2D cudaMemcpy (578 GB/s).
__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Same thing with float4: each thread moves 16 bytes per load/store, so the
// GPU issues 4x fewer memory instructions for the same traffic.
// Mapping: identical grid-stride loop, but over n4 = n / 4 float4 elements.
//   Requires 16-byte-aligned pointers (cudaMalloc guarantees 256 B) and
//   n % 4 == 0 (main rounds n down).
// Memory: each warp moves 512 contiguous bytes per array per iteration; the
//   const __restrict__ inputs compile to ld.global.nc.v4 (read-only path).
// Measured: 590 GB/s vs 569 GB/s scalar, only ~4% faster, because the scalar
//   kernel was already bandwidth-bound; fewer instructions help little once
//   DRAM is the limit.
__global__ void vector_add_f4(const float4* __restrict__ a, const float4* __restrict__ b,
                              float4* __restrict__ c, size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (size_t(1) << 26);  // 64M floats
    n &= ~size_t(3);  // keep divisible by 4 for the float4 variant
    const size_t bytes = n * sizeof(float);
    print_device();
    std::printf("vector_add: n = %zu (%.0f MiB per array)\n\n", n, bytes / double(1 << 20));

    std::vector<float> ha(n), hb(n), hc(n), ref(n);
    fill_random(ha, 1);
    fill_random(hb, 2);
    for (size_t i = 0; i < n; ++i) ref[i] = ha[i] + hb[i];

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemcpy(a, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, hb.data(), bytes, cudaMemcpyHostToDevice));

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256;
    const int blocks = sms * 8;  // a few resident blocks per SM; the loop covers the rest

    bool all_ok = true;
    print_header("GB/s");
    // Copy c back, compare with the CPU reference, print the row (GB/s =
    // 3 arrays * bytes / time), then zero c so the next kernel must rewrite it.
    auto check = [&](const char* name, float ms) {
        CUDA_CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));
        double err = max_rel_err(hc.data(), ref.data(), n);
        bool ok = err == 0.0;  // a single float add is bit-exact on CPU and GPU
        all_ok &= ok;
        print_row(name, ms, 3.0 * bytes / (ms * 1e6), err, ok);
        CUDA_CHECK(cudaMemset(c, 0, bytes));
    };

    float ms = time_ms([&] { vector_add<<<blocks, threads>>>(a, b, c, n); });
    check("vector_add (grid-stride)", ms);

    ms = time_ms([&] {
        vector_add_f4<<<blocks, threads>>>((const float4*)a, (const float4*)b, (float4*)c, n / 4);
    });
    check("vector_add_f4 (float4)", ms);

    // Reference point: a device-to-device copy reads N and writes N bytes.
    ms = time_ms([&] { CUDA_CHECK(cudaMemcpyAsync(c, a, bytes, cudaMemcpyDeviceToDevice)); });
    print_ref_row("cudaMemcpy D2D (ref)", ms, 2.0 * bytes / (ms * 1e6));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
