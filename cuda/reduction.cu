// reduction.cu - three ways to sum an array, from worst to good.
//
// Teaches:
//   1. naive atomic: every element does atomicAdd on ONE address. Correct,
//      but all N updates serialize at a single L2 location.
//   2. shared-memory tree: each thread first sums many elements privately
//      (grid-stride), then the block halves the active threads each step
//      through __shared__ memory with a __syncthreads() per level. One atomic
//      per block instead of one per element.
//   3. warp shuffle: __shfl_down_sync lets the 32 lanes of a warp exchange
//      registers directly, no shared memory or block barrier needed. Only one
//      value per warp goes through shared memory. This is how CUB/cuBLAS do it.
//   Plus cub::DeviceReduce::Sum as the library reference.
//
// A sum reads N * 4 bytes, so it is memory-bound: GB/s = 4 * N / time.
// FP32 summation order differs between methods, so results differ slightly;
// error is reported relative to sum(|x|), the natural bound for rounding error.
//
// Usage: build/reduction [n_floats]
#include <cub/cub.cuh>

#include "common.cuh"

constexpr int THREADS = 256;

__global__ void reduce_atomic_naive(const float* __restrict__ x, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}

__global__ void reduce_smem_tree(const float* __restrict__ x, float* out, size_t n) {
    __shared__ float sdata[THREADS];
    float sum = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        sum += x[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    // Tree: 256 -> 128 -> ... -> 1. Sequential addressing (tid < s) keeps the
    // active threads contiguous, so whole warps retire early (no divergence)
    // and there are no shared-memory bank conflicts.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

__device__ __forceinline__ float warp_sum(float v) {
    // After offsets 16, 8, 4, 2, 1 lane 0 holds the sum of all 32 lanes.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void reduce_warp_shuffle(const float* __restrict__ x, float* out, size_t n) {
    __shared__ float warp_sums[THREADS / 32];
    float sum = 0.f;
    // float4 loads: 4 elements per memory instruction. Assumes n % 4 == 0.
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = x4[i];
        sum += (v.x + v.y) + (v.z + v.w);
    }
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    sum = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < THREADS / 32 ? warp_sums[lane] : 0.f;
        sum = warp_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (size_t(1) << 25);  // 32M floats
    n &= ~size_t(3);
    const size_t bytes = n * sizeof(float);
    print_device();
    std::printf("reduction: n = %zu (%.0f MiB)\n\n", n, bytes / double(1 << 20));

    std::vector<float> hx(n);
    fill_random(hx, 5);
    double ref = 0.0, l1 = 0.0;
    for (float v : hx) {
        ref += v;
        l1 += std::fabs(v);
    }

    float *x, *out;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), bytes, cudaMemcpyHostToDevice));

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = sms * 8;
    const double tol = 1e-6;  // |gpu - ref| / sum|x|

    bool all_ok = true;
    print_header("GB/s");
    // Time `launch` (which must zero `out` first), then run once more and check.
    auto bench = [&](const char* name, auto launch, int iters) {
        float ms = time_ms(launch, iters, 1);
        launch();
        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
        double err = std::fabs(got - ref) / l1;
        bool ok = err < tol;
        all_ok &= ok;
        print_row(name, ms, bytes / (ms * 1e6), err, ok);
    };

    bench("atomic (1 per element)", [&] {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float)));
        reduce_atomic_naive<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS>>>(x, out, n);
    }, 3);

    bench("smem tree (1 atomic/block)", [&] {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float)));
        reduce_smem_tree<<<blocks, THREADS>>>(x, out, n);
    }, 20);

    bench("warp shuffle + float4", [&] {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float)));
        reduce_warp_shuffle<<<blocks, THREADS>>>(x, out, n);
    }, 20);

    // CUB: first call with null temp storage just reports how much it needs.
    void* temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, x, out, (int)n));
    CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    bench("cub::DeviceReduce::Sum", [&] {
        CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, x, out, (int)n));
    }, 20);

    CUDA_CHECK(cudaFree(temp));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(out));
    std::printf("\nCPU double sum = %.6f, sum|x| = %.1f; err = |gpu - cpu| / sum|x|, tol %.0e\n",
                ref, l1, tol);
    std::printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
