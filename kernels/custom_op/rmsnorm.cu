// Fused RMSNorm as a PyTorch custom operator: labops::rmsnorm(x, weight, eps) -> y
//
//   y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * weight
//
// One thread block per row. Each thread accumulates the sum of squares of its strided slice in
// fp32, a warp-shuffle + shared-memory reduction gives the row's inverse RMS, then the same
// threads re-read the row (it is still in L1/L2 for typical hidden sizes) and write the output.
// Inputs are fp32, fp16 or bf16 (accumulation is always fp32); rows must be contiguous.
//
// Registered with TORCH_LIBRARY so it is a real dispatcher op (torch.ops.labops.rmsnorm). The
// Python side (rmsnorm_op.py) adds a fake/meta kernel with torch.library.register_fake, which is
// what lets torch.compile trace through it without graph breaks.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T> __device__ __forceinline__ float to_f(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half from_f(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
                               int cols, float eps) {
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = to_f(xr[c]);
        ss += v * v;
    }
    __shared__ float partial[32];
    ss = warp_sum(ss);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) partial[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.f;
        ss = warp_sum(ss);
        if (lane == 0) partial[0] = rsqrtf(ss / cols + eps);
    }
    __syncthreads();
    const float inv = partial[0];
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = from_f<T>(to_f(xr[c]) * inv * to_f(w[c]));
}

// v2: 16-byte vector loads/stores (8 bf16/fp16 or 4 fp32 per access) and the row kept in
// registers between the two passes, so x is read from DRAM once. Used when the hidden size is a
// multiple of the vector width, the pointers are 16-byte aligned and the row fits in
// blockDim (<= 1024) x MAXV vectors; otherwise the scalar kernel above runs.
constexpr int MAXV = 8;

template <typename T>
__global__ void __launch_bounds__(512) rmsnorm_vec_kernel(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
                                   int cols, float eps) {
    constexpr int VEC = 16 / sizeof(T);
    const int nvec = cols / VEC;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)blockIdx.x * cols);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    uint4* yr = reinterpret_cast<uint4*>(y + (size_t)blockIdx.x * cols);
    uint4 cache[MAXV];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < nvec) {
            cache[k] = xr[i];
            const T* e = reinterpret_cast<const T*>(&cache[k]);
#pragma unroll
            for (int j = 0; j < VEC; ++j) { const float f = to_f(e[j]); ss += f * f; }
        }
    }
    __shared__ float partial[32];
    ss = warp_sum(ss);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) partial[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.f;
        ss = warp_sum(ss);
        if (lane == 0) partial[0] = rsqrtf(ss / cols + eps);
    }
    __syncthreads();
    const float inv = partial[0];
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < nvec) {
            const uint4 wq = wv[i];
            uint4 out;
            const T* e = reinterpret_cast<const T*>(&cache[k]);
            const T* we = reinterpret_cast<const T*>(&wq);
            T* o = reinterpret_cast<T*>(&out);
#pragma unroll
            for (int j = 0; j < VEC; ++j) o[j] = from_f<T>(to_f(e[j]) * inv * to_f(we[j]));
            yr[i] = out;
        }
    }
}

template <typename T>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, double eps, bool allow_vec) {
    const int cols = x.size(-1);
    const int rows = x.numel() / cols;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int VEC = 16 / sizeof(T);
    const int nvec = cols / VEC;
    const bool aligned = (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0);
    if (allow_vec && cols % VEC == 0 && aligned && nvec <= 512 * MAXV) {
        // vectors per thread (VPT): 1 thread per VPT vectors, <= 512 threads (the register
        // cache of MAXV uint4 per thread does not fit 1024 threads: "too many resources").
        // LABOPS_RMSNORM_VPT overrides the default for the tuning sweep in bench_rmsnorm.py.
        static const int vpt_env = [] { const char* e = std::getenv("LABOPS_RMSNORM_VPT"); return e ? std::atoi(e) : 0; }();
        const int vpt = std::min(MAXV, std::max(vpt_env > 0 ? vpt_env : 2, (nvec + 511) / 512));
        const int threads = std::min(512, std::max(32, (((nvec + vpt - 1) / vpt + 31) / 32) * 32));
        rmsnorm_vec_kernel<T><<<rows, threads, 0, stream>>>(
            reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
            reinterpret_cast<T*>(y.data_ptr()), cols, static_cast<float>(eps));
    } else {
        const int threads = cols >= 4096 ? 1024 : cols >= 1024 ? 512 : 256;
        rmsnorm_kernel<T><<<rows, threads, 0, stream>>>(
            reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
            reinterpret_cast<T*>(y.data_ptr()), cols, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor rmsnorm_impl(const at::Tensor& x, const at::Tensor& weight, double eps, bool allow_vec) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "rmsnorm: CUDA tensors expected");
    TORCH_CHECK(x.is_contiguous(), "rmsnorm: x must be contiguous");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1), "rmsnorm: weight must be [hidden]");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "rmsnorm: x and weight dtypes differ");
    const c10::cuda::CUDAGuard guard(x.device());
    auto w = weight.contiguous();
    auto y = at::empty_like(x);
    switch (x.scalar_type()) {
        case at::kFloat: launch<float>(x, w, y, eps, allow_vec); break;
        case at::kHalf: launch<__half>(x, w, y, eps, allow_vec); break;
        case at::kBFloat16: launch<__nv_bfloat16>(x, w, y, eps, allow_vec); break;
        default: TORCH_CHECK(false, "rmsnorm: unsupported dtype ", x.scalar_type());
    }
    return y;
}

at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
    return rmsnorm_impl(x, weight, eps, /*allow_vec=*/true);
}
at::Tensor rmsnorm_naive_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
    return rmsnorm_impl(x, weight, eps, /*allow_vec=*/false);
}

}  // namespace

TORCH_LIBRARY(labops, m) {
    m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("rmsnorm_naive(Tensor x, Tensor weight, float eps) -> Tensor");  // v1, for comparison
}

TORCH_LIBRARY_IMPL(labops, CUDA, m) {
    m.impl("rmsnorm", &rmsnorm_cuda);
    m.impl("rmsnorm_naive", &rmsnorm_naive_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // the op is reached through torch.ops.labops
