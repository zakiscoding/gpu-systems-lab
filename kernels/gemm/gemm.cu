// Step-by-step GEMM on the RTX 5070: C = A * B, row-major, M = N = K (default 4096).
//
// FP32 (CUDA cores), each step changes one thing:
//   1 naive         one thread per C element; threadIdx.x walks rows, so a warp's loads of B are
//                   the same address (broadcast) and its loads of A stride by K: uncoalesced
//   2 coalesced     threadIdx.x walks columns: a warp reads 32 consecutive floats of B and writes
//                   32 consecutive floats of C
//   3 smem tiling   32x32 tiles of A and B staged in shared memory, each value reused 32 times
//   4 1D regtile    64x64 block tile, each thread computes 8 C values in a column (TM=8): one
//                   shared-memory load of B feeds 8 FMAs
//   5 2D regtile    128x128 block tile, each thread an 8x8 C sub-tile: an outer product of 8 A
//                   values and 8 B values per k, 64 FMAs per 16 shared loads
//   6 vectorized    like 5, with float4 global loads and A stored transposed in shared memory so
//                   the inner loop reads both operands with 128-bit loads
// FP16 inputs, FP32 accumulate (tensor cores):
//   7 wmma          each warp computes a 64x64 tile from 16x16x16 WMMA fragments, tiles staged
//                   in shared memory
// References: cuBLAS SGEMM (FP32, tensor cores off) and cuBLAS GEMM FP16->FP32 (tensor cores),
// and CUTLASS 2.x device::Gemm (SIMT FP32 and TensorOp FP16) when built with -DWITH_CUTLASS.
//
// Every kernel is checked against cuBLAS on the same inputs (max |diff| / max |ref|).
// Usage: gemm [N] [reps]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifdef WITH_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s, __FILE__, __LINE__); std::exit(2); } } while (0)

// ---------------------------------------------------------------- 1. naive
__global__ void k1_naive(int N, const float* A, const float* B, float* C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;   // x -> row: uncoalesced
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------- 2. coalesced
__global__ void k2_coalesced(int N, const float* A, const float* B, float* C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> col: coalesced B and C
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------- 3. shared-memory tiling
constexpr int T3 = 32;
__global__ void k3_smem(int N, const float* A, const float* B, float* C) {
    __shared__ float As[T3][T3], Bs[T3][T3];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T3 + ty, col = blockIdx.x * T3 + tx;
    float acc = 0.f;
    for (int t = 0; t < N; t += T3) {
        As[ty][tx] = A[row * N + t + tx];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T3; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------- 4. 1D register tiling
// block tile 64x64, K-step 8, 512 threads, each computes TM=8 rows of one column
template <int BM, int BN, int BK, int TM>
__global__ void k4_1d(int N, const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int tcol = threadIdx.x % BN, trow = threadIdx.x / BN;
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int ar = threadIdx.x / BK, ac = threadIdx.x % BK;    // 512 threads load 64x8 of A
    const int br = threadIdx.x / BN, bc = threadIdx.x % BN;    // and 8x64 of B
    float acc[TM] = {0.f};
    for (int t = 0; t < N; t += BK) {
        As[ar * BK + ac] = A[ar * N + ac];
        Bs[br * BN + bc] = B[br * N + bc];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + tcol];
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[i] += As[(trow * TM + i) * BK + k] * b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) C[(trow * TM + i) * N + tcol] = acc[i];
}

// ---------------------------------------------------------------- 5. 2D register tiling
// block tile 128x128, K-step 8, 256 threads, each computes an 8x8 sub-tile
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) k5_2d(int N, const float* A, const float* B, float* C) {
    constexpr int THREADS = (BM * BN) / (TM * TN);
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int tcol = threadIdx.x % (BN / TN), trow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float acc[TM * TN] = {0.f}, ra[TM], rb[TN];
    for (int t = 0; t < N; t += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += THREADS) As[i] = A[(i / BK) * N + i % BK];
        for (int i = threadIdx.x; i < BK * BN; i += THREADS) Bs[i] = B[(i / BN) * N + i % BN];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) ra[i] = As[(trow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k * BN + tcol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += ra[i] * rb[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) C[(trow * TM + i) * N + tcol * TN + j] = acc[i * TN + j];
}

// ---------------------------------------------------------------- 6. vectorized + transposed As
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) k6_vec(int N, const float* A, const float* B, float* C) {
    constexpr int THREADS = (BM * BN) / (TM * TN);
    __shared__ float As[BK * BM], Bs[BK * BN];          // As is stored transposed: As[k][m]
    const int tcol = threadIdx.x % (BN / TN), trow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float acc[TM * TN] = {0.f}, ra[TM], rb[TN];
    for (int t = 0; t < N; t += BK) {
        for (int i = threadIdx.x; i < BM * BK / 4; i += THREADS) {   // float4 along K, scatter transposed
            const int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
            const float4 v = reinterpret_cast<const float4*>(&A[r * N + c])[0];
            As[(c + 0) * BM + r] = v.x; As[(c + 1) * BM + r] = v.y;
            As[(c + 2) * BM + r] = v.z; As[(c + 3) * BM + r] = v.w;
        }
        for (int i = threadIdx.x; i < BK * BN / 4; i += THREADS) {
            const int r = i / (BN / 4), c = (i % (BN / 4)) * 4;
            reinterpret_cast<float4*>(&Bs[r * BN + c])[0] = reinterpret_cast<const float4*>(&B[r * N + c])[0];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = reinterpret_cast<const float4*>(&As[k * BM + trow * TM + i])[0];
                ra[i] = v.x; ra[i + 1] = v.y; ra[i + 2] = v.z; ra[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 v = reinterpret_cast<const float4*>(&Bs[k * BN + tcol * TN + j])[0];
                rb[j] = v.x; rb[j + 1] = v.y; rb[j + 2] = v.z; rb[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += ra[i] * rb[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; j += 4)
            reinterpret_cast<float4*>(&C[(trow * TM + i) * N + tcol * TN + j])[0] =
                make_float4(acc[i * TN + j], acc[i * TN + j + 1], acc[i * TN + j + 2], acc[i * TN + j + 3]);
}

// ---------------------------------------------------------------- 7. WMMA tensor cores
// block = 4 warps (2x2), block tile 128x128, each warp 64x64 = 4x4 fragments of 16x16, K-step 32
using namespace nvcuda;
constexpr int W_BM = 128, W_BN = 128, W_BK = 32;
__global__ void __launch_bounds__(128) k7_wmma(int N, const half* A, const half* B, float* C) {
    __shared__ half As[W_BM * W_BK], Bs[W_BK * W_BN];
    const int warp = threadIdx.x / 32, wm = warp / 2, wn = warp % 2;
    A += blockIdx.y * W_BM * N;
    B += blockIdx.x * W_BN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    for (int t = 0; t < N; t += W_BK) {
        // 128 threads copy 128x32 of A and 32x128 of B with 16-byte (8 half) loads
        for (int i = threadIdx.x; i < W_BM * W_BK / 8; i += 128) {
            const int r = i / (W_BK / 8), c = (i % (W_BK / 8)) * 8;
            reinterpret_cast<uint4*>(&As[r * W_BK + c])[0] = reinterpret_cast<const uint4*>(&A[r * N + c])[0];
        }
        for (int i = threadIdx.x; i < W_BK * W_BN / 8; i += 128) {
            const int r = i / (W_BN / 8), c = (i % (W_BN / 8)) * 8;
            reinterpret_cast<uint4*>(&Bs[r * W_BN + c])[0] = reinterpret_cast<const uint4*>(&B[r * N + c])[0];
        }
        __syncthreads();
        A += W_BK;
        B += W_BK * N;
#pragma unroll
        for (int k = 0; k < W_BK; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(fa[i], &As[(wm * 64 + i * 16) * W_BK + k], W_BK);
#pragma unroll
            for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], &Bs[k * W_BN + wn * 64 + j * 16], W_BN);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }
    float* Cb = C + (blockIdx.y * W_BM + wm * 64) * N + blockIdx.x * W_BN + wn * 64;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(&Cb[i * 16 * N + j * 16], acc[i][j], N, wmma::mem_row_major);
}

// ---------------------------------------------------------------- harness
static float time_ms(const std::function<void()>& f, int reps) {
    f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ts;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms);
    }
    CK(cudaGetLastError());
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

// max |x - ref| / max |ref|; NaN/Inf anywhere returns +inf (std::max silently drops NaN, which
// made a broken first version of this check report exactly 0)
static double rel_err(const std::vector<float>& x, const std::vector<float>& ref) {
    double md = 0, mr = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(ref[i])) return INFINITY;
        const double d = std::fabs((double)x[i] - (double)ref[i]);
        if (d > md) md = d;
        if (std::fabs((double)ref[i]) > mr) mr = std::fabs((double)ref[i]);
    }
    return mr > 0 ? md / mr : INFINITY;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 15;
    if (N % 128) { std::fprintf(stderr, "N must be a multiple of 128\n"); return 1; }
    const size_t n2 = (size_t)N * N;
    const double flops = 2.0 * N * N * (double)N;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("{\"device\":\"%s\",\"sm\":\"%d.%d\",\"N\":%d,\"reps\":%d,\"results\":[\n", p.name, p.major, p.minor, N, reps);

    std::vector<float> hA(n2), hB(n2), hC(n2), ref32(n2), ref16(n2);
    std::mt19937 g(0);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& v : hA) v = u(g);
    for (auto& v : hB) v = u(g);
    std::vector<half> hA16(n2), hB16(n2);
    for (size_t i = 0; i < n2; ++i) { hA16[i] = __float2half(hA[i]); hB16[i] = __float2half(hB[i]); }

    float *A, *B, *C;
    half *A16, *B16;
    CK(cudaMalloc(&A, n2 * 4)); CK(cudaMalloc(&B, n2 * 4)); CK(cudaMalloc(&C, n2 * 4));
    CK(cudaMalloc(&A16, n2 * 2)); CK(cudaMalloc(&B16, n2 * 2));
    CK(cudaMemcpy(A, hA.data(), n2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, hB.data(), n2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A16, hA16.data(), n2 * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B16, hB16.data(), n2 * 2, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CB(cublasCreate(&h));
    const float one = 1.f, zero = 0.f;
    // row-major C = A*B  ==  column-major C^T = B^T * A^T: pass B first
    auto sgemm = [&] {
        CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));        // true FP32, no TF32 tensor cores
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B, N, A, N, &zero, C, N));
    };
    auto hgemm = [&] {
        CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B16, CUDA_R_16F, N, A16, CUDA_R_16F, N,
                        &zero, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    sgemm(); CK(cudaMemcpy(ref32.data(), C, n2 * 4, cudaMemcpyDeviceToHost));
    hgemm(); CK(cudaMemcpy(ref16.data(), C, n2 * 4, cudaMemcpyDeviceToHost));

    std::fprintf(stderr, "ref32[0]=%g ref32[n/2]=%g ref16[0]=%g\n", ref32[0], ref32[n2 / 2], ref16[0]);
    bool first = true;
    auto report = [&](const std::string& name, const std::string& prec, const std::function<void()>& f,
                      const std::vector<float>* ref) {
        CK(cudaMemset(C, 0, n2 * 4));
        const float ms = time_ms(f, reps);
        CK(cudaMemcpy(hC.data(), C, n2 * 4, cudaMemcpyDeviceToHost));
        const double err = ref ? rel_err(hC, *ref) : 0.0;
        std::printf("%s{\"kernel\":\"%s\",\"precision\":\"%s\",\"ms\":%.4f,\"tflops\":%.3f,\"rel_err_vs_cublas\":%.3e}\n",
                    first ? "" : ",", name.c_str(), prec.c_str(), ms, flops / (ms * 1e9), err);
        std::fflush(stdout);
        first = false;
    };

    const dim3 b16(16, 16), g16(N / 16, N / 16);
    report("1 naive", "fp32", [&] { k1_naive<<<g16, b16>>>(N, A, B, C); }, &ref32);
    report("2 coalesced", "fp32", [&] { k2_coalesced<<<g16, b16>>>(N, A, B, C); }, &ref32);
    report("3 smem tiling", "fp32", [&] { k3_smem<<<dim3(N / T3, N / T3), dim3(T3, T3)>>>(N, A, B, C); }, &ref32);
    report("4 1D register tiling", "fp32", [&] { k4_1d<64, 64, 8, 8><<<dim3(N / 64, N / 64), 512>>>(N, A, B, C); }, &ref32);
    report("5 2D register tiling", "fp32", [&] { k5_2d<128, 128, 8, 8, 8><<<dim3(N / 128, N / 128), 256>>>(N, A, B, C); }, &ref32);
    report("6 vectorized", "fp32", [&] { k6_vec<128, 128, 8, 8, 8><<<dim3(N / 128, N / 128), 256>>>(N, A, B, C); }, &ref32);
    report("cuBLAS SGEMM", "fp32", sgemm, nullptr);
    report("7 wmma tensor cores", "fp16->fp32", [&] { k7_wmma<<<dim3(N / W_BN, N / W_BM), 128>>>(N, A16, B16, C); }, &ref16);
    report("cuBLAS GEMM fp16", "fp16->fp32", hgemm, nullptr);

#ifdef WITH_CUTLASS
    {   // CUTLASS 2.x device-level GEMMs with default tile configs (row-major A/B/C)
        using RM = cutlass::layout::RowMajor;
        using Simt = cutlass::gemm::device::Gemm<float, RM, float, RM, float, RM, float,
                                                 cutlass::arch::OpClassSimt, cutlass::arch::Sm50>;
        Simt simt;
        report("CUTLASS SIMT fp32", "fp32", [&] {
            Simt::Arguments args({N, N, N}, {A, N}, {B, N}, {C, N}, {C, N}, {1.f, 0.f});
            const cutlass::Status st = simt(args);
            if (st != cutlass::Status::kSuccess) { std::fprintf(stderr, "CUTLASS SIMT: %s\n", cutlassGetStatusString(st)); std::exit(3); }
        }, &ref32);
        // explicit 128x128x32 tile, 3 stages (48 KB smem): the default Sm80 config (128x256x64)
        // asks for more shared memory per block than sm_120 allows and fails with "Error Internal"
        using TC = cutlass::gemm::device::Gemm<
            cutlass::half_t, RM, cutlass::half_t, RM, float, RM, float,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
        TC tc;
        report("CUTLASS TensorOp fp16", "fp16->fp32", [&] {
            TC::Arguments args({N, N, N}, {reinterpret_cast<cutlass::half_t*>(A16), N},
                               {reinterpret_cast<cutlass::half_t*>(B16), N}, {C, N}, {C, N}, {1.f, 0.f});
            const cutlass::Status st = tc(args);
            if (st != cutlass::Status::kSuccess) { std::fprintf(stderr, "CUTLASS TensorOp: %s\n", cutlassGetStatusString(st)); std::exit(3); }
        }, &ref16);
    }
#endif
    std::printf("]}\n");
    cublasDestroy(h);
    return 0;
}
