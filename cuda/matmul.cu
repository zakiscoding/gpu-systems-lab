// matmul.cu - naive vs shared-memory tiled SGEMM, checked against cuBLAS.
//
// Teaches:
//   * arithmetic intensity: C = A * B for N x N does 2*N^3 FLOPs on 3*N^2
//     values. The naive kernel re-reads A and B from global memory N times
//     each, so it is memory/latency bound far below the FLOP peak.
//   * shared-memory tiling: a block cooperatively loads a TILE x TILE patch
//     of A and of B into __shared__ memory, syncs, then every thread reuses
//     those values TILE times. Global traffic drops by ~TILE x.
//   * __syncthreads(): one barrier after loading (tile is complete) and one
//     after computing (nobody overwrites a tile another thread still reads).
//   * the gap to cuBLAS: register tiling (each thread computes several outputs),
//     vectorized loads, double buffering and tensor-core paths are what take a
//     GEMM from "tiled" to "vendor library" speed.
//
// All matrices are row-major. Correctness: cuBLAS is compared to a CPU double
// reference on sampled entries, then every kernel is compared to cuBLAS in full.
//
// Usage: build/matmul [N]   (N must be a multiple of 32; default 4096)
#include <cublas_v2.h>

#include "common.cuh"

// cuBLAS counterpart of CUDA_CHECK: print the status code and exit(2).
#define CUBLAS_CHECK(call)                                                          \
    do {                                                                            \
        cublasStatus_t st_ = (call);                                                \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, \
                         __LINE__);                                                 \
            std::exit(2);                                                           \
        }                                                                           \
    } while (0)

// One thread per output element; walks a full row of A and column of B.
// Threads in a warp share `row` and have consecutive `col`, so B[k*n + col]
// is coalesced and A[row*n + k] is a broadcast, but nothing is reused.
// Computes C = A * B (n x n, row-major). Mapping: 2D grid of 16x16 blocks,
//   (n/16) x (n/16) blocks, thread (x, y) -> C[row = y][col = x]; the bounds
//   check makes any n safe.
// Cost: 2 global loads per FMA (intensity 0.25 FLOP/byte as written). Each
//   thread requests 2n floats, ~550 GB per call at N = 4096 = 8.2 TB/s at the
//   measured 67 ms, 12x the DRAM peak, so L1/L2 absorb most of it
//   (docs/profiling/README.md). Measured 1,962-2,047 GFLOPS, ~9-11% of cuBLAS.
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
}

// Classic shared-memory tiling. Block = TILE x TILE threads, one output each.
// Assumes n % TILE == 0 to keep the teaching version free of edge handling.
// Mapping: grid (n/TILE) x (n/TILE), block TILE x TILE; thread (tx, ty) owns
//   C[blockIdx.y*TILE + ty][blockIdx.x*TILE + tx]. The block walks the k
//   dimension in n/TILE phases; each phase stages one TILE x TILE tile of A
//   and of B in shared memory (2 * TILE^2 * 4 B: 2 KiB at 16, 8 KiB at 32).
// Memory: global loads are coalesced along tx (a warp reads one contiguous
//   row segment of each tile). In the inner loop As[ty][k] is the same
//   address for the whole warp row (broadcast) and Bs[k][tx] hits
//   consecutive banks, so no bank conflicts. Global traffic drops ~TILE x,
//   but every FMA still needs 2 shared-memory loads, so the inner loop is
//   load/store-unit bound, not FMA bound.
// Measured at N = 4096 (cuda/README.md, docs/profiling/README.md):
//   TILE=16  51.6 ms, 2,664 GFLOPS: only ~1.4x over naive (caches already
//            gave the naive kernel most of this reuse).
//   TILE=32  59.8 ms, 2,300 GFLOPS: slower, since a 1024-thread block lets
//            only one block fit on an SM (1536-thread limit, 67% occupancy)
//            and each __syncthreads() stalls the whole SM.
//   cuBLAS is ~8x faster with plain SIMT FP32 by giving each thread 128
//   outputs (register tiling): ~5 FMAs per load instead of 0.5.
template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int t = 0; t < n; t += TILE) {
        // Each thread loads one element of each tile (coalesced along tx).
        As[ty][tx] = A[row * n + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * n + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * n + col] = acc;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 4096;
    if (n <= 0 || n % 32 != 0) {
        std::fprintf(stderr, "N must be a positive multiple of 32\n");
        return 2;
    }
    const size_t elems = (size_t)n * n, bytes = elems * sizeof(float);
    const double flops = 2.0 * n * n * (double)n;
    print_device();
    std::printf("matmul: N = %d (FP32, row-major)\n\n", n);

    std::vector<float> hA(elems), hB(elems), hC(elems), ref(elems);
    fill_random(hA, 3);
    fill_random(hB, 4);

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep cuBLAS in strict FP32 (no TF32) so it is a fair correctness reference.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.f, beta = 0.f;
    // cuBLAS is column-major. A row-major matrix is its column-major transpose,
    // so computing C^T = B^T * A^T in column-major yields row-major C = A * B.
    auto run_cublas = [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n,
                                 &beta, C, n));
    };

    // 1) Validate cuBLAS itself against a CPU double-precision reference on a
    //    sample of entries (a full CPU GEMM at N=4096 would take minutes).
    run_cublas();
    CUDA_CHECK(cudaMemcpy(ref.data(), C, bytes, cudaMemcpyDeviceToHost));
    double cpu_err = 0.0;
    unsigned s = 99;
    for (int sample = 0; sample < 512; ++sample) {
        s = s * 1664525u + 1013904223u;
        int r = (s >> 8) % n;
        s = s * 1664525u + 1013904223u;
        int c = (s >> 8) % n;
        double acc = 0.0;
        for (int k = 0; k < n; ++k) acc += (double)hA[(size_t)r * n + k] * hB[(size_t)k * n + c];
        float got = ref[(size_t)r * n + c];
        cpu_err = std::fmax(cpu_err, std::fabs(got - acc) / std::fmax(1.0, std::fabs(acc)));
    }
    const double tol = 1e-3;  // FP32 accumulation over K=4096 terms of magnitude <= 1
    bool all_ok = cpu_err < tol;

    print_header("GFLOPS");
    float ms = time_ms(run_cublas);
    print_row("cuBLAS sgemm (reference)", ms, flops / (ms * 1e6), cpu_err, cpu_err < tol);

    // Copy C back, compare in full with the cuBLAS result (`ref`), print the
    // row, then re-poison C with NaN for the next kernel.
    auto check = [&](const char* name, float t) {
        CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
        double err = max_rel_err(hC.data(), ref.data(), elems);
        bool ok = err < tol;
        all_ok &= ok;
        print_row(name, t, flops / (t * 1e6), err, ok);
        CUDA_CHECK(cudaMemset(C, 0xFF, bytes));
    };

    // Poison C (all-0xFF bytes = NaN) so a kernel that silently fails to write
    // cannot pass by leaving the cuBLAS result in place.
    CUDA_CHECK(cudaMemset(C, 0xFF, bytes));
    dim3 block16(16, 16), grid16(n / 16, n / 16);
    dim3 block32(32, 32), grid32(n / 32, n / 32);

    ms = time_ms([&] { matmul_naive<<<grid16, block16>>>(A, B, C, n); }, 5, 1);
    check("naive (16x16 block)", ms);

    ms = time_ms([&] { matmul_tiled<16><<<grid16, block16>>>(A, B, C, n); }, 10, 2);
    check("tiled TILE=16", ms);

    ms = time_ms([&] { matmul_tiled<32><<<grid32, block32>>>(A, B, C, n); }, 10, 2);
    check("tiled TILE=32", ms);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    std::printf("\nerrors vs cuBLAS (tiled rows) / vs CPU double (cuBLAS row); tol %.0e\n", tol);
    std::printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
