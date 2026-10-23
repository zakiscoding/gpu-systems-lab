// Shared helpers for the MPI experiments (roadmap step 5, "MPI second").
//
// Reuses CUDA_CHECK / print_device from cuda/common.cuh and adds the MPI
// bits: CUDA-awareness detection, a rank-0 printf, and the NCCL-style
// bus-bandwidth formula so MPI numbers line up with the NCCL tests.
#pragma once

#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>  // Open MPI extensions: MPIX_Query_cuda_support
#endif

#include <cstdarg>
#include <cstdio>

#include "../cuda/common.cuh"

#define MPI_CHECK(call)                                                       \
    do {                                                                      \
        int err_ = (call);                                                    \
        if (err_ != MPI_SUCCESS) {                                            \
            char msg_[MPI_MAX_ERROR_STRING];                                  \
            int len_ = 0;                                                     \
            MPI_Error_string(err_, msg_, &len_);                              \
            std::fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, msg_);                                     \
            MPI_Abort(MPI_COMM_WORLD, 2);                                     \
        }                                                                     \
    } while (0)

// True when the MPI library can take device pointers directly (CUDA-aware).
// Compile-time support is not enough: the runtime check catches builds where
// the CUDA component is present but disabled.
inline bool mpi_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() == 1;
#else
    return false;
#endif
}

inline int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

inline int world_size() {
    int n = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

// One GPU per rank on multi-GPU nodes: bind each rank to (local rank % GPU
// count). On a single-GPU node every rank lands on device 0, as before.
// Call right after MPI_Init. Returns the number of GPUs visible on this node.
inline int bind_local_gpu() {
    const char* lr = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local = lr ? std::atoi(lr) : 0, ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(local % ndev));
    return ndev;
}

// printf on rank 0 only, so tables are not duplicated per rank.
inline void root_printf(const char* fmt, ...) {
    if (world_rank() != 0) return;
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

// Same convention as nccl-tests: algbw = bytes / time, and for all-reduce
// busbw = algbw * 2(n-1)/n, the per-link traffic of a ring algorithm. Using it
// here makes MPI and NCCL all-reduce numbers directly comparable.
inline double allreduce_busbw(double algbw, int n) { return algbw * 2.0 * (n - 1) / n; }

inline const char* human_bytes(size_t b, char* buf, size_t len) {
    if (b >= (1u << 20)) std::snprintf(buf, len, "%zu MiB", b >> 20);
    else if (b >= (1u << 10)) std::snprintf(buf, len, "%zu KiB", b >> 10);
    else std::snprintf(buf, len, "%zu B", b);
    return buf;
}
