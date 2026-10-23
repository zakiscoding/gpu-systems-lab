// Ping-pong between two ranks: the classic MPI latency/bandwidth benchmark.
//
// Three buffer paths, to show what "CUDA-aware MPI" buys you:
//   host    pinned host buffers, plain MPI_Send/MPI_Recv
//   staged  data lives on the GPU; each hop copies D2H, sends, then H2D.
//           This is what you must do by hand with a non-CUDA-aware MPI.
//   direct  device pointers handed straight to MPI (only if CUDA-aware).
//
// Latency is half the round-trip time; bandwidth is bytes / one-way time.
// Usage: mpirun -np 2 build/pingpong [max_mib]
#include <cstring>
#include <vector>

#include "mpi_common.cuh"

enum class Path { Host, Staged, Direct };

static const char* path_name(Path p) {
    switch (p) {
        case Path::Host: return "host";
        case Path::Staged: return "staged";
        default: return "direct";
    }
}

// Returns microseconds per one-way message. Rank 0 sends then receives,
// rank 1 receives then sends; the payload is checked after the last hop.
static double pingpong(Path path, size_t bytes, int iters, char* host, char* dev, bool* ok) {
    const int rank = world_rank();
    const int peer = 1 - rank;
    char* buf = (path == Path::Direct) ? dev : host;

    auto hop_send = [&] {
        if (path == Path::Staged) CUDA_CHECK(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
        MPI_CHECK(MPI_Send(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD));
    };
    auto hop_recv = [&] {
        MPI_CHECK(MPI_Recv(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        if (path == Path::Staged) CUDA_CHECK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    };

    // rank 0 owns the pattern; rank 1 echoes whatever it receives
    std::vector<char> expect(bytes);
    for (size_t i = 0; i < bytes; ++i) expect[i] = char((i * 131 + bytes) & 0x7f);
    if (rank == 0) {
        std::memcpy(host, expect.data(), bytes);
        CUDA_CHECK(cudaMemcpy(dev, expect.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        std::memset(host, 0, bytes);
        CUDA_CHECK(cudaMemset(dev, 0, bytes));
    }

    // long enough for GPU clocks to leave idle and MPI connections to settle
    const int warmup = iters / 10 > 20 ? iters / 10 : 20;
    double t0 = 0;
    for (int i = 0; i < warmup + iters; ++i) {
        if (i == warmup) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            t0 = MPI_Wtime();
        }
        if (rank == 0) { hop_send(); hop_recv(); }
        else { hop_recv(); hop_send(); }
    }
    double usec = (MPI_Wtime() - t0) / iters / 2 * 1e6;

    if (rank == 0) {
        std::vector<char> got(bytes);
        if (path == Path::Host) std::memcpy(got.data(), host, bytes);
        else CUDA_CHECK(cudaMemcpy(got.data(), dev, bytes, cudaMemcpyDeviceToHost));
        *ok = std::memcmp(got.data(), expect.data(), bytes) == 0;
    }
    MPI_CHECK(MPI_Bcast(ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD));
    return usec;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    const int ngpu = bind_local_gpu();
    if (world_size() != 2) {
        root_printf("pingpong needs exactly 2 ranks (mpirun -np 2)\n");
        MPI_Finalize();
        return 1;
    }
    const size_t max_bytes = size_t(argc > 1 ? std::atoi(argv[1]) : 64) << 20;
    const bool aware = mpi_cuda_aware();

    if (world_rank() == 0) print_device();
    root_printf("GPUs on node: %d | MPI CUDA-aware: %s\n\n", ngpu, aware ? "yes" : "no (direct path skipped)");

    char *host = nullptr, *dev = nullptr;
    CUDA_CHECK(cudaMallocHost(&host, max_bytes));
    CUDA_CHECK(cudaMalloc(&dev, max_bytes));

    std::vector<Path> paths = {Path::Host, Path::Staged};
    if (aware) paths.push_back(Path::Direct);

    root_printf("%-8s %10s %12s %10s %4s\n", "path", "size", "latency_us", "GB/s", "ok");
    bool all_ok = true;
    char sz[32];
    for (Path p : paths) {
        for (size_t bytes = 8; bytes <= max_bytes; bytes *= 8) {
            int iters = bytes <= (64 << 10) ? 1000 : bytes <= (8 << 20) ? 100 : 20;
            bool ok = false;
            double us = pingpong(p, bytes, iters, host, dev, &ok);
            all_ok &= ok;
            root_printf("%-8s %10s %12.2f %10.2f %4s\n", path_name(p),
                        human_bytes(bytes, sz, sizeof sz), us, bytes / us / 1e3, ok ? "yes" : "NO");
        }
        root_printf("\n");
    }

    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaFree(dev));
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
