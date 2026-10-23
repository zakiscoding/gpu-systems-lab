// MPI_Allreduce bandwidth sweep, reported the same way as nccl-tests
// (algbw and busbw), so it can be compared with modal/ddp_train.py's NCCL
// all-reduce numbers.
//
// Each rank's send buffer holds (rank + 1); the sum must be n(n+1)/2
// everywhere. The reduction is out of place (send -> recv), so inputs never
// need refilling and every timed iteration is pure collective (+ staging).
// With --device the data starts and ends on the GPU and is staged through
// pinned host memory (or passed directly if CUDA-aware) — the hand-written
// version of what NCCL does natively.
//
// Usage: mpirun -np 4 build/allreduce [--device] [max_mib]
#include <cstring>
#include <string>
#include <vector>

#include "mpi_common.cuh"

__global__ void fill(float* x, size_t n, float v) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        x[i] = v;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    bool device = false;
    size_t max_mib = 256;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--device") device = true;
        else max_mib = std::atoi(argv[i]);
    }
    const int n = world_size();
    const bool direct = device && mpi_cuda_aware();
    const size_t max_count = (max_mib << 20) / sizeof(float);
    const float expect = n * (n + 1) / 2.0f;

    if (world_rank() == 0) print_device();
    root_printf("ranks: %d | data: %s | path: %s\n\n", n, device ? "GPU" : "host",
                !device ? "host buffers" : direct ? "CUDA-aware direct" : "staged via pinned host");

    // send/recv pairs on host (always, for staging) and on device (--device)
    const size_t max_bytes = max_count * sizeof(float);
    float *h_send = nullptr, *h_recv = nullptr, *d_send = nullptr, *d_recv = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send, max_bytes));
    CUDA_CHECK(cudaMallocHost(&h_recv, max_bytes));
    if (device) {
        CUDA_CHECK(cudaMalloc(&d_send, max_bytes));
        CUDA_CHECK(cudaMalloc(&d_recv, max_bytes));
        fill<<<256, 256>>>(d_send, max_count, float(world_rank() + 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        for (size_t i = 0; i < max_count; ++i) h_send[i] = float(world_rank() + 1);
    }

    root_printf("%10s %12s %12s %12s %4s\n", "size", "time_us", "algbw_GB/s", "busbw_GB/s", "ok");
    bool all_ok = true;
    char sz[32];
    for (size_t count = 256; count <= max_count; count *= 4) {
        const size_t bytes = count * sizeof(float);
        const int iters = bytes <= (1 << 20) ? 200 : bytes <= (64 << 20) ? 20 : 5;

        auto one = [&] {
            if (direct) {
                MPI_CHECK(MPI_Allreduce(d_send, d_recv, (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            } else if (device) {  // stage: D2H, reduce on host, H2D
                CUDA_CHECK(cudaMemcpy(h_send, d_send, bytes, cudaMemcpyDeviceToHost));
                MPI_CHECK(MPI_Allreduce(h_send, h_recv, (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
                CUDA_CHECK(cudaMemcpy(d_recv, h_recv, bytes, cudaMemcpyHostToDevice));
            } else {
                MPI_CHECK(MPI_Allreduce(h_send, h_recv, (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            }
        };

        for (int i = 0; i < 5; ++i) one();  // warmup: GPU clocks, MPI connection setup
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double t0 = MPI_Wtime();
        for (int i = 0; i < iters; ++i) one();
        double local = (MPI_Wtime() - t0) / iters;
        double secs = 0;  // slowest rank defines collective time
        MPI_CHECK(MPI_Allreduce(&local, &secs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

        // verify a sample of the final result on every rank
        std::vector<float> check(count);
        if (device) CUDA_CHECK(cudaMemcpy(check.data(), d_recv, bytes, cudaMemcpyDeviceToHost));
        else std::memcpy(check.data(), h_recv, bytes);
        int ok = 1;
        for (size_t i = 0; i < count; i += 1 + count / 1024) ok &= check[i] == expect;
        ok &= check[count - 1] == expect;
        int all = 0;
        MPI_CHECK(MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD));
        all_ok &= all;

        double algbw = bytes / secs / 1e9;
        root_printf("%10s %12.1f %12.2f %12.2f %4s\n", human_bytes(bytes, sz, sizeof sz), secs * 1e6,
                    algbw, allreduce_busbw(algbw, n), all ? "yes" : "NO");
    }

    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    if (device) {
        CUDA_CHECK(cudaFree(d_send));
        CUDA_CHECK(cudaFree(d_recv));
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
