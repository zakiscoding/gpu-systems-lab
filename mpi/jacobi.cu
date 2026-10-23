// 2D Jacobi heat diffusion with MPI domain decomposition and halo exchange.
//
// This is the experiment where MPI's model is the point: the grid is split
// into horizontal strips, one per rank. Each step, every rank runs a stencil
// kernel on its strip, then swaps its edge rows ("halos") with the ranks above
// and below via MPI_Sendrecv. NCCL/DDP has no equivalent of this
// neighbour-to-neighbour pattern; it only does collectives.
//
// Boundary: top edge held at 1.0, other edges at 0.0. Correctness: the
// decomposed result must match a single-process reference computed on rank 0
// (bit-identical, since every cell does the same float ops in the same order).
//
// Usage: mpirun -np 4 build/jacobi [N=4096] [steps=500]
#include <cmath>
#include <vector>

#include "mpi_common.cuh"

// One Jacobi sweep over the interior rows [1, rows] of a strip with halo rows
// 0 and rows+1. Columns 0 and N-1 are fixed boundary.
__global__ void jacobi_step(const float* __restrict__ in, float* __restrict__ out, int rows, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (r > rows || c >= N) return;
    size_t i = size_t(r) * N + c;
    if (c == 0 || c == N - 1) { out[i] = in[i]; return; }
    out[i] = 0.25f * (in[i - N] + in[i + N] + in[i - 1] + in[i + 1]);
}

// Row-0 / row-(rows+1) halos of the global top and bottom strips are the
// fixed boundary: top = 1.0, bottom = 0.0.
static void init_strip(std::vector<float>& h, int rows, int N, bool top, bool bottom) {
    std::fill(h.begin(), h.end(), 0.f);
    if (top) std::fill(h.begin(), h.begin() + N, 1.f);
    (void)bottom;  // bottom halo already 0
}

static void exchange_halos(float* d, int rows, int N, int up, int down, float* hbuf, bool direct) {
    const size_t row = size_t(N);
    float* first = d + row;               // first interior row
    float* last = d + row * rows;         // last interior row
    float* halo_top = d;                  // row 0
    float* halo_bot = d + row * (rows + 1);

    if (direct) {
        MPI_CHECK(MPI_Sendrecv(first, N, MPI_FLOAT, up, 0, halo_bot, N, MPI_FLOAT, down, 0,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Sendrecv(last, N, MPI_FLOAT, down, 1, halo_top, N, MPI_FLOAT, up, 1,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        return;
    }
    // staged: hbuf = [send_up | send_down | recv_top | recv_bot]
    float *su = hbuf, *sd = hbuf + row, *rt = hbuf + 2 * row, *rb = hbuf + 3 * row;
    CUDA_CHECK(cudaMemcpy(su, first, row * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sd, last, row * 4, cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Sendrecv(su, N, MPI_FLOAT, up, 0, rb, N, MPI_FLOAT, down, 0, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(sd, N, MPI_FLOAT, down, 1, rt, N, MPI_FLOAT, up, 1, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(halo_top, rt, row * 4, cudaMemcpyHostToDevice));
    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(halo_bot, rb, row * 4, cudaMemcpyHostToDevice));
}

// Runs `steps` sweeps on a strip of `rows` interior rows. Returns seconds,
// split into compute and halo-exchange time.
static void run(float* a, float* b, int rows, int N, int steps, int up, int down, float* hbuf,
                bool direct, double* compute_s, double* comm_s) {
    dim3 block(32, 8), grid((N + 31) / 32, (rows + 7) / 8);
    *compute_s = *comm_s = 0;
    for (int s = 0; s < steps; ++s) {
        double t0 = MPI_Wtime();
        jacobi_step<<<grid, block>>>(a, b, rows, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = MPI_Wtime();
        exchange_halos(b, rows, N, up, down, hbuf, direct);
        *compute_s += t1 - t0;
        *comm_s += MPI_Wtime() - t1;
        std::swap(a, b);
    }
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    const int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 500;
    const int rank = world_rank(), n = world_size();
    const bool direct = mpi_cuda_aware();

    // interior rows are 1..N-2 globally (rows 0 and N-1 are boundary)
    const int interior = N - 2;
    const int base = interior / n, extra = interior % n;
    const int rows = base + (rank < extra ? 1 : 0);
    const int start = 1 + rank * base + std::min(rank, extra);  // first global interior row
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank == n - 1 ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) print_device();
    root_printf("grid %dx%d, %d steps, %d ranks (%d-%d rows each), halo path: %s\n\n", N, N, steps, n,
                base, base + (extra ? 1 : 0), direct ? "CUDA-aware direct" : "staged via host");

    const size_t strip = size_t(rows + 2) * N;
    std::vector<float> h(strip);
    init_strip(h, rows, N, rank == 0, rank == n - 1);
    float *a, *b, *hbuf;
    CUDA_CHECK(cudaMalloc(&a, strip * 4));
    CUDA_CHECK(cudaMalloc(&b, strip * 4));
    CUDA_CHECK(cudaMallocHost(&hbuf, size_t(N) * 4 * 4));
    CUDA_CHECK(cudaMemcpy(a, h.data(), strip * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h.data(), strip * 4, cudaMemcpyHostToDevice));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double t0 = MPI_Wtime(), compute_s, comm_s;
    run(a, b, rows, N, steps, up, down, hbuf, direct, &compute_s, &comm_s);
    double wall = MPI_Wtime() - t0;
    float* result = (steps % 2 == 0) ? a : b;

    // gather interior rows on rank 0
    std::vector<float> mine(size_t(rows) * N);
    CUDA_CHECK(cudaMemcpy(mine.data(), result + N, mine.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> counts(n), displs(n);
    for (int r = 0; r < n; ++r) {
        counts[r] = (base + (r < extra ? 1 : 0)) * N;
        displs[r] = (1 + r * base + std::min(r, extra)) * N;
    }
    std::vector<float> global;
    if (rank == 0) global.assign(size_t(N) * N, 0.f);
    MPI_CHECK(MPI_Gatherv(mine.data(), (int)mine.size(), MPI_FLOAT, global.data(), counts.data(),
                          displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    (void)start;

    // wall time is the slowest rank; the compute/halo split is averaged over
    // ranks (maxima of each part can come from different ranks and overlap)
    double max_wall = 0, avg_comm = 0, avg_compute = 0;
    MPI_CHECK(MPI_Reduce(&wall, &max_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&comm_s, &avg_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&compute_s, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    avg_comm /= n;
    avg_compute /= n;

    int ok = 1;
    if (rank == 0) {
        // single-process reference: the whole interior as one strip, no neighbours
        std::vector<float> ref(size_t(N) * N, 0.f);
        std::fill(ref.begin(), ref.begin() + N, 1.f);
        float *ra, *rb;
        CUDA_CHECK(cudaMalloc(&ra, ref.size() * 4));
        CUDA_CHECK(cudaMalloc(&rb, ref.size() * 4));
        CUDA_CHECK(cudaMemcpy(ra, ref.data(), ref.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rb, ref.data(), ref.size() * 4, cudaMemcpyHostToDevice));
        double c1, c2;
        run(ra, rb, interior, N, steps, MPI_PROC_NULL, MPI_PROC_NULL, hbuf, true, &c1, &c2);
        CUDA_CHECK(cudaMemcpy(ref.data(), (steps % 2 == 0) ? ra : rb, ref.size() * 4,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(ra));
        CUDA_CHECK(cudaFree(rb));

        double max_diff = 0, heat = 0;
        for (size_t i = size_t(N); i < size_t(N) * (N - 1); ++i) {
            max_diff = std::fmax(max_diff, std::fabs(double(global[i]) - ref[i]));
            heat += global[i];
        }
        ok = max_diff == 0.0;
        double cells = double(interior) * N * steps;
        std::printf("%-22s %10.3f s\n", "wall (slowest rank)", max_wall);
        double split = avg_compute + avg_comm;
        std::printf("%-22s %10.3f s  (%.0f%%, rank avg)\n", "  stencil compute", avg_compute, 100 * avg_compute / split);
        std::printf("%-22s %10.3f s  (%.0f%%, rank avg)\n", "  halo exchange", avg_comm, 100 * avg_comm / split);
        std::printf("%-22s %10.2f Gcell/s\n", "throughput", cells / max_wall / 1e9);
        std::printf("%-22s %10.4e  (total heat %.2f)\n", "max |diff| vs serial", max_diff, heat);
        std::printf("%-22s %10s\n", "match", ok ? "yes" : "NO");
    }
    MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFreeHost(hbuf));
    MPI_Finalize();
    return ok ? 0 : 1;
}
