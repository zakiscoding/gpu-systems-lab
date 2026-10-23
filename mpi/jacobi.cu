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
// out[r][c] = average of the 4 neighbours of in[r][c] (5-point stencil);
// boundary columns are copied through unchanged.
// Mapping: 2D grid of 32x8 blocks, thread (x, y) -> column c = x, strip row
//   r = y + 1 (skipping halo row 0). One cell per thread, no loop.
// Memory: a warp is 32 consecutive columns of one row, so the in[i +- N],
//   in[i] and out[i] accesses are coalesced and the i +- 1 neighbours mostly
//   hit the same cache lines; the vertical neighbours are reused through
//   L1/L2 by the blocks above/below. ~5 loads + 1 store per 4 FLOPs, so it
//   is memory-bound: 66.6 Gcell/s on 1 rank, 4096^2 (mpi/README.md).
// in and out must be different buffers (Jacobi, not Gauss-Seidel).
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
// h holds (rows + 2) * N floats (halo, rows interior rows, halo); zeroed,
// then row 0 set to 1.0 on the topmost rank. Host-only, no MPI.
static void init_strip(std::vector<float>& h, int rows, int N, bool top, bool bottom) {
    std::fill(h.begin(), h.end(), 0.f);
    if (top) std::fill(h.begin(), h.begin() + N, 1.f);
    (void)bottom;  // bottom halo already 0
}

// Halo exchange for a device strip d of (rows + 2) x N floats:
//   my first interior row -> `up`'s bottom halo; `down`'s first row -> my
//   bottom halo (tag 0), and symmetrically last row -> `down` / top halo
//   from `up` (tag 1).
// up/down are neighbour ranks, MPI_PROC_NULL at the global edges (sends and
// receives with MPI_PROC_NULL complete immediately and leave the buffer
// alone, so edge halos keep their fixed boundary values).
// MPI_Sendrecv pairs the send and receive in one call, so the chain of ranks
// cannot deadlock the way blocking Send-then-Recv on every rank could.
// It is point-to-point, but every rank must still make both calls each step
// (in the same tag order) or its neighbours block.
//   direct  (CUDA-aware MPI) device pointers go straight to MPI.
//   staged  hbuf (pinned, >= 4 * N floats) holds two send and two receive
//           rows: 2 blocking D2H copies, the two Sendrecvs on host memory,
//           then 2 blocking H2D copies (skipped at the global edges).
// Each message is one N-float row (16 KiB at N = 4096). On one shared GPU
// this exchange is 40-68% of runtime at 2-8 ranks (mpi/README.md), mostly
// context-switch cost that MPS removes.
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
// a holds the current state, b the next (both device, same strip layout);
// they are ping-ponged, so after an even step count the result is in a, odd
// in b (the caller picks with steps % 2). Each step: launch the stencil,
// cudaDeviceSynchronize (so compute time is real kernel time and the halo
// rows are final), then exchange_halos on the freshly written buffer.
// In the decomposed run every rank must call it with the same `steps`. With
// up = down = MPI_PROC_NULL it communicates with nobody, which is how rank 0
// alone computes the serial reference.
// compute_s / comm_s are out-params, wall-clock seconds via MPI_Wtime.
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
    const int ngpu = bind_local_gpu();
    const int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 500;
    const int rank = world_rank(), n = world_size();
    const bool direct = mpi_cuda_aware();

    // interior rows are 1..N-2 globally (rows 0 and N-1 are boundary);
    // block split: the first `extra` ranks get one extra row. Rank r's
    // neighbours are r-1 (up) and r+1 (down), MPI_PROC_NULL at the ends.
    const int interior = N - 2;
    const int base = interior / n, extra = interior % n;
    const int rows = base + (rank < extra ? 1 : 0);
    const int start = 1 + rank * base + std::min(rank, extra);  // first global interior row
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank == n - 1 ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) print_device();
    root_printf("grid %dx%d, %d steps, %d ranks (%d-%d rows each) on %d GPU(s), halo path: %s\n\n", N, N, steps,
                n, base, base + (extra ? 1 : 0), std::min(n, ngpu),
                direct ? "CUDA-aware direct" : "staged via host");

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

    // gather interior rows on rank 0 (MPI_Gatherv is a collective: every rank
    // must reach it; counts/displs are in floats and mirror the row split)
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
    // Collective: every rank waits here while rank 0 runs the reference, then
    // all ranks exit with rank 0's verdict.
    MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFreeHost(hbuf));
    MPI_Finalize();
    return ok ? 0 : 1;
}
