// Multi-machine Jacobi heat diffusion: the same halo-exchange pattern as
// ../jacobi.cu, but each rank may run on a different kind of machine.
//
// Built with -DUSE_CUDA (nvcc -x cu) the stencil runs on the GPU; built as
// plain C++ it runs on the CPU. So one run can mix the RTX 5070 node and the
// M1 iMac. Rows are split by per-rank weights (--weight), so a slow CPU rank
// gets a thinner strip than a GPU rank.
//
// Correctness: rank 0 gathers the grid and compares it with a serial CPU
// reference. FMA contraction is disabled on both builds (-fmad=false /
// -ffp-contract=off) so GPU and CPU do identical float ops: the match is exact.
//
// Usage (MPMD, each host passes its own weight):
//   mpirun -np 1 --host zak-gpu-node jacobi_hetero --weight 9 : \
//          -np 1 --host mos-imac     jacobi_hetero --weight 1
// Optional: --n 2048 --steps 200
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                         __LINE__);                                                       \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                 \
        }                                                                                 \
    } while (0)

__global__ void jacobi_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (r > rows || c >= N) return;
    size_t i = size_t(r) * N + c;
    if (c == 0 || c == N - 1) { out[i] = in[i]; return; }
    out[i] = 0.25f * (in[i - N] + in[i + N] + in[i - 1] + in[i + 1]);
}
#endif

// Same arithmetic as the kernel, on the CPU (also used for the reference).
static void jacobi_cpu(const float* in, float* out, int rows, int N) {
    for (int r = 1; r <= rows; ++r) {
        size_t row = size_t(r) * N;
        out[row] = in[row];
        out[row + N - 1] = in[row + N - 1];
        for (int c = 1; c < N - 1; ++c) {
            size_t i = row + c;
            out[i] = 0.25f * (in[i - N] + in[i + N] + in[i - 1] + in[i + 1]);
        }
    }
}

// A strip of `rows` interior rows plus halo rows 0 and rows+1. On the GPU
// path the strip lives in device memory and halos are staged through host
// buffers; on the CPU path everything is host memory.
struct Strip {
    int rows, N;
    bool gpu;
    std::vector<float> a, b;  // host (CPU path, and initial upload)
    float *da = nullptr, *db = nullptr;
    float* cur() { return gpu ? da : a.data(); }
    float* nxt() { return gpu ? db : b.data(); }
    void swap() {
        if (gpu) std::swap(da, db);
        else std::swap(a, b);
    }
};

static void step(Strip& s) {
#ifdef USE_CUDA
    if (s.gpu) {
        dim3 block(32, 8), grid((s.N + 31) / 32, (s.rows + 7) / 8);
        jacobi_kernel<<<grid, block>>>(s.cur(), s.nxt(), s.rows, s.N);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
#endif
    jacobi_cpu(s.cur(), s.nxt(), s.rows, s.N);
}

static void copy_row(float* dst, const float* src, int N, bool gpu, bool to_device) {
#ifdef USE_CUDA
    if (gpu) {
        CUDA_CHECK(cudaMemcpy(dst, src, size_t(N) * 4, to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
        return;
    }
#endif
    (void)gpu;
    (void)to_device;
    std::memcpy(dst, src, size_t(N) * 4);
}

// Swap edge rows of the freshly written buffer with the neighbours.
static void exchange(Strip& s, int up, int down, std::vector<float>& hb) {
    const int N = s.N;
    float* d = s.nxt();
    float *su = hb.data(), *sd = su + N, *rt = sd + N, *rb = rt + N;
    copy_row(su, d + size_t(N), N, s.gpu, false);
    copy_row(sd, d + size_t(N) * s.rows, N, s.gpu, false);
    MPI_Sendrecv(su, N, MPI_FLOAT, up, 0, rb, N, MPI_FLOAT, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sd, N, MPI_FLOAT, down, 1, rt, N, MPI_FLOAT, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up != MPI_PROC_NULL) copy_row(d, rt, N, s.gpu, true);
    if (down != MPI_PROC_NULL) copy_row(d + size_t(N) * (s.rows + 1), rb, N, s.gpu, true);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    int N = 2048, steps = 200;
    double weight = 1.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--n") N = std::atoi(argv[i + 1]);
        else if (k == "--steps") steps = std::atoi(argv[i + 1]);
        else if (k == "--weight") weight = std::atof(argv[i + 1]);
    }

    bool gpu = false;
    char device[64] = "CPU";
#ifdef USE_CUDA
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        std::snprintf(device, sizeof device, "%s", p.name);
        gpu = true;
    }
#endif

    // split interior rows 1..N-2 by weight; last rank absorbs rounding
    std::vector<double> w(n);
    MPI_Allgather(&weight, 1, MPI_DOUBLE, w.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double wsum = 0;
    for (double x : w) wsum += x;
    const int interior = N - 2;
    std::vector<int> counts(n), starts(n);
    int acc = 0;
    for (int r = 0; r < n; ++r) {
        starts[r] = 1 + acc;
        counts[r] = (r == n - 1) ? interior - acc : std::max(1, int(interior * w[r] / wsum + 0.5));
        acc += counts[r];
    }
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank == n - 1 ? MPI_PROC_NULL : rank + 1;

    Strip s{counts[rank], N, gpu, {}, {}};
    const size_t cells = size_t(s.rows + 2) * N;
    s.a.assign(cells, 0.f);
    if (rank == 0) std::fill(s.a.begin(), s.a.begin() + N, 1.f);  // hot top edge
    s.b = s.a;
#ifdef USE_CUDA
    if (gpu) {
        CUDA_CHECK(cudaMalloc(&s.da, cells * 4));
        CUDA_CHECK(cudaMalloc(&s.db, cells * 4));
        CUDA_CHECK(cudaMemcpy(s.da, s.a.data(), cells * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.db, s.a.data(), cells * 4, cudaMemcpyHostToDevice));
    }
#endif
    std::vector<float> hb(size_t(N) * 4);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime(), compute = 0, comm = 0;
    for (int it = 0; it < steps; ++it) {
        double c0 = MPI_Wtime();
        step(s);
        double c1 = MPI_Wtime();
        exchange(s, up, down, hb);
        compute += c1 - c0;
        comm += MPI_Wtime() - c1;
        s.swap();
    }
    double wall = MPI_Wtime() - t0;

    // bring the final interior rows back to host
    std::vector<float> mine(size_t(s.rows) * N);
#ifdef USE_CUDA
    if (gpu) CUDA_CHECK(cudaMemcpy(mine.data(), s.cur() + N, mine.size() * 4, cudaMemcpyDeviceToHost));
    else
#endif
        std::memcpy(mine.data(), s.cur() + N, mine.size() * 4);

    std::vector<int> rc(n), rd(n);
    for (int r = 0; r < n; ++r) {
        rc[r] = counts[r] * N;
        rd[r] = starts[r] * N;
    }
    std::vector<float> global;
    if (rank == 0) global.assign(size_t(N) * N, 0.f);
    MPI_Gatherv(mine.data(), (int)mine.size(), MPI_FLOAT, global.data(), rc.data(), rd.data(), MPI_FLOAT, 0,
                MPI_COMM_WORLD);

    // per-rank report
    char host[64] = {};
    gethostname(host, sizeof host - 1);
    struct Row { char host[64]; char dev[64]; double compute, comm; } me{}, *all = nullptr;
    std::snprintf(me.host, sizeof me.host, "%s", host);
    std::snprintf(me.dev, sizeof me.dev, "%s", device);
    me.compute = compute;
    me.comm = comm;
    std::vector<Row> rows(rank == 0 ? n : 0);
    if (rank == 0) all = rows.data();
    MPI_Gather(&me, sizeof(Row), MPI_BYTE, all, sizeof(Row), MPI_BYTE, 0, MPI_COMM_WORLD);
    double max_wall = 0;
    MPI_Reduce(&wall, &max_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (rank == 0) {
        std::printf("grid %dx%d, %d steps, %d ranks\n\n", N, N, steps, n);
        std::printf("%-4s %-26s %-24s %6s %10s %10s %6s\n", "rank", "host", "device", "rows", "compute_s",
                    "halo_s", "halo%");
        for (int r = 0; r < n; ++r)
            std::printf("%-4d %-26.26s %-24.24s %6d %10.3f %10.3f %5.0f%%\n", r, rows[r].host, rows[r].dev,
                        counts[r], rows[r].compute, rows[r].comm,
                        100 * rows[r].comm / (rows[r].compute + rows[r].comm));

        // serial CPU reference over the whole grid
        std::vector<float> ra(size_t(N) * N, 0.f);
        std::fill(ra.begin(), ra.begin() + N, 1.f);
        std::vector<float> rb2 = ra;
        double r0 = MPI_Wtime();
        for (int it = 0; it < steps; ++it) {
            jacobi_cpu(ra.data(), rb2.data(), interior, N);
            std::swap(ra, rb2);
        }
        double serial = MPI_Wtime() - r0;
        double max_diff = 0;
        for (size_t i = size_t(N); i < size_t(N) * (N - 1); ++i)
            max_diff = std::fmax(max_diff, std::fabs(double(global[i]) - ra[i]));
        ok = max_diff == 0.0;
        double gcell = double(interior) * N * steps / 1e9;
        std::printf("\nwall (slowest rank) %8.3f s   %6.2f Gcell/s\n", max_wall, gcell / max_wall);
        std::printf("serial CPU, rank 0  %8.3f s   %6.2f Gcell/s\n", serial, gcell / serial);
        std::printf("max |diff| vs serial %.3e  -> %s\n", max_diff, ok ? "match" : "MISMATCH");
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef USE_CUDA
    if (gpu) {
        cudaFree(s.da);
        cudaFree(s.db);
    }
#endif
    MPI_Finalize();
    return ok ? 0 : 1;
}
