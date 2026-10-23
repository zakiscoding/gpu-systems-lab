// Network ping-pong and all-reduce between machines (host buffers only, no
// CUDA), so it builds on both the Linux GPU node and the M1 iMac.
//
// Same output conventions as ../pingpong.cu and ../allreduce.cu, so the
// cross-machine numbers line up against the single-node shared-memory ones.
//
// Usage: mpirun -np 2 --host zak-gpu-node,mos-imac ... netbench [max_mib]
#include <mpi.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

static int rank_, size_;

static void root_printf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void root_printf(const char* fmt, ...) {
    if (rank_ != 0) return;
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

static const char* human(size_t b, char* buf, size_t n) {
    if (b >= (1u << 20)) std::snprintf(buf, n, "%zu MiB", b >> 20);
    else if (b >= (1u << 10)) std::snprintf(buf, n, "%zu KiB", b >> 10);
    else std::snprintf(buf, n, "%zu B", b);
    return buf;
}

// Ranks 0 and 1 bounce a buffer; returns one-way microseconds. Rank 0 checks
// the echoed payload byte for byte.
static double pingpong(size_t bytes, int iters, std::vector<char>& buf, bool* ok) {
    const int peer = 1 - rank_;
    std::vector<char> expect(bytes);
    for (size_t i = 0; i < bytes; ++i) expect[i] = char((i * 131 + bytes) & 0x7f);
    if (rank_ == 0) std::memcpy(buf.data(), expect.data(), bytes);
    else std::memset(buf.data(), 0, bytes);

    const int warmup = iters / 10 > 5 ? iters / 10 : 5;
    double t0 = 0;
    for (int i = 0; i < warmup + iters; ++i) {
        if (i == warmup) {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
        if (rank_ == 0) {
            MPI_Send(buf.data(), (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf.data(), (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf.data(), (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }
    }
    double us = (MPI_Wtime() - t0) / iters / 2 * 1e6;
    if (rank_ == 0) *ok = std::memcmp(buf.data(), expect.data(), bytes) == 0;
    MPI_Bcast(ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return us;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    const size_t max_bytes = size_t(argc > 1 ? std::atoi(argv[1]) : 16) << 20;

    // who is where: every rank reports its host to rank 0
    char host[64] = {};
    gethostname(host, sizeof host - 1);
    std::vector<char> hosts(64 * size_t(size_));
    MPI_Gather(host, 64, MPI_CHAR, hosts.data(), 64, MPI_CHAR, 0, MPI_COMM_WORLD);
    for (int r = 0; r < size_; ++r) root_printf("rank %d on %s\n", r, &hosts[64 * size_t(r)]);
    root_printf("\n");

    bool all_ok = true;
    char sz[32];
    std::vector<char> buf(max_bytes);

    if (size_ == 2) {  // ping-pong is defined for exactly two ranks
        root_printf("ping-pong rank 0 <-> rank 1\n%10s %12s %10s %4s\n", "size", "latency_us", "MB/s", "ok");
        for (size_t b = 8; b <= max_bytes; b *= 8) {
            int iters = b <= (64 << 10) ? 200 : b <= (2 << 20) ? 20 : 5;
            bool ok = false;
            double us = pingpong(b, iters, buf, &ok);
            all_ok &= ok;
            root_printf("%10s %12.1f %10.1f %4s\n", human(b, sz, sizeof sz), us, b / us, ok ? "yes" : "NO");
        }
        root_printf("\n");
    }

    // all-reduce sweep, nccl-tests busbw convention
    const float expect = size_ * (size_ + 1) / 2.0f;
    std::vector<float> send(max_bytes / 4, float(rank_ + 1)), recv(max_bytes / 4);
    root_printf("all-reduce, %d ranks\n%10s %12s %12s %12s %4s\n", size_, "size", "time_us", "algbw_MB/s",
                "busbw_MB/s", "ok");
    for (size_t count = 256; count * 4 <= max_bytes; count *= 8) {
        const int iters = count * 4 <= (1 << 20) ? 50 : 5;
        for (int i = 0; i < 3; ++i)
            MPI_Allreduce(send.data(), recv.data(), (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < iters; ++i)
            MPI_Allreduce(send.data(), recv.data(), (int)count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double local = (MPI_Wtime() - t0) / iters, secs = 0;
        MPI_Allreduce(&local, &secs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        int ok = recv[0] == expect && recv[count - 1] == expect, all = 0;
        MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        all_ok &= all;
        double algbw = count * 4 / secs / 1e6;
        root_printf("%10s %12.1f %12.1f %12.1f %4s\n", human(count * 4, sz, sizeof sz), secs * 1e6, algbw,
                    algbw * 2.0 * (size_ - 1) / size_, all ? "yes" : "NO");
    }

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
