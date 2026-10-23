# Multi-machine MPI: RTX 5070 + M1 iMac over Tailscale

**[homelab](https://github.com/zakiscoding/homelab)** (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../../README.md) / `mpi/multinode/` · [← repo README](../../README.md) · [Results](../../docs/RESULTS.md) · [Code guide](../../docs/CODE-GUIDE.md)

Real two-node MPI across a home network. Rank 0 runs on the Ubuntu node (RTX 5070, x86_64), rank 1
on an M1 iMac (arm64, macOS, no CUDA). The two machines talk over Tailscale.

| Program | What it measures |
|---|---|
| `netbench.cpp` | Ping-pong latency/bandwidth and an all-reduce sweep between machines (host buffers) |
| `jacobi_hetero.cpp` | The Jacobi halo exchange from `../jacobi.cu`, with the GPU stencil on Linux and a CPU stencil on macOS, rows split by `--weight`. Checked bit-for-bit against a serial CPU run |

## What it took

1. **Matching Open MPI on both machines, built from source.** Ubuntu ships 4.1.6. The Mac has no
   package of that version, so 4.1.6 is built into `~/opt/openmpi-4.1.6-het` on **both** machines.
2. **`--enable-heterogeneous`.** x86_64 Linux and arm64 macOS disagree on `sizeof(long double)`
   (16 vs 8 bytes), and a stock build refuses to mix the two.
3. **Bundled hwloc, libevent and PMIx on both sides.** The first Linux build picked up Ubuntu's
   system hwloc and libevent, while the Mac used Open MPI's bundled copies. The daemons connected,
   then failed decoding the job map (`Data unpack had inadequate space`). Rebuilding Linux with
   `--with-hwloc=internal --with-libevent=internal --with-pmix=internal` fixed it.
4. **Passwordless SSH from the launching node to the Mac.** A dedicated key (`~/.ssh/id_mpi`) plus a
   `Host mos-imac` entry setting `User momo`. `mpirun` starts the Mac's daemon over SSH.
5. **Explicit paths.** The Mac's non-interactive shell has no Open MPI on `PATH`, so
   `orte_launch_agent` names `orted` by absolute path, and MPMD syntax (`-np 1 --host A prog : -np 1
   --host B prog`) gives each host its own binary path.
6. **TCP on the tailnet only.** `btl_tcp_if_include` and `oob_tcp_if_include` are set to
   `100.64.0.0/10`, Tailscale's address range. That works with Linux's `tailscale0` and macOS's
   `utunN` alike.

```bash
# on each machine (Linux: CUDA build; macOS: CPU build)
make -C mpi/multinode
# from the GPU node
make -C mpi/multinode net
make -C mpi/multinode jacobi GPU_W=9 MAC_W=1
```

## Results

The link is a direct LAN path (`tailscale ping`: "via 10.0.0.190"), but the iMac is on **Wi-Fi**,
with RTT jittering between 5 and 60 ms.

### Ping-pong and all-reduce (Tailscale over Wi-Fi)

| Size | one-way latency | bandwidth |
|---|---:|---:|
| 8 B | 2.6 ms | — |
| 4 KiB | 7.9 ms | 0.5 MB/s |
| 256 KiB | 108 ms | 2.4 MB/s |
| 16 MiB | 1.71 s | 9.8 MB/s |

All-reduce peaked at 4.8 MB/s busbw (4 MiB). For scale against the single-node results in
`../README.md`:

| Path | small-message latency | peak bandwidth |
|---|---:|---:|
| Shared memory, one node | 0.17 µs | 21 GB/s |
| Tailscale over Wi-Fi, two nodes | 2,600 µs | 0.01 GB/s |
| *ratio* | *~15,000x* | *~2,000x* |

### Heterogeneous Jacobi (2048², 200 steps)

| Split (GPU:Mac) | GPU rows | Mac rows | compute GPU / Mac | halo time | wall | matches serial |
|---|---:|---:|---|---:|---:|---|
| 9:1 | 1841 | 205 | 0.007 s / 0.013 s | 4.83 s | 4.84 s | yes, bit-exact |
| 50:1 | 2006 | 40 | 0.007 s / 0.003 s | 3.55 s | 3.56 s | yes, bit-exact |
| serial CPU, one node | — | — | 0.21 s | — | 0.21 s | reference |

## Takeaways

- **It is correct across GPU and CPU, x86 and ARM.** With FMA contraction disabled on both builds
  (`-fmad=false`, `-ffp-contract=off`), the RTX 5070 and the M1 do identical float operations, and
  the gathered grid matches the serial run exactly.
- **The network is everything.** Each step swaps two 8 KiB halo rows, about 24 ms per step on this
  link, so halo exchange is ~100% of runtime and two machines run **17-23x slower than one CPU**.
  Compute is under 15 ms total on either side.
- **Load balancing can't fix latency.** Moving work from 9:1 to 50:1 barely changes the total,
  because the halo exchange happens every step whatever the strip size. This is why HPC clusters use
  InfiniBand (~1 µs latency, 25+ GB/s), and why codes overlap communication with compute or exchange
  halos less often (deeper halos / communication-avoiding stencils).
- **Practical lessons:** every node needs the same MPI version *and* the same bundled-library
  choices; heterogeneous architectures need an explicit build flag; and the launcher's assumptions
  (identical paths, `PATH` in non-interactive shells) break first.
