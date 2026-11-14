# Ray: Ray Serve vs plain vLLM, and a Ray cluster across the homelab

[gpu-systems-lab](../../README.md) / `serving/ray/` · [← serving](../README.md) · [Results](../../docs/RESULTS.md)

## 1. Ray Serve LLM vs plain vLLM

What does Ray Serve's LLM layer cost on top of the engine it wraps? Both sides run in the **same
image** (`rayproject/ray-llm:2.60.1`: Ray 2.60.1 with vLLM 0.29.0). They serve the same model
(Qwen2.5-0.5B-Instruct) with the same engine settings: 15% of the RTX 5070, `max_model_len` 2048,
`max_num_seqs` 64, prefix caching off. So the only difference is Ray Serve's proxy, ingress and
replica actors.

| File | What it is |
|---|---|
| [serve_llm.py](serve_llm.py) | `ray.serve.llm.LLMConfig` + `build_openai_app`, one replica |
| [serve_config.yaml](serve_config.yaml) | Ray Serve on 127.0.0.1:8002. Port 8000 belongs to the live `vllm-chat` on the same host network |
| [bench_serve.sh](bench_serve.sh) | Starts `vllm serve`, benchmarks it, stops it, then the same for `serve run`. `vllm bench serve`, random 256 in / 128 out, `--ignore-eos`, 400 prompts, concurrency 1/16/64, 3 repeats each, after a warm-up burst at concurrency 64 |

Measured 2026-11-14 on the GPU PC (WSL2), next to the idle `vllm-chat`. Each cell is the mean of 3
runs, ± the population SD. Raw JSON: [results/ray/](../../results/ray/).

| concurrency | | output tok/s | TTFT mean / p99 | TPOT mean | inter-token p99 |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM | **343.6 ± 0.0** | **18.1** / 20.8 ms | 2.79 ms | **3.4 ms** |
| | Ray Serve | 331.1 ± 0.6 (−3.6%) | 61.0 / 67.7 ms | 2.55 ms | 50.5 ms |
| 16 | vLLM | **4,590 ± 30** | **64** / 182 ms | 3.00 ms | **3.9 ms** |
| | Ray Serve | 3,754 ± 42 (−18%) | 97 / 224 ms | 3.44 ms | 53.7 ms |
| 64 | vLLM | **8,071 ± 48** | **241** / 849 ms | 5.62 ms | **31.9 ms** |
| | Ray Serve | 6,048 ± 37 (−25%) | 531 / 1,249 ms | 5.75 ms | 90.2 ms |

### What it shows

1. **Ray Serve's cost grows with load.** It costs 3.6% throughput for one user, 18% at 16 and 25%
   at 64. Every request and every streamed chunk crosses the HTTP proxy, an ingress replica and the
   `LLMServer` replica (separate processes, serialized between them). With a 0.5B model each decode
   step is only ~3–6 ms, so per-request Python overhead is a large share. A bigger model spends
   more time per step on the GPU, and the relative overhead should shrink.
2. **Time to first token: +43 ms at concurrency 1.** That's the extra hops before the first
   chunk reaches the client.
3. **Tokens arrive in bursts.** p99 inter-token latency is 50–90 ms through Ray Serve against
   3–32 ms direct. Mean TPOT even looks slightly *better* at concurrency 1 (2.55 vs 2.79 ms),
   because Ray Serve's LLM router coalesces streamed tokens and sends them in batches. The average
   holds up, but a chat UI sees tokens arrive in clumps.
4. **What you get for it:** autoscaling replicas across nodes, multi-model routing, and Ray's
   placement and fault tolerance. Those don't help on a single GPU, which is why plain
   `vllm serve` (here, and as `vllm-chat`) wins on one card.

**WSL2 gotcha found on the way:** vLLM 0.29 defaults to its new "V2" model runner, which needs
pinned host memory (CUDA unified virtual addressing). vLLM turns pinned memory off under WSL2, so
the engine died with `RuntimeError: UVA is not available`. `VLLM_USE_V2_MODEL_RUNNER=0` fixes it.
The repo's vLLM 0.32 image isn't affected.

## 2. A Ray cluster across the GPU PC and the iMac

| File | What it is |
|---|---|
| [cluster/Dockerfile](cluster/Dockerfile), [cluster/ts-ray.sh](cluster/ts-ray.sh) | Ray head image for the GPU PC. The container runs its **own `tailscaled`** and joins the tailnet as an ephemeral node with an auth key (read from a mounted file, never on the command line). Ray then advertises a routable 100.x address. WSL2's NAT and Docker's bridge would otherwise break Ray's many node-to-node ports |
| [cluster/cluster_bench.py](cluster/cluster_bench.py) | Task round trip per node, object transfer between nodes, Ray Train (TorchTrainer, DDP over gloo) with 1 and 2 workers |
| [cluster/gloo_probe.py](cluster/gloo_probe.py) | Bare-bones cross-node gloo `init_process_group` + `all_reduce`, with a 60 s timeout |

**Where it runs:**
- **Head:** a container on the GPU PC (WSL2): Ray 2.60.0, Python 3.13.18, 8 CPUs plus the GPU.
- **Worker:** the iMac, running Ray **natively on macOS** in a uv-managed Python 3.13.18 venv, 4 CPUs.
  Its Colima containers have no working outbound network, so a container worker wasn't an option.
- **Laptop:** not part of the cluster (it's only the controller).

Getting the two to join took four fixes, all real gotchas:
1. **macOS clusters are unsupported.** Ray refuses to join a macOS node to a multi-node cluster unless
   `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` is set.
2. **Python must match to the patch level.** Every node needs the same version: 3.13.18 (conda, in
   the image) vs 3.13.2 (Homebrew) was rejected with `Version mismatch`, so the iMac uses
   `uv python install 3.13.18`.
3. **The auth key must be reusable.** A single-use key works once; restarting the head needs a
   reusable one.
4. **gloo must use the tailnet interface.** Otherwise it advertises a Docker-bridge or LAN address:
   `GLOO_SOCKET_IFNAME=tailscale0` on the head, `utun4` on the iMac.

### Measured (2026-11-14)

Raw JSON: [results/ray-cluster/](../../results/ray-cluster/).

| | head (local) | iMac (over the tailnet) |
|---|---:|---:|
| empty task round trip p50 / p99 | 1.67 / 2.70 ms | **24.4 / 47.2 ms** |
| `ray.get` of an array created on the other node | | 3.2 MB/s (16 MB), 8.9 (64 MB), **10.0 MB/s** (256 MB) |

| Ray Train, 2-layer MLP (1.6M params), batch 256/worker, gloo on CPU | step | samples/s |
|---|---:|---:|
| 1 worker (head) | 7.72 ms | 33,145 |
| 2 workers, **both on the head** | 11.07 ms | **46,241** (1.40x, 70% efficiency) |
| 2 workers, **one per machine** | **hung at process-group setup** | — |

1. **Ray Core works across the machines, and the network sets the cost.** A task on the iMac costs
   ~24 ms round trip (vs 1.7 ms locally), and objects move at ~10 MB/s. Tailscale reports a
   *direct* path (not a relay), so ~10 MB/s is the iMac's Wi-Fi link, the same ~10 MB/s the
   repo's MPI run measured between these machines. Anything that ships data per task to the iMac
   will be network-bound.
2. **Cross-node DDP between the Linux head and the macOS worker didn't work.** `gloo_probe.py`
   shows both ranks reaching the TCP store on the head, then each timing out after 60 s waiting for
   the other's key (`DistStoreError: wait timeout ... keys: /default_pg/0//cpu//0/rank_0`). Both run
   torch 2.11.0, so it isn't a version mismatch. It's left as an open issue rather than guessed
   at. Ray marks macOS cluster nodes unsupported for a reason, and on Wi-Fi the iMac would be a
   ~10 MB/s straggler for every all-reduce anyway.
3. **Ray Train itself is fine:** 2 DDP workers on one node scale 1.40x on this small model.
4. **KubeRay:** not attempted (optional in the brief). The k3s node is on the Ubuntu boot, which is
   offline.

### Teardown

`docker stop ray-head` (the entrypoint logs the container out of the tailnet; ephemeral nodes are
removed automatically), `ray stop` on the iMac, and the auth-key copy deleted from the GPU PC.
