# HAProxy in front of vLLM replicas: health checks, failover, overhead

[gpu-systems-lab](../../README.md) / `serving/lb/` · [← serving](../README.md) · [Results](../../docs/RESULTS.md)

Two vLLM replicas (Qwen2.5-0.5B-Instruct, `vllm/vllm-openai:v0.32.0`) behind HAProxy 3.2 on the
RTX 5070. They run next to the live `vllm-chat` without stopping it: each replica gets 12% of the
card. Everything is bound to 127.0.0.1 and torn down after the run.

| File | What it does |
|---|---|
| [compose.yml](compose.yml) | Project `lb-lab`: replicas `r1` and `r2`, `haproxy`, a `client` container |
| [haproxy.cfg](haproxy.cfg) | `leastconn`, an active `GET /health` check every 1 s (`fall 2 rise 2`), and L7 retries (`retry-on conn-failure empty-response response-timeout`, `retries 2`, `option redispatch`) |
| [loadtest.py](loadtest.py) | `overhead`: latency direct vs through HAProxy, plus a concurrency burst. `failover`: constant load at concurrency 16, one JSON line per request |
| [run.sh](run.sh) | Brings it up, runs `overhead`, then two 100 s failover runs (retries on, then off). Each kills `r1` (SIGKILL) at 20 s and starts it again at 40 s |

Measured 2026-11-14 (UTC) on the GPU PC (WSL2). Raw per-request logs: [results/lb/](../../results/lb/).

## Failover: kill one replica under load

Concurrency 16, 32-token completions. `r1` is SIGKILLed 20 s in, restarted at 40 s, and is
`healthy` again 26–31 s later.

| | requests | failed | failure types | throughput, r1 down |
|---|---:|---:|---|---:|
| **HAProxy with L7 retries** | 3,432 | **8 (0.23%)** | 8 connection closed mid-response | 34.5 req/s |
| HAProxy without retries | 3,730 | 218 (5.8%) | **202 HTTP 503** + 16 connection closed | 33.2 req/s |

- **Retries absorb the detection gap.** Health checks need 2 failed probes 1 s apart to mark `r1`
  DOWN. Without retries, every request routed to the dead server in that window failed with a 503:
  202 errors within 1 s of the kill. With `retry-on conn-failure`, HAProxy re-sent them to `r2` and
  the client never saw it.
- **In-flight requests are lost either way.** The 8 (or 16) requests that `r1` was generating when
  it died were cut off mid-response. HAProxy can't safely replay a response that has already
  started. A client-side retry is the only fix.
- **No latency cliff.** p50 stayed at 434–477 ms before, during and after the outage, and p99 stayed
  under 700 ms with only `r2` serving. With retries on, no request took over 2 s.
- **The replica rejoins on its own.** After `docker start`, HAProxy's `rise 2` put `r1` back in
  rotation as soon as `/health` answered, and traffic spread across both again.

## Overhead

1,200 sequential 1-token completions per path, alternated in 3 rounds:

| | p50 | p99 |
|---|---:|---:|
| direct to `r1` | 16.10 ms | 18.55 ms |
| through HAProxy | 17.19 ms | 19.64 ms |
| **added by the LB** | **+1.09 ms** | +1.10 ms |

A second run measured +0.92 ms. That's an extra hop through Docker's bridge network plus HTTP
parsing. It's small next to a single decode step, and negligible next to a real completion.

**Throughput doesn't scale with replicas on one GPU.** With warm replicas, 128 concurrent
64-token requests ran at 62 req/s direct to one replica and 63 req/s through HAProxy to two (p50
1,532 vs 1,106 ms). The replicas share one GPU, so a second one on the same card buys availability
and a slightly lower median, not throughput.

## What broke: a healthy replica that isn't ready

On a **fresh** start, the first burst of 128 concurrent 64-token requests timed out (30 s client
timeout). This happened going direct to `r1` (100 timeouts + 28 waiting in the client's connection
pool) and through HAProxy (50 of 128). An identical burst a minute later finished in 2–3 s with no
errors. It reproduced on a second fresh start.

`/health` returned 200 the whole time, so HAProxy counted the replica as up. The likely cause is
vLLM compiling kernels the first time it sees a large decode batch: the warm-up had only sent
batch-size-1 requests. Consistent with that, **restarting the same container** (failover runs,
compile cache still on its filesystem) gave no stall. Not proven: the replica's logs from the cold
run weren't kept.

In production: don't count a replica as ready on `/health` alone. Run a warm-up batch at the batch
sizes you'll serve before adding it to the pool (HAProxy `agent-check` or a Kubernetes readiness
probe that does a real completion), add `slowstart`, and persist vLLM's compile cache
(`~/.cache/vllm`) in a volume so new containers start warm.

## Run it

On the GPU node (WSL), with `vllm/vllm-openai:v0.32.0` and the model in the HF cache:

```bash
./run.sh            # ~6 min; writes ./out (copy into results/lb/), then docker compose down
```
