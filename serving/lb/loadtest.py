"""Load generator for the HAProxy + vLLM replicas lab (runs in the `client` container).

    python loadtest.py overhead            # latency added by the LB: direct to r1 vs through HAProxy
    python loadtest.py failover NAME SECS  # constant load through the LB; the driver kills a replica

Every request is one JSON line in /out/<name>.jsonl: start time, latency, HTTP status or error,
and which replica answered (vLLM doesn't say, so HAProxy's server name comes back in a header
only if configured; we infer nothing and just record status/latency).
"""

import asyncio
import json
import statistics
import sys
import time

import httpx

LB, DIRECT = "http://haproxy:8100", "http://r1:8000"
BODY = {"model": "qwen0.5b", "prompt": "The capital of France is", "max_tokens": 1, "temperature": 0}


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


async def one(client, base, body):
    t0 = time.time()
    try:
        r = await client.post(f"{base}/v1/completions", json=body)
        return {"t": t0, "ms": (time.time() - t0) * 1e3, "status": r.status_code}
    except httpx.HTTPError as e:
        return {"t": t0, "ms": (time.time() - t0) * 1e3, "status": None, "error": type(e).__name__}


async def overhead(n=400, conc=16):
    out = {}
    async with httpx.AsyncClient(timeout=30) as c:
        for base in (DIRECT, LB):           # warm both paths
            for _ in range(20):
                await one(c, base, BODY)
        for rnd in range(3):                # alternate to cancel drift
            for name, base in (("direct", DIRECT), ("haproxy", LB)):
                lat = [(await one(c, base, BODY))["ms"] for _ in range(n)]
                out.setdefault(name, []).extend(lat)
        res = {k: {"n": len(v), "p50_ms": round(statistics.median(v), 3), "p99_ms": round(pct(v, 99), 3),
                   "mean_ms": round(statistics.mean(v), 3)} for k, v in out.items()}
        res["added_p50_ms"] = round(res["haproxy"]["p50_ms"] - res["direct"]["p50_ms"], 3)
        # throughput at concurrency `conc`, 64-token completions
        body = {**BODY, "max_tokens": 64}
        for name, base in (("direct_r1_only", DIRECT), ("haproxy_2_replicas", LB)):
            t0 = time.time()
            rs = await asyncio.gather(*[one(c, base, body) for _ in range(conc * 8)])
            dt = time.time() - t0
            ok = [r for r in rs if r["status"] == 200]
            from collections import Counter
            res[f"throughput_{name}"] = {"requests": len(rs), "ok": len(ok), "req_per_s": round(len(ok) / dt, 2),
                                         "outcomes": dict(Counter(str(r.get("status") or r.get("error")) for r in rs)),
                                         "p50_ms": round(statistics.median(r["ms"] for r in ok), 1) if ok else None}
    print(json.dumps(res, indent=1))
    with open("/out/overhead.json", "w") as f:
        json.dump(res, f, indent=1)


async def failover(name, secs, conc=16):
    body = {**BODY, "max_tokens": 32}
    stop = time.time() + secs
    f = open(f"/out/{name}.jsonl", "w", buffering=1)

    async def worker(c):
        while time.time() < stop:
            r = await one(c, LB, body)
            f.write(json.dumps(r) + "\n")

    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=conc)) as c:
        await asyncio.gather(*[worker(c) for _ in range(conc)])
    f.close()


if __name__ == "__main__":
    if sys.argv[1] == "overhead":
        asyncio.run(overhead())
    else:
        asyncio.run(failover(sys.argv[2], float(sys.argv[3])))
