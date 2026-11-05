# Worker LLM gateway

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `gateway/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

A Cloudflare Worker that gives the [vLLM server](../serving/README.md) one public, OpenAI-compatible URL,
with per-client API keys, rate limits, and a clear error when the GPU PC is offline or on Windows.

```
client --gateway key--> Worker (api.<domain>) --VPC Service--> Tunnel --> cloudflared --> vllm:8000
                                               adds VLLM_API_KEY
```

vLLM has no public hostname. The Worker reaches it through a [Workers VPC](https://developers.cloudflare.com/workers-vpc/)
Service limited to one host and port. `homelab-services` keeps using the tailnet.

**Status: deployed on workers.dev, with vLLM and the tunnel in [WSL2](../cloudflare/README.md#wsl2-on-the-windows-boot) on the GPU PC.** See
[spike results](#spike-results). The Ubuntu boot and a custom domain aren't set up. Workers VPC is in beta and its APIs may change.

| File | What it does |
|---|---|
| [src/index.js](src/index.js) | The Worker |
| [test/gateway.test.js](test/gateway.test.js) | Unit tests with a stubbed binding and rate limiter (`npm test`) |
| [wrangler.jsonc](wrangler.jsonc) | Domain, bindings and limits |
| [make-key.sh](make-key.sh) | Creates a client key and its `GATEWAY_KEYS` entry |

## Behaviour

| Case | Result |
|---|---|
| Routes | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`. Public `GET /health` reports the gateway only. Anything else is 404, so other vLLM endpoints stay private |
| Bad or missing key | 401 |
| Over the rate limit | 429 with `Retry-After`. Default 60 requests/min per client |
| Rate limiter failing | 503 `rate_limiter_unavailable`. Requests are refused rather than let through unlimited |
| Body over 256 KiB, or not a JSON object | 413, or 400 |
| `max_tokens` / `max_completion_tokens` | Clamped to 1024. Chat requests with neither get 1024 |
| GPU PC offline | 503 `upstream_unavailable` with `Retry-After` |
| Gateway's `VLLM_API_KEY` is wrong | 502 `gateway_misconfigured` |
| Logs | Client name, route, status and latency. Never prompts, completions or keys |

## Setup

Requires the [tunnel](../cloudflare/README.md) running. A domain is optional: the Worker serves on workers.dev by default.

```bash
cd gateway
npm ci
npx wrangler login
```

Create the VPC Service using the tunnel's UUID. The account needs the Connectivity Directory Admin role.

```bash
npx wrangler vpc service create vllm --type http --hostname vllm --tunnel-id <TUNNEL_UUID> --http-port 8000
```

In [wrangler.jsonc](wrangler.jsonc), set `service_id` to the returned ID. To use your own domain, set
`"workers_dev": false` and add `"routes": [{ "pattern": "api.<domain>", "custom_domain": true }]`.
Then set the secrets. `VLLM_API_KEY` is the value in `serving/.env` on the machine running vLLM.

```bash
./make-key.sh my-laptop
npx wrangler secret put VLLM_API_KEY
npx wrangler secret put GATEWAY_KEYS
npm run deploy
```

To revoke a client, remove its entry and run `wrangler secret put GATEWAY_KEYS` again.

## Spike results

Measured against the deployed Worker, with vLLM and the tunnel in WSL2 on the RTX 5070.

| Check | Result |
|---|---|
| Worker reaches vLLM through the VPC Service; `vllm` resolves through Docker DNS | Works |
| Streaming | 400 tokens arrive in over 440 separate chunks, median gap 5–6 ms, not buffered |
| Added latency | 200 tokens: 1.31 s direct, 1.53 s through the gateway |
| No key, wrong key, unlisted path | 401, 401, 404 |
| `cloudflared` stopped, or vLLM stopped | 503 `upstream_unavailable`; recovers after restart (vLLM reload 55 s) |
| Sequential requests over the rate limit | 429 starts at request 61 |
| Concurrent requests over the rate limit | **Not limited:** 300 of 300 allowed at 15 at a time |
| CPU time | 0–2 ms per request on a short stream and `/v1/models`; longer streams not measured |
| Prompts in Worker logs | None found |
| Request over 125 s | Not reached: the longest allowed completion (1024 tokens) took 7 s when idle |

**Fallback, not needed so far:** if the VPC binding stops working, make `llm.<domain>` a tunnel hostname behind an Access **Service Auth** policy, and have the Worker
call it with `CF-Access-Client-Id` and `CF-Access-Client-Secret`. Worker-to-Access requests in the same zone would
need their own test.

## End-to-end check

```bash
export GW=https://api.<domain> KEY=<client key>
curl -s "$GW/health"
curl -s -o /dev/null -w '%{http_code}\n' "$GW/v1/models"                       # 401
curl -s -H "Authorization: Bearer $KEY" "$GW/v1/models"                        # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" "$GW/tokenize"  # 404
curl -N "$GW/v1/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-instruct","stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
```

Then run `docker compose stop` in `serving/` and confirm a 503, and that it recovers after restarting. Confirm the
tailnet path still works, `10.0.0.21:{3000,8000,9090}` still refuses connections, and `ss -ltn` shows no new listeners.

## Limits

- **Privacy:** Cloudflare terminates TLS, so prompts and completions pass through it decrypted.
- **Windows boots:** the gateway returns 503 while the PC is on Windows unless vLLM and the tunnel are running in [WSL2](../cloudflare/README.md#wsl2-on-the-windows-boot). The Ollama fallback is only in the iMac API.
- **Long requests:** Cloudflare documents a 125 s proxy read timeout (error 524) for proxied traffic below Enterprise. Long completions should stream.
- **Workers Free:** 100,000 requests/day (Error 1027 beyond that) and 10 ms CPU per request. Waiting on the network doesn't count as CPU time.
- **Rate limiting:** approximate and counted per Cloudflare location. It stops sequential overuse but did not stop concurrent bursts (see [spike results](#spike-results)), so it won't protect the GPU from a client sending many requests at once. A strict limit would need a Durable Object.
- **`n` is not limited:** a client can multiply GPU load with `n` up to what vLLM accepts, so only give keys to clients you trust.
