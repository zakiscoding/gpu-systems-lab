# Cloudflare Tunnel

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `cloudflare/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

A `cloudflared` container on `zak-gpu-node`. It makes outbound connections to Cloudflare, so no router
ports are opened and nothing is published on the node. It serves two paths:

| Path | Origin | Who gets in |
|---|---|---|
| Private, through the [Worker gateway](../gateway/README.md) | `vllm:8000` (no public hostname) | clients with a gateway key |
| `grafana.<domain>` | `grafana:3000` | emails on an Access allow policy |

The tailnet addresses in [serving/](../serving/README.md) and [observability/](../observability/README.md)
don't change, and `homelab-services` keeps calling vLLM over Tailscale.

**Status: running in [WSL2](#wsl2-on-the-windows-boot) for the gateway path only.** The Ubuntu node and the
Grafana route haven't been set up or tested. `serving_default` is confirmed in WSL2; `gpu-observability_default`
comes from the compose project name, so confirm it with `docker network ls` on the node.

## Setup

1. In the Cloudflare One dashboard, create a remotely managed tunnel with the Docker connector and copy its token.
2. On the node, start the connector:

   ```bash
   cd ~/gpu-systems-lab/serving && docker compose up -d
   cd ~/gpu-systems-lab/observability && docker compose up -d
   cd ~/gpu-systems-lab/cloudflare && cp .env.example .env && chmod 600 .env
   # edit .env: paste TUNNEL_TOKEN
   docker compose up -d && docker compose logs --tail 20 cloudflared
   ```

3. **Grafana, in this order.** Create an Access application for `grafana.<domain>` with an Allow policy for
   your email first. Only then add the tunnel's public hostname `grafana.<domain>` -> `http://grafana:3000`.
   Grafana allows anonymous viewers, so without Access the dashboard would be public.
4. **vLLM:** add no public hostname. Follow the [gateway setup](../gateway/README.md#setup).

## WSL2 on the Windows boot

The GPU PC can run this path from Ubuntu in WSL2 (`zak-ts`) while it is booted into Windows. Only `serving/`
runs there, so the tunnel carries the gateway path and has no Grafana origin:

```bash
cd ~/gpu-systems-lab/serving && docker compose -f compose.yml -f compose.wsl.yml up -d
cd ~/gpu-systems-lab/cloudflare && cp .env.example .env && chmod 600 .env
# edit .env: paste TUNNEL_TOKEN
docker compose -f compose.yml -f compose.wsl.yml up -d
```

**The distro stops seconds after its last session closes**, even with containers running and `restart: unless-stopped`
(tested: PID 1, dockerd and vLLM all restarted on each new `wsl.exe` session). The gateway is offline until something
starts the distro again, then vLLM takes from about 45 s to 3 minutes to reload.

The Windows scheduled task **WSL Keep Alive** holds a session open by running `wsl.exe -d Ubuntu --exec sleep infinity`.
It runs at startup as `zakar` with S4U logon, so no password is stored. With it running, the distro and vLLM stayed up
through 90 s with no other session (tested). Starting at boot before anyone logs in hasn't been tested with a reboot.
To undo, run this in an elevated PowerShell, then `wsl.exe --terminate Ubuntu`:

```powershell
Unregister-ScheduledTask -TaskName 'WSL Keep Alive' -Confirm:$false
```

## Ubuntu boot

**Written but not tested:** the PC hasn't been booted into Ubuntu since this was added. Only one boot runs at a time, and both
can use the same tunnel. The Worker holds one `VLLM_API_KEY`, and `homelab-services` already uses the node's key, so make the
node's key the shared one.

1. Get this code onto the node (`git pull` once it's pushed), then on the node:

   ```bash
   cd ~/gpu-systems-lab/serving && docker compose up -d
   cd ~/gpu-systems-lab/observability && docker compose up -d
   cd ~/gpu-systems-lab/cloudflare && cp .env.example .env && chmod 600 .env
   # edit .env: paste TUNNEL_TOKEN
   docker compose up -d && docker compose logs --tail 20 cloudflared
   ```

2. From the laptop, in `gateway/`, give the Worker the node's key. The value is piped, so it isn't displayed:

   ```bash
   ssh gpu-ts "grep ^VLLM_API_KEY= ~/gpu-systems-lab/serving/.env | cut -d= -f2" | tr -d '\r\n' | npx wrangler secret put VLLM_API_KEY
   ```

3. Next time the PC is on Windows, put the same value in `serving/.env` in WSL and recreate that container, or the gateway gets a 502.

4. Confirm chat replies carry `x-gateway-backend: gpu`.

## Rules

- **Never add a hostname for Prometheus or node-exporter.** `cloudflared` shares a Docker network with both.
  They aren't reachable from outside unless a hostname is configured for them. Prometheus has no auth and
  has `--web.enable-lifecycle` on.
- Keep the tunnel's catch-all rule at 404.
- k3s, SSH and MPI stay on Tailscale.

## Rollback

```bash
cd ~/gpu-systems-lab/cloudflare && docker compose down
```

This takes both paths offline at once and leaves the tailnet untouched. Delete the tunnel in the dashboard
to retire it for good.
