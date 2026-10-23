"""Homelab job-queue worker: claims routed jobs from the iMac API and runs them via the router.

The homelab control plane (homelab-services on the iMac, Postgres as source of truth) queues
`train` jobs. Jobs submitted with a `route` spec are claimed only by this worker; the Windows
QLoRA worker skips them. For each job:

  heartbeat -> claim (routed=true) -> decide (live probe of the RTX node) -> run locally over SSH
  or on Modal -> record the decision as the job's experiment -> mark succeeded / failed

Runs on the laptop (the homelab "controller"): it has SSH to the node and the Modal client.

Config (environment, or router/worker.local.env, which is gitignored by *.local.*):
  API_BASE_URL    default http://100.91.113.98:8000
  WORKER_API_KEY  the iMac API's worker key (required)
  WORKER_ID       default router-<hostname>
  MAX_USD_PER_JOB cap for jobs that don't set max_usd (default 1.0)

  python -m router.homelab_worker            # poll forever
  python -m router.homelab_worker --once     # one claim attempt (tests)

Submit a job (from anywhere on the tailnet, with the user API key):
  curl -X POST http://100.91.113.98:8000/jobs -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
    -d '{"route": {"command": "python benchmarks/smoke_test.py --size 4096", "vram_gb": 1, "local_minutes": 0.5}}'

Leases last 15 min (LEASE_MINUTES_TRAIN on the API). While a job runs, a background thread
renews the lease every LEASE_RENEW_SEC via POST /internal/worker/jobs/{id}/renew, so long jobs
aren't requeued by another worker's stale-lease sweep.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .__main__ import build_command
from .catalog import REPO, load_targets
from .probe import probe_local
from .route import Decision, Job, decide

ENV_FILE = Path(__file__).with_name("worker.local.env")
LEASE_RENEW_SEC = 300  # train leases are 15 min on the API; renew every 5
# Job fields a `route` spec may set; anything else in the spec is ignored (except `command`)
ROUTE_KEYS = ("vram_gb", "local_minutes", "gpus", "max_usd", "deadline_min")


def _load_env_file() -> None:
    """Load KEY=VALUE lines from ENV_FILE into os.environ, if the file exists.
    Real environment variables win (setdefault); '#' lines are skipped."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Api:
    """Thin client for the homelab API's worker endpoints, authenticated with the
    worker key and identified by WORKER_ID / hostname headers."""

    def __init__(self) -> None:
        """Read config from the environment; exits the process if WORKER_API_KEY is unset."""
        self.base = os.environ.get("API_BASE_URL", "http://100.91.113.98:8000").rstrip("/")
        key = os.environ.get("WORKER_API_KEY", "")
        if not key:
            raise SystemExit(f"WORKER_API_KEY not set (environment or {ENV_FILE})")
        self.worker_id = os.environ.get("WORKER_ID", f"router-{socket.gethostname()}")
        self.headers = {"X-Worker-Key": key, "X-Worker-Id": self.worker_id, "X-Worker-Host": socket.gethostname()}
        self.http = httpx.Client(timeout=30.0)

    def post(self, path: str, **kw) -> httpx.Response:
        """POST to the API (30 s timeout); raises httpx.HTTPStatusError on 4xx/5xx."""
        r =self.http.post(f"{self.base}{path}", headers=self.headers, **kw)
        r.raise_for_status()
        return r

    def claim(self) -> dict | None:
        """Lease the next routed job, or None when the queue is empty (HTTP 204)."""
        r =self.post("/internal/worker/jobs/claim", params={"routed": "true"})
        return None if r.status_code == 204 else r.json()


def _renew_lease_loop(api: "Api", job_id: str, stop: threading.Event) -> None:
    """Background keep-alive: POST /renew every LEASE_RENEW_SEC until `stop` is set.

    Uses its own HTTP client (httpx.Client isn't shared across threads here). A 409 means the
    job was requeued or taken over; it's logged, and the run's final report will then fail.
    """
    with httpx.Client(timeout=30.0) as http:
        while not stop.wait(LEASE_RENEW_SEC):
            try:
                http.post(f"{api.base}/internal/worker/jobs/{job_id}/renew", headers=api.headers).raise_for_status()
            except httpx.HTTPError as e:
                print(f"[router-worker] lease renew failed for {job_id}: {e}", flush=True)


def job_from_payload(job_id: str, payload: dict) -> Job:
    """Build a router Job from a queued job's `route` spec.

    `command` is required; vram_gb and local_minutes default to 1, and max_usd
    falls back to MAX_USD_PER_JOB so a job without a cap can't run up a bill.
    Raises KeyError/TypeError/ValueError on a malformed spec.
    """
    spec =payload["route"]
    kw = {k: spec[k] for k in ROUTE_KEYS if spec.get(k) is not None}
    kw.setdefault("max_usd", float(os.environ.get("MAX_USD_PER_JOB", "1.0")))
    return Job(name=f"job-{job_id[:8]}", command=str(spec["command"]),
               vram_gb=float(kw.pop("vram_gb", 1)), local_minutes=float(kw.pop("local_minutes", 1)), **kw)


def decision_record(d: Decision, exit_code: int | None, seconds: float) -> dict:
    """The routing decision and its outcome as JSON for the job's experiment record:
    chosen target, estimates (min, USD), local state, why each other target was
    rejected, exit code and wall time (s)."""
    c =d.choice
    return {
        "router": {
            "target": c.target.name if c else None,
            "gpus": d.job.gpus,
            "est_minutes": round(c.total_min, 2) if c else None,
            "est_usd": round(c.cost_usd, 4) if c else None,
            "local_state": "online" if d.local.online else f"offline: {d.local.error}",
            "local_busy": d.local.busy if d.local.online else None,
            "rejected": {o.target.name: o.reasons for o in d.options if not o.feasible},
            "exit_code": exit_code,
            "wall_seconds": round(seconds, 1),
        }
    }


def run_job(api: Api, job: dict) -> None:
    """Route and run one claimed job, then report the result to the API.

    Side effects: an SSH probe of the node, then the job itself as a blocking
    subprocess (SSH to the node, or a paid `modal run`; no --yes gate here, the
    max_usd cap is the guard). Bad specs and unroutable jobs fail as
    "permanent"; a non-zero exit fails as "transient" so the API may retry.
    The experiment record is best-effort: an API error there only warns.
    """
    job_id = job["id"]
    payload = job.get("payload") or {}
    try:
        spec = job_from_payload(job_id, payload)
    except (KeyError, TypeError, ValueError) as e:
        api.post(f"/internal/worker/jobs/{job_id}/failed",
                 json={"error_message": f"bad route spec: {e}", "error_class": "permanent"})
        return

    d = decide(spec, load_targets(), probe_local())
    if d.choice is None:
        why = "; ".join(f"{o.target.name}: {', '.join(o.reasons)}" for o in d.options)
        api.post(f"/internal/worker/jobs/{job_id}/failed",
                 json={"error_message": f"no feasible target ({why})"[:2000], "error_class": "permanent"})
        print(f"[router-worker] {job_id}: no feasible target")
        return

    cmd = build_command(d)
    print(f"[router-worker] {job_id}: -> {d.choice.target.name} "
          f"(~{d.choice.total_min:.1f} min, ~${d.choice.cost_usd:.2f}): {spec.command}", flush=True)
    # output is decoded as UTF-8 below, so make the child write UTF-8 too (Windows
    # defaults to a legacy codepage) and drop deprecation noise from the captured log
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore::DeprecationWarning")
    t0 = time.perf_counter()
    stop = threading.Event()
    keepalive = threading.Thread(target=_renew_lease_loop, args=(api, job_id, stop), daemon=True)
    keepalive.start()
    try:
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        stop.set()
        keepalive.join(timeout=5)
    secs = time.perf_counter() - t0
    out = (p.stdout + p.stderr).strip()
    print(out[-3000:], flush=True)

    try:
        api.post(f"/internal/worker/jobs/{job_id}/experiment",
                 json={"hyperparams": decision_record(d, p.returncode, secs) | {"route": payload["route"]},
                       "model_version": d.choice.target.name})
    except httpx.HTTPError as e:
        print(f"[router-worker] experiment record warning: {e}")

    if p.returncode == 0:
        api.post(f"/internal/worker/jobs/{job_id}/succeeded")
        print(f"[router-worker] {job_id}: succeeded on {d.choice.target.name} in {secs:.0f}s")
    else:
        api.post(f"/internal/worker/jobs/{job_id}/failed",
                 json={"error_message": f"exit {p.returncode} on {d.choice.target.name}: {out[-1500:]}",
                       "error_class": "transient"})
        print(f"[router-worker] {job_id}: failed (exit {p.returncode})")


def main(argv: list[str] | None = None) -> int:
    """Poll loop: heartbeat, claim, run, sleep `--poll` seconds (WORKER_POLL_SEC,
    default 10). API errors are logged and the loop carries on; jobs run one at a time."""
    ap =argparse.ArgumentParser(prog="python -m router.homelab_worker")
    ap.add_argument("--once", action="store_true", help="one claim attempt, then exit")
    ap.add_argument("--poll", type=float, default=float(os.environ.get("WORKER_POLL_SEC", "10")))
    a = ap.parse_args(argv)
    _load_env_file()
    api = Api()
    print(f"[router-worker] {api.worker_id} polling {api.base}", flush=True)
    while True:
        try:
            api.post("/internal/worker/heartbeat")
            job = api.claim()
            if job:
                run_job(api, job)
            elif a.once:
                print("[router-worker] no routed job pending")
        except httpx.HTTPError as e:
            print(f"[router-worker] API error: {e}", flush=True)
        if a.once:
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
