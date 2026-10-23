#!/usr/bin/env python3
"""Submit a GPU job to the k3s cluster, wait for it, print logs and timings.

Stdlib only; talks to the cluster through `kubectl` (run on the GPU node, KUBECONFIG=~/.kube/config).

    python3 submit.py --name smoke --image nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04 -- nvidia-smi
    python3 submit.py --name mm --env DURATION=20 --configmap gpu-bench:/bench -- python /bench/bench.py
    python3 submit.py --spec job.json          # {"name":..., "image":..., "command":[...], "gpus":1, "env":{...}}
    python3 submit.py --name mm --no-wait ...  # submit and return (the Job queues if the GPU is busy)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"


def kubectl(*args, input=None, check=True):
    r = subprocess.run(["kubectl", *args], input=input, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"kubectl {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout


def build_job(spec):
    name, ns = spec["name"], spec.get("namespace", "gpu-jobs")
    container = {
        "name": "main",
        "image": spec.get("image", DEFAULT_IMAGE),
        "command": spec["command"],
        "env": [{"name": k, "value": str(v)} for k, v in spec.get("env", {}).items()],
        "resources": {"limits": {"nvidia.com/gpu": int(spec.get("gpus", 1))}},
    }
    pod = {"restartPolicy": "Never", "runtimeClassName": "nvidia", "containers": [container]}
    if spec.get("gpus", 1) == 0:  # CPU-only job: no GPU limit, no NVIDIA runtime
        del container["resources"], pod["runtimeClassName"]
    vols = []
    for i, m in enumerate(spec.get("configmaps", [])):
        cm, path = m.split(":", 1)
        vols.append({"name": f"cm{i}", "configMap": {"name": cm}})
        container.setdefault("volumeMounts", []).append({"name": f"cm{i}", "mountPath": path})
    if vols:
        pod["volumes"] = vols
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": ns, "labels": {"submitted-by": "submit.py"}},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": int(spec.get("ttl", 3600)),
            "activeDeadlineSeconds": int(spec.get("timeout", 3600)),
            "template": {"metadata": {"labels": {"submitted-by": "submit.py"}}, "spec": pod},
        },
    }


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def pod_for(name, ns):
    out = json.loads(kubectl("-n", ns, "get", "pods", "-l", f"job-name={name}", "-o", "json"))
    items = out["items"]
    return items[0] if items else None


def wait(name, ns, timeout):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        job = json.loads(kubectl("-n", ns, "get", "job", name, "-o", "json"))
        pod = pod_for(name, ns)
        state = "no pod yet"
        if pod:
            phase = pod["status"].get("phase")
            conds = {c["type"]: c for c in pod["status"].get("conditions", [])}
            sched = conds.get("PodScheduled", {})
            if sched.get("status") == "False":
                state = f"{phase} ({sched.get('reason')}: {sched.get('message')})"
            else:
                cs = (pod["status"].get("containerStatuses") or [{}])[0].get("state", {})
                waiting = cs.get("waiting", {}).get("reason")
                state = f"{phase}" + (f" ({waiting})" if waiting else "")
        if state != last:
            print(f"[{datetime.now():%H:%M:%S}] {state}", flush=True)
            last = state
        for c in job["status"].get("conditions", []):
            if c["type"] in ("Complete", "Failed") and c["status"] == "True":
                return c["type"], job, pod
        time.sleep(2)
    return "Timeout", job, pod_for(name, ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="JSON job spec file")
    ap.add_argument("--name")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--namespace", default="gpu-jobs")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--configmap", action="append", default=[], metavar="NAME:PATH")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds (also the Job activeDeadline)")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the Job JSON and exit")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if a.spec:
        with open(a.spec) as f:
            spec = json.load(f)
    else:
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not (a.name and cmd):
            ap.error("--name and a command (after --) are required without --spec")
        spec = {"name": a.name, "image": a.image, "command": cmd, "gpus": a.gpus,
                "namespace": a.namespace, "timeout": a.timeout, "configmaps": a.configmap,
                "env": dict(e.split("=", 1) for e in a.env)}
    job = build_job(spec)
    name, ns = job["metadata"]["name"], job["metadata"]["namespace"]
    if a.dry_run:
        print(json.dumps(job, indent=2))
        return
    print(kubectl("create", "-f", "-", input=json.dumps(job)).strip())
    if a.no_wait:
        return

    result, job, pod = wait(name, ns, spec.get("timeout", a.timeout) + 60)
    if pod:
        print("---- logs ----")
        print(kubectl("-n", ns, "logs", pod["metadata"]["name"], check=False).rstrip())
        print("--------------")
        created = ts(pod["metadata"]["creationTimestamp"])
        cs = (pod["status"].get("containerStatuses") or [{}])[0].get("state", {})
        term = cs.get("terminated", {})
        started, finished = ts(term.get("startedAt")), ts(term.get("finishedAt"))
        if started and finished:
            print(f"queue+pull wait {(started - created).total_seconds():.0f}s | "
                  f"run {(finished - started).total_seconds():.0f}s | "
                  f"total {(finished - created).total_seconds():.0f}s | exit {term.get('exitCode')}")
    print(f"result: {result}")
    sys.exit(0 if result == "Complete" else 1)


if __name__ == "__main__":
    main()
