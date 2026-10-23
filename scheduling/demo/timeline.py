#!/usr/bin/env python3
"""Per-pod timeline for the queue demo: created / scheduled / container start / end, plus the
benchmark's own JSON line (TFLOPS). Usage: python3 timeline.py [label-selector] [namespace]"""
import json
import subprocess
import sys
from datetime import datetime

sel = sys.argv[1] if len(sys.argv) > 1 else "demo=gpu-queue"
ns = sys.argv[2] if len(sys.argv) > 2 else "gpu-jobs"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


pods = json.loads(subprocess.check_output(
    ["kubectl", "-n", ns, "get", "pods", "-l", sel, "-o", "json"], text=True))["items"]
rows = []
for p in pods:
    name = p["metadata"]["name"]
    created = ts(p["metadata"]["creationTimestamp"])
    sched = next((ts(c["lastTransitionTime"]) for c in p["status"].get("conditions", [])
                  if c["type"] == "PodScheduled" and c["status"] == "True"), None)
    term = (p["status"].get("containerStatuses") or [{}])[0].get("state", {}).get("terminated", {})
    start, end = term.get("startedAt"), term.get("finishedAt")
    bench = {}
    for line in subprocess.run(["kubectl", "-n", ns, "logs", name], capture_output=True,
                               text=True).stdout.splitlines():
        if line.startswith("{"):
            bench = json.loads(line)
    rows.append((created, name, sched, ts(start) if start else None, ts(end) if end else None, bench))

rows.sort(key=lambda r: (r[3] or r[0]))
t0 = min(r[0] for r in rows)
f = lambda t: f"{(t - t0).total_seconds():6.0f}s" if t else "     -"
print(f"{'pod':<20} {'created':>7} {'sched':>7} {'start':>7} {'end':>7} {'run_s':>6} {'TFLOPS':>7}")
for created, name, sched, start, end, b in rows:
    run = f"{(end - start).total_seconds():6.0f}" if start and end else "     -"
    print(f"{name:<20} {f(created)} {f(sched)} {f(start)} {f(end)} {run} {b.get('tflops_avg', '-'):>7}")
ends = [r[4] for r in rows if r[4]]
if ends:
    print(f"makespan (first created -> last finished): {(max(ends) - t0).total_seconds():.0f}s")
