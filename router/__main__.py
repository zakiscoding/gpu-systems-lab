"""CLI: python -m router {route,run,targets} ...

  route   decide and explain (never runs anything)
  run     decide, then execute: local over SSH, or Modal via modal/run_job.py
          (Modal runs cost money, so they need --yes)
  targets show the catalog, marking measured vs estimated speeds
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from .catalog import REPO, load_targets
from .jobs import JOBS
from .probe import LocalState, probe_local
from .route import Decision, Job, decide

MODAL = [sys.executable, "-m", "modal", "run", "modal/run_job.py"]


def job_from_args(a: argparse.Namespace) -> Job:
    base = JOBS[a.job] if a.job else Job("custom", vram_gb=1, local_minutes=1)
    over = {k: getattr(a, k) for k in ("vram_gb", "local_minutes", "gpus", "max_usd", "deadline_min", "command")
            if getattr(a, k) is not None}
    return Job(**{**base.__dict__, **over})


def show(d: Decision) -> None:
    j, loc = d.job, d.local
    print(f"job {j.name}: {j.vram_gb:g} GB/GPU x{j.gpus}, {j.local_minutes:g} min on RTX 5070, "
          f"cap ${j.max_usd:g}, deadline {j.deadline_min:g} min")
    state = (f"online, {loc.util_pct:.0f}% util, {loc.compute_procs} CUDA procs, "
             f"{loc.vram_free_gb:.1f}/{loc.vram_total_gb:.1f} GB free"
             if loc.online else f"offline ({loc.error})")
    print(f"local RTX 5070: {state}\n")
    print(f"  {'target':<10} {'TFLOPS':>7} {'run min':>8} {'+queue':>7} {'est $':>7}  verdict")
    for o in d.options:
        t = o.target
        tf = f"{t.bf16_tflops:.0f}{'' if t.measured else '*'}"
        mark = "<- CHOSEN" if d.choice is o else ("ok" if o.feasible else "; ".join(o.reasons))
        print(f"  {t.name:<10} {tf:>7} {o.runtime_min:>8.1f} {o.total_min:>7.1f} {o.cost_usd:>7.2f}  {mark}")
    print("\n  * estimated TFLOPS (not measured here); prices approximate, verify at modal.com/pricing")
    if d.choice is None:
        print("\nno target satisfies every constraint: relax the deadline, cost cap, or VRAM")
    else:
        c = d.choice
        why = "free and idle" if c.target.local else f"cheapest feasible (~${c.cost_usd:.2f})"
        print(f"\n=> {c.target.name}{f' x{j.gpus}' if j.gpus > 1 else ''}: {why}, "
              f"~{c.total_min:.1f} min to result")


def execute(d: Decision, yes: bool) -> int:
    c, j = d.choice, d.job
    if c is None:
        return 2
    if not j.command:
        print("job has no command to run")
        return 2
    if c.target.local:
        remote = j.command.replace("python ", ".venv/bin/python ", 1) if j.command.startswith("python ") else j.command
        cmd = ["ssh", "-o", "BatchMode=yes", "gpu-ts", f"cd ~/gpu-systems-lab && {remote}"]
    else:
        if not yes:
            print(f"\nModal run costs money (~${c.cost_usd:.2f}); re-run with --yes to launch")
            return 3
        gpu = c.target.name + (f":{j.gpus}" if j.gpus > 1 else "")
        cmd = MODAL + ["--gpu", gpu, "--command", j.command]
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m router", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("route", "run"):
        s = sub.add_parser(name)
        s.add_argument("--job", choices=sorted(JOBS))
        s.add_argument("--vram-gb", type=float)
        s.add_argument("--local-minutes", type=float)
        s.add_argument("--gpus", type=int)
        s.add_argument("--max-usd", type=float)
        s.add_argument("--deadline-min", type=float)
        s.add_argument("--command")
        s.add_argument("--assume-local", choices=("probe", "idle", "busy", "offline"), default="probe",
                       help="skip the SSH probe and pretend (for demos and tests)")
        if name == "run":
            s.add_argument("--yes", action="store_true", help="allow a paid Modal launch")
    sub.add_parser("targets")
    a = p.parse_args(argv)

    targets = load_targets()
    if a.cmd == "targets":
        for t in targets:
            src = "measured" if t.measured else "estimated"
            print(f"{t.name:<10} {t.vram_gb:>4g} GB  {t.bf16_tflops:>6.1f} TFLOPS ({src})  "
                  f"${t.usd_per_gpu_hour:.2f}/h  queue ~{t.queue_s:g}s  {t.note}")
        return 0

    local = {
        "probe": None,
        "idle": LocalState(True, 12.0, 1.2, 0, 0),
        "busy": LocalState(True, 12.0, 9.5, 97, 1),
        "offline": LocalState(False, error="simulated"),
    }[a.assume_local] or probe_local()
    d = decide(job_from_args(a), targets, local)
    show(d)
    return execute(d, a.yes) if a.cmd == "run" else (0 if d.choice else 2)


if __name__ == "__main__":
    sys.exit(main())
