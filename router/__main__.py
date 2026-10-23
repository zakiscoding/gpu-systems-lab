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

# argv prefix for a Modal launch; run from the repo root (execute() sets cwd=REPO)
MODAL = [sys.executable, "-m", "modal", "run", "modal/run_job.py"]


def job_from_args(a: argparse.Namespace) -> Job:
    """Build the Job to route: a named job from JOBS (or a 1 GB / 1 min "custom"
    placeholder), with any flags given on the command line overriding its fields."""
    base = JOBS[a.job] if a.job else Job("custom", vram_gb=1, local_minutes=1)
    over = {k: getattr(a, k) for k in ("vram_gb", "local_minutes", "gpus", "max_usd", "deadline_min", "command")
            if getattr(a, k) is not None}
    return Job(**{**base.__dict__, **over})


def show(d: Decision) -> None:
    """Print the decision as a table: the job, the local card's live state, every
    target's runtime / time-to-result (min) / estimated cost (USD) / verdict, and
    the final choice with the reason it won. Estimated TFLOPS are marked with '*'."""
    j, loc = d.job, d.local
    print(f"job {j.name}: {j.vram_gb:g} GB/GPU x{j.gpus}, {j.local_minutes:g} min on RTX 5070, "
          f"cap ${j.max_usd:g}, deadline {j.deadline_min:g} min")
    resident = f", resident: {', '.join(loc.resident)}" if loc.resident else ""
    state = (f"online, {loc.util_pct:.0f}% util, {loc.compute_procs} other CUDA procs{resident}, "
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


def build_command(d: Decision) -> list[str]:
    """argv that runs the job on the chosen target: SSH to the node, or modal/run_job.py."""
    c, j = d.choice, d.job
    if c.target.local:
        # on the node, run Python jobs with the repo's own venv interpreter
        remote = j.command.replace("python ", ".venv/bin/python ", 1) if j.command.startswith("python ") else j.command
        return ["ssh", "-o", "BatchMode=yes", "gpu-ts", f"cd ~/gpu-systems-lab && {remote}"]
    gpu = c.target.name + (f":{j.gpus}" if j.gpus > 1 else "")
    return MODAL + ["--gpu", gpu, "--command", j.command]


def execute(d: Decision, yes: bool) -> int:
    """Run the chosen job in a subprocess (SSH or `modal run`) and return its exit code.

    Refuses without running anything when there is no feasible target or no command
    (exit 2), or when the choice is a paid Modal GPU and `yes` is False (exit 3).
    """
    c, j = d.choice, d.job
    if c is None:
        return 2
    if not j.command:
        print("job has no command to run")
        return 2
    if not c.target.local and not yes:
        print(f"\nModal run costs money (~${c.cost_usd:.2f}); re-run with --yes to launch")
        return 3
    cmd = build_command(d)
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 ok, 2 nothing feasible / no command,
    3 paid Modal run not confirmed with --yes, otherwise the job's own exit code."""
    p =argparse.ArgumentParser(prog="python -m router", description=__doc__,
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

    # canned node states for --assume-local; "probe" (None) falls through to a live SSH probe.
    # LocalState(online, vram_total_gb, vram_used_gb, util_pct, compute_procs)
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
