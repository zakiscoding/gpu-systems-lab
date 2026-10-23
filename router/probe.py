"""Live state of the local GPU node, read over SSH.

The node dual-boots (Ubuntu = zak-gpu-node, Windows = zak-ts), so "offline"
is a normal answer, not an error: the router then simply skips the local card.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

SSH_HOST = "gpu-ts"
QUERY = "memory.total,memory.used,utilization.gpu"


@dataclass(frozen=True)
class LocalState:
    online: bool
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    util_pct: float = 0.0
    compute_procs: int = 0
    error: str = ""

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    @property
    def busy(self) -> bool:
        # The desktop session alone holds ~1.2 GB for graphics, so used VRAM
        # isn't a signal; a running CUDA process or real utilization is.
        return self.util_pct >= 20 or self.compute_procs > 0


def probe_local(host: str = SSH_HOST, timeout_s: float = 10.0) -> LocalState:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout_s)}", host,
           f"nvidia-smi --query-gpu={QUERY} --format=csv,noheader,nounits"
           " && nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 5)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return LocalState(False, error=str(e))
    if out.returncode != 0:
        lines = (out.stderr or out.stdout).strip().splitlines()
        return LocalState(False, error=lines[-1] if lines else f"exit {out.returncode}")
    gpu_line, procs = out.stdout.strip().splitlines()[:2]
    total, used, util = (float(x) for x in gpu_line.split(","))
    return LocalState(True, total / 1024, used / 1024, util, int(procs))
