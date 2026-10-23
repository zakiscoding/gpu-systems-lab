"""Live state of the local GPU node, read over SSH.

The node dual-boots (Ubuntu = zak-gpu-node, Windows = zak-ts), so "offline"
is a normal answer, not an error: the router then simply skips the local card.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

SSH_HOST = "gpu-ts"
# Long-running inference servers that hold VRAM but idle most of the time (the homelab's
# always-on vLLM chat backend). Their memory counts against free VRAM, but they don't
# make the card "busy". Regex on nvidia-smi's process name.
RESIDENT_PROCS = os.environ.get("ROUTER_RESIDENT_PROCS", r"VLLM::EngineCore")

# three utilization samples ~0.5 s apart (max), then memory and the compute processes
REMOTE = (
    "for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 0.5; done;"
    " echo ---; nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits;"
    " echo ---; nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits"
)


@dataclass(frozen=True)
class LocalState:
    online: bool
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    util_pct: float = 0.0
    compute_procs: int = 0  # CUDA processes other than resident servers
    error: str = ""
    resident: tuple[str, ...] = field(default=())  # resident servers found (e.g. vLLM)

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    @property
    def busy(self) -> bool:
        # The desktop (~1.2 GB) and a resident vLLM server (~5 GB) hold VRAM without
        # computing, so used VRAM isn't a signal; another job or real utilization is.
        return self.util_pct >= 20 or self.compute_procs > 0


def parse(stdout: str) -> LocalState:
    util_part, mem_part, procs_part = (stdout.strip() + "\n").split("---", 2)
    util = max(float(x) for x in util_part.split())
    total, used = (float(x) for x in mem_part.strip().splitlines()[0].split(","))
    names = [ln.split(",")[0].strip() for ln in procs_part.strip().splitlines() if ln.strip()]
    resident = tuple(n for n in names if re.search(RESIDENT_PROCS, n))
    return LocalState(True, total / 1024, used / 1024, util, len(names) - len(resident), resident=resident)


def probe_local(host: str = SSH_HOST, timeout_s: float = 10.0) -> LocalState:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout_s)}", host, REMOTE]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 10)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return LocalState(False, error=str(e))
    if out.returncode != 0:
        lines = (out.stderr or out.stdout).strip().splitlines()
        return LocalState(False, error=lines[-1] if lines else f"exit {out.returncode}")
    try:
        return parse(out.stdout)
    except (ValueError, IndexError) as e:
        return LocalState(False, error=f"unparseable nvidia-smi output: {e}")
