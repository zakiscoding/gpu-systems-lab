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
    """Snapshot of the RTX 5070 node as the router sees it (VRAM in GiB, as
    nvidia-smi MiB / 1024). `online=False` carries the reason in `error`."""

    online: bool
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    util_pct: float = 0.0
    compute_procs: int = 0  # CUDA processes other than resident servers
    error: str = ""
    resident: tuple[str, ...] = field(default=())  # resident servers found (e.g. vLLM)

    @property
    def vram_free_gb(self) -> float:
        """VRAM not held by anything right now, resident servers included."""
        return self.vram_total_gb - self.vram_used_gb

    @property
    def busy(self) -> bool:
        """True when another job is using the card: >= 20% utilization or any
        non-resident CUDA process."""
        # The desktop (~1.2 GB) and a resident vLLM server (~5 GB) hold VRAM without
        # computing, so used VRAM isn't a signal; another job or real utilization is.
        return self.util_pct >= 20 or self.compute_procs > 0


def parse(stdout: str) -> LocalState:
    """Turn the output of REMOTE into a LocalState.

    Takes the max of the utilization samples (one quiet sample shouldn't hide a
    running job), converts MiB to GiB, and splits compute processes into
    resident servers (matching RESIDENT_PROCS) and everything else. Raises
    ValueError/IndexError on malformed output; probe_local() handles that.
    """
    util_part, mem_part, procs_part = (stdout.strip() + "\n").split("---", 2)
    util = max(float(x) for x in util_part.split())
    total, used = (float(x) for x in mem_part.strip().splitlines()[0].split(","))
    names = [ln.split(",")[0].strip() for ln in procs_part.strip().splitlines() if ln.strip()]
    resident = tuple(n for n in names if re.search(RESIDENT_PROCS, n))
    return LocalState(True, total / 1024, used / 1024, util, len(names) - len(resident), resident=resident)


def probe_local(host: str = SSH_HOST, timeout_s: float = 10.0) -> LocalState:
    """SSH to the node (non-interactive, BatchMode) and read its GPU state.

    Never raises: an SSH timeout, missing ssh binary, non-zero exit or
    unparseable output all come back as an offline LocalState with the reason.
    Takes ~1.5 s plus SSH setup when the node is up.
    """
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
