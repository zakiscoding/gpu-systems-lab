"""NVIDIA nccl-tests on a 2x H100 Modal container, compared with what NVLink should give.

Builds nccl-tests (pinned tag) against the CUDA image's NCCL and runs the standard
collectives with both GPUs driven by one process (-g 2), 8 B to 8 GiB, with
correctness checking on (-c 1). Also records the topology (nvidia-smi topo -m,
NVLink status) and the transport NCCL picked (NCCL_DEBUG=INFO), and repeats
all_reduce with peer-to-peer disabled to show what the same collective costs
through host shared memory.

Raw logs and the parsed JSON go to results/nccl_tests/ on the laptop.

Usage (from the repo root):
    modal run modal/nccl_tests.py                  # H100:2, full sweep (~10 min, ~$1.5)
    modal run modal/nccl_tests.py --quick          # 8 B .. 64 MiB, pipeline check
    python modal/nccl_parse.py results/nccl_tests/<run>/all_reduce_perf.log
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NCCL_TESTS_TAG = "v2.21.1"
DEFAULT_GPU = os.environ.get("MODAL_LAB_NCCL_GPU", "H100:2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")  # the devel image already ships (and apt-holds) libnccl2 + libnccl-dev
    .run_commands(
        f"git clone --depth 1 --branch {NCCL_TESTS_TAG} https://github.com/NVIDIA/nccl-tests /opt/nccl-tests",
        # sm_90 only: we only run on H100, and it keeps the build short
        "cd /opt/nccl-tests && make -j8 MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr "
        "NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'",
    )
    .add_local_file(HERE / "nccl_parse.py", "/root/nccl_parse.py")
)

app = modal.App("gpu-lab-nccl-tests", image=image)

TESTS = ["all_reduce_perf", "all_gather_perf", "reduce_scatter_perf",
         "broadcast_perf", "alltoall_perf", "sendrecv_perf"]


def sh(cmd: str, env: dict | None = None, timeout: int = 600) -> str:
    """Run a shell command, return stdout+stderr (never raises on a non-zero exit)."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    return p.stdout + p.stderr


@app.function(gpu=DEFAULT_GPU, timeout=25 * 60)
def run_tests(max_bytes: str, iters: int):
    """In the container: topology, then each test, then all_reduce without P2P.
    Returns {name: raw log}."""
    import sys
    sys.path.insert(0, "/root")
    logs = {
        "topo": sh("nvidia-smi topo -m"),
        "nvlink": sh("nvidia-smi nvlink -s"),
        "gpus": sh("nvidia-smi --query-gpu=name,pci.bus_id,driver_version,clocks.max.sm,power.limit --format=csv"),
    }
    base = f"-b 8 -e {max_bytes} -f 2 -g 2 -w 5 -n {iters} -c 1"
    env = {**os.environ}
    # one INFO run so the log shows which transport (P2P/CUMEM, NVLS, SHM) NCCL chose
    logs["all_reduce_perf"] = sh(f"/opt/nccl-tests/build/all_reduce_perf {base}",
                                 env={**env, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH"})
    for t in TESTS[1:]:
        logs[t] = sh(f"/opt/nccl-tests/build/{t} {base}", env=env)
    logs["all_reduce_perf_no_p2p"] = sh(f"/opt/nccl-tests/build/all_reduce_perf {base}",
                                        env={**env, "NCCL_P2P_DISABLE": "1"})
    return logs


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, quick: bool = False, iters: int = 20):
    """Run on Modal, save raw logs + parsed summary under results/nccl_tests/<ts>/."""
    import sys
    sys.path.insert(0, str(HERE))
    import nccl_parse  # local module next to this file

    t0 = time.perf_counter()
    logs = run_tests.with_options(gpu=gpu).remote("64M" if quick else "8G", iters)
    wall = time.perf_counter() - t0

    out = REPO / "results" / "nccl_tests" / f"{gpu.replace(':', 'x')}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in logs.items():
        (out / f"{name}.log").write_text(text)

    summary = {"gpu_request": gpu, "nccl_tests": NCCL_TESTS_TAG, "wall_seconds": round(wall, 1),
               "quick": quick, "tests": {}}
    for name, text in logs.items():
        if name.endswith("_perf") or name.endswith("_no_p2p"):
            parsed = nccl_parse.parse(text)
            summary["tests"][name] = parsed
            summary.setdefault("nccl_version", parsed.get("nccl_version"))
    summary["transport"] = nccl_parse.transports(logs["all_reduce_perf"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'test':<26}{'peak busbw GB/s':>16}{'at size':>12}{'wrong':>7}")
    for name, s in summary["tests"].items():
        pk = s.get("peak") or {}
        print(f"{name:<26}{pk.get('busbw_gbs', '-'):>16}{nccl_parse.human(pk.get('bytes')):>12}{s.get('wrong_total', '-'):>7}")
    print(f"transport: {summary['transport']}")
    print(f"[modal] wall {wall:.0f}s, wrote {out}")
