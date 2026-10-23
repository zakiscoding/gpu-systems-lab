"""MPI with one rank per GPU in a single multi-GPU Modal container.

The local RTX 5070 node has one GPU, so its MPI runs share it and never speed
up (see mpi/README.md). Here each rank gets its own GPU: the repo's mpi/
programs are built inside the container with nvcc + Open MPI and run at 1 and
2 ranks, so the Jacobi strong-scaling result is real.

The container uses Ubuntu 24.04's Open MPI 4.1.6 (same as the local node) and
CUDA 12.8; the GPU's compute capability is read at runtime to pick -arch.
Output is saved to results/modal_mpi_<gpu>_<ts>.txt.

Usage (from the repo root):
    modal run modal/mpi_multigpu.py                 # default L4:2
    modal run modal/mpi_multigpu.py --gpu H100:2
    modal run modal/mpi_multigpu.py --gpu H100:2 --cuda-aware   # Open MPI built --with-cuda
    MODAL_LAB_MPI_GPU=H100:2 modal run modal/mpi_multigpu.py
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/lab"
DEFAULT_GPU = os.environ.get("MODAL_LAB_MPI_GPU", "L4:2")

CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
OMPI_TARBALL = "openmpi-4.1.6.tar.bz2"
OMPI_SHA256 = "f740994485516deb63b5311af122c265179f5328a0d857a567b85db00b11e415"


def _add_repo(img: modal.Image) -> modal.Image:
    return img.add_local_dir(REPO / "mpi", f"{REMOTE_REPO}/mpi", ignore=["build", "multinode"]).add_local_file(
        REPO / "cuda" / "common.cuh", f"{REMOTE_REPO}/cuda/common.cuh"
    )


# Stock: Ubuntu's Open MPI, not CUDA-aware -> programs stage GPU data via host.
image = _add_repo(
    modal.Image.from_registry(CUDA_BASE, add_python="3.12").apt_install("openmpi-bin", "libopenmpi-dev", "make")
)

def _build_cuda_ompi():
    """Image build step: Open MPI 4.1.6 from source --with-cuda into /opt/ompi-cuda.

    A run_function step (not run_commands) so the build can request real CPUs;
    on the default image builder the compile took 45+ minutes.
    """
    import subprocess

    subprocess.run(
        f"cd /tmp && wget -q https://download.open-mpi.org/release/open-mpi/v4.1/{OMPI_TARBALL}"
        f" && echo '{OMPI_SHA256}  {OMPI_TARBALL}' | sha256sum -c"
        f" && tar xjf {OMPI_TARBALL} && cd openmpi-4.1.6"
        " && ./configure --prefix=/opt/ompi-cuda --with-cuda=/usr/local/cuda --disable-mpi-fortran"
        " --without-ucx > /tmp/ompi-configure.log"
        " && make -j$(nproc) > /tmp/ompi-make.log && make install > /dev/null"
        " && rm -rf /tmp/openmpi-4.1.6*",
        shell=True,
        check=True,
    )


# CUDA-aware: the same Open MPI 4.1.6, built from source --with-cuda, so MPI
# calls accept device pointers (smcuda BTL on-node). The programs detect this
# with MPIX_Query_cuda_support() and hand GPU buffers straight to MPI.
cuda_aware_image = _add_repo(
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .apt_install("build-essential", "make", "wget", "bzip2", "file")
    .run_function(_build_cuda_ompi, cpu=16, memory=16384, timeout=40 * 60)
    # mpirun's rsh launcher wants an ssh binary even for single-node jobs
    # (Ubuntu's openmpi-bin pulls this in; the source build doesn't)
    .apt_install("openssh-client")
    .env({
        "PATH": "/opt/ompi-cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/opt/ompi-cuda/lib:/usr/local/cuda/lib64",
    })
)

app = modal.App("gpu-lab-mpi-multigpu", image=image)

# Runs inside the container, in order. Jacobi uses a large grid so compute,
# not halo exchange, dominates and the 2-GPU split has something to win.
JACOBI = "8192 500"


def _steps(ngpu: int) -> list[tuple[str, str]]:
    mpirun = "mpirun --allow-run-as-root --bind-to none"
    return [
        ("gpus", "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"),
        # gVisor sandboxes hide the PCI topology, so this is informational only
        ("topology", "nvidia-smi topo -m || echo '(topology not visible in this sandbox)'"),
        ("open mpi", "mpirun --version | head -1 && ompi_info --parsable --all | grep mpi_built_with_cuda_support:value"),
        ("build", "make -C mpi ARCH=$ARCH 2>&1 | grep -vE '^(/usr/local/cuda|mkdir)' || true"),
        ("jacobi, 1 rank / 1 GPU", f"{mpirun} -np 1 mpi/build/jacobi {JACOBI}"),
        (f"jacobi, {ngpu} ranks / {ngpu} GPUs", f"{mpirun} -np {ngpu} mpi/build/jacobi {JACOBI}"),
        ("pingpong, 2 ranks on 2 GPUs", f"{mpirun} -np 2 mpi/build/pingpong 64"),
        (f"allreduce GPU data, {ngpu} ranks", f"{mpirun} -np {ngpu} mpi/build/allreduce --device 256"),
    ]


@app.function(gpu=DEFAULT_GPU, timeout=30 * 60)
def run_mpi():
    """Stock Open MPI. Yields ('started', host) once, then (label, output, exit) per step."""
    yield from _run_steps()


@app.function(gpu=DEFAULT_GPU, image=cuda_aware_image, timeout=30 * 60)
def run_mpi_cuda_aware():
    """Same steps against the CUDA-aware Open MPI build."""
    yield from _run_steps()


def _run_steps():
    import shutil
    import socket

    yield ("started", socket.gethostname(), 0)

    # mounts are read-only; build in a writable copy
    work = Path("/tmp/lab")
    shutil.copytree(REMOTE_REPO, work, dirs_exist_ok=True)

    cc = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.split()
    ngpu = len(cc)
    env = dict(os.environ, ARCH="sm_" + cc[0].replace(".", ""))

    for label, cmd in _steps(ngpu):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, shell=True, cwd=work, env=env, capture_output=True, text=True)
        out = (p.stdout + p.stderr).rstrip() + f"\n[{time.perf_counter() - t0:.1f}s]"
        yield (label, out, p.returncode)


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, cuda_aware: bool = False):
    if ":" not in gpu or int(gpu.split(":")[1]) < 2:
        raise SystemExit(f"--gpu needs a multi-GPU request like L4:2 or H100:2, got {gpu!r}")

    fn = run_mpi_cuda_aware if cuda_aware else run_mpi
    flavor = "cuda-aware" if cuda_aware else "stock"
    t_call = time.perf_counter()
    log = [f"# modal/mpi_multigpu.py --gpu {gpu} ({flavor} Open MPI)  ({datetime.now():%Y-%m-%d %H:%M})"]
    failed = []
    for label, out, code in fn.with_options(gpu=gpu).remote_gen():
        if label == "started":
            print(f"[modal] container up on {gpu} after {time.perf_counter() - t_call:.1f}s")
            continue
        block = f"\n=== {label} (exit {code}) ===\n{out}"
        print(block, flush=True)
        log.append(block)
        if code != 0:
            failed.append(label)

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"modal_mpi_{flavor}_{gpu.replace(':', 'x')}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(log) + "\n")
    print(f"\n[modal] wall {time.perf_counter() - t_call:.1f}s, wrote {path}")
    if failed:
        raise SystemExit(f"failed steps: {failed}")
