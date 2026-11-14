"""What a two-machine Ray cluster over Tailscale costs: task latency, object transfer, Ray Train.

Run on the head (GPU PC) after the iMac worker has joined:
    docker exec ray-head python /lab/cluster_bench.py --out /out/cluster.json

1. Cluster view: every node's tailnet IP and resources.
2. Task round trip: an empty remote task pinned to the head vs pinned to the iMac (NodeAffinity).
3. Object transfer: a numpy array created on the iMac and ray.get() on the head (Ray's object
   manager pulls it over the tailnet), 16 to 256 MB.
4. Ray Train: a small MLP with TorchTrainer, DDP over gloo on CPU, 1 worker (head only) vs 2
   workers placed one per machine (SPREAD). Same per-worker batch (weak scaling).
"""

import argparse
import json
import statistics
import time

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def node_by_arch():
    """{"head": node_id, "imac": node_id} from the nodes' architecture resource labels."""
    out = {}
    for n in ray.nodes():
        if not n["Alive"]:
            continue
        key = "head" if n["NodeManagerAddress"] == ray.util.get_node_ip_address() else "imac"
        out[key] = n
    return out


@ray.remote(num_cpus=0.1)
def noop():
    return 0


@ray.remote(num_cpus=1)
def make_array(mb):
    return np.ones(mb * 2**20 // 8, dtype=np.float64)


def pinned(fn, node):
    return fn.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False))


def task_latency(node, n=200):
    f = pinned(noop, node)
    ray.get([f.remote() for _ in range(10)])           # warm the worker
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        ray.get(f.remote())
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    return {"p50_ms": round(statistics.median(lat), 2), "p99_ms": round(lat[int(0.99 * n) - 1], 2)}


def transfer(node, sizes=(16, 64, 256)):
    rows = []
    f = pinned(make_array, node)
    for mb in sizes:
        ref = f.remote(mb)
        ray.wait([ref], fetch_local=False)               # created on the iMac, not yet pulled
        t0 = time.perf_counter()
        arr = ray.get(ref)
        dt = time.perf_counter() - t0
        assert arr.nbytes == mb * 2**20 and float(arr[-1]) == 1.0
        rows.append({"mb": mb, "seconds": round(dt, 3), "MB_per_s": round(mb / dt, 1)})
        del arr, ref
    return rows


def train_loop(config):
    import torch
    import torch.nn as nn
    from ray import train
    from ray.train.torch import prepare_model

    torch.manual_seed(0)
    model = prepare_model(nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
                                        nn.Linear(512, 10)))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(train.get_context().get_world_rank())
    times = []
    for step in range(config["steps"]):
        x = torch.randn(config["batch"], 512, generator=g)
        y = torch.randint(0, 10, (config["batch"],), generator=g)
        t0 = time.perf_counter()
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()                                    # DDP all-reduce of ~1.5M grads (6 MB) here
        opt.step()
        if step >= 5:
            times.append(time.perf_counter() - t0)
    rec = {"rank": train.get_context().get_world_rank(), "step_ms": statistics.median(times) * 1e3,
           "loss": loss.item(), "world": train.get_context().get_world_size(),
           "node_ip": ray.util.get_node_ip_address()}
    ray.get(ray.get_actor("train_metrics").put.remote(rec))   # Train v2's Result.metrics came back None
    train.report({"step_ms": rec["step_ms"]})


@ray.remote(num_cpus=0)
class Metrics:
    def __init__(self):
        self.rows = []

    def put(self, rec):
        self.rows.append(rec)

    def get(self):
        return self.rows


def ray_train(workers, steps=60, batch=256, spread=True):
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    trainer = TorchTrainer(
        train_loop, train_loop_config={"steps": steps, "batch": batch},
        scaling_config=ScalingConfig(num_workers=workers, use_gpu=False, resources_per_worker={"CPU": 2},
                                     placement_strategy="SPREAD" if (workers > 1 and spread) else "PACK"),
        torch_config=TorchConfig(backend="gloo"),
        run_config=RunConfig(storage_path="/tmp/ray_results", name=f"mlp_w{workers}"))
    coll = Metrics.options(name="train_metrics").remote()
    t0 = time.perf_counter()
    trainer.fit()
    wall = time.perf_counter() - t0
    rows = ray.get(coll.get.remote())
    ray.kill(coll)
    step = max(r["step_ms"] for r in rows)           # the slowest rank sets the pace
    return {"workers": workers, "step_ms": round(step, 2), "samples_per_s": round(workers * batch / (step / 1e3)),
            "wall_s": round(wall, 1), "ranks": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                                                for r in sorted(rows, key=lambda r: r["rank"])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out/cluster.json")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--spread", action="store_true", help="put the 2 Train workers on different nodes")
    args = ap.parse_args()
    ray.init(address="auto")
    nodes = node_by_arch()
    rec = {"ray": ray.__version__,
           "nodes": [{"ip": n["NodeManagerAddress"], "cpus": n["Resources"].get("CPU"), "gpus": n["Resources"].get("GPU", 0),
                      "memory_gb": round(n["Resources"].get("memory", 0) / 2**30, 1)} for n in nodes.values()]}
    print(json.dumps(rec, indent=1), flush=True)
    if not args.train_only:
        rec["task_roundtrip"] = {"head (local)": task_latency(nodes["head"]), "imac (over tailnet)": task_latency(nodes["imac"])}
        print(json.dumps(rec["task_roundtrip"]), flush=True)
        rec["object_pull_imac_to_head"] = transfer(nodes["imac"])
        print(json.dumps(rec["object_pull_imac_to_head"]), flush=True)
    # cross-node (SPREAD) DDP hangs at gloo rendezvous between the Linux head and the macOS
    # node (see gloo_probe.py), so the 2-worker run is placed on the head (PACK) unless --spread
    rec["ray_train"] = [ray_train(1), ray_train(2, spread=args.spread)]
    print(json.dumps(rec["ray_train"]), flush=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
