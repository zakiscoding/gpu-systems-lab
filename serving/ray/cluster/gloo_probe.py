"""Can a gloo process group form across the two Ray nodes? One actor per node, rank 0 on the
head, explicit MASTER_ADDR (head tailnet IP) and a 60 s timeout, so a failure is an error, not a
hang. Then a 1M-float all_reduce and its time.

    docker exec ray-head python /lab/gloo_probe.py
"""

import datetime
import json
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
class Rank:
    def run(self, rank, master, port):
        import os
        import socket

        import torch
        import torch.distributed as dist
        os.environ.update(MASTER_ADDR=master, MASTER_PORT=str(port))
        info = {"rank": rank, "host": socket.gethostname(), "ip": ray.util.get_node_ip_address(),
                "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"), "torch": torch.__version__}
        try:
            t0 = time.perf_counter()
            dist.init_process_group("gloo", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60))
            info["init_s"] = round(time.perf_counter() - t0, 2)
            x = torch.ones(1 << 20)
            dist.all_reduce(x)
            t0 = time.perf_counter()
            for _ in range(5):
                dist.all_reduce(x)
            dt = (time.perf_counter() - t0) / 5
            info.update(ok=bool(x[0].item() == 2 ** 6), allreduce_4MB_ms=round(dt * 1e3, 1),
                        MB_per_s=round(4 / dt, 1))
            dist.destroy_process_group()
        except Exception as e:  # noqa: BLE001 - report whatever broke
            info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return info


ray.init(address="auto")
head_ip = ray.util.get_node_ip_address()
nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
other = next(ip for ip in nodes if ip != head_ip)
actors = [Rank.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip], soft=False)).remote()
          for ip in (head_ip, other)]
out = ray.get([a.run.remote(r, head_ip, 29555) for r, a in enumerate(actors)])
print(json.dumps(out, indent=1))
