"""Routing rules, checked without SSH or Modal: python -m unittest router.test_route"""

import unittest

from router.catalog import DEFAULT_TARGETS
from router.probe import LocalState, parse
from router.route import Job, decide

IDLE = LocalState(True, 12.0, 1.2, 0, 0)  # desktop holds ~1.2 GB even when idle
BUSY = LocalState(True, 12.0, 9.0, 95, 1)
OFFLINE = LocalState(False, error="test")
TARGETS = list(DEFAULT_TARGETS)


def pick(job, local=IDLE):
    d = decide(job, TARGETS, local)
    return d.choice.target.name if d.choice else None


class RouteTest(unittest.TestCase):
    def test_small_job_stays_local_when_idle(self):
        self.assertEqual(pick(Job("s", vram_gb=1, local_minutes=1)), "local")

    def test_busy_or_offline_local_goes_to_cheapest_cloud(self):
        job = Job("s", vram_gb=1, local_minutes=1)
        for state in (BUSY, OFFLINE):
            self.assertNotEqual(pick(job, state), "local")

    def test_vram_headroom_uses_free_memory(self):
        # 10.8 GB free, 90% headroom -> 9.72 GB usable on the local card
        self.assertNotEqual(pick(Job("v", vram_gb=10, local_minutes=1)), "local")
        self.assertEqual(pick(Job("v", vram_gb=9.5, local_minutes=1)), "local")

    def test_desktop_vram_alone_is_not_busy(self):
        self.assertFalse(IDLE.busy)

    def test_resident_vllm_is_not_busy_but_takes_vram(self):
        # live output shape: 3 util samples, memory, compute apps (vLLM holding 5 GB at 0%)
        s = parse("0\n0\n3\n---\n12227, 6291\n---\nVLLM::EngineCore, 5090\n")
        self.assertTrue(s.online and not s.busy)
        self.assertEqual(s.resident, ("VLLM::EngineCore",))
        self.assertEqual(pick(Job("small", vram_gb=1, local_minutes=1), s), "local")
        self.assertNotEqual(pick(Job("big", vram_gb=8, local_minutes=1), s), "local")  # ~5.8 GB free

    def test_other_cuda_job_or_load_is_busy(self):
        self.assertTrue(parse("0\n0\n0\n---\n12227, 9000\n---\npython3, 8000\n").busy)
        self.assertTrue(parse("5\n90\n10\n---\n12227, 2000\n---\n").busy)  # peak util counts

    def test_big_model_needs_80gb_card(self):
        self.assertIn(pick(Job("big", vram_gb=60, local_minutes=10)), {"A100-80GB", "H100"})

    def test_deadline_forces_faster_gpu(self):
        slow_ok = pick(Job("d", vram_gb=8, local_minutes=240))
        rushed = pick(Job("d", vram_gb=8, local_minutes=240, deadline_min=60))
        self.assertEqual(slow_ok, "local")
        self.assertNotEqual(rushed, "local")

    def test_multi_gpu_requires_nvlink(self):
        d = decide(Job("m", vram_gb=10, local_minutes=60, gpus=2), TARGETS, IDLE)
        self.assertTrue(d.choice.target.nvlink)
        l4 = next(o for o in d.options if o.target.name == "L4")
        self.assertIn("no NVLink for multi-GPU", l4.reasons)

    def test_cost_cap_can_make_job_unroutable(self):
        self.assertIsNone(pick(Job("c", vram_gb=60, local_minutes=600, max_usd=0.01)))


if __name__ == "__main__":
    unittest.main()
