"""Routing rules, checked without SSH or Modal: python -m unittest router.test_route"""

import unittest

from router.catalog import DEFAULT_TARGETS
from router.probe import LocalState
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
