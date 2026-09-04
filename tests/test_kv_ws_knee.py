from __future__ import annotations

import unittest

from TokenSim.kv_working_set.knee import (
    is_stable,
    little_concurrency,
    search_knee_qps,
    snap_qps,
)


class KneeSearchTest(unittest.TestCase):
    def test_snap_qps_to_resolution(self):
        self.assertEqual(snap_qps(0.02), 0.02)
        self.assertEqual(snap_qps(0.03), 0.04)
        self.assertEqual(snap_qps(1.0), 1.0)

    def test_stable_requires_goodput_and_ttft(self):
        light = {"output_qps": 0.02, "ttft_p99": 1.0}
        self.assertTrue(is_stable({"output_qps": 0.019, "ttft_p99": 2.0}, 0.02, 1.0))
        self.assertFalse(is_stable({"output_qps": 0.01, "ttft_p99": 1.0}, 0.02, 1.0))
        self.assertFalse(is_stable({"output_qps": 0.02, "ttft_p99": 4.0}, 0.02, 1.0))
        self.assertTrue(is_stable(light, 0.02, 1.0))

    def test_search_finds_last_stable_qps(self):
        def evaluate(qps: float) -> dict:
            ttft = 1.0 if qps <= 0.16 else 10.0
            return {"output_qps": qps, "ttft_p99": ttft}

        star, points = search_knee_qps(evaluate)
        self.assertEqual(star, 0.16)
        self.assertIn(0.02, points)
        self.assertIn(0.32, points)

    def test_little_n(self):
        self.assertAlmostEqual(little_concurrency(0.2, 10.0), 2.0)


if __name__ == "__main__":
    unittest.main()
