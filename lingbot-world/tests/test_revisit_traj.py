#!/usr/bin/env python3
import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "realcamvid"))
from make_revisit_traj import (  # noqa: E402
    MULTI_PASSES,
    interp_pose,
    multi_revisit_indices,
    pingpong_indices,
    resample_poses,
)
import numpy as np


class TestRevisitTraj(unittest.TestCase):
    def test_length_and_bounds(self):
        for n in (32, 53, 129, 279):
            idx = multi_revisit_indices(n, target=481, dwell=8)
            self.assertEqual(len(idx), 481)
            self.assertTrue(min(idx) >= 0)
            self.assertTrue(max(idx) <= n - 1)
            self.assertEqual(idx[0], 0)

    def test_returns_to_seed(self):
        idx = multi_revisit_indices(129, target=481, dwell=8)
        self.assertGreater(idx.count(0), 8)  # start + later revisits of seed

    def test_not_a_single_palindrome(self):
        n = 129
        pp = pingpong_indices(n, 481)
        mv = multi_revisit_indices(n, 481, dwell=8)
        self.assertNotEqual(pp, mv)
        # ping-pong period is 2n-2; multi_revisit should not be that period repeated
        period = 2 * n - 2
        self.assertNotEqual(mv[:period], list(range(n)) + list(range(n - 2, 0, -1)))

    def test_interp_identity(self):
        p = np.eye(4, dtype=np.float64)
        p[:3, 3] = [1, 2, 3]
        q = p.copy()
        q[:3, 3] = [3, 2, 1]
        mid = interp_pose(p, q, 0.5)
        np.testing.assert_allclose(mid[:3, 3], [2, 2, 2])

    def test_resample_len(self):
        poses = np.repeat(np.eye(4)[None], 10, axis=0)
        out = resample_poses(poses, 481)
        self.assertEqual(len(out), 481)

    def test_passes_cover_far_end(self):
        self.assertEqual(MULTI_PASSES[0][-1], 0.0)
        self.assertIn(1.0, MULTI_PASSES[0])


if __name__ == "__main__":
    unittest.main()
