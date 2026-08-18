"""CPU-only sanity for Matrix-Game 2.0 KV CR (no weights)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_retention import (
    KVRetentionPlanner,
    MousePoseTracker,
    camera_affinity,
    fifo_keep,
)


def test_fifo_matches_official_six_frame_window():
    old = [0, 1, 2, 3, 4, 5]
    assert fifo_keep(old, 3, sink_frames=0) == [3, 4, 5]


def test_cr_keeps_sink_on_overflow():
    planner = KVRetentionPlanner(policy="world_state_cr", sink_frames=1, recent_frames=1)
    planner.cached_frames = [0, 1, 2, 3, 4, 5]
    for i, yaw in enumerate([0, 10, 20, 30, 40, 50]):
        planner.frame_pose[i] = np.array([0, 0, i, 0, yaw * 0.01, 0], dtype=np.float64)
        planner.frame_motion[i] = 0.1 * i
        planner.frame_ema[i] = 0.1 * i
    planner.pose_tracker.x = 0.0
    planner.pose_tracker.yaw = 0.0
    kept = planner.plan_keep(planner.cached_frames, n_keep=3, query_pose=planner.pose_tracker.as_array())
    assert 0 in kept
    assert 5 in kept
    assert len(kept) == 3


def test_revisit_prefers_sink_over_far_archive():
    planner = KVRetentionPlanner(
        policy="world_state_cr", sink_frames=1, recent_frames=1, rank_alpha=0.0)
    planner.cached_frames = [0, 1, 2, 3, 4, 5]
    planner.frame_pose[0] = np.zeros(6)
    for i in range(1, 6):
        planner.frame_pose[i] = np.array([10.0, 0, 10.0, 0, 2.0, 0], dtype=np.float64)
        planner.frame_motion[i] = 1.0
        planner.frame_ema[i] = 1.0
    planner.pose_tracker.reset()
    kept = planner.plan_keep(planner.cached_frames, 3, planner.pose_tracker.as_array())
    assert kept[0] == 0


def test_write_kv_repack_keeps_sink_tokens():
    planner = KVRetentionPlanner(policy="world_state_cr", sink_frames=1, recent_frames=1)
    frame_seqlen = 4
    cap = 6 * frame_seqlen
    kv = {
        "k": torch.zeros(1, cap, 2, 2),
        "v": torch.zeros(1, cap, 2, 2),
        "global_end_index": torch.tensor([6 * frame_seqlen]),
        "local_end_index": torch.tensor([6 * frame_seqlen]),
    }
    for f in range(6):
        kv["k"][:, f * frame_seqlen:(f + 1) * frame_seqlen] = float(f)
        kv["v"][:, f * frame_seqlen:(f + 1) * frame_seqlen] = float(f)
        planner.cached_frames.append(f)
        planner.frame_pose[f] = np.array([0, 0, float(f), 0, 0, 0], dtype=np.float64)
        planner.frame_motion[f] = 0.0
        planner.frame_ema[f] = 0.0
    new_k = torch.ones(1, 3 * frame_seqlen, 2, 2) * 9
    new_v = torch.ones(1, 3 * frame_seqlen, 2, 2) * 9
    local_end = planner.write_kv(
        kv, new_k, new_v, current_start=6 * frame_seqlen, frame_seqlen=frame_seqlen)
    assert local_end == cap
    assert 0 in planner.cached_frames
    assert planner.cached_frames[-3:] == [6, 7, 8]
    sink_vals = kv["k"][0, :frame_seqlen, 0, 0]
    assert torch.allclose(sink_vals, torch.zeros(frame_seqlen))


def test_window_policy_does_not_enable():
    planner = KVRetentionPlanner(policy="window")
    assert not planner.enabled
    assert fifo_keep([0, 1, 2, 3, 4, 5], 3) == [3, 4, 5]


def test_mouse_pose_accumulates():
    tr = MousePoseTracker()
    tr.step([0.0, 0.2], [1, 0, 0, 0])
    tr.step([0.0, 0.2], [1, 0, 0, 0])
    assert tr.yaw > 0
    assert camera_affinity(tr.as_array(), np.zeros(6)) < 1.0


if __name__ == "__main__":
    test_fifo_matches_official_six_frame_window()
    test_cr_keeps_sink_on_overflow()
    test_revisit_prefers_sink_over_far_archive()
    test_write_kv_repack_keeps_sink_tokens()
    test_window_policy_does_not_enable()
    test_mouse_pose_accumulates()
    print("ok")
