"""World-State CR on Matrix-Game 2.0 self-attn KV.

MG2 already has a rolling KV buffer (local_attn_size frames). Official overflow
is FIFO. This module only replaces *which* old frames stay in that buffer.

  window         — official FIFO (sink_size=0)
  world_state_cr — pack [sink | ranked archive | recent] then append the new block

Action-module KV is left FIFO. Selector checkpoints are not loaded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def camera_affinity(query: Optional[np.ndarray], key: Optional[np.ndarray]) -> float:
    if query is None or key is None:
        return 0.0
    dt = float(np.linalg.norm(query[:3] - key[:3]))
    dr = float(np.linalg.norm(query[3:] - key[3:]))
    return float(math.exp(-dt) * math.exp(-dr))


def _as_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


class MousePoseTracker:
    """Integrate mouse look + WASD into a coarse 6D pose for CR ranking."""

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def reset(self) -> None:
        self.__init__()

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z, self.pitch, self.yaw, self.roll],
            dtype=np.float64,
        )

    def step(self, mouse_xy: Any = None, keyboard: Any = None) -> np.ndarray:
        mouse = _as_numpy(mouse_xy)
        if mouse is not None and mouse.size >= 2:
            mouse = mouse.reshape(-1)
            self.pitch += float(mouse[0])
            self.yaw += float(mouse[1])
        keys = _as_numpy(keyboard)
        if keys is not None and keys.size >= 2:
            keys = keys.reshape(-1)
            fwd = float(keys[0]) - (float(keys[1]) if keys.size > 1 else 0.0)
            strafe = 0.0
            if keys.size >= 4:
                strafe = float(keys[3]) - float(keys[2])
            cy = math.cos(self.yaw)
            sy = math.sin(self.yaw)
            self.x += fwd * sy + strafe * cy
            self.z += fwd * cy - strafe * sy
        return self.as_array()


def fifo_keep(old_frames: Sequence[int], n_keep: int, sink_frames: int = 0) -> List[int]:
    """Official MG2 overflow: keep sink prefix, then the newest remainder."""
    old = list(old_frames)
    if n_keep <= 0:
        return []
    if n_keep >= len(old):
        return old
    sink = [f for f in old if f < sink_frames][:sink_frames]
    rest = [f for f in old if f not in set(sink)]
    need = max(0, n_keep - len(sink))
    return sink + rest[-need:]


@dataclass
class KVRetentionPlanner:
    policy: str = "window"
    sink_frames: int = 1
    recent_frames: int = 1
    keep_ratio: float = 0.5
    min_keep: int = 2
    rank_alpha: float = 0.5
    ema_beta: float = 0.7
    cached_frames: List[int] = field(default_factory=list)
    frame_pose: Dict[int, np.ndarray] = field(default_factory=dict)
    frame_motion: Dict[int, float] = field(default_factory=dict)
    frame_ema: Dict[int, float] = field(default_factory=dict)
    pose_tracker: MousePoseTracker = field(default_factory=MousePoseTracker)
    last_choice: str = "window"
    last_kept: List[int] = field(default_factory=list)
    _plan_key: Optional[Tuple[int, int, int, int]] = None
    _plan: Optional[Dict[str, Any]] = None
    _committed_key: Optional[Tuple[int, int, int, int]] = None

    @property
    def enabled(self) -> bool:
        return self.policy not in (None, "", "window", "off", "fifo")

    def reset(self) -> None:
        self.cached_frames = []
        self.frame_pose.clear()
        self.frame_motion.clear()
        self.frame_ema.clear()
        self.pose_tracker.reset()
        self.last_choice = "window"
        self.last_kept = []
        self._plan_key = None
        self._plan = None
        self._committed_key = None

    def observe_block(
        self,
        start_frame: int,
        num_frames: int,
        mouse: Any = None,
        keyboard: Any = None,
    ) -> None:
        """Record per-latent-frame pose from RGB-rate mouse/keyboard."""
        mouse_np = _as_numpy(mouse)
        key_np = _as_numpy(keyboard)
        if mouse_np is not None and mouse_np.ndim == 3:
            mouse_np = mouse_np[0]
        if key_np is not None and key_np.ndim == 3:
            key_np = key_np[0]
        for f in range(int(start_frame), int(start_frame) + int(num_frames)):
            rgb_end = 1 + 4 * f
            rgb_start = 0 if f == 0 else 1 + 4 * (f - 1)
            m = None
            k = None
            if mouse_np is not None and mouse_np.ndim >= 2:
                sl = mouse_np[rgb_start:max(rgb_start + 1, rgb_end)]
                if sl.size:
                    m = sl.mean(axis=0)
            if key_np is not None and key_np.ndim >= 2:
                sl = key_np[rgb_start:max(rgb_start + 1, rgb_end)]
                if sl.size:
                    k = sl.mean(axis=0)
            pose = self.pose_tracker.step(m, k)
            self.frame_pose[f] = pose.copy()
            motion = 0.0
            if m is not None:
                motion += float(np.linalg.norm(np.asarray(m).reshape(-1)[:2]))
            if k is not None:
                motion += float(np.linalg.norm(np.asarray(k).reshape(-1)))
            self.frame_motion[f] = motion
            prev = self.frame_ema.get(f)
            beta = float(self.ema_beta)
            self.frame_ema[f] = motion if prev is None else beta * prev + (1.0 - beta) * motion

    def score_frame(self, frame_id: int, query_pose: Optional[np.ndarray]) -> float:
        aff = camera_affinity(query_pose, self.frame_pose.get(int(frame_id)))
        motion = float(self.frame_ema.get(int(frame_id), self.frame_motion.get(int(frame_id), 0.0)))
        alpha = max(0.0, min(1.0, float(self.rank_alpha)))
        return (1.0 - alpha) * aff + alpha * min(1.0, motion)

    def plan_keep(self, old_frames: Sequence[int], n_keep: int, query_pose: Optional[np.ndarray]) -> List[int]:
        old = list(old_frames)
        if n_keep <= 0:
            self.last_choice = "empty"
            return []
        if not self.enabled:
            self.last_choice = "window"
            return fifo_keep(old, n_keep, sink_frames=0)

        sink_n = max(0, int(self.sink_frames))
        recent_n = max(0, int(self.recent_frames))
        sink = [f for f in old if f < sink_n]
        recent = [f for f in old if f not in set(sink)][-recent_n:] if recent_n else []
        reserved = set(sink) | set(recent)
        archive = [f for f in old if f not in reserved]
        slots = max(0, n_keep - len(sink) - len(recent))
        if slots < len(archive):
            ranked = sorted(archive, key=lambda f: self.score_frame(f, query_pose), reverse=True)
            archive = ranked[:slots]
            archive.sort()
        kept = sink + archive + recent
        if len(kept) > n_keep:
            extra = set(kept[n_keep:])
            kept = [f for f in kept if f not in extra]
        self.last_choice = "world_state_cr"
        if sink and sink[0] in kept:
            self.last_choice = "sink+archive" if archive else "sink+recent"
        return kept

    def prepare_write(
        self,
        *,
        current_start: int,
        num_new_tokens: int,
        frame_seqlen: int,
        kv_cache_size: int,
        global_end: int,
        local_end: int,
    ) -> Dict[str, Any]:
        current_end = int(current_start) + int(num_new_tokens)
        key = (int(current_start), current_end, int(global_end), int(local_end))
        if self._plan_key == key and self._plan is not None:
            return self._plan

        new_frames = list(range(int(current_start) // frame_seqlen, current_end // frame_seqlen))
        is_new_extent = current_end > int(global_end)
        overflow = is_new_extent and (int(num_new_tokens) + int(local_end) > int(kv_cache_size))
        cap_frames = int(kv_cache_size) // int(frame_seqlen)

        if not is_new_extent:
            plan = {
                "mode": "overwrite",
                "new_frames": new_frames,
                "keep_old": list(self.cached_frames),
                "src_token_index": None,
            }
        elif not overflow:
            plan = {
                "mode": "append",
                "new_frames": new_frames,
                "keep_old": list(self.cached_frames),
                "src_token_index": None,
            }
        else:
            n_keep = max(0, cap_frames - len(new_frames))
            query = self.pose_tracker.as_array()
            keep_old = self.plan_keep(self.cached_frames, n_keep, query)
            src: List[int] = []
            pos = {fid: i for i, fid in enumerate(self.cached_frames)}
            for fid in keep_old:
                if fid not in pos:
                    continue
                start = pos[fid] * frame_seqlen
                src.extend(range(start, start + frame_seqlen))
            plan = {
                "mode": "repack",
                "new_frames": new_frames,
                "keep_old": keep_old,
                "src_token_index": src,
            }
            self.last_kept = keep_old + new_frames

        self._plan_key = key
        self._plan = plan
        return plan

    def commit_cached_frames(self, plan: Dict[str, Any]) -> None:
        if self._committed_key == self._plan_key:
            return
        mode = plan["mode"]
        if mode != "overwrite":
            self.cached_frames = list(plan["keep_old"]) + list(plan["new_frames"])
        self._committed_key = self._plan_key

    def write_kv(
        self,
        kv_cache: Dict[str, torch.Tensor],
        roped_key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        frame_seqlen: int,
    ) -> int:
        """In-place KV write. Returns new local_end_index."""
        num_new = int(roped_key.shape[1])
        current_end = int(current_start) + num_new
        kv_size = int(kv_cache["k"].shape[1])
        global_end = int(kv_cache["global_end_index"].item())
        local_end = int(kv_cache["local_end_index"].item())
        plan = self.prepare_write(
            current_start=current_start,
            num_new_tokens=num_new,
            frame_seqlen=int(frame_seqlen),
            kv_cache_size=kv_size,
            global_end=global_end,
            local_end=local_end,
        )

        if plan["mode"] == "repack" and self.enabled:
            src = plan["src_token_index"] or []
            n_kept = len(src)
            if n_kept:
                idx = torch.as_tensor(src, device=kv_cache["k"].device, dtype=torch.long)
                k_kept = kv_cache["k"][:, idx].clone()
                v_kept = kv_cache["v"][:, idx].clone()
                kv_cache["k"][:, :n_kept] = k_kept
                kv_cache["v"][:, :n_kept] = v_kept
            local_end_index = n_kept + num_new
            local_start = n_kept
            kv_cache["k"][:, local_start:local_end_index] = roped_key
            kv_cache["v"][:, local_start:local_end_index] = value
        elif (current_end > global_end) and (num_new + local_end > kv_size):
            sink_tokens = 0
            num_evicted = num_new + local_end - kv_size
            num_rolled = local_end - num_evicted - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                kv_cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                kv_cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
            local_end_index = local_end + current_end - global_end - num_evicted
            local_start = local_end_index - num_new
            kv_cache["k"][:, local_start:local_end_index] = roped_key
            kv_cache["v"][:, local_start:local_end_index] = value
        else:
            local_end_index = local_end + current_end - global_end
            local_start = local_end_index - num_new
            kv_cache["k"][:, local_start:local_end_index] = roped_key
            kv_cache["v"][:, local_start:local_end_index] = value

        self.commit_cached_frames(plan)
        return int(local_end_index)

    def summary(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "cached_frames": list(self.cached_frames),
            "last_choice": self.last_choice,
            "last_kept": list(self.last_kept),
        }


def attach_planner(model: torch.nn.Module, planner: Optional[KVRetentionPlanner]) -> None:
    from wan.modules.causal_model import CausalWanSelfAttention
    for mod in model.modules():
        if isinstance(mod, CausalWanSelfAttention):
            mod.kv_retention = planner


def build_planner(args: Any) -> KVRetentionPlanner:
    policy = str(getattr(args, "memory_policy", "window") or "window").lower()
    return KVRetentionPlanner(
        policy=policy,
        sink_frames=int(getattr(args, "cr_sink_frames", 1)),
        recent_frames=int(getattr(args, "cr_recent_frames", 1)),
        keep_ratio=float(getattr(args, "cr_keep_ratio", 0.5)),
        min_keep=int(getattr(args, "cr_min_keep", 2)),
        rank_alpha=float(getattr(args, "consol_rank_alpha", 0.5)),
        ema_beta=float(getattr(args, "consol_beta", 0.7)),
    )
