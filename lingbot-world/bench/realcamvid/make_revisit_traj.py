#!/usr/bin/env python3
"""Build 481-frame revisit trajectories on RealCam-Vid source poses.

RealCam-Vid clips are pose-limited (RE10K ≤279, Mira/DL3DV ≤130). New SE(3)
viewpoints cannot be invented; we only replay (and linearly interpolate) the
captured pose manifold.

Ping-pong (clips_long) is a single palindrome. ``multi_revisit`` (default) is a
WorldKV-style schedule: full out-and-back, a staggered mid-path loop, a local
loop near the far end, and short dwells at turnaround viewpoints.

Writes archive ``clips_revisit/<id>/{image.jpg, poses.npy, intrinsics.npy,
prompt.txt, schedule.json}``. Images/prompts are copied from ``clips/``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # Lingbot/
ARCH = ROOT / "artifacts/archive_pre_paper/bench/realcamvid"
SRC = ARCH / "clips"
DST = ARCH / "clips_revisit"
SUBSETS = Path(__file__).resolve().parent / "subsets"
TARGET = 481
DWELL = 8

# Fractions of the captured path. Each inner list is a pass.
# Inspired by WorldKV: explore → return → partial re-explore → local loop.
MULTI_PASSES = (
    (0.0, 1.0, 0.0),           # full tour + return to seed (global revisit)
    (0.0, 0.72, 0.28, 1.0),    # staggered: miss the seed, hit mid then far
    (1.0, 0.45, 1.0),          # local loop at the far end
)


def _rot_to_quat(Rm: np.ndarray) -> np.ndarray:
    m00, m01, m02 = float(Rm[0, 0]), float(Rm[0, 1]), float(Rm[0, 2])
    m10, m11, m12 = float(Rm[1, 0]), float(Rm[1, 1]), float(Rm[1, 2])
    m20, m21, m22 = float(Rm[2, 0]), float(Rm[2, 1]), float(Rm[2, 2])
    t = m00 + m11 + m22
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return np.array([(t + 1.0) * 2.0 * s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s])
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)
    th = math.acos(dot)
    s0 = math.sin((1.0 - a) * th) / math.sin(th)
    s1 = math.sin(a * th) / math.sin(th)
    return s0 * q0 + s1 * q1


def interp_pose(p0: np.ndarray, p1: np.ndarray, a: float) -> np.ndarray:
    a = float(np.clip(a, 0.0, 1.0))
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (1.0 - a) * p0[:3, 3] + a * p1[:3, 3]
    out[:3, :3] = _quat_to_rot(_slerp(_rot_to_quat(p0[:3, :3]), _rot_to_quat(p1[:3, :3]), a))
    return out


def resample_poses(poses: np.ndarray, target: int) -> np.ndarray:
    """Uniform arc-index resample with SE(3) interpolation."""
    n = len(poses)
    if n == target:
        return poses.astype(np.float32)
    if n == 1:
        return np.repeat(poses[:1], target, axis=0).astype(np.float32)
    xs = np.linspace(0, n - 1, num=target)
    out = np.zeros((target, 4, 4), dtype=np.float64)
    for i, x in enumerate(xs):
        j = int(np.floor(x))
        k = min(j + 1, n - 1)
        out[i] = interp_pose(poses[j], poses[k], x - j)
    return out.astype(np.float32)


def _walk(a: int, b: int) -> List[int]:
    if a == b:
        return [a]
    step = 1 if b > a else -1
    return list(range(a, b, step)) + [b]


def pingpong_indices(n: int, target: int) -> List[int]:
    if n >= target:
        return list(range(target))
    seq = list(range(n)) + list(range(n - 2, 0, -1))
    return [seq[i % len(seq)] for i in range(target)]


def multi_revisit_indices(n: int, target: int = TARGET, dwell: int = DWELL) -> List[int]:
    """Index schedule on a captured path of length n, then trimmed/padded to target."""
    if n <= 1:
        return [0] * target
    last = n - 1

    def at(frac: float) -> int:
        return int(np.clip(round(frac * last), 0, last))

    idx: List[int] = []
    passes = list(MULTI_PASSES)
    # Short clips (Mira ~129, DL3DV ~50) need extra cycles to fill 481.
    guard = 0
    while len(idx) < target and guard < 8:
        for pass_ in passes:
            pts = [at(f) for f in pass_]
            for i, p in enumerate(pts):
                if idx and p == idx[-1]:
                    idx.extend([p] * dwell)
                    continue
                start = idx[-1] if idx else p
                if not idx:
                    idx.append(p)
                    idx.extend([p] * dwell)
                    continue
                idx.extend(_walk(start, p)[1:])
                idx.extend([p] * dwell)
        guard += 1
        # Subsequent cycles drop the first full 0→1→0 if we already have it,
        # to avoid a purely periodic palindrome.
        passes = list(MULTI_PASSES[1:]) + [MULTI_PASSES[0]]

    if len(idx) >= target:
        pick = np.linspace(0, len(idx) - 1, num=target)
        return [idx[int(round(i))] for i in pick]
    # Still short: dwell-pad the last viewpoint.
    idx.extend([idx[-1]] * (target - len(idx)))
    return idx[:target]


def apply_indices(poses: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return poses[np.asarray(idx, dtype=np.int64)].astype(np.float32)


def load_clip_ids(list_path: Path) -> List[str]:
    return [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def write_clip(cid: str, idx: Sequence[int], src_dir: Path, dst_dir: Path, pattern: str) -> dict:
    poses = np.load(src_dir / "poses.npy")
    kin_path = src_dir / "intrinsics.npy"
    K = np.load(kin_path) if kin_path.is_file() else None
    out_poses = apply_indices(poses, idx)
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / "poses.npy", out_poses)
    if K is not None:
        k0 = K[0] if K.ndim == 2 else K
        np.save(dst_dir / "intrinsics.npy", np.tile(np.asarray(k0, dtype=np.float32), (len(idx), 1)))
    for name in ("image.jpg", "prompt.txt"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
    meta = {
        "clip_id": cid,
        "pattern": pattern,
        "source_frames": int(len(poses)),
        "out_frames": int(len(idx)),
        "n_unique_src_idx": int(len(set(idx))),
        "indices_head": list(map(int, idx[:12])),
        "indices_tail": list(map(int, idx[-12:])),
    }
    (dst_dir / "schedule.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize multi-revisit 481-frame pose schedules")
    ap.add_argument("--ids", default=str(SUBSETS / "default_loop.txt"))
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--pattern", choices=["multi_revisit", "pingpong"], default="multi_revisit")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--dwell", type=int, default=DWELL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = load_clip_ids(Path(args.ids))
    if args.limit:
        ids = ids[: args.limit]
    src_root, dst_root = Path(args.src), Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    for cid in ids:
        src_dir = src_root / cid
        if not (src_dir / "poses.npy").is_file():
            print(f"[skip] missing source {src_dir}")
            continue
        poses = np.load(src_dir / "poses.npy")
        n = len(poses)
        if args.pattern == "pingpong":
            idx = pingpong_indices(n, args.target)
        else:
            idx = multi_revisit_indices(n, target=args.target, dwell=args.dwell)
        write_clip(cid, idx, src_dir, dst_root / cid, args.pattern)
        ok += 1
        print(f"[ok] {cid}  srcT={n}  unique={len(set(idx))}/{args.target}")
    print(f"wrote {ok}/{len(ids)} -> {dst_root}  pattern={args.pattern}")


if __name__ == "__main__":
    main()
