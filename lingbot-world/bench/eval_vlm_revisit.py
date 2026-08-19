#!/usr/bin/env python3
"""VLM Revisit Memory Judge for World-State CR.

Protocol (WS-CR qualitative plan A):
  Inputs per WorldKV pose pair:
    condition image + first-visit frame + revisit frame
  Outputs:
    same_place ∈ {0,1,2,3}
    identity_drift ∈ {layout, appearance, object_missing, flicker, ok}
    preference ∈ {A, B, tie}   (pairwise)
    rationale

Default local judge: Qwen3-VL-8B-Instruct (transformers ≥ 5.12).
Self-contained: does not import eval_worldkv_memory (avoids lpips/skimage).

Example:
  /DATA/miniconda3_corl/envs_corl/mus3d-sim/bin/python bench/eval_vlm_revisit.py \\
    --out_dir output/realcamvid_vlm_revisit_default_loop \\
    --clips_dir bench/realcamvid/clips_default_loop \\
    --methods window,worldkv,ws_v3_a05_g64 \\
    --model_path /DATA/YuanZhen/models/Qwen3-VL-8B-Instruct \\
    --max_pairs_per_clip 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import random
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DRIFT_LABELS = {"layout", "appearance", "object_missing", "flicker", "ok"}
PREF_LABELS = {"A", "B", "tie"}
_BENCH = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pose pairing (copied from eval_worldkv_memory; keep independent of lpips)
# ---------------------------------------------------------------------------

def _rot_geodesic(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(tr))


def se3_distance_c2w(pa: np.ndarray, pb: np.ndarray, translation_scale: float = 1.0) -> float:
    d_trans = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3])) / max(translation_scale, 1e-6)
    d_rot = _rot_geodesic(pa[:3, :3], pb[:3, :3]) / math.pi
    return d_trans + d_rot


def scene_translation_scale(poses: np.ndarray) -> float:
    t = poses[:, :3, 3]
    return max(float(np.linalg.norm(t.max(0) - t.min(0))), 1e-3)


def find_revisit_pairs(
    poses: np.ndarray,
    radius: float = 0.15,
    min_gap: int = 30,
    translation_scale: float = 1.0,
) -> List[Tuple[int, int, float]]:
    T = len(poses)
    if T <= min_gap + 1:
        return []
    ts = float(translation_scale)
    pairs: List[Tuple[int, int, float]] = []
    for t in range(min_gap, T):
        best_s = -1
        best_d = float("inf")
        for s in range(0, t - min_gap + 1):
            d = se3_distance_c2w(poses[s], poses[t], translation_scale=ts)
            if d <= radius and (d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and s < best_s)):
                best_d = d
                best_s = s
        if best_s >= 0:
            pairs.append((best_s, t, float(best_d)))
    return pairs


def subsample_pairs(pairs: Sequence[Tuple[int, int, float]], max_pairs: int) -> List[Tuple[int, int, float]]:
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return list(pairs)
    idx = np.linspace(0, len(pairs) - 1, num=max_pairs, dtype=np.int64)
    return [pairs[int(i)] for i in idx]


def read_video_frames(path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Decode requested frame indices as RGB uint8. Clamp OOB to last frame."""
    want = sorted(set(int(i) for i in indices if int(i) >= 0))
    if not want:
        return {}
    import av
    container = av.open(path)
    want_set = set(want)
    decoded: Dict[int, np.ndarray] = {}
    last_i = -1
    last_arr = None
    for i, frame in enumerate(container.decode(video=0)):
        arr = frame.to_ndarray(format="rgb24")
        last_i, last_arr = i, arr
        if i in want_set:
            decoded[i] = arr
        if last_i >= max(want) and len(decoded) == len(want):
            break
    container.close()
    if last_arr is None:
        return {}
    out: Dict[int, np.ndarray] = {}
    for i in want:
        src = i if i in decoded else last_i
        out[i] = decoded.get(src, last_arr)
    return out


# ---------------------------------------------------------------------------
# JSON parse / image helpers
# ---------------------------------------------------------------------------

def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        soft = m.group(0).replace("'", '"')
        soft = re.sub(r",\s*}", "}", soft)
        try:
            obj = json.loads(soft)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _resize_max(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


class Qwen3VLJudge:
    def __init__(self, model_path: str, device: str = "cuda:0", max_side: int = 512):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.max_side = max_side
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[vlm] loading {model_path} on {device} dtype={dtype}", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("[vlm] ready", flush=True)

    def _prep(self, images: Sequence[Image.Image]) -> List[Image.Image]:
        return [_resize_max(im.convert("RGB"), self.max_side) for im in images]

    @torch.inference_mode()
    def generate(self, images: Sequence[Image.Image], prompt: str, max_new_tokens: int = 192) -> str:
        imgs = self._prep(images)
        content: List[Dict[str, Any]] = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def normalize_absolute(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not obj:
        return None
    try:
        sp = int(obj.get("same_place"))
    except Exception:
        return None
    if sp not in (0, 1, 2, 3):
        return None
    drift = str(obj.get("identity_drift", "")).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "objectmissing": "object_missing",
        "missing_object": "object_missing",
        "objects_missing": "object_missing",
        "geometry": "layout",
        "structure": "layout",
        "texture": "appearance",
        "color": "appearance",
        "none": "ok",
        "no_drift": "ok",
    }
    drift = aliases.get(drift, drift)
    if drift not in DRIFT_LABELS:
        drift = "ok" if sp == 3 else "appearance"
    return {
        "same_place": sp,
        "identity_drift": drift,
        "rationale": str(obj.get("rationale", "")).strip(),
    }


def normalize_preference(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not obj:
        return None
    pref = str(obj.get("preference", "")).strip()
    pu = pref.upper()
    if pu in {"A", "METHOD A", "METHOD_A"}:
        pref = "A"
    elif pu in {"B", "METHOD B", "METHOD_B"}:
        pref = "B"
    elif pu.lower() in {"tie", "equal", "same"}:
        pref = "tie"
    else:
        return None
    return {"preference": pref, "rationale": str(obj.get("rationale", "")).strip()}


def judge_absolute(
    judge: Qwen3VLJudge,
    prompt: str,
    cond: Image.Image,
    first: Image.Image,
    revisit: Image.Image,
    retries: int = 1,
) -> Tuple[Optional[Dict[str, Any]], str]:
    last = ""
    for _ in range(retries + 1):
        last = judge.generate([cond, first, revisit], prompt)
        norm = normalize_absolute(_parse_json_obj(last))
        if norm:
            return norm, last
    return None, last


def judge_preference(
    judge: Qwen3VLJudge,
    prompt: str,
    a_first: Image.Image,
    a_rev: Image.Image,
    b_first: Image.Image,
    b_rev: Image.Image,
    retries: int = 1,
) -> Tuple[Optional[Dict[str, Any]], str]:
    last = ""
    for _ in range(retries + 1):
        last = judge.generate([a_first, a_rev, b_first, b_rev], prompt)
        norm = normalize_preference(_parse_json_obj(last))
        if norm:
            return norm, last
    return None, last


def discover_clips(videos_dir: Path, methods: Sequence[str]) -> List[str]:
    suffix = f"_{methods[0]}.mp4"
    clips = []
    for p in sorted(videos_dir.glob(f"*{suffix}")):
        clip = p.name[: -len(suffix)]
        if all((videos_dir / f"{clip}_{m}.mp4").is_file() for m in methods):
            clips.append(clip)
    return clips


def aggregate(results: Dict[str, Any], methods: Sequence[str]) -> Dict[str, Any]:
    abs_by = {m: [] for m in methods}
    drift_by = {m: defaultdict(int) for m in methods}
    pref_counts = defaultdict(lambda: {"A": 0, "B": 0, "tie": 0, "n": 0})
    parse_fail = {"absolute": 0, "preference": 0}

    for clip_rec in results["per_clip"].values():
        for rec in clip_rec.get("absolute", []):
            m = rec["method"]
            if rec.get("parsed") is None:
                parse_fail["absolute"] += 1
                continue
            p = rec["parsed"]
            abs_by[m].append(int(p["same_place"]))
            drift_by[m][p["identity_drift"]] += 1
        for rec in clip_rec.get("preference", []):
            key = f"{rec['method_a']}__vs__{rec['method_b']}"
            if rec.get("parsed") is None:
                parse_fail["preference"] += 1
                continue
            pref = rec["parsed"]["preference"]
            pref_counts[key][pref] += 1
            pref_counts[key]["n"] += 1

    by_method = {}
    for m in methods:
        scores = abs_by[m]
        by_method[m] = {
            "n": len(scores),
            "same_place_mean": float(np.mean(scores)) if scores else None,
            "same_place_std": float(np.std(scores)) if scores else None,
            "same_place_hist": {str(k): int(scores.count(k)) for k in range(4)},
            "identity_drift_hist": dict(drift_by[m]),
        }

    prefs = {}
    for key, c in pref_counts.items():
        n = max(c["n"], 1)
        a_name, b_name = key.split("__vs__")
        prefs[key] = {
            "method_a": a_name,
            "method_b": b_name,
            "n": c["n"],
            "A": c["A"],
            "B": c["B"],
            "tie": c["tie"],
            "win_rate_A": c["A"] / n,
            "win_rate_B": c["B"] / n,
            "tie_rate": c["tie"] / n,
        }
    return {"by_method": by_method, "pairwise": prefs, "parse_fail": parse_fail}


def main():
    ap = argparse.ArgumentParser(description="VLM Revisit Memory Judge")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--methods", default="window,worldkv,ws_v3_a05_g64")
    ap.add_argument("--model_path", default="/DATA/YuanZhen/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--min_gap", type=int, default=30)
    ap.add_argument("--max_pairs_per_clip", type=int, default=8)
    ap.add_argument("--translation_scale_mode", choices=["fixed", "scene"], default="fixed")
    ap.add_argument("--translation_scale", type=float, default=1.0)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_clips", type=int, default=0)
    ap.add_argument("--skip_preference", action="store_true")
    ap.add_argument("--skip_absolute", action="store_true")
    ap.add_argument("--save_name", default="vlm_revisit_eval.json")
    ap.add_argument("--prompt_abs", default=str(_BENCH / "prompts" / "revisit_memory_v1.txt"))
    ap.add_argument("--prompt_pref", default=str(_BENCH / "prompts" / "revisit_preference_v1.txt"))
    ap.add_argument("--save_panels", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    clips_dir = Path(args.clips_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    clips = discover_clips(videos_dir, methods)
    if clips_dir.is_dir():
        keep = {p.name for p in clips_dir.iterdir() if p.is_dir()}
        clips = [c for c in clips if c in keep]
    if args.limit_clips > 0:
        clips = clips[: args.limit_clips]
    if not clips:
        raise SystemExit(f"No complete clip set in {videos_dir} ∩ {clips_dir} for {methods}")

    prompt_abs = Path(args.prompt_abs).read_text(encoding="utf-8").strip()
    prompt_pref = Path(args.prompt_pref).read_text(encoding="utf-8").strip()
    judge = Qwen3VLJudge(args.model_path, device=args.device, max_side=args.max_side)

    panels_dir = out_dir / "panels"
    if args.save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "protocol": {
            "name": "vlm_revisit_memory_judge_v1",
            "judge_model": args.model_path,
            "methods": methods,
            "clips_dir": str(clips_dir),
            "out_dir": str(out_dir),
            "radius": args.radius,
            "min_gap": args.min_gap,
            "max_pairs_per_clip": args.max_pairs_per_clip,
            "seed": args.seed,
            "n_clips": len(clips),
        },
        "per_clip": {},
    }

    pair_combos = list(combinations(methods, 2))
    t0 = time.time()
    n_abs = n_pref = 0

    for ci, clip in enumerate(clips):
        print(f"\n[{ci + 1}/{len(clips)}] {clip}", flush=True)
        pose_path = clips_dir / clip / "poses.npy"
        cond_path = clips_dir / clip / "image.jpg"
        if not pose_path.is_file() or not cond_path.is_file():
            print(f"  skip: missing poses/image under {clips_dir / clip}")
            continue
        poses = np.load(str(pose_path))
        ts = (
            scene_translation_scale(poses)
            if args.translation_scale_mode == "scene"
            else float(args.translation_scale)
        )
        pairs = find_revisit_pairs(poses, radius=args.radius, min_gap=args.min_gap, translation_scale=ts)
        pairs = subsample_pairs(pairs, args.max_pairs_per_clip)
        if not pairs:
            print("  no pairs")
            continue

        cond = Image.open(cond_path).convert("RGB")
        need = sorted({int(s) for s, _, _ in pairs} | {int(t) for _, t, _ in pairs})
        method_maps: Dict[str, Dict[int, np.ndarray]] = {}
        for m in methods:
            vpath = videos_dir / f"{clip}_{m}.mp4"
            method_maps[m] = read_video_frames(str(vpath), need)
            print(f"  loaded {m}: {len(method_maps[m])}/{len(need)} frames", flush=True)

        clip_rec: Dict[str, Any] = {"n_pairs": len(pairs), "absolute": [], "preference": []}

        for pi, (s, t, dist) in enumerate(pairs):
            if not args.skip_absolute:
                for m in methods:
                    fmap = method_maps[m]
                    if s not in fmap or t not in fmap:
                        continue
                    first = _to_pil(fmap[s])
                    rev = _to_pil(fmap[t])
                    parsed, raw = judge_absolute(judge, prompt_abs, cond, first, rev)
                    clip_rec["absolute"].append({
                        "pair_index": pi,
                        "first_idx": int(s),
                        "revisit_idx": int(t),
                        "pose_dist": float(dist),
                        "method": m,
                        "parsed": parsed,
                        "raw": raw[:1000],
                    })
                    n_abs += 1
                    if args.save_panels and parsed is not None:
                        pdir = panels_dir / clip / f"p{pi}_{m}"
                        pdir.mkdir(parents=True, exist_ok=True)
                        cond.save(pdir / "cond.jpg")
                        first.save(pdir / "first.jpg")
                        rev.save(pdir / "revisit.jpg")
                        (pdir / "judge.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")

            if not args.skip_preference:
                for a, b in pair_combos:
                    fa, fb = method_maps[a], method_maps[b]
                    if s not in fa or t not in fa or s not in fb or t not in fb:
                        continue
                    # Randomize left/right to cancel VLM position bias; map back to a/b.
                    h = hashlib.md5(f"{clip}|{pi}|{a}|{b}|{args.seed}".encode()).digest()[0]
                    swap = (h & 1) == 1
                    left, right = (b, a) if swap else (a, b)
                    fl, fr = method_maps[left], method_maps[right]
                    parsed, raw = judge_preference(
                        judge, prompt_pref,
                        _to_pil(fl[s]), _to_pil(fl[t]),
                        _to_pil(fr[s]), _to_pil(fr[t]),
                    )
                    mapped = None
                    if parsed:
                        pref = parsed["preference"]
                        if swap and pref in ("A", "B"):
                            pref = "B" if pref == "A" else "A"
                        winner = a if pref == "A" else b if pref == "B" else "tie"
                        mapped = {
                            "preference": pref,
                            "winner": winner,
                            "swapped_display": swap,
                            "rationale": parsed["rationale"],
                        }
                    clip_rec["preference"].append({
                        "pair_index": pi,
                        "first_idx": int(s),
                        "revisit_idx": int(t),
                        "pose_dist": float(dist),
                        "method_a": a,
                        "method_b": b,
                        "parsed": mapped,
                        "raw": raw[:1000],
                    })
                    n_pref += 1

            if (pi + 1) % 2 == 0 or pi == 0:
                print(
                    f"  pair {pi + 1}/{len(pairs)}  abs={n_abs} pref={n_pref}  "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

        results["per_clip"][clip] = clip_rec
        results["summary"] = aggregate(results, methods)
        out_path = out_dir / args.save_name
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  checkpoint → {out_path}", flush=True)

    results["summary"] = aggregate(results, methods)
    results["timing"] = {
        "total_s": time.time() - t0,
        "n_absolute_calls": n_abs,
        "n_preference_calls": n_pref,
    }
    out_path = out_dir / args.save_name
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n======== VLM Revisit Memory Judge ========")
    print(f"model: {args.model_path}")
    print(f"clips: {len(results['per_clip'])}  abs_calls: {n_abs}  pref_calls: {n_pref}")
    print(f"{'method':22s} {'n':>5} {'same_place':>10}  drift_top")
    for m, v in results["summary"]["by_method"].items():
        drift = v.get("identity_drift_hist") or {}
        top = max(drift.items(), key=lambda x: x[1])[0] if drift else "-"
        sp = v.get("same_place_mean")
        print(f"{m:22s} {v.get('n', 0):5d} {(sp if sp is not None else float('nan')):10.3f}  {top}")
    print("\nPairwise win-rates (A vs B):")
    for v in results["summary"]["pairwise"].values():
        print(
            f"  {v['method_a']} vs {v['method_b']}: "
            f"A={v['win_rate_A']:.3f} B={v['win_rate_B']:.3f} "
            f"tie={v['tie_rate']:.3f} (n={v['n']})"
        )
    print(f"\nsaved {out_path}")

    lines = [
        f"model={args.model_path}",
        f"methods={methods}",
        f"n_clips={len(results['per_clip'])}",
        "",
        "absolute same_place:",
    ]
    for m, v in results["summary"]["by_method"].items():
        lines.append(
            f"  {m}: mean={v.get('same_place_mean')} n={v.get('n')} drift={v.get('identity_drift_hist')}"
        )
    lines.append("\npairwise:")
    for v in results["summary"]["pairwise"].values():
        lines.append(
            f"  {v['method_a']} vs {v['method_b']}: "
            f"A={v['win_rate_A']:.3f} B={v['win_rate_B']:.3f} tie={v['tie_rate']:.3f} n={v['n']}"
        )
    (out_dir / "vlm_revisit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
