"""Per-block CUDA-event split: DiT vs VAE vs leftover Python."""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import torch


def _pct(xs: List[float], q: float) -> Optional[float]:
    xs = [float(v) for v in xs if math.isfinite(float(v))]
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return float(s[i])


def _mean(xs: List[float]) -> Optional[float]:
    xs = [float(v) for v in xs if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


class E2EProfile:
    def __init__(self, *, enabled: bool = False, warmup_blocks: int = 2) -> None:
        self.enabled = bool(enabled)
        self.warmup_blocks = int(warmup_blocks)
        self.n_block = 0
        self.time_ms: Dict[str, List[float]] = {
            "dit": [],
            "vae": [],
            "block": [],
            "other": [],
            "rope": [],
            "attn": [],
            "ffn": [],
        }

    def timed(self, bucket: str, fn: Callable[[], Any]) -> Any:
        if not self.enabled:
            return fn()
        if not torch.cuda.is_available():
            t0 = time.perf_counter()
            out = fn()
            self.time_ms.setdefault(bucket, []).append(
                (time.perf_counter() - t0) * 1000.0)
            return out
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        self.time_ms.setdefault(bucket, []).append(float(start.elapsed_time(end)))
        return out

    def finish_block(self) -> None:
        if not self.enabled:
            return
        dit = self.time_ms["dit"][-1] if self.time_ms["dit"] else 0.0
        vae = self.time_ms["vae"][-1] if self.time_ms["vae"] else 0.0
        block = self.time_ms["block"][-1] if self.time_ms["block"] else dit + vae
        self.time_ms["other"].append(max(0.0, block - dit - vae))
        self.n_block += 1

    def summary(self) -> Dict[str, Any]:
        skip = min(self.warmup_blocks, max(0, self.n_block - 1))

        def pack(xs: List[float], drop: int = 0) -> Dict[str, Any]:
            ys = xs[drop:] if drop else xs
            return {
                "n": len(ys),
                "mean_ms": _mean(ys),
                "p50_ms": _pct(ys, 50),
                "p95_ms": _pct(ys, 95),
                "sum_ms": (float(sum(ys)) if ys else None),
            }

        dit = self.time_ms["dit"]
        vae = self.time_ms["vae"]
        block = self.time_ms["block"]
        p50_block = _pct(block[skip:], 50) or 0.0
        p50_dit = _pct(dit[skip:], 50) or 0.0
        p50_vae = _pct(vae[skip:], 50) or 0.0
        p50_rope = _pct(self.time_ms.get("rope", [])[skip:], 50) or 0.0
        p50_attn = _pct(self.time_ms.get("attn", [])[skip:], 50) or 0.0
        p50_ffn = _pct(self.time_ms.get("ffn", [])[skip:], 50) or 0.0
        return {
            "enabled": self.enabled,
            "warmup_dropped": skip,
            "all": {k: pack(v) for k, v in self.time_ms.items()},
            "steady": {k: pack(v, skip) for k, v in self.time_ms.items()},
            "share_p50": {
                "dit": (p50_dit / p50_block if p50_block else None),
                "vae": (p50_vae / p50_block if p50_block else None),
                "rope_of_dit": (p50_rope / p50_dit if p50_dit else None),
                "attn_of_dit": (p50_attn / p50_dit if p50_dit else None),
                "ffn_of_dit": (p50_ffn / p50_dit if p50_dit else None),
                "other": (
                    max(0.0, p50_block - p50_dit - p50_vae) / p50_block
                    if p50_block else None
                ),
            },
            "note": (
                "Drop first warmup_dropped blocks for steady p50. "
                "other = block - dit - vae (Python / sync / leftover)."
            ),
        }
