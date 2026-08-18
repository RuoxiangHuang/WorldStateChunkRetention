"""Causal RoPE: fp64 (original), fp32, and fused freqs_i cache.

Not process-global: one CausalRoPE instance is attached to the DiT and its
self-attention modules. Profile uses CUDA events flushed once per temporal block.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


def _pct(xs: List[float], q: float) -> Optional[float]:
    xs = [float(v) for v in xs if math.isfinite(float(v))]
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return float(s[i])


def _grid_fhw(grid_sizes: torch.Tensor) -> Tuple[int, int, int]:
    vals = grid_sizes.tolist()
    if isinstance(vals[0], list):
        vals = vals[0]
    return int(vals[0]), int(vals[1]), int(vals[2])


def _split_freqs(freqs: torch.Tensor, c: int):
    return freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)


def build_freqs_i(
    freqs: torch.Tensor,
    f: int,
    h: int,
    w: int,
    start_frame: int,
    c: int,
) -> torch.Tensor:
    freq_f, freq_h, freq_w = _split_freqs(freqs, c)
    return torch.cat(
        [
            freq_f[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)


def rope_apply_original(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    real_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Match the previous causal_rope_apply Python loop (B0 when float64)."""
    n, c = x.size(2), x.size(3) // 2
    freq_f, freq_h, freq_w = _split_freqs(freqs, c)
    output = []
    f, h, w = _grid_fhw(grid_sizes)
    for i in range(len(x)):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(real_dtype).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freq_f[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        if freqs_i.dtype != x_i.dtype:
            freqs_i = freqs_i.to(x_i.dtype)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalRoPE:
    def __init__(self, mode: str = "fp64", profile: bool = False) -> None:
        if mode not in ("fp64", "fp32", "fp32_fused"):
            raise ValueError(f"unknown rope mode: {mode}")
        self.mode = mode
        self.profile = bool(profile)
        self._cache: Dict[Any, torch.Tensor] = {}
        self._ev_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.block_ms: List[float] = []

    def apply(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        start_frame: int = 0,
    ) -> torch.Tensor:
        if self.mode == "fp32_fused":
            return self._apply_fused(x, grid_sizes, freqs, int(start_frame))
        real_dtype = torch.float64 if self.mode == "fp64" else torch.float32
        return rope_apply_original(
            x, grid_sizes, freqs, int(start_frame), real_dtype=real_dtype
        )

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        start_frame: int = 0,
    ):
        if not self.profile or (not q.is_cuda):
            return (
                self.apply(q, grid_sizes, freqs, start_frame),
                self.apply(k, grid_sizes, freqs, start_frame),
            )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        rq = self.apply(q, grid_sizes, freqs, start_frame)
        rk = self.apply(k, grid_sizes, freqs, start_frame)
        end.record()
        self._ev_pairs.append((start, end))
        return rq, rk

    def finish_block(self) -> Optional[float]:
        if not self._ev_pairs:
            if self.profile:
                self.block_ms.append(0.0)
            return 0.0 if self.profile else None
        self._ev_pairs[-1][1].synchronize()
        ms = float(sum(s.elapsed_time(e) for s, e in self._ev_pairs))
        self._ev_pairs.clear()
        self.block_ms.append(ms)
        return ms

    def summary(self) -> Dict[str, Any]:
        skip = min(2, max(0, len(self.block_ms) - 1))
        ys = self.block_ms[skip:] if skip else list(self.block_ms)
        return {
            "mode": self.mode,
            "profile": self.profile,
            "n_block": len(self.block_ms),
            "warmup_dropped": skip,
            "p50_ms": _pct(ys, 50),
            "p95_ms": _pct(ys, 95),
            "sum_ms": float(sum(ys)) if ys else None,
            "all_sum_ms": float(sum(self.block_ms)) if self.block_ms else None,
        }

    def _apply_fused(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        start_frame: int,
    ) -> torch.Tensor:
        b, _, n, d = x.shape
        c = d // 2
        f, h, w = _grid_fhw(grid_sizes)
        seq_len = f * h * w
        freqs_i = self._freqs_i(freqs, f, h, w, start_frame, c)
        x_main = x[:, :seq_len].to(torch.float32).reshape(b, seq_len, n, c, 2).contiguous()
        xc = torch.view_as_complex(x_main)
        out = torch.view_as_real(xc * freqs_i).reshape(b, seq_len, n, d)
        out = out.type_as(x)
        if x.size(1) > seq_len:
            out = torch.cat([out, x[:, seq_len:]], dim=1)
        return out

    def _freqs_i(
        self,
        freqs: torch.Tensor,
        f: int,
        h: int,
        w: int,
        start_frame: int,
        c: int,
    ) -> torch.Tensor:
        key = (
            int(start_frame),
            f,
            h,
            w,
            c,
            freqs.device,
            freqs.dtype,
            int(freqs.data_ptr()),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        freqs_i = build_freqs_i(freqs, f, h, w, start_frame, c).to(torch.complex64)
        self._cache[key] = freqs_i
        return freqs_i


def attach_causal_rope(
    model,
    *,
    mode: str = "fp64",
    profile: bool = False,
    quiet: bool = False,
) -> CausalRoPE:
    rope = CausalRoPE(mode=mode, profile=profile)
    model.causal_rope = rope
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "self_attn"):
            blk.self_attn.causal_rope = rope
    if not quiet:
        print(f"[ROPE] mode={mode} profile={profile}", flush=True)
    return rope
