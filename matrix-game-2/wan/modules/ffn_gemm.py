"""FFN GEMM path: bf16 baseline, optional FP8 W8A8 via torch._scaled_mm.

Quality-equivalent, not bit-exact. No Transformer Engine / extra packages.
Weight is quantized once; activations are quantized every forward.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# e4m3fn finite max
_FP8_E4M3_MAX = 448.0


def _pct(xs: List[float], q: float) -> Optional[float]:
    xs = [float(v) for v in xs if math.isfinite(float(v))]
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return float(s[i])


class GemmTimer:
    """CUDA-event split: self-attn vs FFN, flushed once per temporal block."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "attn": [],
            "ffn": [],
        }
        self.block_ms: Dict[str, List[float]] = {"attn": [], "ffn": []}

    def timed(self, name: str, fn: Callable[[], Any]) -> Any:
        if (not self.enabled) or (not torch.cuda.is_available()):
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        self._pairs.setdefault(name, []).append((start, end))
        return out

    def finish_block(self) -> Dict[str, float]:
        out = {"attn": 0.0, "ffn": 0.0}
        if not self.enabled:
            return out
        for name, pairs in self._pairs.items():
            if not pairs:
                self.block_ms.setdefault(name, []).append(0.0)
                continue
            pairs[-1][1].synchronize()
            ms = float(sum(s.elapsed_time(e) for s, e in pairs))
            self._pairs[name] = []
            self.block_ms.setdefault(name, []).append(ms)
            out[name] = ms
        return out

    def summary(self) -> Dict[str, Any]:
        skip = min(2, max(0, len(self.block_ms.get("ffn", [])) - 1))

        def pack(xs: List[float]) -> Dict[str, Any]:
            ys = xs[skip:] if skip else list(xs)
            return {
                "n": len(ys),
                "p50_ms": _pct(ys, 50),
                "p95_ms": _pct(ys, 95),
                "sum_ms": float(sum(ys)) if ys else None,
            }

        return {
            "profile": self.enabled,
            "warmup_dropped": skip,
            "attn": pack(self.block_ms.get("attn", [])),
            "ffn": pack(self.block_ms.get("ffn", [])),
        }


def quantize_fp8_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach()
    amax = w.abs().amax().clamp(min=1e-12)
    scale = (amax.float() / _FP8_E4M3_MAX).to(dtype=torch.float32)
    w8 = (w.float() / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    return w8, scale


def quantize_fp8_act(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax.float() / _FP8_E4M3_MAX).to(dtype=torch.float32)
    x8 = (x.float() / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    return x8, scale.reshape(())


def scaled_mm_fp8(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """x [M,K] @ w [N,K]. cuBLASLt wants row-major A and column-major B.

    Keep ``w_fp8`` as contiguous ``[N,K]`` and pass ``w_fp8.t()`` without
    ``contiguous()``, so B is a column-major ``[K,N]`` view.
    """
    x8, x_scale = quantize_fp8_act(x)
    if not x8.is_contiguous():
        x8 = x8.contiguous()
    w_fp8 = w_fp8.contiguous()
    w_col = w_fp8.t()
    kwargs = dict(
        scale_a=x_scale,
        scale_b=w_scale.reshape(()),
        out_dtype=x.dtype,
        use_fast_accum=True,
    )
    if bias is not None:
        kwargs["bias"] = bias.to(dtype=x.dtype).contiguous()
    out = torch._scaled_mm(x8, w_col, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


class Fp8Ffn(nn.Module):
    def __init__(self, ffn: nn.Sequential) -> None:
        super().__init__()
        self.orig = ffn
        self._w1 = None
        self._s1 = None
        self._w2 = None
        self._s2 = None
        self._ready = False

    def prepare(self) -> None:
        fc1: nn.Linear = self.orig[0]
        fc2: nn.Linear = self.orig[2]
        self._w1, self._s1 = quantize_fp8_weight(fc1.weight)
        self._w2, self._s2 = quantize_fp8_weight(fc2.weight)
        self._s1 = self._s1.to(device=fc1.weight.device)
        self._s2 = self._s2.to(device=fc2.weight.device)
        self._ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._ready:
            self.prepare()
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        fc1: nn.Linear = self.orig[0]
        fc2: nn.Linear = self.orig[2]
        y = scaled_mm_fp8(x2, self._w1, self._s1, fc1.bias)
        y = self.orig[1](y)
        y = scaled_mm_fp8(y, self._w2, self._s2, fc2.bias)
        return y.view(shape)


def attach_gemm_timer(model, *, enabled: bool = False) -> GemmTimer:
    timer = GemmTimer(enabled=enabled)
    model.gemm_timer = timer
    for blk in getattr(model, "blocks", []):
        blk.gemm_timer = timer
    return timer


def configure_ffn(model, *, mode: str = "bf16") -> None:
    if mode not in ("bf16", "fp8", "compile"):
        raise ValueError(f"unknown ffn mode: {mode}")
    n = 0
    for blk in getattr(model, "blocks", []):
        ffn = getattr(blk, "ffn", None)
        if ffn is None:
            continue
        if mode == "fp8":
            if not isinstance(ffn, Fp8Ffn):
                blk.ffn = Fp8Ffn(ffn)
            blk.ffn.prepare()
            n += 1
        elif mode == "compile":
            if isinstance(ffn, Fp8Ffn):
                raise RuntimeError("cannot compile after fp8 wrap")
            blk.ffn.compile(mode="reduce-overhead", fullgraph=False)
            n += 1
        else:
            n += 1
    print(f"[FFN] mode={mode} modules={n}", flush=True)


def ffn_flops(tokens: int = 2640, dim: int = 1536, ffn_dim: int = 8960) -> int:
    return 2 * tokens * (dim * ffn_dim + ffn_dim * dim)
