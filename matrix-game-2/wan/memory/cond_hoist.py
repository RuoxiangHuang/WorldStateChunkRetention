"""TICH for Matrix-Game 2.0.

Exact elimination of timestep-invariant conditioning work inside the
S-step denoising loop. Orthogonal to KV CR. Default off.

MG2 has no LingBot camera Linear injectors. Hoistable subgraphs:

  1. I2V patch Conv3d split: Conv([x, cond]) = Conv_x(x) + Conv_c(cond) + b
  2. CLIP ``img_emb(visual_context)`` MLP output only.

Do not hoist ActionModule. Do not cache cross-attn K/V — MG2 already does
that in ``crossattn_cache``; TICH only caches the CLIP MLP tokens.

State is per pipeline/model instance (``TICHState``), never a process-global
dict. Caches are keyed and invalidated explicitly.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


TensorKey = Tuple[Any, ...]


def tensor_id_key(x: Optional[torch.Tensor]) -> TensorKey:
    """shape / dtype / device / storage ptr / autograd version.

    Uses ``Tensor._version`` so in-place writes invalidate the cache without
    reading GPU values (``.item()`` / ``.mean()`` would sync the hot path).
    """
    if x is None:
        return ("none",)
    ptr = 0
    try:
        # Tensor.data_ptr() is the view start. Storage data_ptr() is shared
        # by all slices of the same parent and would collide across blocks.
        ptr = int(x.data_ptr()) if x.numel() else 0
    except Exception:
        ptr = 0
    return (
        tuple(int(s) for s in x.shape),
        str(x.dtype),
        str(x.device),
        ptr,
        int(getattr(x, "_version", 0)),
    )


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


def _conv_kwargs(conv: torch.nn.Conv3d) -> Dict[str, Any]:
    return {
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
        "groups": conv.groups,
    }


def conv3d_static(conv: torch.nn.Conv3d, cond: torch.Tensor, x_channels: int) -> torch.Tensor:
    """Conv_c(cond) + bias. ``cond`` is the trailing channels of the Conv3d input."""
    w = conv.weight
    in_ch = int(w.shape[1])
    cy = int(cond.shape[1])
    cx = int(x_channels)
    if cx + cy != in_ch:
        raise ValueError(f"channel split mismatch {cx}+{cy} vs in_ch={in_ch}")
    w_c = w[:, cx:cx + cy]
    return F.conv3d(cond, w_c, bias=conv.bias, **_conv_kwargs(conv))


def conv3d_dynamic(conv: torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """Conv_x(x) without bias."""
    w = conv.weight
    cx = int(x.shape[1])
    w_x = w[:, :cx]
    return F.conv3d(x, w_x, bias=None, **_conv_kwargs(conv))


class TICHState:
    """Per-instance hoist cache. Attach to CausalWanModel via ``attach_tich``."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        conv_split: bool = True,
        img_emb: bool = True,
        profile: bool = False,
        assert_counts: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.conv_split = bool(conv_split)
        self.img_emb_on = bool(img_emb)
        self.profile = bool(profile)
        self.assert_counts = bool(assert_counts)
        self.rollout_id: int = 0
        self.block_id: int = -1
        self.n_conv_compute = 0
        self.n_conv_reuse = 0
        self.n_img_compute = 0
        self.n_img_reuse = 0
        self._img_key: Optional[TensorKey] = None
        self._img_value: Optional[torch.Tensor] = None
        self._conv_key: Optional[Tuple[Any, ...]] = None
        self._conv_value: Optional[torch.Tensor] = None
        self._block_compute0 = 0
        self._block_reuse0 = 0
        self.time_ms: Dict[str, List[float]] = {
            "conv_c": [],
            "conv_x": [],
            "conv_full": [],
            "img_emb": [],
            "denoise": [],
            "block": [],
            "rollout": [],
        }
        self._rollout_t0 = 0.0

    def reset(self) -> None:
        self.block_id = -1
        self.n_conv_compute = 0
        self.n_conv_reuse = 0
        self.n_img_compute = 0
        self.n_img_reuse = 0
        self._img_key = None
        self._img_value = None
        self._conv_key = None
        self._conv_value = None
        for k in self.time_ms:
            self.time_ms[k] = []

    def begin_rollout(self, rollout_id: int = 0) -> None:
        self.reset()
        self.rollout_id = int(rollout_id)
        self._rollout_t0 = time.perf_counter()

    def end_rollout(self) -> None:
        elapsed = (time.perf_counter() - self._rollout_t0) * 1000.0
        self.time_ms["rollout"].append(elapsed)
        self._conv_key = None
        self._conv_value = None
        self.block_id = -1

    def begin_block(
        self,
        block_id: int,
        *,
        model=None,
        cond_concat: Optional[torch.Tensor] = None,
        x_channels: int = 16,
    ) -> None:
        self.block_id = int(block_id)
        self._block_compute0 = self.n_conv_compute
        self._block_reuse0 = self.n_conv_reuse
        self._conv_key = None
        self._conv_value = None
        if (
            self.enabled
            and self.conv_split
            and model is not None
            and cond_concat is not None
        ):
            self.precompute_conv_static(model, cond_concat, x_channels=x_channels)

    def end_block(self, *, expected_forwards: Optional[int] = None) -> None:
        if (
            self.enabled
            and self.assert_counts
            and expected_forwards is not None
            and expected_forwards > 0
        ):
            d_compute = self.n_conv_compute - self._block_compute0
            d_reuse = self.n_conv_reuse - self._block_reuse0
            # Conv_c precomputed once; every generator forward reuses it.
            if d_compute != 1 or d_reuse != int(expected_forwards):
                raise AssertionError(
                    "TICH per-block conv counts: "
                    f"compute={d_compute} reuse={d_reuse}, "
                    f"expected compute=1 reuse={int(expected_forwards)} "
                    f"(block_id={self.block_id})"
                )
        self._conv_key = None
        self._conv_value = None
        self.block_id = -1

    def timed(self, bucket: str, fn: Callable[[], Any]) -> Any:
        if (not self.profile) or (not torch.cuda.is_available()):
            if not self.profile:
                return fn()
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

    def _img_lookup_key(self, visual_context: torch.Tensor) -> TensorKey:
        return (self.rollout_id, tensor_id_key(visual_context))

    def _conv_lookup_key(self, cond_concat: torch.Tensor) -> Tuple[Any, ...]:
        return (self.rollout_id, int(self.block_id), tensor_id_key(cond_concat))

    def precompute_img_emb(self, model, visual_context: torch.Tensor) -> torch.Tensor:
        return self.get_img_emb(model, visual_context)

    def get_img_emb(self, model, visual_context: torch.Tensor) -> torch.Tensor:
        if (not self.enabled) or (not self.img_emb_on):
            return model.img_emb(visual_context)
        key = self._img_lookup_key(visual_context)
        if self._img_value is not None and self._img_key == key:
            self.n_img_reuse += 1
            return self._img_value
        y = self.timed("img_emb", lambda: model.img_emb(visual_context))
        self._img_key = key
        self._img_value = y
        self.n_img_compute += 1
        return y

    def precompute_conv_static(
        self, model, cond_concat: torch.Tensor, *, x_channels: int = 16
    ) -> torch.Tensor:
        conv = model.patch_embedding
        key = self._conv_lookup_key(cond_concat)
        if self._conv_value is not None and self._conv_key == key:
            return self._conv_value

        def _run():
            return conv3d_static(conv, cond_concat, x_channels=x_channels)

        static = self.timed("conv_c", _run)
        self._conv_key = key
        self._conv_value = static
        self.n_conv_compute += 1
        if self.profile:
            # Same-shape full Conv probe, outside the denoise timer.
            x_ch = int(x_channels)
            dummy = torch.zeros(
                cond_concat.shape[0],
                x_ch,
                *cond_concat.shape[2:],
                device=cond_concat.device,
                dtype=cond_concat.dtype,
            )
            self.timed(
                "conv_full",
                lambda: conv(torch.cat([dummy, cond_concat], dim=1)),
            )
        return static

    def patch_embed(self, model, x: torch.Tensor, cond_concat: torch.Tensor) -> torch.Tensor:
        conv = model.patch_embedding
        if (not self.enabled) or (not self.conv_split):
            return conv(torch.cat([x, cond_concat], dim=1))
        key = self._conv_lookup_key(cond_concat)
        static = self._conv_value
        if static is None or self._conv_key != key:
            static = self.precompute_conv_static(
                model, cond_concat, x_channels=int(x.shape[1]))
        else:
            self.n_conv_reuse += 1
        dyn = self.timed("conv_x", lambda: conv3d_dynamic(conv, x))
        return dyn + static

    def summary(self) -> Dict[str, Any]:
        profile = {}
        for k, xs in self.time_ms.items():
            profile[k] = {
                "n": len(xs),
                "mean_ms": _mean(xs),
                "p50_ms": _pct(xs, 50),
                "p95_ms": _pct(xs, 95),
                "sum_ms": (float(sum(xs)) if xs else None),
            }
        return {
            "enabled": self.enabled,
            "rollout_id": self.rollout_id,
            "flags": {
                "conv_split": self.conv_split,
                "img_emb": self.img_emb_on,
                "profile": self.profile,
            },
            "counts": {
                "conv_compute": self.n_conv_compute,
                "conv_reuse": self.n_conv_reuse,
                "img_compute": self.n_img_compute,
                "img_reuse": self.n_img_reuse,
            },
            "profile_ms": profile,
            "note": (
                "TICH caches CLIP MLP tokens only; cross-attn K/V stays in "
                "pipeline.crossattn_cache. Conv split is an engineering cache, "
                "not a guaranteed wall-clock win on 3-step MG2. "
                "After block-level precompute, expected per denoise block is "
                "conv_compute=1 and conv_reuse=num_steps+refresh."
            ),
        }


def attach_tich(model, state: Optional[TICHState]) -> None:
    model.tich = state
