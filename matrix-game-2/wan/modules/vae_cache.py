"""Bit-exact helpers for causal VAE streaming (no third-party deps)."""
from typing import List

import torch

# Sequential A/B/C only: one active decoder at a time. True = in-place copy_
# (C). False = original ``feat_cache[idx] = value`` (A/B legacy).
_USE_PREALLOC = True


def set_vae_prealloc(enabled: bool) -> None:
    global _USE_PREALLOC
    _USE_PREALLOC = bool(enabled)


def vae_prealloc_enabled() -> bool:
    return bool(_USE_PREALLOC)


def store_feat_cache(feat_cache, idx, value):
    """Keep causal VAE cache slots at a stable storage when the shape matches."""
    if not _USE_PREALLOC:
        feat_cache[idx] = value
        return
    slot = feat_cache[idx]
    if (
        torch.is_tensor(slot)
        and torch.is_tensor(value)
        and slot.shape == value.shape
        and slot.dtype == value.dtype
        and slot.device == value.device
    ):
        if slot.data_ptr() != value.data_ptr():
            slot.copy_(value)
        return
    feat_cache[idx] = value


def join_time(pieces: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-frame decoder outputs without a growing ``torch.cat``."""
    if len(pieces) == 1:
        return pieces[0]
    t0 = pieces[0].shape[2]
    if all(p.shape[2] == t0 for p in pieces):
        b, c, _, h, w = pieces[0].shape
        out = pieces[0].new_empty(b, c, t0 * len(pieces), h, w)
        for i, p in enumerate(pieces):
            out[:, :, i * t0:(i + 1) * t0].copy_(p)
        return out
    return torch.cat(pieces, 2)


def feat_cache_ready(feat_cache) -> bool:
    return bool(feat_cache) and all(torch.is_tensor(c) for c in feat_cache)
