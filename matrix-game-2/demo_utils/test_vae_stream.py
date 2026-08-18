"""CPU sanity for VAE stream opt (prealloc join + in-place cache)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan", "modules"))
from vae_cache import join_time, set_vae_prealloc, store_feat_cache, vae_prealloc_enabled


def test_join_time_matches_cat():
    torch.manual_seed(0)
    a = torch.randn(1, 3, 4, 8, 8)
    b = torch.randn(1, 3, 4, 8, 8)
    c = torch.randn(1, 3, 4, 8, 8)
    old = torch.cat([a, b], 2)
    old = torch.cat([old, c], 2)
    new = join_time([a, b, c])
    assert torch.equal(old, new)


def test_join_time_ragged_falls_back_to_cat():
    a = torch.randn(1, 3, 2, 8, 8)
    b = torch.randn(1, 3, 4, 8, 8)
    assert torch.equal(join_time([a, b]), torch.cat([a, b], 2))


def test_store_feat_reuses_storage():
    prev = vae_prealloc_enabled()
    try:
        set_vae_prealloc(True)
        slot = torch.zeros(1, 4, 2, 2, 2)
        cache = [slot]
        nxt = torch.ones_like(slot)
        store_feat_cache(cache, 0, nxt)
        assert cache[0] is slot
        assert torch.equal(slot, nxt)
    finally:
        set_vae_prealloc(prev)


def test_store_feat_first_write_allocates():
    cache = [None]
    val = torch.randn(1, 2, 2, 2, 2)
    store_feat_cache(cache, 0, val)
    assert cache[0] is val


def test_store_feat_legacy_replaces_storage():
    prev = vae_prealloc_enabled()
    try:
        set_vae_prealloc(False)
        slot = torch.zeros(1, 4, 2, 2, 2)
        cache = [slot]
        nxt = torch.ones_like(slot)
        store_feat_cache(cache, 0, nxt)
        assert cache[0] is nxt
    finally:
        set_vae_prealloc(prev)


def test_join_time_aliased_pieces_collapse():
    """Graph replay returning one buffer makes later join see only the last frame."""
    buf = torch.randn(1, 3, 4, 8, 8)
    pieces = [buf, buf]
    last = torch.ones_like(buf)
    buf.copy_(last)
    out = join_time(pieces)
    assert torch.equal(out[:, :, :4], last)
    assert torch.equal(out[:, :, 4:], last)


if __name__ == "__main__":
    test_join_time_matches_cat()
    test_join_time_ragged_falls_back_to_cat()
    test_store_feat_reuses_storage()
    test_store_feat_first_write_allocates()
    test_store_feat_legacy_replaces_storage()
    test_join_time_aliased_pieces_collapse()
    print("ok")
