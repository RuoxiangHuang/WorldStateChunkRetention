"""CPU-only sanity for Matrix-Game 2.0 TICH (no weights)."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cond_hoist import (
    TICHState,
    attach_tich,
    conv3d_dynamic,
    conv3d_static,
    tensor_id_key,
)


class _FakePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(36, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def img_emb(self, x):
        return x * 3 + 1


def test_split_matches_full_conv():
    torch.manual_seed(0)
    model = _FakePatch()
    x = torch.randn(2, 16, 3, 8, 8)
    cond = torch.randn(2, 20, 3, 8, 8)
    full = model.patch_embedding(torch.cat([x, cond], dim=1))
    static = conv3d_static(model.patch_embedding, cond, x_channels=16)
    split = conv3d_dynamic(model.patch_embedding, x) + static
    assert torch.allclose(full, split, atol=1e-5, rtol=1e-5)


def test_disabled_never_reuses():
    state = TICHState(enabled=False, assert_counts=False)
    model = _FakePatch()
    attach_tich(model, state)
    x = torch.randn(1, 16, 3, 8, 8)
    cond = torch.randn(1, 20, 3, 8, 8)
    state.begin_block(0, model=model, cond_concat=cond)
    state.patch_embed(model, x, cond)
    state.patch_embed(model, x * 0.5, cond)
    assert state.n_conv_compute == 0
    assert state.n_conv_reuse == 0


def test_two_instances_do_not_share_cache():
    a = TICHState(enabled=True, assert_counts=False)
    b = TICHState(enabled=True, assert_counts=False)
    ma, mb = _FakePatch(), _FakePatch()
    attach_tich(ma, a)
    attach_tich(mb, b)
    xa = torch.randn(1, 16, 3, 8, 8)
    xb = torch.randn(1, 16, 3, 8, 8)
    ca = torch.randn(1, 20, 3, 8, 8)
    cb = torch.randn(1, 20, 3, 8, 8)
    a.begin_rollout(1)
    b.begin_rollout(2)
    a.begin_block(0, model=ma, cond_concat=ca)
    b.begin_block(0, model=mb, cond_concat=cb)
    a.patch_embed(ma, xa, ca)
    assert a.n_conv_compute == 1
    assert b.n_conv_compute == 1
    assert a._conv_value is not b._conv_value


def test_precompute_then_reuse_within_block():
    state = TICHState(enabled=True, assert_counts=True)
    model = _FakePatch()
    attach_tich(model, state)
    x1 = torch.randn(1, 16, 3, 8, 8)
    x2 = torch.randn(1, 16, 3, 8, 8)
    cond = torch.randn(1, 20, 3, 8, 8)
    state.begin_rollout(0)
    state.begin_block(0, model=model, cond_concat=cond)
    y1 = state.patch_embed(model, x1, cond)
    y2 = state.patch_embed(model, x2, cond)
    assert state.n_conv_compute == 1
    assert state.n_conv_reuse == 2
    full1 = model.patch_embedding(torch.cat([x1, cond], dim=1))
    full2 = model.patch_embedding(torch.cat([x2, cond], dim=1))
    assert torch.allclose(y1, full1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(y2, full2, atol=1e-5, rtol=1e-5)
    state.end_block(expected_forwards=2)


def test_new_block_or_new_cond_invalidates_conv():
    state = TICHState(enabled=True, assert_counts=False)
    model = _FakePatch()
    x = torch.randn(1, 16, 3, 8, 8)
    cond_a = torch.randn(1, 20, 3, 8, 8)
    cond_b = torch.randn(1, 20, 3, 8, 8)
    state.begin_rollout(0)
    state.begin_block(0, model=model, cond_concat=cond_a)
    state.patch_embed(model, x, cond_a)
    state.begin_block(1, model=model, cond_concat=cond_b)
    state.patch_embed(model, x, cond_b)
    assert state.n_conv_compute == 2
    assert tensor_id_key(cond_a) != tensor_id_key(cond_b)


def test_img_emb_keyed_and_survives_blocks():
    state = TICHState(enabled=True, assert_counts=False)
    model = _FakePatch()
    ctx = torch.randn(1, 4, 8)
    other = torch.randn(1, 4, 8)
    state.begin_rollout(0)
    state.begin_block(0)
    a = state.get_img_emb(model, ctx)
    state.begin_block(1)
    b = state.get_img_emb(model, ctx)
    assert torch.equal(a, b)
    assert state.n_img_compute == 1
    assert state.n_img_reuse == 1
    c = state.get_img_emb(model, other)
    assert state.n_img_compute == 2
    assert not torch.equal(a, c)


def test_rollout_id_invalidates_img():
    state = TICHState(enabled=True, assert_counts=False)
    model = _FakePatch()
    ctx = torch.randn(1, 4, 8)
    state.begin_rollout(0)
    state.get_img_emb(model, ctx)
    state.begin_rollout(1)
    state.get_img_emb(model, ctx)
    assert state.n_img_compute == 1  # reset() on begin_rollout


def test_slices_of_parent_have_distinct_keys():
    parent = torch.randn(1, 20, 6, 8, 8)
    a = parent[:, :, 0:3]
    b = parent[:, :, 3:6]
    assert tensor_id_key(a) != tensor_id_key(b)


def test_inplace_version_invalidates_img():
    state = TICHState(enabled=True, assert_counts=False)
    model = _FakePatch()
    ctx = torch.randn(1, 4, 8)
    state.begin_rollout(0)
    state.get_img_emb(model, ctx)
    ctx.add_(1.0)
    state.get_img_emb(model, ctx)
    assert state.n_img_compute == 2


if __name__ == "__main__":
    test_split_matches_full_conv()
    test_disabled_never_reuses()
    test_two_instances_do_not_share_cache()
    test_precompute_then_reuse_within_block()
    test_new_block_or_new_cond_invalidates_conv()
    test_img_emb_keyed_and_survives_blocks()
    test_rollout_id_invalidates_img()
    test_slices_of_parent_have_distinct_keys()
    test_inplace_version_invalidates_img()
    print("ok")
