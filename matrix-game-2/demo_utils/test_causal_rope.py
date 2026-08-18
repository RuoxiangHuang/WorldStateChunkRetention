"""CPU checks for causal RoPE modes (no DiT, no CUDA required)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan", "modules"))
from causal_rope import CausalRoPE, rope_apply_original  # noqa: E402


def _rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _wan_freqs(head_dim=128):
    d = head_dim
    return torch.cat(
        [
            _rope_params(64, d - 4 * (d // 6)),
            _rope_params(64, 2 * (d // 6)),
            _rope_params(64, 2 * (d // 6)),
        ],
        dim=1,
    )


def test_fp64_matches_original():
    torch.manual_seed(0)
    f, h, w, n, d = 2, 4, 4, 12, 128
    x = torch.randn(1, f * h * w, n, d, dtype=torch.float32)
    freqs = _wan_freqs(d)
    grid = torch.tensor([f, h, w])
    old = rope_apply_original(x, grid, freqs, start_frame=1, real_dtype=torch.float64)
    rope = CausalRoPE(mode="fp64")
    new = rope.apply(x, grid, freqs, start_frame=1)
    assert torch.equal(old, new)


def test_fp32_fused_matches_fp32_loop():
    torch.manual_seed(1)
    f, h, w, n, d = 2, 4, 4, 12, 128
    x = torch.randn(1, f * h * w, n, d, dtype=torch.float32)
    freqs = _wan_freqs(d)
    grid = torch.tensor([f, h, w])
    loop = CausalRoPE(mode="fp32").apply(x, grid, freqs, start_frame=0)
    fused = CausalRoPE(mode="fp32_fused").apply(x, grid, freqs, start_frame=0)
    diff = (loop.float() - fused.float()).abs().max().item()
    assert diff == 0.0, diff


def test_fp32_close_to_fp64():
    torch.manual_seed(2)
    f, h, w, n, d = 2, 4, 4, 12, 128
    x = torch.randn(1, f * h * w, n, d, dtype=torch.float32)
    freqs = _wan_freqs(d)
    grid = torch.tensor([f, h, w])
    a = CausalRoPE(mode="fp64").apply(x, grid, freqs, start_frame=0)
    b = CausalRoPE(mode="fp32").apply(x, grid, freqs, start_frame=0)
    max_abs = (a.float() - b.float()).abs().max().item()
    assert max_abs < 2e-5, max_abs


if __name__ == "__main__":
    test_fp64_matches_original()
    test_fp32_fused_matches_fp32_loop()
    test_fp32_close_to_fp64()
    print("ok")
