"""CPU checks for FFN FP8 quantize (no scaled_mm)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan", "modules"))
from ffn_gemm import quantize_fp8_weight  # noqa: E402


def test_quantize_roundtrip_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    w8, scale = quantize_fp8_weight(w)
    recon = w8.float() * scale
    max_abs = (w.float() - recon).abs().max().item()
    assert max_abs < 0.25, max_abs


if __name__ == "__main__":
    test_quantize_roundtrip_bounded()
    print("ok")
