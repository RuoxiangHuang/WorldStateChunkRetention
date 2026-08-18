"""FFN GEMM microbench: bf16 Linear vs FP8 W8A8 scaled_mm. No DiT."""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wan.modules.ffn_gemm import Fp8Ffn, ffn_flops


def _ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    print(f"cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    print(f"device={torch.cuda.get_device_name(0)}")
    tokens, dim, ffn_dim = 2640, 1536, 8960
    print(f"shape tokens={tokens} dim={dim} ffn_dim={ffn_dim}")
    print(f"ffn_flops={ffn_flops(tokens, dim, ffn_dim):.3e}")
    x = torch.randn(1, tokens, dim, device="cuda", dtype=torch.bfloat16)
    ffn = nn.Sequential(
        nn.Linear(dim, ffn_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(ffn_dim, dim),
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        y_bf16 = ffn(x)
        bf16_ms = _ms(lambda: ffn(x))
        print(f"bf16_ms={bf16_ms:.3f}")
        try:
            fp8 = Fp8Ffn(ffn)
            fp8.prepare()
            y_fp8 = fp8(x)
            fp8_ms = _ms(lambda: fp8(x))
            diff = (y_bf16.float() - y_fp8.float()).abs()
            print(f"fp8_ms={fp8_ms:.3f} speedup={bf16_ms / fp8_ms:.3f}x")
            print(
                f"fp8_vs_bf16 max_abs={float(diff.max()):.6g} "
                f"mse={float(diff.pow(2).mean()):.6g}"
            )
        except Exception as e:
            print(f"fp8 unavailable: {type(e).__name__}: {e}")
            print("Do not enable --ffn-mode fp8 until this microbench works.")
        try:
            compiled = torch.compile(ffn, mode="reduce-overhead", fullgraph=False)
            compile_ms = _ms(lambda: compiled(x), warmup=30, iters=50)
            print(f"compile_ms={compile_ms:.3f} speedup={bf16_ms / compile_ms:.3f}x")
        except Exception as e:
            print(f"compile unavailable: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
