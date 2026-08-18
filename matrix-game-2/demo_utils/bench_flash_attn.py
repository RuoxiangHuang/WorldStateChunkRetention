"""Attention microbench for B2. Does not change DiT device detection."""
from __future__ import annotations

import time

import torch


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
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"device={name}")
        low = name.lower()
        hopper_str = any(k in low for k in ("h100", "h800", "hopper", "l20y"))
        print(f"name_matches_current_fa3_gate={hopper_str} h20_in_name={'h20' in low}")
    try:
        import flash_attn
        print(f"flash_attn={getattr(flash_attn, '__version__', 'yes')}")
        fa2 = True
    except Exception as e:
        print(f"flash_attn unavailable: {e}")
        fa2 = False
    try:
        import flash_attn_interface
        print(f"flash_attn_interface={flash_attn_interface}")
        fa3 = True
    except Exception as e:
        print(f"flash_attn_interface unavailable: {e}")
        fa3 = False

    if not torch.cuda.is_available():
        return
    # MG2 causal decode: q=3*880, kv window=6*880, heads=12, dim=128
    b, hq, hk, n, d = 1, 2640, 5280, 12, 128
    q = torch.randn(b, hq, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hk, n, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, hk, n, d, device="cuda", dtype=torch.bfloat16)
    if fa2:
        from flash_attn import flash_attn_func

        def run_fa2():
            return flash_attn_func(q, k, v, causal=False)

        print(f"fa2_ms={_ms(run_fa2):.3f}  q={tuple(q.shape)} k={tuple(k.shape)}")
    if fa3:
        from flash_attn_interface import flash_attn_func as fa3_func

        def run_fa3():
            return fa3_func(q, k, v, causal=False)

        print(f"fa3_ms={_ms(run_fa3):.3f}")
    print("Note: this does not enable FA3 in DiT. Change attention.py only after a win here.")


if __name__ == "__main__":
    main()
