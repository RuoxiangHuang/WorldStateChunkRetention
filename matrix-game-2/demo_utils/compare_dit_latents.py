"""Compare dumped denoised latents from two B0/B1 rollouts."""
from __future__ import annotations

import argparse
import math

import torch


def _metrics(a: torch.Tensor, b: torch.Tensor):
    diff = a.float() - b.float()
    mse = float(diff.pow(2).mean().item())
    max_abs = float(diff.abs().max().item())
    psnr = None if mse <= 0 else 10.0 * math.log10(1.0 / mse)
    return max_abs, mse, psnr, bool(torch.equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    da = torch.load(args.a, map_location="cpu")
    db = torch.load(args.b, map_location="cpu")
    print(f"blocks a={len(da['blocks'])} b={len(db['blocks'])}")
    for i, (x, y) in enumerate(zip(da["blocks"], db["blocks"])):
        max_abs, mse, psnr, eq = _metrics(x, y)
        psnr_s = "inf" if psnr is None else f"{psnr:.3f}"
        print(
            f"block {i}: equal={eq} max_abs={max_abs:.6g} mse={mse:.6g} psnr={psnr_s}"
        )
    max_abs, mse, psnr, eq = _metrics(da["output"], db["output"])
    psnr_s = "inf" if psnr is None else f"{psnr:.3f}"
    print(f"full: equal={eq} max_abs={max_abs:.6g} mse={mse:.6g} psnr={psnr_s}")


if __name__ == "__main__":
    main()
