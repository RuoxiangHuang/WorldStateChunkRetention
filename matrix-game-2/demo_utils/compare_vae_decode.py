"""GPU VAE A/B/C regression: no DiT. Same latents, same causal cache timeline.

A: compile + legacy cat/assign
B: eager + legacy cat/assign
C: eager + prealloc/copy_
D (Graph) is not run.

C vs B should be ~element-wise identical. A vs B may differ (compile kernels).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from demo_utils.vae_block3 import VAEDecoderWrapper, configure_vae_decoder


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    mse = float(diff.pow(2).mean().item())
    max_abs = float(diff.abs().max().item())
    # decoder outputs are clamped to [-1, 1]
    psnr = None if mse <= 0 else 10.0 * math.log10(4.0 / mse)
    return {
        "max_abs": max_abs,
        "mse": mse,
        "psnr_db": psnr,
        "equal": bool(torch.equal(a, b)),
        "shape": list(a.shape),
    }


def _load_decoder(pretrained: str, device: torch.device) -> VAEDecoderWrapper:
    decoder = VAEDecoderWrapper()
    state = torch.load(os.path.join(pretrained, "Wan2.1_VAE.pth"), map_location="cpu")
    decoder_state = {
        k: v for k, v in state.items() if "decoder." in k or "conv2" in k
    }
    decoder.load_state_dict(decoder_state)
    decoder.to(device, torch.float16)
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder


def _decode_blocks(
    decoder: VAEDecoderWrapper,
    latents: torch.Tensor,
    frames_per_block: int,
) -> List[torch.Tensor]:
    n_cache = int(decoder.decoder.decoder_conv_num)
    cache = [None] * n_cache
    outs: List[torch.Tensor] = []
    n = latents.shape[1]
    with torch.no_grad():
        for start in range(0, n, frames_per_block):
            chunk = latents[:, start:start + frames_per_block]
            video, cache = decoder(chunk, *cache)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outs.append(video.detach().contiguous().cpu())
    return outs


def _pair_report(
    name: str,
    left: List[torch.Tensor],
    right: List[torch.Tensor],
) -> Dict[str, Any]:
    blocks = [_metrics(a, b) for a, b in zip(left, right)]
    cat_l = torch.cat(left, 1)
    cat_r = torch.cat(right, 1)
    overall = _metrics(cat_l, cat_r)
    return {"pair": name, "overall": overall, "blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--num_latent_frames", type=int, default=24)
    parser.add_argument("--frames_per_block", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Only B vs C (no A). Use if compile tax is not needed this run.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("compare_vae_decode.py needs CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    # Match inference decoder input: [B, F, C, H, W] after DiT transpose.
    latents = torch.randn(
        1, args.num_latent_frames, 16, 44, 80,
        device=device, dtype=torch.float16,
    )

    def run_mode(tag: str, *, legacy: bool, compile_vae: bool) -> List[torch.Tensor]:
        print(f"[VAE-ABC] decode {tag}", flush=True)
        decoder = _load_decoder(args.pretrained_model_path, device)
        configure_vae_decoder(decoder, legacy=legacy, compile_vae=compile_vae)
        return _decode_blocks(decoder, latents, args.frames_per_block)

    outs_b = run_mode("B_eager_legacy", legacy=True, compile_vae=False)
    outs_c = run_mode("C_eager_prealloc", legacy=False, compile_vae=False)
    report: Dict[str, Any] = {
        "seed": args.seed,
        "num_latent_frames": args.num_latent_frames,
        "frames_per_block": args.frames_per_block,
        "pairs": {
            "C_vs_B": _pair_report("C_vs_B", outs_c, outs_b),
        },
    }
    if not args.skip_compile:
        outs_a = run_mode("A_compile_legacy", legacy=True, compile_vae=True)
        report["pairs"]["B_vs_A"] = _pair_report("B_vs_A", outs_b, outs_a)
        report["pairs"]["C_vs_A"] = _pair_report("C_vs_A", outs_c, outs_a)

    def _line(pair: Dict[str, Any]) -> str:
        o = pair["overall"]
        psnr = "inf" if o["psnr_db"] is None else f"{o['psnr_db']:.3f}"
        return (
            f"{pair['pair']}: equal={o['equal']} max_abs={o['max_abs']:.6g} "
            f"mse={o['mse']:.6g} psnr_db={psnr} (range=[-1,1])"
        )

    print("[VAE-ABC] summary", flush=True)
    for pair in report["pairs"].values():
        print("  " + _line(pair), flush=True)
    cb = report["pairs"]["C_vs_B"]["overall"]
    if cb["equal"] or cb["max_abs"] == 0.0:
        print("[VAE-ABC] C vs B: copy_/prealloc matches legacy assign (expected).", flush=True)
    else:
        print(
            "[VAE-ABC] C vs B DIVERGED: copy_/prealloc path is not bit-exact. "
            "Do not treat C as a lossless stand-in for B.",
            flush=True,
        )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[VAE-ABC] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
