#!/usr/bin/env python3
"""Merge WorldKV-style quantitative JSON + VLM qualitative JSON into one report.

Usage:
  python bench/summarize_wscr_benchmark.py \\
    --quant output/.../worldkv_memory_eval_default_loop.json \\
    --vlm   output/.../vlm_revisit_eval.json \\
    --out   output/.../wscr_benchmark_default_loop.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _f(v: Any, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"missing file: {p}")
    return json.loads(p.read_text())


def merge(
    quant: Optional[Dict[str, Any]],
    vlm: Optional[Dict[str, Any]],
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    aliases = aliases or {}
    q_raw = (quant or {}).get("by_method") or {}
    v_sum = (vlm or {}).get("summary") or {}
    v_raw = v_sum.get("by_method") or {}
    occupied = set(q_raw) | set(v_raw)

    def canon(name: str) -> str:
        dst = aliases.get(name, name)
        if dst != name and dst in occupied:
            return name
        return dst

    q_by = {canon(k): v for k, v in q_raw.items()}
    v_by = {canon(k): v for k, v in v_raw.items()}
    methods = list(dict.fromkeys(list(q_by) + list(v_by)))
    by_method = {}
    for m in methods:
        row: Dict[str, Any] = {}
        if m in q_by:
            row.update(q_by[m])
        if m in v_by:
            row["same_place_mean"] = v_by[m].get("same_place_mean")
            row["same_place_std"] = v_by[m].get("same_place_std")
            row["same_place_hist"] = v_by[m].get("same_place_hist")
            row["identity_drift_hist"] = v_by[m].get("identity_drift_hist")
            row["n_vlm_pairs"] = v_by[m].get("n")
        by_method[m] = row
    pairwise: Dict[str, Any] = {}
    for v in (v_sum.get("pairwise") or {}).values():
        a, b = canon(v["method_a"]), canon(v["method_b"])
        pairwise[f"{a}__vs__{b}"] = {**v, "method_a": a, "method_b": b}
    return {
        "protocol": {
            "name": "wscr_revisit_benchmark_v1",
            "quantitative": "WorldKV-style PSNR/SSIM/LPIPS/FID + FPS/ctx/peak",
            "qualitative": "VLM Revisit Memory Judge (same_place 0-3, optional pairwise)",
            "aliases": aliases,
        },
        "quantitative_source": (quant or {}).get("protocol"),
        "qualitative_source": (vlm or {}).get("protocol"),
        "by_method": by_method,
        "pairwise": pairwise or None,
        "parse_fail": v_sum.get("parse_fail"),
        "quantitative": quant,
        "qualitative": None if vlm is None else {
            "protocol": vlm.get("protocol"),
            "summary": vlm.get("summary"),
            "timing": vlm.get("timing"),
        },
    }


def render_md(merged: Dict[str, Any]) -> str:
    lines = [
        "# WS-CR Revisit Benchmark",
        "",
        "## Quantitative (WorldKV protocol)",
        "",
        "| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FPS↑ | ctx | peakGB | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in merged["by_method"].items():
        ctx = r.get("mean_ctx")
        ctx_s = f"{ctx:.0f}" if isinstance(ctx, (int, float)) else "—"
        lines.append(
            f"| {m} | {_f(r.get('psnr'))} | {_f(r.get('ssim'), 4)} | "
            f"{_f(r.get('lpips'), 4)} | {_f(r.get('fid'), 2)} | "
            f"{_f(r.get('throughput_fps'))} | {ctx_s} | "
            f"{_f(r.get('mean_peak_gb'), 2)} | {r.get('n_clips', '—')} |"
        )
    lines += [
        "",
        "## Qualitative (VLM same_place)",
        "",
        "| Method | same_place mean | n pairs | hist {0,1,2,3} |",
        "|---|---:|---:|---|",
    ]
    any_vlm = False
    for m, r in merged["by_method"].items():
        if r.get("same_place_mean") is None:
            lines.append(f"| {m} | — | — | — |")
            continue
        any_vlm = True
        hist = r.get("same_place_hist") or {}
        h = ", ".join(f"{k}:{hist.get(str(k), 0)}" for k in range(4))
        lines.append(
            f"| {m} | {_f(r.get('same_place_mean'), 3)} | "
            f"{r.get('n_vlm_pairs', '—')} | {h} |"
        )
    if not any_vlm:
        lines.append("")
        lines.append("_VLM not run (pass `--vlm` JSON or set RUN_VLM=1)._")
    pw = merged.get("pairwise") or {}
    if pw:
        lines += ["", "## Pairwise preference (position-balanced)", ""]
        for v in pw.values():
            lines.append(
                f"- {v['method_a']} vs {v['method_b']}: "
                f"A={v['win_rate_A']:.3f} B={v['win_rate_B']:.3f} "
                f"tie={v['tie_rate']:.3f} (n={v['n']})"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge WS-CR quant + VLM eval dumps")
    ap.add_argument("--quant", required=True, help="eval_worldkv_memory JSON")
    ap.add_argument("--vlm", default=None, help="eval_vlm_revisit JSON (optional)")
    ap.add_argument("--out", required=True, help="merged JSON path")
    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Rename method keys, e.g. --alias ws_v3_a05_g64=world_state_cr (repeatable)",
    )
    args = ap.parse_args()

    aliases = {}
    for item in args.alias:
        if "=" not in item:
            raise SystemExit(f"bad --alias {item!r}; expected src=dst")
        src, dst = item.split("=", 1)
        aliases[src.strip()] = dst.strip()

    quant = _load(args.quant)
    vlm = _load(args.vlm) if args.vlm else None
    merged = merge(quant, vlm, aliases=aliases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    md = render_md(merged)
    md_path = out.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[saved] {out}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
