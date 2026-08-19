#!/usr/bin/env bash
# VLM Revisit Memory Judge on RealCam-Vid default_loop.
# Judge: Qwen3-VL-8B-Instruct (local). Methods: window / worldkv / WSCRv3.
set -euo pipefail
cd /DATA/YuanZhen/Lingbot/lingbot-world

PY=${PY:-/DATA/miniconda3_corl/envs_corl/mus3d-sim/bin/python}
OUT=${OUT:-output/realcamvid_vlm_revisit_default_loop}
CLIPS=${CLIPS:-bench/realcamvid/clips_default_loop}
MODEL=${MODEL:-/DATA/YuanZhen/models/Qwen3-VL-8B-Instruct}
METHODS=${METHODS:-window,worldkv,ws_v3_a05_g64}
DEVICE=${DEVICE:-cuda:0}
PAIRS=${PAIRS:-8}

mkdir -p "$OUT/logs"
echo "[vlm-revisit] model=$MODEL methods=$METHODS pairs/clip=$PAIRS"
"$PY" bench/eval_vlm_revisit.py \
  --out_dir "$OUT" \
  --clips_dir "$CLIPS" \
  --methods "$METHODS" \
  --model_path "$MODEL" \
  --device "$DEVICE" \
  --max_pairs_per_clip "$PAIRS" \
  --save_name vlm_revisit_eval_default_loop.json \
  "$@"
