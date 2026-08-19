#!/usr/bin/env bash
# Unified WS-CR revisit benchmark: quantitative (WorldKV protocol) + optional VLM judge.
#
# Same pose pairing (radius / min_gap) for both tracks. Outputs land in $OUT:
#   worldkv_memory_eval_${SUBSET}.json
#   vlm_revisit_eval_${SUBSET}.json          (if RUN_VLM=1)
#   wscr_benchmark_${SUBSET}.json / .md      (merged report)
#
# Usage:
#   REALCAMVID_OUT=output/realcamvid_ws_v3_default_loop \
#   REALCAMVID_SUBSET=default_loop \
#   METHODS=window,worldkv,world_state_cr \
#     bash bench/realcamvid/run_wscr_benchmark.sh
#
# VLM uses a newer transformers env (Qwen3-VL). Quant uses lingbot.
set -euo pipefail

ROOT=/DATA/YuanZhen/Lingbot
cd "$ROOT/lingbot-world"
source "$ROOT/env.sh" 2>/dev/null || true

if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
fi
export TORCH_HOME="${TORCH_HOME:-/DATA/YuanZhen/.cache/torch}"
export INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth}"

OUT="${REALCAMVID_OUT:?set REALCAMVID_OUT to a generation dir with videos/ (+ stats/)}"
SUBSET="${REALCAMVID_SUBSET:-default_loop}"
METHODS="${METHODS:-window,world_state_cr,worldkv}"
RADIUS="${RADIUS:-0.15}"
MIN_GAP="${MIN_GAP:-30}"
MAX_PAIRS_Q="${MAX_PAIRS_Q:-64}"
MAX_PAIRS_V="${MAX_PAIRS_V:-8}"
DEVICE="${DEVICE:-cuda:0}"
RUN_VLM="${RUN_VLM:-1}"
SKIP_PREF="${SKIP_PREF:-0}"
QUANT_PY="${QUANT_PY:-python}"
VLM_PY="${VLM_PY:-/DATA/miniconda3_corl/envs_corl/mus3d-sim/bin/python}"
VLM_MODEL="${VLM_MODEL:-/DATA/YuanZhen/models/Qwen3-VL-8B-Instruct}"

case "$SUBSET" in
  default_loop)   CLIPS_DIR=bench/realcamvid/clips_default_loop ;;
  default_random) CLIPS_DIR=bench/realcamvid/clips_default_random ;;
  default_all)    CLIPS_DIR=bench/realcamvid/clips_default_all ;;
  *)              CLIPS_DIR="$SUBSET" ;;
esac

if [[ ! -d "$OUT/videos" ]]; then
  echo "[error] missing $OUT/videos" >&2
  exit 1
fi
bash bench/realcamvid/build_subsets.sh >/dev/null || true
mkdir -p "$OUT/logs"

Q_JSON="$OUT/worldkv_memory_eval_${SUBSET}.json"
V_JSON="$OUT/vlm_revisit_eval_${SUBSET}.json"
M_JSON="$OUT/wscr_benchmark_${SUBSET}.json"

echo "[wscr-bench] quant  out=$OUT subset=$SUBSET methods=$METHODS"
"$QUANT_PY" bench/eval_worldkv_memory.py \
  --out_dir "$OUT" \
  --clips_dir "$CLIPS_DIR" \
  --methods "$METHODS" \
  --radius "$RADIUS" \
  --min_gap "$MIN_GAP" \
  --max_pairs_per_clip "$MAX_PAIRS_Q" \
  --device "$DEVICE" \
  --save_name "$(basename "$Q_JSON")"

VLM_ARG=()
if [[ "$RUN_VLM" == "1" ]]; then
  if [[ ! -x "$VLM_PY" && ! -f "$VLM_PY" ]]; then
    echo "[warn] VLM python not found ($VLM_PY); skipping qualitative" >&2
  elif [[ ! -d "$VLM_MODEL" ]]; then
    echo "[warn] VLM weights missing ($VLM_MODEL); skipping qualitative" >&2
  else
    echo "[wscr-bench] vlm    model=$VLM_MODEL pairs/clip=$MAX_PAIRS_V"
    VLM_EXTRA=()
    if [[ "$SKIP_PREF" == "1" ]]; then
      VLM_EXTRA+=(--skip_preference)
    fi
    "$VLM_PY" bench/eval_vlm_revisit.py \
      --out_dir "$OUT" \
      --clips_dir "$CLIPS_DIR" \
      --methods "$METHODS" \
      --model_path "$VLM_MODEL" \
      --device "$DEVICE" \
      --radius "$RADIUS" \
      --min_gap "$MIN_GAP" \
      --max_pairs_per_clip "$MAX_PAIRS_V" \
      --save_name "$(basename "$V_JSON")" \
      "${VLM_EXTRA[@]}"
    VLM_ARG=(--vlm "$V_JSON")
  fi
else
  echo "[wscr-bench] vlm skipped (RUN_VLM=0)"
fi

"$QUANT_PY" bench/summarize_wscr_benchmark.py \
  --quant "$Q_JSON" \
  "${VLM_ARG[@]}" \
  --out "$M_JSON"

echo "[wscr-bench] DONE -> $M_JSON"
echo "             table -> ${M_JSON%.json}.md"
