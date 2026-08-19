# RealCam-Vid default test subsets

Official **default evaluation battery** for Chunk Retention / MoSaiC.

| Subset | Clips | Role |
|--------|-------|------|
| `default_loop` | **60** | Revisit / loop-closure (World-State CR primary). Original 24 native indoor loops + 36 long ping-pong clips (RE10K outdoor / DL3DV / Mira) |
| `default_random` | 40 | Random forward tours, seed=42 (generalization; often **no** revisit) |
| `default_all` | 100 | loop + random (disjoint IDs) |

Clip IDs are listed under `subsets/*.txt`; metadata in `subsets/manifest.json`.

## Setup (once)

Large converted clips live in the archive. Link default subsets:

```bash
cd lingbot-world/bench/realcamvid
chmod +x build_subsets.sh
./build_subsets.sh
```

Override archive location:

```bash
REALCAMVID_ARCHIVE=/path/to/realcamvid ./build_subsets.sh
```

This creates:

- `clips_default_loop/`
- `clips_default_random/`
- `clips_default_all/`

## Run

```bash
source env.sh
cd "$LINGBOT_WORLD"

# Recommended battery (64 clips)
torchrun --nproc_per_node=8 bench/batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --clips_dir bench/realcamvid/clips_default_all \
  --out_dir output/realcamvid_default_all \
  --methods window,heuristic_cr,learned_cr,world_state_cr \
  --frame_num 481 --ulysses_size 8

# Loop-only (24 clips) — World-State CR focus
# --clips_dir bench/realcamvid/clips_default_loop
```

Or set `REALCAMVID_SUBSET=default_all` (see `batch_generate.py --subset`).

## Default eval = unified WS-CR revisit benchmark

One entry point runs **quantitative** WorldKV protocol (PSNR / SSIM / LPIPS / FID
+ FPS / ctx / peak) and **qualitative** VLM `same_place` on the **same pose pairs**.

```bash
REALCAMVID_OUT=output/realcamvid_ws_v3_default_loop \
REALCAMVID_SUBSET=default_loop \
METHODS=window,worldkv,world_state_cr \
  bash bench/realcamvid/run_wscr_benchmark.sh
# → $OUT/wscr_benchmark_default_loop.json  and  .md
```

| Env | Default | Meaning |
|-----|---------|---------|
| `RUN_VLM` | `1` | Set `0` for pixels-only |
| `SKIP_PREF` | `0` | Set `1` to skip pairwise VLM (absolute only, ~3× faster) |
| `MAX_PAIRS_Q` | `64` | Quantitative pairs/clip |
| `MAX_PAIRS_V` | `8` | VLM pairs/clip (subsample of the same pairing rule) |
| `QUANT_PY` | `python` | Env with lpips / skimage (`lingbot`) |
| `VLM_PY` | mus3d-sim python | Env with `transformers>=5.12` |

Standalone tracks (same scripts the launcher calls):

```bash
# quantitative only
REALCAMVID_OUT=... REALCAMVID_SUBSET=default_loop \
  METHODS=window,world_state_cr,worldkv \
  bash bench/realcamvid/run_worldkv_eval.sh

# qualitative only
PAIRS=8 DEVICE=cuda:0 bash bench/realcamvid/run_vlm_revisit.sh
```

**Coverage.** `default_loop` is **60** clips (same videos), all 481-frame
**multi_revisit** schedules on the captured pose path (`clips_revisit/`, see
`make_revisit_traj.py`): full out-and-back, a staggered mid-path loop, a local
loop at the far end, and short dwells — not ping-pong and not free-space SE(3).
`default_random` (40) remains short forward tours.

## Official WorldKV baseline (8×H20)

```bash
REALCAMVID_SUBSET=default_loop \
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
  bash bench/realcamvid/run_worldkv_official.sh
```

See [`docs/experiments/REALCAMVID.md`](../../../docs/experiments/REALCAMVID.md) and [`bench/worldkv/README.md`](../worldkv/README.md).
