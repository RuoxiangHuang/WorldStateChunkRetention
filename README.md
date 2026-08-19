# World-State Chunk Retention for Long-Horizon Video World Models

Sparse KV inference for camera-conditioned **LingBot-World-Fast**: keep a
bounded attention context without forgetting revisited viewpoints.

This repository is the research / collaboration tree around **Chunk Retention
(CR)** — especially **World-State CR** (default **v3** = future-use selector +
Memory Consolidation + SWTP).

> **Sole source tree:** [`lingbot-world/`](lingbot-world/)  
> Paper LaTeX: [`paper/`](paper/) · Docs: [`docs/`](docs/) · Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Why

Autoregressive world models store history in KV caches. A **sliding window**
is fast but drops past viewpoints; full history is faithful but too expensive.
**Chunk Retention** answers:

> *Under a fixed on-device KV budget, which historical chunks stay after each
> new chunk is generated?*

Default answer: tiered **sink / recent / archive**, with archive ranked by an
11-D **World-State** ChunkSelector trained for **future coverage**, plus
in-cache Memory Consolidation (EMA + L1/L2 demotion via SWTP).

---

## Methods (CLI)

| Name | Role | Flag |
|------|------|------|
| Sliding Window | FIFO local attention baseline | `--memory_policy window` |
| Heuristic CR | Motion-score archive ranking (ablation) | `--memory_policy heuristic_cr` |
| Learned CR | 5-D ChunkSelector (`selector_all4.pt`) | `--memory_policy learned_cr` |
| **World-State CR (v3)** | **Default**: future-use selector + consolidation + SWTP + TICH | `--memory_policy world_state_cr` |
| World-State CR v2 | Selector only (former default) | `--memory_policy world_state_cr_v2` |
| World-State CR v1 | Attention-mass ablation (`selector_ws_v1.pt`) | `--memory_policy world_state_cr_v1` |
| SWTP-only | Token pruning without CR | `--enable_swtp` (no CR policy) |
| **TICH** | Timestep-invariant condition hoisting (exact compute-axis; on by default in WS-CR v3) | `--enable_cond_hoist` / `--disable_cond_hoist` |

All CR variants are **post-generation** retention: generate a chunk → score
history → keep a budgeted archive.

---

## Repository layout

```
.
├── README.md                 ← you are here
├── LICENSE                   ← Apache-2.0
├── CONTRIBUTING.md
├── env.sh                    ← paths: LINGBOT_WORLD, CKPT_DIR, WORLDKV_ROOT, …
├── docs/
│   ├── method/               ← CHUNK_RETENTION / SWTP / MEMORY_CONSOLIDATION / TICH
│   └── experiments/          ← RealCam-Vid protocol & notes
├── paper/                    ← ICLR paper LaTeX
├── lingbot-world/            ← ★ sole source (edit here)
│   ├── generate_fast.py      ← CLI entry
│   ├── train_selector.py     ← ChunkSelector training
│   ├── wan/                  ← CR / SWTP / consolidation runtime
│   ├── assets/selectors/     ← small .pt checkpoints (tracked)
│   ├── examples/             ← demo clips 00–05
│   ├── scripts/              ← run_*.sh
│   ├── bench/                ← RealCam-Vid + WorldKV baseline
│   └── tests/
├── weights/                  ← symlink stubs only (see weights/README.md)
├── third_party/              ← clone WorldKV here (not vendored)
└── artifacts/                ← local experiment dumps (gitignored)
```

**Not in git (and should stay local):** model weights, generated videos,
`lingbot-world/output/`, RealCam-Vid raw clips, `third_party/WorldKV` checkout.

---

## Requirements

- Linux + NVIDIA GPU(s); multi-GPU recommended (scripts default to 8×)
- Python **3.10–3.11**, CUDA-compatible PyTorch **≥ 2.4**
- LingBot-World-Fast checkpoint ([HuggingFace `robbyant/lingbot-world-base-cam`](https://huggingface.co/collections/robbyant/lingbot-world))
- Optional: [WorldKV](https://github.com/cvlab-kaist/WorldKV) for the retrieval baseline

---

## Quick start

### 1. Clone & environment

```bash
git clone <this-repo-url>
cd <this-repo>
source env.sh                    # sets LINGBOT_WORLD, CKPT_DIR, …

conda create -n lingbot python=3.11 -y
conda activate lingbot
pip install -r lingbot-world/requirements.txt
# Install flash-attn / sageattention matching your CUDA if needed.
```

### 2. Place weights

```bash
# Option A: export path
export CKPT_DIR=/path/to/lingbot-world-base-cam

# Option B: follow weights/README.md and symlink under weights/
```

Checkpoint path for WorldKV must contain the substring `cam`
(control-type detection). Prefer `lingbot-world-base-cam`.

### 3. Smoke run (example01, World-State CR)

```bash
source env.sh
cd "$LINGBOT_WORLD"
./scripts/run_world_state_cr.sh
# outputs under lingbot-world/output/example01/videos/
```

Root convenience symlinks (same scripts):

```bash
./run_window.sh
./run_heuristic_cr.sh
./run_learned_cr.sh
./run_world_state_cr.sh
./run_swtp.sh
```

### 4. Direct CLI

```bash
cd lingbot-world
torchrun --nproc_per_node=8 generate_fast.py \
  --task i2v-A14B --size 480*832 \
  --ckpt_dir "$CKPT_DIR" \
  --image examples/01/image.jpg \
  --action_path examples/01 \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --frame_num 481 --convert_model_dtype \
  --sink_size 1 --max_attention_size 47000 \
  --memory_policy world_state_cr \
  --ma_kv_recent_window 1 --ma_kv_keep_ratio 0.5 \
  --ma_kv_min_keep_chunks 2 \
  --ma_kv_latent_rescue --ma_kv_latent_rescue_thr 0.08 \
  --save_dir output/example01/videos \
  --save_file output/example01/videos/world_state_cr.mp4 \
  --prompt "..."
```

---

## RealCam-Vid evaluation

See [`docs/experiments/REALCAMVID.md`](docs/experiments/REALCAMVID.md) and
`lingbot-world/bench/realcamvid/`. Typical multi-GPU launch:

```bash
cd lingbot-world
N=4 U=2 METHODS=window,world_state_cr \
  OUT=output/realcamvid_run CLIPS=bench/realcamvid/clips_default_loop \
  bash bench/run_dp2.sh --frame_num 481
```

---

## Tests

```bash
source env.sh
cd "$LINGBOT_WORLD"
python -m unittest discover -s tests -v
```

---

## Documentation map

| Topic | Path |
|-------|------|
| Chunk Retention (CR variants) | [`docs/method/CHUNK_RETENTION.md`](docs/method/CHUNK_RETENTION.md) |
| SWTP | [`docs/method/SWTP.md`](docs/method/SWTP.md) |
| Memory Consolidation | [`docs/method/MEMORY_CONSOLIDATION.md`](docs/method/MEMORY_CONSOLIDATION.md) |
| TICH (condition hoisting) | [`docs/method/TICH_Condition_Hoisting.md`](docs/method/TICH_Condition_Hoisting.md) |
| RealCam-Vid protocol | [`docs/experiments/REALCAMVID.md`](docs/experiments/REALCAMVID.md) |
| Experiment index | [`docs/experiments/INDEX.md`](docs/experiments/INDEX.md) |
| Weights layout | [`weights/README.md`](weights/README.md) |
| WorldKV third-party | [`third_party/README.md`](third_party/README.md) |
| Source package README | [`lingbot-world/README.md`](lingbot-world/README.md) |
| Paper | [`paper/iclr2027_conference.tex`](paper/iclr2027_conference.tex) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

---

## Collaboration notes

1. **Edit only `lingbot-world/`** (+ `docs/`, `paper/` as needed).
2. Never commit weights, videos, or `output/` / `artifacts/`.
3. Keep PRs small; run unit tests before review.
4. Prefer discussing API changes (`--memory_policy` names, selector defaults)
   before merging.

---

## Upstream & citation

Built on [LingBot-World](https://github.com/robbyant/lingbot-world)
(Robbyant Team). WorldKV baseline: [Yi et al., WorldKV](https://arxiv.org/abs/2605.22718).

```bibtex
@misc{worldstatecr2026,
  title  = {World-State Chunk Retention for Long-Horizon Video World Models},
  author = {Anonymous},
  year   = {2026},
  note   = {ICLR submission; code in this repository}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
