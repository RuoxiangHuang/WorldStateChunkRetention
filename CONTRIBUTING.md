# Contributing

Thanks for collaborating on **World-State Chunk Retention**.

## Source of truth

| Path | Role |
|------|------|
| **`lingbot-world/`** | **Only** editable source tree (runtime, bench, tests, selectors) |
| `docs/` | Method + experiment docs |
| `paper/` | Paper LaTeX |

## Setup (dev)

```bash
git clone <this-repo>
cd <this-repo>
source env.sh

# Python env (example)
conda create -n lingbot python=3.11 -y
conda activate lingbot
pip install -r lingbot-world/requirements.txt
# flash-attn / sageattention: install per your CUDA stack

# Place LingBot-World-Fast checkpoint and point CKPT_DIR
export CKPT_DIR=/path/to/lingbot-world-base-cam
```

Weights layout: [`weights/README.md`](weights/README.md).

## Branch & PR hygiene

1. Create a feature branch from `main` (or the agreed default branch).
2. Keep PRs focused: one method change / one bench change / one docs change.
3. Do **not** commit:
   - model weights, videos, `output/`, `artifacts/`
   - RealCam-Vid raw clips or large binaries
4. Run tests before opening a PR:
   ```bash
   cd lingbot-world && python -m unittest discover -s tests -v
   ```

## Naming

- Prefer `--memory_policy world_state_cr` (default v3) over raw flag piles.
- Ablations: `world_state_cr_v2` (selector only), `world_state_cr_v1` (attn-mass).
- Paper sources live under `paper/` (not a method name).
