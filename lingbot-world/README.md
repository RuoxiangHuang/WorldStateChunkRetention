# lingbot-world (sole source tree)

Inference runtime for **World-State Chunk Retention** on LingBot-World-Fast.
Edit code **here only**.

Parent overview: [`../README.md`](../README.md).  
Paper LaTeX: [`../paper/`](../paper/).

## Methods

| CLI `--memory_policy` | Meaning |
|-----------------------|---------|
| `window` | Sliding-window baseline |
| `heuristic_cr` | Motion-score CR |
| `learned_cr` | 5-D ChunkSelector (`assets/selectors/selector_all4.pt`) |
| `world_state_cr` | **Default v3**: future-use selector + consolidation + SWTP + TICH |
| `world_state_cr_v2` | Selector only (former default) |
| `world_state_cr_v1` | Attention-mass ablation (`selector_ws_v1.pt`) |
| `world_state_cr_future` / `_v3` / `_consol` | Aliases of `world_state_cr` |

Orthogonal compute-axis: **TICH** (on by default with `world_state_cr`; `--disable_cond_hoist` to opt out).

## Entry points

| Script | Method |
|--------|--------|
| `scripts/run_window.sh` | Sliding window |
| `scripts/run_heuristic_cr.sh` | Heuristic CR |
| `scripts/run_learned_cr.sh` | Learned CR |
| `scripts/run_world_state_cr.sh` | World-State CR v3 (default) |
| `scripts/run_swtp.sh` | SWTP only |
| `scripts/run_cond_hoist.sh` | TICH (exact condition hoisting) |

Batch: `bench/batch_generate.py`  
Eval: `bench/eval_worldkv_memory.py` / `bench/realcamvid/run_worldkv_eval.sh`  
WorldKV baseline: `bench/worldkv/` + `bench/realcamvid/run_worldkv_official.sh`

## Layout

```
generate_fast.py          CLI
train_selector.py         ChunkSelector training
wan/                      model + CR / SWTP / consolidation + TICH
assets/selectors/         tracked .pt checkpoints
examples/                 demo clips
scripts/                  run_*.sh
bench/                    RealCam-Vid + WorldKV
tests/
```
