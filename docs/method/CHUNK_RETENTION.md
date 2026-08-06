# Chunk Retention — Three CR Variants

LingBot-World-Fast generates video in latent chunks. Historical K/V is the only
long-range memory. **Chunk Retention** decides which past chunks stay in the
active attention context after each new chunk is produced.

## Shared mechanism

All CR variants use the same three-tier layout:

| Tier | Rule |
|------|------|
| Sink | First `sink_size` chunks, always kept |
| Recent | Last `recent_window` chunks, FIFO |
| Archive | Older candidates ranked by a utility score; keep top-K under token budget |

Default tier ratio (**1:1:2**): `sink_size=1`, `ma_kv_recent_window=1`, `ma_kv_min_keep_chunks=2`
(from RealCam-Vid ratio sweep; `ma_kv_keep_ratio=0.5`).

Ranking happens **after** the chunk is generated (post-generation retention).
Dropped archive chunks are gone for the rest of the rollout.

## Heuristic CR (`heuristic_cr`)

Archive score = camera/action motion peak-mean, fused with latent residual
rescue (`max(camera_motion, latent_motion)`). No learned weights.

## Learned CR (`learned_cr`)

5-D `ChunkSelector` MLP (`selector_all4.pt`, schema `learned.v0`):

`motion_norm, age_norm, cam_angular, kcent_affinity, value_norm`

Frozen baseline for ablations against World-State CR.

## World-State CR (`world_state_cr`, default = former `world_state_cr_future`)

Default learned retention policy. Same Sink–Recent–Resident-Archive skeleton,
with:

- **P0 labels:** Future Coverage Oracle (`future_use_v1`) — discounted mix of
  future attention mass and pose/frustum reuse over horizon `H` (default 8).
- **P1 features:** schema `world_state.v2` with corrected
  `reachability` / `relative_motion` / `time_since_last_observed` formulas.

Checkpoint: `selector_ws_future_v1.pt`.  
`world_state_cr_future` remains a **back-compat alias** of `world_state_cr`.

```bash
python train_selector.py --label_type future_use_v1 --feature_schema v2 \
  --oracle path/to/oracle_dense_*.pt \
  --out assets/selectors/selector_ws_future_v1.pt

python generate_fast.py ... --memory_policy world_state_cr
```

## World-State CR v1 ablation (`world_state_cr_v1`)

Frozen attention-mass selector (`selector_ws_v1.pt`, schema `world_state.v1`)
with SE(3) geometry / frustum features. Kept only for ablations against the
default future-use World-State CR.

## CLI

```bash
python generate_fast.py ... --memory_policy heuristic_cr
python generate_fast.py ... --memory_policy learned_cr
python generate_fast.py ... --memory_policy world_state_cr
python generate_fast.py ... --memory_policy world_state_cr_v1
```

Baseline without CR: `--memory_policy window` (sliding local attention).
