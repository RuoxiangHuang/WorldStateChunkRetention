# SWTP — Saliency-Weighted Token Pruning

**SWTP** sparsifies the **token dimension inside archive chunks**. It is
orthogonal to Chunk Retention (which sparsifies the chunk dimension).

## Idea

An archive chunk is a long-range memory anchor. Most of its ~4680 tokens are
redundant once sink + recent already cover the scene. SWTP keeps:

1. Top-K tokens by latent residual saliency
2. A small set of summary tokens (`swtp_num_summary`)

Applied **lazily at archive promotion** (when a chunk leaves the recent window
and enters the archive tier), so recent chunks stay dense for local coherence.

## CLI

```bash
# Standalone (no MoCE)
python generate_fast.py ... --enable_swtp \
  --swtp_keep_ratio 0.5 --swtp_num_summary 64 --swtp_min_saliency_gini 0.20

# Or use scripts/run_swtp.sh
```

Key flags: `swtp_keep_ratio`, `swtp_num_summary`, `swtp_min_saliency_gini`.
Optional `archive_diversity_pool` adds trajectory-diversity reranking among
archive candidates (used in MoSaiC).

## Implementation

Primary logic lives in `lingbot-world/wan/image2video_fast.py`
(`_compute_token_saliency`, `_apply_swtp_to_kv`, `_apply_swtp_to_archive_segments`).
