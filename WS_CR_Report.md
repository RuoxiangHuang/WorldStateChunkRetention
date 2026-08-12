# World-State Chunk Retention: Method Report

**Status.** Default inference policy `world_state_cr` = **v3** (`ws_v3_a05_g64`).  
**Scope.** LingBot-World-Fast long-horizon video generation with bounded KV memory.  
**Primary benchmark.** RealCam-Vid WorldKV-style revisit protocol (`default_loop`, 24 clips; supporting runs on `default_all`, 64 clips).  
**Date.** 2026-08.

---

## 1. Motivation

Autoregressive world models generate video in latent **chunks**. The only long-range memory available at inference is the transformer’s historical key/value (KV) cache. A naive sliding window discards early context and degrades geometry under camera revisit; retaining all history is computationally infeasible. **Chunk Retention (CR)** therefore selects which past chunks remain in the active attention context after each new chunk is produced.

**World-State Chunk Retention (WS-CR)** specializes CR for camera-driven world simulation: archive utility is estimated from *world-state* features and future-use supervision, then managed over time via consolidation and intra-chunk token compression. The current default package is **WS-CR v3**.

---

## 2. Core Components

WS-CR is a three-layer memory stack. Layers are complementary: the first sparsifies **which chunks** stay; the second decides **how long and in what form** they persist; the third sparsifies **tokens inside** retained archive chunks.

```text
new chunk
   │
   ▼
┌──────────────────────────────────────────────┐
│  (A) Three-tier Chunk Retention              │
│      Sink · Recent · Archive (top-K by score)│
└─────────────────────┬────────────────────────┘
                      │ archive candidates
                      ▼
┌──────────────────────────────────────────────┐
│  (B) Memory Consolidation (C1 + C2)          │
│      EMA utility · rank mix α·s+(1-α)·u      │
│      L0 → L1(SWTP) → L2(gist) → L3(drop)     │
└─────────────────────┬────────────────────────┘
                      │ L1 / L2 compression
                      ▼
┌──────────────────────────────────────────────┐
│  (C) SWTP — Saliency-Weighted Token Pruning  │
│      energy-cover keep + spatial-cell gist   │
└──────────────────────────────────────────────┘
```

### 2.1 Three-tier layout (shared CR substrate)

| Tier | Role |
|------|------|
| **Sink** | First `sink_size` chunks; permanent scene anchors |
| **Recent** | Last `recent_window` chunks; dense local coherence (FIFO) |
| **Archive** | Older candidates ranked by a utility score; keep top-K under a token budget |

Default tier ratio **1:1:2** (`sink=1`, `recent=1`, `archive_min=2`), with `ma_kv_keep_ratio=0.5`, selected on a RealCam-Vid ratio sweep. Ranking is **post-generation**: a chunk is scored after it is written, then retained or demoted for the remainder of the rollout.

Related CR variants (ablations / baselines):

| Policy | Archive utility |
|--------|-----------------|
| `window` | No CR; sliding local attention |
| `heuristic_cr` | Camera / latent motion peaks |
| `learned_cr` | 5-D MLP (`selector_all4.pt`) |
| `world_state_cr` (**default v3**) | Future-use world-state selector + consolidation + SWTP |
| `world_state_cr_v2` | Selector only (former default) |
| `world_state_cr_v1` | Attention-mass selector |

### 2.2 World-state selector (v2 backbone of v3)

Archive ranking starts from a learned **ChunkSelector** trained against a **Future Coverage Oracle**: labels emphasize chunks that will be useful under subsequent pose revisits rather than instantaneous attention mass alone.

- Checkpoint: `assets/selectors/selector_ws_future_v1.pt`
- Feature schema: `world_state.v2`
- Instantaneous selector score denoted \(s\)

This selector alone defines **WS-CR v2**. Version **v3** keeps the same selector and adds consolidation + SWTP.

### 2.3 Memory Consolidation

Consolidation manages the *lifecycle* of in-cache archive segments. It does **not** maintain an external retrieval bank (contrast: WorldKV pose-top-\(k\) store).

| ID | Mechanism | Role |
|----|-----------|------|
| **C1** | EMA utility \(u \leftarrow \beta u + (1-\beta)s\) with hysteresis | Scores accumulate temporal evidence; stabilized chunks receive a small keep bonus |
| **C2** | Tiered demotion L0→L1→L2→L3 | Compress or drop low-utility archive instead of binary discard |
| Probe | Revisit coverage | Diagnostic: fraction of pose revisits whose target chunk remains cached |
| **C3** | Cross-chunk merge | **Deferred** (RoPE / index semantics) |

**Ranking under consolidation.**

\[
\mathrm{rank} = \alpha\, s + (1-\alpha)\, u_{\mathrm{ema}}
\]

**Default knobs (`ws_v3_a05_g64`).**

| Knob | Default | Meaning |
|------|---------|---------|
| `consol_rank_alpha` \(\alpha\) | **0.5** | Mix instantaneous score and EMA |
| `consol_beta` \(\beta\) | **0.7** | EMA inertia |
| `consol_gist_tokens` | **64** | L2 gist length per demoted chunk |
| `consol_l2_bottom_ratio` | **0.5** | Bottom fraction of kept archive → L2 |
| `consol_gist_budget` | 512 | Soft cap on total L2 tokens |
| `consol_patience` | 2 | Stabilization / demotion patience |

### 2.4 SWTP (intra-chunk compression)

**SWTP** sparsifies the token axis *inside* archive chunks and is orthogonal to CR’s chunk-axis selection. Applied lazily when a chunk is promoted from Recent → Archive (sink never pruned; recent stays dense).

Default settings used by WS-CR v3:

- `swtp_keep_ratio=0.5`, `swtp_energy_cover=0.9`
- `swtp_num_summary=64` (spatial-cell summaries with key-norm compensation)
- `archive_diversity_pool=4`

L1 demotion uses standard SWTP; L2 uses a more aggressive **gist** mode with `consol_gist_tokens` summaries.

---

## 3. Default Method Specification (WS-CR v3)

**CLI.**

```bash
python generate_fast.py ... --memory_policy world_state_cr
# aliases: world_state_cr_v3 | world_state_cr_consol | world_state_cr_future | ws_v3_a05_g64
```

**Equivalent package.**

1. Motion-adaptive three-tier CR with world-state future-use selector  
2. Consolidation `full` with \(\alpha=0.5\), \(\beta=0.7\), gist \(=64\), L2 bottom ratio \(=0.5\)  
3. SWTP enabled for L1/L2 archive compression  

**Implementation map.**

| Module | Path |
|--------|------|
| Policy wiring | `lingbot-world/generate_fast.py` (`_apply_memory_policy`) |
| Eviction + consol | `lingbot-world/wan/image2video_fast.py` |
| Consolidation | `lingbot-world/wan/utils/memory_consolidation.py` |
| SWTP | `lingbot-world/wan/utils/swtp.py` |
| Batch methods | `lingbot-world/bench/batch_generate.py` |

**Peak-memory protocol.** Multi-method runs in one process leave FSDP teardown residue (~15–20 GB) that falsely inflates subsequent `peak_memory_allocated_gb`. Fair comparisons launch **one method per process** (`bench/run_dp2.sh`).

---

## 4. Benchmark and Metrics

This section specifies the evaluation substrate used throughout §5. The design goal is to measure **world memory under camera revisit**—not generic short-clip video quality in isolation.

### 4.1 Dataset: RealCam-Vid

[RealCam-Vid](docs/experiments/REALCAMVID.md) is an external camera-controllable video / world-model test corpus. Each clip is converted into the LingBot inference layout:

```text
clip_id/
  image.jpg    # conditioning first frame
  poses.npy    # camera-to-world (c2w) trajectory
  prompt.txt   # text prompt (caption)
```

Official default test batteries used in this repository (`lingbot-world/bench/realcamvid/subsets/`):

| Subset | Size | Role |
|--------|------|------|
| **`default_loop`** | 24 | Loop / revisit trajectories; **primary** WS-CR battery |
| **`default_random`** | 40 | Seeded forward roaming; holdout / generalization sanity |
| **`default_all`** | 64 | `loop ∪ random`; recommended full paper battery |

Symlink materialization: `bash lingbot-world/bench/realcamvid/build_subsets.sh`  
→ `clips_default_loop/`, `clips_default_random/`, `clips_default_all/`.

**Generation settings used for reported WS-CR tables.**

| Setting | Value | Notes |
|---------|-------|-------|
| Horizon | `frame_num=481` | ≈30 s / ~40 latent chunks; CR gains are clearer than short clips |
| Resolution | `480×832` (typical) | Matches Fast A14B I2V config |
| Seed | `base_seed=42` | Fixed unless stated |
| Parallelism | `N=4` DP groups × `U=2` Ulysses | Fair peak protocol via `run_dp2.sh` |

Sources / provenance notes: `docs/experiments/REALCAMVID.md`.

### 4.2 Quality protocol: WorldKV-style revisit

Default quality evaluation follows the **WorldKV world-memory protocol** (Yi et al., [arXiv:2605.22718](https://arxiv.org/abs/2605.22718)), implemented in `lingbot-world/bench/eval_worldkv_memory.py`.

**Pose pairing (method-independent).** For each clip with GT poses \(\{P_t\}\):

1. For every frame \(t\), search earlier frames \(s \le t - \texttt{min\_gap}\) whose SE(3) distance to \(P_t\) is \(\le \texttt{radius}\).  
2. Among matches, take the closest pose; ties prefer the earliest visit (true first-visit).  
3. Emit pair \((s, t)\). Optionally subsample to at most `max_pairs_per_clip` pairs via uniform index linspace so all methods share the **same** pairs on a clip.

**Default pairing hyperparameters.**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `radius` | 0.15 | SE(3) acceptance radius (translation scaled + geodesic rotation) |
| `min_gap` | 30 | Minimum temporal separation (frames) between first-visit and revisit |
| `max_pairs_per_clip` | 64 | Cap on evaluated pairs per clip |

**Pairwise fidelity** (averaged over selected revisit pairs):

| Metric | Direction | Definition (this codebase) |
|--------|-----------|------------------------------|
| **PSNR** | ↑ | Pixel MSE on uint8 RGB; \(10\log_{10}(255^2 / \mathrm{MSE})\) |
| **SSIM** | ↑ | `skimage` structural similarity, channel-wise RGB |
| **LPIPS** | ↓ | AlexNet LPIPS on \([-1,1]\) normalized frames |

**Distributional consistency:**

| Metric | Direction | Definition |
|--------|-----------|------------|
| **FID** | ↓ | Fréchet distance between Inception-v3 features of the *set of revisit frames* and the *set of first-visit frames* (set-level; not paired) |

FID is reported as a single set statistic per method; optional clip-level bootstrap CIs live in `bench/fid_bootstrap.py`. Paired significance for PSNR/SSIM/LPIPS can be computed with `bench/paired_significance.py`.

**Invocation.**

```bash
REALCAMVID_OUT=<out_dir> REALCAMVID_SUBSET=default_loop \
  METHODS=window,world_state_cr \
  bash bench/realcamvid/run_worldkv_eval.sh
# → <out_dir>/worldkv_memory_eval_default_loop.json
```

### 4.3 Efficiency and memory diagnostics

Reported alongside quality (from per-clip `stats/*.json` written at generation time):

| Metric | Direction | Meaning |
|--------|-----------|---------|
| **Throughput (FPS)** | ↑ | Wall-clock frames / total generation time (WorldKV eval aggregate) |
| **avg attention context tokens** (`ctx`) | ↓* | Mean active KV tokens attending at generation steps |
| **peak memory (GB)** | ↓* | `peak_memory_allocated_gb` under one-method-per-process launches |
| **avg chunk time (s)** | ↓ | Per-chunk wall time |
| **revisit coverage** | ↑ | Diagnostic: fraction of pose revisits whose target chunk is still in cache |
| **tier_counts (L1/L2/L3)** | — | Consolidation demotion occupancy (sanity for C2) |

\*Lower is better only under a quality constraint: WS-CR aims to **cut ctx / peak without losing revisit fidelity**.

**Fairness caveat.** Peak memory must not be compared across methods packed into a single FSDP process; use `bench/run_dp2.sh` (one method per wave).

### 4.4 What we optimize vs. what we monitor

| Priority | Metrics | Use in decision making |
|----------|---------|------------------------|
| Primary | Revisit **PSNR** (then SSIM / LPIPS) | Promote / reject default candidates |
| Secondary | **FID**, FPS, ctx, peak GB | Break ties; reject gains that inflate FID or memory |
| Diagnostic | revisit coverage, L2 tier counts | Debug consolidation / SWTP; not paper headline numbers |

**Current practice for WS-CR defaults.** Sweep and select on **`default_loop`**; require a clear multi-metric win (e.g. ΔPSNR ≳ 0.05 without >5% peak/ctx regression and without FID collapse) before changing `world_state_cr`. Holdout on `default_random` / `default_all` is recommended before paper-facing claims (§6).

---

## 5. Experiments

Unless noted, runs follow §4: RealCam-Vid **`default_loop`** (24 clips), `frame_num=481`, WorldKV revisit protocol, and one-method-per-process launches.

### 5.1 Main comparison: WSCRv3 vs baselines

Protocol: RealCam-Vid `default_loop` (24 clips), WorldKV revisit eval.  
**WSCRv3** = current default `ws_v3_a05_g64`. **WorldKV** / **window** from the same-subset baseline run; **Context** = mean attention context tokens; **Peak GB** = mean `peak_memory_allocated_gb` (one-method-per-process).

| Method | PSNR | SSIM | LPIPS | FID | FPS | Context | Peak GB |
|--------|------|------|-------|-----|-----|---------|---------|
| **WSCRv3** | **10.107** | **0.346** | **0.713** | **60.28** | **1.690** | **16096** | **39.84** |
| WorldKV | 10.008 | 0.326 | 0.715 | 62.85 | 1.115 | 39000 | 60.57 |
| window | 9.830 | 0.300 | 0.726 | 66.98 | 1.178 | 41471 | 54.67 |
| Heuristic CR (early MoCE) | —† | — | — | — | — | — | — |

Sources: WSCRv3 — `output/realcamvid_consol_sweep_default_loop/`; window / WorldKV — `output/realcamvid_ws_v3_default_loop/`.

**Naming note.** “WSCRv1” in casual discussion usually means the **first MoCE stage = Heuristic CR** (`heuristic_cr`, motion-score archive ranking)—not the code flag `world_state_cr_v1` (attention-mass World-State selector).

**† Why the Heuristic row is blank here.** Heuristic / MoCE **was** evaluated historically, but under a **different protocol** that cannot be merged into this WorldKV table:

| Item | Historical MoCE-vs-Window | This table |
|------|---------------------------|------------|
| Method | MoCE = Heuristic CR | WSCRv3 / WorldKV / window |
| Clips | 10 RealEstate10K clips | `default_loop` (24) |
| Horizon | 273 frames (~17 s) | 481 frames |
| Quality | `eval_semantics` (LPIPS-to-seed, DINO, CLIP) | WorldKV revisit PSNR/SSIM/LPIPS/FID |
| Headline result | time **−5.6%**, peak mem **−18.7%**, ctx **−17.9%**; quality mixed | see numbers above |

Artifacts: `artifacts/archive_pre_paper/docs/experiments/realcamvid/REPORT_10clip.md`.  
Filling the Heuristic row requires a fresh `heuristic_cr` generate+eval on `default_loop` under `eval_worldkv_memory.py`.

For reference, selector-only **WSCRv2** on the same Phase-1 WorldKV sweep: PSNR 9.854 / SSIM 0.324 / LPIPS 0.720 / FID 61.24 / FPS 1.657 / Context 21489.

### 5.2 Baseline comparison (context)

On `default_loop`, sliding-window attention vs. an earlier WS-CR checkpoint (selector-era `world_state_cr`):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ | Notes |
|--------|--------|--------|---------|-------|-------|
| `window` | 9.830 | 0.300 | 0.726 | 1.18 | Dense local window; high peak mem (~55 GB) |
| WS-CR (pre-v3 package) | 10.026 | 0.327 | 0.719 | 1.47 | Selector CR without full consol stack |
| WorldKV (official bank) | 10.008 | 0.326 | 0.715 | 1.12 | External pose bank; higher peak |

On the 64-clip `default_all` battery, WS-CR reduced mean context (~19 k vs ~30 k for window) and peak memory (~38 GB vs ~53 GB) while improving revisit quality relative to window.

### 5.3 Phase-1 consolidation sweep (α × gist)

Grid over rank mix \(\alpha\) and L2 gist size (fixed L2 bottom ratio 0.5, \(\beta=0.7\)), plus frozen v2:

| Method | \(\alpha\) | gist | PSNR | SSIM | LPIPS | FID | FPS | ctx | peak GB |
|--------|------------|------|------|------|-------|-----|-----|-----|---------|
| `world_state_cr_v2` | — | — | 9.854 | 0.324 | 0.720 | 61.24 | 1.657 | 21 489 | 39.85 |
| `ws_v3_a0` | 0.0 | 96 | 10.013 | 0.337 | 0.713 | 63.14 | 1.690 | 15 998 | 39.84 |
| `ws_v3_a05` | 0.5 | 96 | 9.988 | 0.341 | 0.720 | 60.30 | 1.687 | 16 049 | 39.83 |
| `ws_v3_a1` | 1.0 | 96 | 10.033 | 0.343 | 0.718 | 60.39 | 1.679 | 16 027 | 39.83 |
| **`ws_v3_a05_g64`** | **0.5** | **64** | **10.107** | **0.346** | **0.713** | **60.28** | **1.690** | **16 096** | **39.84** |
| `ws_v3_a05_g128` | 0.5 | 128 | 9.936 | 0.336 | 0.717 | 59.76 | 1.674 | 15 879 | 39.83 |

**Finding.** v3 with consolidation improves revisit PSNR over v2 (+0.25) while cutting context (~21 k → ~16 k). Among the grid, **\(\alpha=0.5\), gist\(=64\)** is the Pareto choice and was promoted to the repository default.

### 5.4 Phase-2 refinement (L2 ratio, fine gist, \(\beta\))

Holding the Phase-1 winner fixed and sweeping previously untouched knobs:

| Method | Change vs default | PSNR | SSIM | LPIPS | FID | FPS |
|--------|-------------------|------|------|-------|-----|-----|
| **`ws_v3_a05_g64`** | *(default)* | **10.107** | **0.346** | **0.713** | **60.28** | 1.690 |
| `ws_v3_b09` | \(\beta=0.9\) | 10.164 | 0.344 | 0.712 | 64.69 | 1.695 |
| `ws_v3_b05` | \(\beta=0.5\) | 10.036 | 0.340 | 0.721 | 60.97 | 1.693 |
| `ws_v3_g80` | gist=80 | 10.074 | 0.337 | 0.719 | 62.52 | 1.685 |
| `ws_v3_g32` / `g48` | gist∈{32,48} | ≤9.981 | 0.337 | ≥0.717 | ≥62.11 | ~1.70 |
| `ws_v3_l25` / `l75` | L2 ratio ∈{0.25,0.75} | 10.107 | identical to default | | | |

**Findings.**

1. Finer gist sizes confirm a local optimum near **64**; smaller gists hurt PSNR/SSIM.  
2. \(\beta=0.9\) yields a marginal PSNR gain (+0.057) but **degrades FID** (60.28 → 64.69); under a joint quality rule it was **not** adopted.  
3. Varying `consol_l2_bottom_ratio` produced **identical** metrics to the anchor, indicating L2 demotion intensity is currently under-effective (implementation / trigger investigation outstanding).

**Default decision.** Retain **`ws_v3_a05_g64`** as `world_state_cr`.

### 5.5 Efficiency summary (default vs window / v2)

| System | Approx. ctx tokens | Peak GB | Revisit PSNR (`default_loop`) |
|--------|--------------------|---------|--------------------------------|
| `window` | ~41 k | ~55 | 9.830 |
| WS-CR v2 | ~21 k | ~40 | 9.854 |
| **WS-CR v3 (`a05_g64`)** | **~16 k** | **~40** | **10.107** |

---

## 6. Todo / Open Research Agenda

Items are ordered by expected scientific impact.

### 6.1 Near term (engineering → paper reliability)

| ID | Item | Rationale |
|----|------|-----------|
| T1 | **Diagnose inert L2 ratio** | Phase-2 `l25`/`l75` matched the default exactly; verify `consol_l2_bottom_ratio` is applied in `assign_archive_tiers` and reflected in `tier_counts` |
| T2 | **Holdout validation** | Re-evaluate default vs `ws_v3_b09` on `default_random` / `default_all` before any further default change |
| T3 | **Fair peak protocol in all tables** | Enforce one-method-per-process in every reported ablation; document residue artifact |
| T4 | **Seed / schedule stability** | Repeat key cells under alternate seeds or Fast-sampling timestep indices |

### 6.2 Method (paper-facing)

| ID | Item | Rationale |
|----|------|-----------|
| T5 | **C3: cross-chunk merge** | Deferred consolidation step; requires careful RoPE / position-index handling |
| T6 | **Adaptive gist budget** | Condition `consol_gist_tokens` / L2 ratio on revisit density or remaining token budget |
| T7 | **Selector–consolidation co-design** | Train or calibrate the future-use selector with consolidation-induced rank targets, not instantaneous keep labels alone |
| T8 | **Diversity-aware archive** | Strengthen `archive_diversity_pool` / pose coverage objectives to reduce redundant near-duplicates |

### 6.3 Evaluation & reporting

| ID | Item | Rationale |
|----|------|-----------|
| T9 | **Unified paper table** | Window / heuristic / learned / WS-CR v1–v3 / WorldKV on identical `default_all` protocol |
| T10 | **Qualitative revisit panels** | Side-by-side frames at matched revisit pairs for v2 vs v3 vs window |
| T11 | **Ablation completeness** | Report SWTP-only, EMA-only (`world_state_cr_ema`), and full v3 as a nested stack |

### 6.4 Explicitly out of scope (for now)

- Retraining the 14B DiT backbone under CR objectives  
- Replacing the dynamic KV path with a learned external memory controller  
- Production serving / quantization stacks unrelated to memory policy

---

## 7. Reproduction Cheatsheet

```bash
# Default WS-CR v3 generation (batch, fair peaks)
cd lingbot-world
N=4 U=2 METHODS=window,world_state_cr \
  OUT=output/my_run CLIPS=bench/realcamvid/clips_default_loop \
  bash bench/run_dp2.sh --frame_num 481 --base_seed 42

# WorldKV-style eval
REALCAMVID_OUT=output/my_run REALCAMVID_SUBSET=default_loop \
  METHODS=window,world_state_cr \
  bash bench/realcamvid/run_worldkv_eval.sh

# Consolidation sweep summaries
python bench/summarize_consol_sweep.py output/realcamvid_consol_sweep_default_loop
python bench/summarize_consol_sweep.py output/realcamvid_consol_sweep2_default_loop
```

**Artifacts.**

- Phase-1: `lingbot-world/output/realcamvid_consol_sweep_default_loop/`  
- Phase-2: `lingbot-world/output/realcamvid_consol_sweep2_default_loop/`  
- Method notes: `docs/method/CHUNK_RETENTION.md`, `MEMORY_CONSOLIDATION.md`, `SWTP.md`

---

## 8. Takeaways

1. **WS-CR** couples world-state future-use selection with temporal consolidation and SWTP token compression to bound KV growth under camera revisit.  
2. The shipping default is **v3 = `ws_v3_a05_g64`**: \(\alpha=0.5\), gist \(=64\), \(\beta=0.7\), L2 bottom ratio \(=0.5\).  
3. On RealCam-Vid `default_loop`, v3 improves revisit PSNR over window and over selector-only v2 while reducing attention context by roughly **2.5×** vs window.  
4. Further \(\beta\) tuning can trade PSNR for FID; the present default prefers the more balanced Phase-1 winner pending holdout confirmation and L2-trigger fixes.
