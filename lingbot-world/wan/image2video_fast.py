import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import (
    WanModelFast,
    oracle_reset, oracle_set, oracle_state,
)
from .modules.chunk_selector import (
    ChunkSelector, load_selector, build_chunk_features, FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION, features_for_schema, se3_distance,
)
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    compute_chunk_motion_score,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


class WanI2VFast:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
        enable_motion_adaptive_kv_eviction=False,
        ma_kv_recent_window=1,
        ma_kv_keep_ratio=0.5,
        ma_kv_min_keep_chunks=2,
        ma_kv_latent_rescue=False,
        ma_kv_latent_rescue_thr=0.08,
        enable_swtp=False,
        swtp_keep_ratio=0.5,
        swtp_num_summary=64,
        swtp_min_saliency_gini=0.20,
        swtp_energy_cover=0.9,
        archive_diversity_pool=0,
        selector="learned",
        selector_ckpt=None,
        collect_oracle=False,
        oracle_probe_every=8,
        oracle_out=None,
        oracle_label_type="attention_mass",
        oracle_future_horizon=8,
        oracle_future_gamma=0.9,
        oracle_future_alpha=0.5,
        # Memory Consolidation (World-State CR lifecycle; not WorldKV retrieval).
        consolidation="off",
        consol_beta=0.7,
        consol_patience=2,
        consol_stabilize_thr=0.6,
        consol_gist_tokens=64,
        consol_gist_budget=512,
        consol_rank_alpha=0.5,
        consol_l2_bottom_ratio=0.5,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.enable_motion_adaptive_kv_eviction = enable_motion_adaptive_kv_eviction
        self.ma_kv_recent_window = ma_kv_recent_window
        self.ma_kv_keep_ratio = ma_kv_keep_ratio
        self.ma_kv_min_keep_chunks = ma_kv_min_keep_chunks
        self.ma_kv_latent_rescue = ma_kv_latent_rescue
        self.ma_kv_latent_rescue_thr = ma_kv_latent_rescue_thr
        # CR-only runtime: skip per-chunk CUDA sync / hot-path empty_cache / tqdm.
        # Window baseline keeps the stricter profiling path unchanged.
        self._cr_fast_runtime = bool(enable_motion_adaptive_kv_eviction)
        self.enable_swtp = enable_swtp
        self.swtp_keep_ratio = swtp_keep_ratio
        self.swtp_num_summary = swtp_num_summary
        self.swtp_min_saliency_gini = swtp_min_saliency_gini
        self.swtp_energy_cover = float(swtp_energy_cover)
        self.archive_diversity_pool = int(archive_diversity_pool)
        from .utils.memory_consolidation import ConsolidationConfig
        self.consolidation = ConsolidationConfig(
            mode=str(consolidation or "off").lower(),
            beta=float(consol_beta),
            patience=int(consol_patience),
            stabilize_thr=float(consol_stabilize_thr),
            gist_tokens=int(consol_gist_tokens),
            gist_budget_tokens=int(consol_gist_budget),
            rank_alpha=float(consol_rank_alpha),
            l2_bottom_ratio=float(consol_l2_bottom_ratio),
            swtp_keep_ratio=float(swtp_keep_ratio),
            swtp_num_summary=int(swtp_num_summary),
            swtp_energy_cover=float(swtp_energy_cover),
        )

        # ── Chunk selector (Learned / World-State CR) ───────────────────────
        # selector='learned' => ChunkSelector MLP archive ranking.
        # selector='heuristic' => motion_score (Heuristic CR).
        self.selector = selector or "learned"
        self.selector_ckpt = selector_ckpt
        self.chunk_selector = None
        # Feature normalization refs (populated from ckpt meta; fallbacks below).
        self._sel_motion_ref = 0.5
        self._sel_vnorm_ref = 1.0
        self._sel_translation_scale = 1.0
        self._revisit_radius = 0.15
        self._sel_feature_names = list(FEATURE_NAMES)
        self._sel_schema_version = FEATURE_SCHEMA_VERSION
        if self.selector == "learned" and self.enable_motion_adaptive_kv_eviction:
            from .utils.selector_defaults import resolve_selector_ckpt
            selector_ckpt = resolve_selector_ckpt(selector_ckpt)
            self.selector_ckpt = selector_ckpt
            assert selector_ckpt is not None, "--selector learned requires --selector_ckpt"
            self.chunk_selector = load_selector(selector_ckpt, map_location="cpu").to(self.device)
            _meta = getattr(self.chunk_selector, "_meta", {}) or {}
            payload_meta = torch.load(selector_ckpt, map_location="cpu", weights_only=False).get("meta", {})
            self._sel_motion_ref = float(payload_meta.get("motion_ref", self._sel_motion_ref))
            self._sel_vnorm_ref = float(payload_meta.get("vnorm_ref", self._sel_vnorm_ref))
            self._sel_translation_scale = float(
                payload_meta.get("translation_scale", self._sel_translation_scale))
            self._sel_feature_names = list(
                getattr(self.chunk_selector, "_feature_names", FEATURE_NAMES))
            self._sel_schema_version = getattr(
                self.chunk_selector, "_schema_version", FEATURE_SCHEMA_VERSION)
            logging.info(
                f"ChunkSelector loaded ({self._sel_schema_version}, "
                f"dim={len(self._sel_feature_names)}) from {selector_ckpt}")
        if self._cr_fast_runtime and self.rank == 0:
            logging.info(
                "CR fast runtime ON: no per-chunk cuda.synchronize / "
                "no hot-path empty_cache / tqdm disabled (CR path only).")

        # Pose bookkeeping for World-State CR features (retention only).
        self._chunk_poses = {}  # chunk_id -> {translation, rotation, intrinsics, camera_forward}

        # Oracle collection (teacher pass that logs per-chunk attention mass labels).
        # Default label_type remains attention_mass (World-State CR). Optional
        # future_use_v1 post-aggregates discounted future coverage without changing
        # the CR eviction path.
        self.collect_oracle = bool(collect_oracle)
        self.oracle_probe_every = int(oracle_probe_every)
        self.oracle_out = oracle_out
        self.oracle_label_type = str(oracle_label_type or "attention_mass")
        self.oracle_future_horizon = int(oracle_future_horizon)
        self.oracle_future_gamma = float(oracle_future_gamma)
        self.oracle_future_alpha = float(oracle_future_alpha)
        # Per-chunk metadata table (chunk_id -> {motion_score, camera_forward,
        # k_centroid(layer0), value_norm}) captured during collection AND reused as
        # the live decision context at inference.
        self._chunk_meta = {}
        self._chunk_size = 3

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        # Infer control modality from checkpoint path; default to camera control
        # when the directory name has neither marker (e.g. lingbot_world_fast).
        if 'act' in checkpoint_dir and 'cam' not in checkpoint_dir:
            self.control_type = 'act'
        else:
            self.control_type = 'cam'

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModelFast from {checkpoint_dir}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_dir,
            subfolder=config.fast_noise_checkpoint,
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size)

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        if os.environ.get("WSCR_COMPILE", os.environ.get("MOSAIC_COMPILE", "0")) == "1":
            # Opt-in (CUDA opt ①): torch.compile each DiT block. The attention call graph-breaks
            # (dynamic KV size + custom flash kernel), so only the token-shape-invariant compute
            # (QKVO/FFN/modulation/cam-MLP, all on the fixed-size current chunk) is compiled.
            # Best-effort: never break generation if compile is unavailable/incompatible.
            try:
                m = getattr(self.model, "module", self.model)
                if hasattr(m, "get_base_model"):
                    m = m.get_base_model()
                blocks = getattr(m, "blocks", None)
                if blocks is not None:
                    for i in range(len(blocks)):
                        blocks[i] = torch.compile(blocks[i], dynamic=False)
                    logging.info(f"WSCR_COMPILE: torch.compile applied to {len(blocks)} DiT blocks")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"WSCR_COMPILE failed, running eager: {e}")

        self.sample_neg_prompt = config.sample_neg_prompt
        self.debug_motion_adaptive_kv = os.environ.get("LINGBOT_DEBUG_MA_KV", "0") == "1"

        if self.enable_motion_adaptive_kv_eviction and self.local_attn_size != -1:
            logging.info(
                "Motion-adaptive KV eviction is enabled; `local_attn_size` is treated as an optional hard attention cap instead of static rolling cache size."
            )

        if self.enable_swtp:
            mode = ("World-State CR + SWTP"
                    if self.enable_motion_adaptive_kv_eviction else "SWTP standalone")
            logging.info(
                f"{mode} is ENABLED. "
                f"keep_ratio={self.swtp_keep_ratio}  "
                f"num_summary={self.swtp_num_summary}  "
                f"energy_cover={self.swtp_energy_cover}  "
                f"min_gini={self.swtp_min_saliency_gini} (below → uniform lattice, not skip). "
                f"Summary tokens use spatial-cell pooling + norm compensation. "
                f"With SWTP enabled, is applied only when a chunk transitions into the archive tier; "
                f"sink and recent chunks remain uncompressed."
            )
        if self.consolidation.enabled:
            logging.info(
                f"Memory Consolidation ENABLED mode={self.consolidation.mode} "
                f"beta={self.consolidation.beta} patience={self.consolidation.patience} "
                f"rank_alpha={self.consolidation.rank_alpha} "
                f"l2_bottom_ratio={self.consolidation.l2_bottom_ratio} "
                f"gist_tokens={self.consolidation.gist_tokens} "
                f"gist_budget={self.consolidation.gist_budget_tokens}. "
                f"(In-cache lifecycle; not WorldKV bank retrieval.)"
            )

        if self.archive_diversity_pool > 0 and self.enable_motion_adaptive_kv_eviction:
            logging.info(
                f"Trajectory-diversity-aware archive selection is ENABLED. "
                f"archive_diversity_pool={self.archive_diversity_pool}. "
                f"Archive chunks are picked via Farthest-Point Sampling over the top-N motion candidates "
                f"using per-chunk camera forward direction (requires action_path with poses)."
            )

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        # PeftModel wraps the DiT; SP hooks must bind to the underlying WanModelFast.
        dit = model.get_base_model() if hasattr(model, "get_base_model") else model
        if use_sp:
            for block in dit.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            dit.forward = types.MethodType(sp_dit_forward_causal, dit)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)


    def _compute_latent_motion_score(self, current_latent, previous_latent):
        if previous_latent is None:
            return float("inf")
        return (current_latent.float() - previous_latent.float()).abs().mean().item()


    def _initialize_motion_adaptive_kv_cache(self, num_layers, device):
        self_kv_cache = []
        for layer_idx in range(num_layers):
            self_kv_cache.append({
                'mode': 'dynamic',
                'layer_idx': layer_idx,   # used by oracle probe-layer gating
                'segments': [],
                'commit_current': False,
                'current_chunk_id': -1,
                'current_motion_score': 0.0,
                'current_is_sink': False,
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })
        return self_kv_cache


    # ════════════════════════════════════════════════════════════════════
    # SWTP (Saliency-Weighted Token Pruning) helpers
    # ════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _compute_token_saliency(self, x0_current, x0_previous):
        """
        Compute per-token saliency from latent residual.

        Args:
            x0_current:  [C, F, H, W] latent at end of current chunk
            x0_previous: [C, F, H, W] same shape, OR None for first chunk

        Returns:
            (saliency, token_grid) where saliency is [F*Ht*Wt] on CPU and
            token_grid is (F, Ht, Wt). Or (None, None) if no previous chunk.
        """
        if x0_previous is None:
            return None, None
        diff = (x0_current.float() - x0_previous.float()).abs().mean(dim=0)  # [F, H, W]
        F, H, W = diff.shape
        # Spatial 2x2 pooling → token grid
        H_p, W_p = H // 2, W // 2
        diff = diff[:, :H_p * 2, :W_p * 2].view(F, H_p, 2, W_p, 2).mean(dim=(2, 4))
        return diff.flatten().cpu(), (int(F), int(H_p), int(W_p))

    @torch.no_grad()
    def _gini_coefficient(self, x):
        from .utils.swtp import gini_coefficient
        return gini_coefficient(x)

    @torch.no_grad()
    def _apply_swtp_to_kv(self, k, v, saliency, keep_ratio, num_summary,
                          token_grid=None, mode="standard"):
        from .utils.swtp import apply_swtp_to_kv
        return apply_swtp_to_kv(
            k, v, saliency,
            keep_ratio=keep_ratio,
            num_summary=num_summary,
            token_grid=token_grid,
            energy_cover=self.swtp_energy_cover,
            mode=mode,
            compensate_summary_norm=True,
        )

    @torch.no_grad()
    def _append_swtp_kv_segments(
        self,
        kv_cache,
        pending_segments,
        chunk_id,
        token_saliency,
        token_grid=None,
        is_sink=False,
    ):
        """
        Append a new chunk's K, V to the dynamic KV cache, optionally applying
        SWTP token pruning based on token_saliency.

        Sink chunks are never pruned (matches SWTP docs). Low-Gini chunks
        fall back to uniform lattice pooling instead of storing full KV forever.
        """
        if pending_segments is None:
            return

        do_swtp = (token_saliency is not None) and (not is_sink)
        mode = "standard"
        if do_swtp:
            gini = self._gini_coefficient(token_saliency)
            if gini < self.swtp_min_saliency_gini:
                mode = "uniform"

        for layer_cache, pending in zip(kv_cache, pending_segments):
            if pending is None:
                continue
            k_full = pending['k']
            v_full = pending['v']
            if do_swtp:
                k_red, v_red, kept_idx = self._apply_swtp_to_kv(
                    k_full, v_full, token_saliency,
                    self.swtp_keep_ratio, self.swtp_num_summary,
                    token_grid=token_grid, mode=mode,
                )
                payload = {
                    'chunk_id': int(chunk_id),
                    'is_sink': bool(is_sink),
                    'is_swtp': True,
                    'is_gist': False,
                    'memory_tier': 'L1',
                    'token_count': int(k_red.shape[1]),
                    'k': k_red,
                    'v': v_red,
                    'num_kept': int(kept_idx.numel()) if hasattr(kept_idx, 'numel') else 0,
                    'num_summary': int(k_red.shape[1]) - (int(kept_idx.numel()) if hasattr(kept_idx, 'numel') else 0),
                    'token_grid': token_grid,
                }
            else:
                payload = {
                    'chunk_id': int(chunk_id),
                    'is_sink': bool(is_sink),
                    'is_swtp': False,
                    'is_gist': False,
                    'memory_tier': 'L0',
                    'token_count': int(k_full.shape[1]),
                    'k': k_full,
                    'v': v_full,
                    'token_grid': token_grid,
                }
            layer_cache['segments'].append(payload)
            local_end_index = sum(s['token_count'] for s in layer_cache['segments'])
            layer_cache['local_end_index'].fill_(local_end_index)

    def _set_motion_adaptive_kv_runtime(self, kv_cache, chunk_id, motion_score, commit_current, is_sink):
        for layer_cache in kv_cache:
            layer_cache['commit_current'] = commit_current
            layer_cache['current_chunk_id'] = int(chunk_id)
            layer_cache['current_motion_score'] = float(motion_score)
            layer_cache['current_is_sink'] = bool(is_sink)


    def _enforce_attention_budget(
        self,
        sink_segments,
        archive_segments,
        recent_segments,
        attention_budget,
    ):
        """
        Drop additional non-sink chunks so total tokens fit ``attention_budget``.

        Sink is always preserved (anchor invariant). Recent chunks are admitted
        newest-first; archive chunks are then admitted by descending motion
        score until the budget is exhausted. Chronological order is restored
        on return.
        """
        sink_tokens = sum(segment['token_count'] for segment in sink_segments)
        if attention_budget <= sink_tokens:
            return list(sink_segments)
        remaining = attention_budget - sink_tokens
        admitted_recent_ids = set()
        for segment in reversed(recent_segments):
            if segment['token_count'] <= remaining:
                admitted_recent_ids.add(segment['chunk_id'])
                remaining -= segment['token_count']
            else:
                break
        admitted_archive_ids = set()
        for segment in sorted(archive_segments, key=self._archive_rank_key, reverse=True):
            if segment['token_count'] <= remaining:
                admitted_archive_ids.add(segment['chunk_id'])
                remaining -= segment['token_count']
        kept_archive = [s for s in archive_segments if s['chunk_id'] in admitted_archive_ids]
        kept_recent = [s for s in recent_segments if s['chunk_id'] in admitted_recent_ids]
        return list(sink_segments) + kept_archive + kept_recent


    @staticmethod
    def _angular_distance(a, b):
        """1 - cos(angle between a and b). Range [0, 2]. Inputs are unit-normalized (3,) tensors."""
        cos = torch.dot(a, b).clamp(-1.0, 1.0).item()
        return 1.0 - cos

    def _select_archive_by_diversity(self, archive_candidates_sorted, archive_budget):
        """
        Two-stage archive selection.
          Stage 1: take top `archive_diversity_pool` from motion-sorted candidates.
          Stage 2: Farthest-Point Sampling on `camera_forward` to pick `archive_budget`
                   chunks that maximize spatial coverage.
        Falls back to pure motion ranking when:
          - archive_diversity_pool is 0 (disabled, backward compat)
          - pool size <= archive_budget (FPS would degenerate)
          - any candidate is missing camera_forward (no pose data)
        """
        pool_size = self.archive_diversity_pool
        if pool_size <= 0 or pool_size <= archive_budget:
            return archive_candidates_sorted[:archive_budget]

        pool = archive_candidates_sorted[:pool_size]
        if any(c.get('camera_forward') is None for c in pool):
            return archive_candidates_sorted[:archive_budget]

        # Seed FPS with highest-motion candidate
        selected = [pool[0]]
        remaining = list(pool[1:])
        while len(selected) < archive_budget and remaining:
            best_idx = 0
            best_dist = -1.0
            for i, cand in enumerate(remaining):
                min_dist = min(
                    self._angular_distance(cand['camera_forward'], s['camera_forward'])
                    for s in selected
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected


    @staticmethod
    def _archive_rank_key(seg):
        """Archive ranking score: learned utility if the selector set it, else the
        motion score (so heuristic MoCE is unchanged)."""
        return seg.get('_sel_score', seg['motion_score'])

    def _archive_rank_key_consol(self, seg):
        from .utils.memory_consolidation import rank_score
        meta = self._chunk_meta.get(int(seg['chunk_id']))
        return rank_score(seg, meta, self.consolidation)

    @torch.no_grad()
    def _score_archive_learned(self, archive_segments, current_segment):
        """Populate seg['_sel_score'] for each archive candidate using the learned
        ChunkSelector. `current_segment` is the most-recent chunk (the query context)."""
        if not archive_segments:
            return
        cur_id = int(current_segment['chunk_id'])
        cur_meta = self._chunk_meta.get(cur_id, {
            'chunk_id': cur_id,
            'camera_forward': current_segment.get('camera_forward'),
            'k_centroid': current_segment.get('k_centroid'),
            'motion_score': current_segment.get('motion_score', 0.0),
            'value_norm': current_segment.get('value_norm', 0.0),
        })
        rows = []
        for seg in archive_segments:
            cand_meta = self._chunk_meta.get(int(seg['chunk_id']), {
                'chunk_id': int(seg['chunk_id']),
                'camera_forward': seg.get('camera_forward'),
                'k_centroid': seg.get('k_centroid'),
                'motion_score': seg.get('motion_score', 0.0),
                'value_norm': seg.get('value_norm', 0.0),
            })
            if cand_meta.get('pose') is None:
                cand_meta = {
                    **cand_meta,
                    'pose': self._chunk_poses.get(int(seg['chunk_id'])),
                }
            if cur_meta.get('pose') is None:
                cur_meta = {
                    **cur_meta,
                    'pose': self._chunk_poses.get(cur_id),
                }
            rows.append(features_for_schema(
                getattr(self, "_sel_feature_names", FEATURE_NAMES),
                cand_meta, cur_meta, cur_id,
                motion_ref=self._sel_motion_ref, vnorm_ref=self._sel_vnorm_ref,
                translation_scale=self._sel_translation_scale,
                schema_version=getattr(
                    self, "_sel_schema_version", FEATURE_SCHEMA_VERSION),
            ))
        feats = torch.tensor(rows, dtype=torch.float32, device=self.device)
        util = self.chunk_selector(feats).detach().float().cpu().tolist()
        if not isinstance(util, list):
            util = [util]
        for seg, u in zip(archive_segments, util):
            seg['_sel_score'] = float(u)

    def _update_consolidation_utilities(self, archive_segments, keep_count):
        """C1: refresh EMA utilities and hysteresis streaks for archive candidates."""
        from .utils.memory_consolidation import update_utility_ema, rank_score
        cfg = self.consolidation
        if not cfg.enabled or not archive_segments:
            return None
        for seg in archive_segments:
            cid = int(seg['chunk_id'])
            meta = self._chunk_meta.setdefault(cid, {'chunk_id': cid})
            score = float(seg.get('_sel_score', seg.get('motion_score', 0.0)))
            update_utility_ema(
                meta, score,
                beta=cfg.beta,
                patience=cfg.patience,
                stabilize_thr=cfg.stabilize_thr,
                keep_threshold=None,
            )
        ranked = sorted(
            archive_segments,
            key=lambda s: rank_score(s, self._chunk_meta.get(int(s['chunk_id'])), cfg),
            reverse=True,
        )
        thr = None
        if keep_count > 0 and ranked:
            thr = rank_score(
                ranked[min(keep_count, len(ranked)) - 1],
                self._chunk_meta.get(int(ranked[min(keep_count, len(ranked)) - 1]['chunk_id'])),
                cfg,
            )
        for seg in archive_segments:
            meta = self._chunk_meta.setdefault(int(seg['chunk_id']), {'chunk_id': int(seg['chunk_id'])})
            u = float(meta.get('u_ema', seg.get('_sel_score', seg.get('motion_score', 0.0))))
            if thr is not None and u < thr:
                meta['low_streak'] = int(meta.get('low_streak', 0)) + 1
            else:
                meta['low_streak'] = 0
        return thr

    def _dump_oracle(self, path):
        """Persist teacher records + per-chunk metadata.

        Always collects attention-mass first. When ``oracle_label_type`` is
        ``future_use_v1``, post-aggregates Future Coverage Oracle labels
        (does not alter CR eviction). Default remains ``attention_mass``.
        """
        recs = oracle_state().get("records", [])
        agg = {}
        for r in recs:
            g = r["gen_chunk_id"]
            for sid, m in zip(r["seg_ids"], r["seg_mass"]):
                agg.setdefault((g, sid), []).append(float(m))
        merged = []
        for (g, sid), v in agg.items():
            merged.append({
                "gen_chunk_id": g,
                "seg_id": sid,
                "mass": sum(v) / len(v),
                "n_layers": len(v),
            })
        # Attach poses into chunk_meta for world-state selector training.
        chunk_meta = dict(self._chunk_meta)
        for cid, pose in self._chunk_poses.items():
            if cid in chunk_meta:
                chunk_meta[cid] = {**chunk_meta[cid], "pose": pose}
            else:
                chunk_meta[cid] = {"chunk_id": cid, "pose": pose}
        payload = {
            "records": merged,
            "chunk_meta": chunk_meta,
            "config": {
                "sink_chunk_count": int(math.ceil(self.sink_size / 3)) if self.sink_size else 0,
                "recent_window": int(self.ma_kv_recent_window),
                "feature_names": FEATURE_NAMES,
                "schema_version": FEATURE_SCHEMA_VERSION,
                "translation_scale": float(self._sel_translation_scale),
                "probe_every": int(self.oracle_probe_every),
                "label_type": "attention_mass",
                "label_version": FEATURE_SCHEMA_VERSION,
            },
        }
        label_type = getattr(self, "oracle_label_type", "attention_mass")
        if label_type == "future_use_v1":
            from .utils.future_use_labels import convert_oracle_payload
            payload = convert_oracle_payload(
                payload,
                horizon=int(getattr(self, "oracle_future_horizon", 8)),
                gamma=float(getattr(self, "oracle_future_gamma", 0.9)),
                alpha=float(getattr(self, "oracle_future_alpha", 0.5)),
            )
            merged = payload["records"]
            chunk_meta = payload["chunk_meta"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)
        logging.info(
            f"[oracle/{payload['config'].get('label_type', label_type)}] "
            f"saved {len(merged)} (gen_chunk,seg) records + "
            f"{len(chunk_meta)} chunk-meta rows -> {path}")

    def _evict_motion_adaptive_kv_cache(self, kv_cache, attention_budget=None):
        recent_window = max(0, self.ma_kv_recent_window)
        keep_ratio = self.ma_kv_keep_ratio
        min_keep_chunks = max(0, self.ma_kv_min_keep_chunks)
        evicted_segments = []
        keep_chunk_ids = None
        self._pending_tier_map = None
        for layer_cache in kv_cache:
            segments = layer_cache['segments']
            if not segments:
                continue
            sink_segments = [segment for segment in segments if segment['is_sink']]
            non_sink_segments = [segment for segment in segments if not segment['is_sink']]
            if len(non_sink_segments) <= recent_window:
                if keep_chunk_ids is None:
                    if attention_budget is not None:
                        kept_segments = self._enforce_attention_budget(
                            sink_segments=sink_segments,
                            archive_segments=[],
                            recent_segments=non_sink_segments,
                            attention_budget=attention_budget,
                        )
                    else:
                        kept_segments = sink_segments + non_sink_segments
                    keep_chunk_ids = {segment['chunk_id'] for segment in kept_segments}
                    evicted_segments = [
                        segment for segment in segments if segment['chunk_id'] not in keep_chunk_ids
                    ]
                else:
                    kept_segments = [
                        segment for segment in (sink_segments + non_sink_segments)
                        if segment['chunk_id'] in keep_chunk_ids
                    ]
            else:
                recent_segments = non_sink_segments[-recent_window:] if recent_window > 0 else []
                archive_segments = non_sink_segments[:-recent_window] if recent_window > 0 else non_sink_segments
                if keep_chunk_ids is None:
                    # World-State CR: learned selector overrides the archive ranking signal. It
                    # sets seg['_sel_score'] (query-aware utility); the heuristic path
                    # leaves it unset and everything falls back to motion_score, so
                    # MoCE behaviour is byte-identical when selector=='heuristic'.
                    if self.selector == "learned" and self.chunk_selector is not None:
                        self._score_archive_learned(archive_segments, non_sink_segments[-1])
                    keep_count = max(min_keep_chunks, int(math.ceil(len(archive_segments) * keep_ratio)))
                    keep_count = min(len(archive_segments), keep_count)
                    rank_key = (
                        self._archive_rank_key_consol
                        if self.consolidation.enabled else self._archive_rank_key
                    )
                    if self.consolidation.enabled:
                        self._update_consolidation_utilities(archive_segments, keep_count)
                    if keep_count > 0:
                        sorted_archive = sorted(
                            archive_segments, key=rank_key, reverse=True,
                        )
                        if self.consolidation.enabled:
                            from .utils.memory_consolidation import (
                                assign_archive_tiers, enforce_gist_budget,
                            )
                            tier_map = assign_archive_tiers(
                                archive_segments,
                                keep_count=keep_count,
                                chunk_meta=self._chunk_meta,
                                cfg=self.consolidation,
                            )
                            tier_map = enforce_gist_budget(
                                tier_map, archive_segments, self._chunk_meta,
                                self.consolidation,
                            )
                            self._pending_tier_map = tier_map
                            kept_archive = sorted(
                                [s for s in archive_segments
                                 if tier_map.get(int(s['chunk_id'])) in ('L1', 'L2')],
                                key=lambda seg: seg['chunk_id'],
                            )
                        else:
                            self._pending_tier_map = None
                            selected = self._select_archive_by_diversity(
                                sorted_archive, archive_budget=keep_count,
                            )
                            kept_archive = sorted(selected, key=lambda seg: seg['chunk_id'])
                    else:
                        self._pending_tier_map = None
                        kept_archive = []
                    if attention_budget is not None:
                        # Budget enforcement must use the same ranking key.
                        if self.consolidation.enabled:
                            _orig = self._archive_rank_key
                            self._archive_rank_key = self._archive_rank_key_consol  # type: ignore
                            try:
                                kept_segments = self._enforce_attention_budget(
                                    sink_segments=sink_segments,
                                    archive_segments=kept_archive,
                                    recent_segments=recent_segments,
                                    attention_budget=attention_budget,
                                )
                            finally:
                                self._archive_rank_key = _orig  # type: ignore
                        else:
                            kept_segments = self._enforce_attention_budget(
                                sink_segments=sink_segments,
                                archive_segments=kept_archive,
                                recent_segments=recent_segments,
                                attention_budget=attention_budget,
                            )
                    else:
                        kept_segments = sink_segments + kept_archive + recent_segments
                    keep_chunk_ids = {segment['chunk_id'] for segment in kept_segments}
                    evicted_segments = [
                        segment for segment in segments if segment['chunk_id'] not in keep_chunk_ids
                    ]
                else:
                    kept_archive = [
                        segment for segment in archive_segments
                        if segment['chunk_id'] in keep_chunk_ids
                    ]
                    kept_segments = [
                        segment for segment in (sink_segments + kept_archive + recent_segments)
                        if segment['chunk_id'] in keep_chunk_ids
                    ]
            layer_cache['segments'] = kept_segments
            # Compress archive tiers: SWTP and/or Consolidation L1/L2.
            self._apply_archive_compression(
                kept_segments, recent_window=max(0, self.ma_kv_recent_window),
            )
            local_end_index = sum(segment['token_count'] for segment in kept_segments)
            layer_cache['local_end_index'].fill_(local_end_index)
        return evicted_segments


    def _apply_archive_compression(self, segments, recent_window):
        """Apply L1 SWTP / L2 gist compression to archive segments in place."""
        if not segments:
            return
        non_sink = [s for s in segments if not s['is_sink']]
        recent_chunk_ids = set(
            s['chunk_id'] for s in non_sink[-recent_window:]
        ) if recent_window > 0 else set()
        tier_map = getattr(self, '_pending_tier_map', None) or {}

        for seg in segments:
            if seg['is_sink'] or seg['chunk_id'] in recent_chunk_ids:
                seg.setdefault('memory_tier', 'L0')
                continue
            cid = int(seg['chunk_id'])
            tier = tier_map.get(cid)
            if tier is None:
                # Legacy SWTP path (consolidation off): SWTP every archive chunk once.
                if not self.enable_swtp or seg.get('is_swtp', False):
                    continue
                target = 'L1'
            else:
                target = tier
                if target == 'L3':
                    continue  # should already be dropped
                if target == 'L1' and seg.get('is_swtp') and not seg.get('is_gist'):
                    seg['memory_tier'] = 'L1'
                    continue
                if target == 'L2' and seg.get('is_gist'):
                    seg['memory_tier'] = 'L2'
                    continue
                if target == 'L1' and not self.enable_swtp and not self.consolidation.tiers_enabled:
                    seg['memory_tier'] = 'L1'
                    continue

            sal = seg.get('token_saliency', None)
            grid = seg.get('token_grid', None)
            if target == 'L2':
                # Gist demotion: summary tokens only. Synthesize flat saliency if missing.
                if sal is None:
                    sal = torch.ones(int(seg['token_count']), dtype=torch.float32)
                    grid = None
                mode = 'gist'
                keep_ratio = 0.0
                num_summary = int(self.consolidation.gist_tokens)
            else:
                # L1 SWTP
                if sal is None:
                    # No saliency → uniform lattice rather than leaving full forever.
                    sal = torch.ones(int(seg['k'].shape[1]), dtype=torch.float32)
                    mode = 'uniform'
                else:
                    gini = self._gini_coefficient(sal)
                    mode = 'uniform' if gini < self.swtp_min_saliency_gini else 'standard'
                keep_ratio = self.swtp_keep_ratio
                num_summary = self.swtp_num_summary
                if not self.enable_swtp and not self.consolidation.tiers_enabled:
                    continue

            k_red, v_red, _ = self._apply_swtp_to_kv(
                seg['k'], seg['v'], sal,
                keep_ratio, num_summary,
                token_grid=grid, mode=mode,
            )
            seg['k'] = k_red
            seg['v'] = v_red
            seg['token_count'] = int(k_red.shape[1])
            seg['is_swtp'] = True
            seg['is_gist'] = (target == 'L2')
            seg['memory_tier'] = target
            seg['token_saliency'] = None

    def _apply_swtp_to_archive_segments(self, segments, recent_window):
        """Backward-compatible alias for SWTP (no consolidation tiers)."""
        self._pending_tier_map = None
        self._apply_archive_compression(segments, recent_window)

    def _append_motion_adaptive_kv_segments(
        self,
        kv_cache,
        pending_segments,
        chunk_id,
        motion_score,
        is_sink,
        token_saliency=None,
        token_grid=None,
        camera_forward=None,
        runtime_stats=None,
    ):
        """
        Append a new chunk's K/V to the dynamic cache. If `token_saliency` is
        provided (SWTP), stash it in the segment payload so that SWTP
        can be applied lazily when the chunk transitions into the archive
        tier (via _evict_motion_adaptive_kv_cache). If `camera_forward` is
        provided, stash for trajectory-diversity archive selection.
        """
        if pending_segments is None:
            return
        # Learned selector / oracle / consolidation need per-chunk metadata.
        need_meta = (
            (self.selector == "learned")
            or self.collect_oracle
            or self.consolidation.enabled
        )
        # Content features are decision-context only (layer-0). Never materialize
        # float32 copies of every layer's K/V — that was the main learned-vs-heuristic
        # peak-memory gap, not the world-state feature schema itself.
        k_centroid = None
        value_norm = 0.0
        if need_meta:
            layer0 = next(
                (ps for lc, ps in zip(kv_cache, pending_segments)
                 if ps is not None and int(lc.get('layer_idx', -1)) == 0),
                None,
            )
            if layer0 is not None:
                with torch.no_grad():
                    # Reduce first, then cast — avoid full-tensor .float() temps.
                    k_centroid = layer0['k'].detach().mean(dim=(0, 1, 2)).float().cpu()
                    value_norm = float(layer0['v'].detach().abs().mean().float().item())

        for layer_cache, pending_segment in zip(kv_cache, pending_segments):
            if pending_segment is None:
                continue
            segments = layer_cache['segments']
            seg_saliency = token_saliency
            is_layer0 = int(layer_cache.get('layer_idx', -1)) == 0
            cf_seg = camera_forward
            if cf_seg is not None and torch.is_tensor(cf_seg):
                # Keep segment payload CPU-side to avoid retaining CUDA tensors
                # across every layer's segment list.
                cf_seg = cf_seg.detach().float().cpu()
            segment_payload = {
                'chunk_id': int(chunk_id),
                'motion_score': float(motion_score),
                'is_sink': bool(is_sink),
                'token_count': int(pending_segment['token_count']),
                'token_saliency': seg_saliency,      # SWTP: stash for lazy prune at archive promotion
                'token_grid': token_grid,
                'is_swtp': False,                    # not compressed yet
                'is_gist': False,
                'memory_tier': 'L0',
                'camera_forward': cf_seg,            # trajectory-diversity: unit forward axis (3,)
                # Only layer-0 needs content features for the shared keep decision.
                'k_centroid': k_centroid if is_layer0 else None,
                'value_norm': value_norm if is_layer0 else 0.0,
                'k': pending_segment['k'],
                'v': pending_segment['v'],
            }
            if len(segments) > 0 and int(segments[-1]['chunk_id']) == int(chunk_id):
                segments[-1] = segment_payload
            else:
                segments.append(segment_payload)
            local_end_index = sum(segment['token_count'] for segment in segments)
            layer_cache['local_end_index'].fill_(local_end_index)
            # Populate the per-chunk decision-context table from layer 0 only
            # (the learned keep decision in _evict is made on layer-0 segments).
            if need_meta and is_layer0:
                cf = cf_seg
                if cf is not None and not isinstance(cf, list):
                    cf = cf.detach().float().cpu().tolist() if torch.is_tensor(cf) else list(cf)
                pose = self._chunk_poses.get(int(chunk_id))
                revisited = []
                if pose is not None:
                    for oid, om in list(self._chunk_meta.items()):
                        op = om.get('pose')
                        if op is None:
                            continue
                        if se3_distance(pose, op, self._sel_translation_scale) <= self._revisit_radius:
                            om['revisit_count'] = int(om.get('revisit_count', 0)) + 1
                            om['last_observed'] = int(chunk_id)
                            revisited.append(int(oid))
                if runtime_stats is not None and revisited:
                    from .utils.memory_consolidation import update_revisit_coverage
                    retained_ids = {int(s['chunk_id']) for s in segments}
                    update_revisit_coverage(runtime_stats, revisited, retained_ids)
                self._chunk_meta[int(chunk_id)] = {
                    'chunk_id': int(chunk_id),
                    'motion_score': float(motion_score),
                    'is_sink': bool(is_sink),
                    'camera_forward': cf,
                    'k_centroid': k_centroid,
                    'value_norm': value_norm,
                    'pose': pose,
                    'revisit_count': 0,
                    'last_observed': int(chunk_id),
                    'u_ema': None,
                    'low_streak': 0,
                    'high_streak': 0,
                    'stabilized': False,
                }


    def _get_retained_token_count(self, kv_cache):
        if not kv_cache:
            return 0
        first_layer = kv_cache[0]
        if first_layer.get('mode') == 'dynamic':
            return sum(segment['token_count'] for segment in first_layer['segments'])
        return int(first_layer['local_end_index'].item())


    def _reduce_max_across_ranks(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())


    def _format_stat_value(self, value, precision=4):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            if math.isinf(value):
                return "inf"
            return f"{value:.{precision}f}"
        return str(value)


    def _log_generation_summary(self, stats):
        if self.rank != 0:
            return
        logging.info("[FastInferenceSummary]")
        ordered_keys = [
            "motion_adaptive_enabled",
            "selector",
            "feature_schema",
            "p50_chunk_time_s",
            "p95_chunk_time_s",
            "total_chunks",
            "retained_chunks",
            "evicted_chunks",
            "retained_chunk_ratio",
            "avg_motion_score_kept",
            "avg_motion_score_evicted",
            "rescue_count",
            "avg_attention_context_tokens",
            "max_attention_context_tokens",
            "total_generation_time_s",
            "avg_chunk_time_s",
            "tail_chunk_time_s",
            "peak_memory_allocated_gb",
            "peak_memory_reserved_gb",
            "timesteps_index",
        ]
        for key in ordered_keys:
            if key in stats:
                logging.info(f"  {key}: {self._format_stat_value(stats[key])}")


    def _debug_log_dynamic_kv_state(self, tag, kv_cache, chunk_id, extra=None):
        if not self.debug_motion_adaptive_kv or not kv_cache:
            return
        first_layer = kv_cache[0]
        segments = first_layer.get("segments", [])
        segment_chunk_ids = [segment["chunk_id"] for segment in segments]
        logging.info(
            f"[MA-KV][rank{self.rank}] {tag} chunk={chunk_id} "
            f"commit_current={first_layer.get('commit_current')} "
            f"segments={len(segments)} "
            f"chunk_ids={segment_chunk_ids} "
            f"local_end={int(first_layer['local_end_index'].item())} "
            f"global_end={int(first_layer['global_end_index'].item())} "
            f"extra={extra}"
        )


    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 179, 358, 679],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        generation_start_time = time.perf_counter()
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            if self.control_type == 'act':
                # In 'act' mode, use rotation of c2ws to control orientation and wasd_action to drive movement.
                wasd_action = np.load(os.path.join(action_path, "action.npy")) # wasd action
                wasd_action = wasd_action[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        self._last_timesteps_index = list(timesteps_index)
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        # cam preparation (only if action_path is provided)
        dit_cond_dict = None
        c2ws_plucker_emb = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(Ks,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=h,
                                    width_resize=w,
                                    height_final=h,
                                    width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            # Snapshot absolute SE(3) + forward BEFORE compute_relative_poses
            # overwrites c2ws_infer with framewise deltas. Forward axis = third
            # column of c2w rotation (OpenCV +Z). Full pose feeds World-State CR features.
            absolute_c2w_per_frame = c2ws_infer.clone()  # (lat_f, 4, 4) or (lat_f, 3, 4+)
            absolute_forward_per_frame = c2ws_infer[:, :3, 2].clone()  # (lat_f, 3)
            absolute_Ks_per_frame = Ks.clone() if torch.is_tensor(Ks) else None
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            if self.control_type == 'act':
                wasd_action = torch.from_numpy(wasd_action[::4]).float().to(self.device)
            else:
                wasd_action = None
            only_rays_d = wasd_action is not None
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            if wasd_action is not None:
                wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
                wasd_action_tensor = rearrange(
                    wasd_action_tensor,
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                    c1=int(h // lat_h),
                    c2=int(w // lat_w),
                )
                wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
                wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        if self.enable_motion_adaptive_kv_eviction or self.enable_swtp:
            kv_size = None
            self_kv_cache = self._initialize_motion_adaptive_kv_cache(
                num_layers=model_args.num_layers,
                device=self.device)
        else:
            if self.local_attn_size > -1:
                kv_size = frame_seqlen * self.local_attn_size
            else:
                kv_size = frame_seqlen * lat_f
            self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
            self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                          shape=self_kv_shape,
                                                          dtype=transformer_dtype,
                                                          device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                         shape=cross_kv_shape,
                                                         dtype=transformer_dtype,
                                                         device=self.device)
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            if c2ws_plucker_emb is not None:
                c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            else:
                c2ws_plucker_emb_chunk = [None] * len(latents_chunk)

            # Per-chunk camera forward + full SE(3) pose for World-State CR / diversity.
            # None when no action_path / no poses available.
            camera_pose_per_chunk = None
            if action_path is not None:
                num_chunks = len(latents_chunk)
                fwd = absolute_forward_per_frame[: num_chunks * chunk_size]
                fwd = fwd.view(num_chunks, chunk_size, 3).mean(dim=1)
                fwd = torch.nn.functional.normalize(fwd, dim=-1).to(self.device)
                camera_forward_per_chunk = fwd  # (num_chunks, 3)
                # Chunk-centre absolute pose (mean translation, mean-frame rotation).
                poses = []
                c2w_abs = absolute_c2w_per_frame[: num_chunks * chunk_size]
                mid = chunk_size // 2
                translations = []
                for ci in range(num_chunks):
                    frame = c2w_abs[ci * chunk_size + mid]
                    R = frame[:3, :3].detach().float().cpu()
                    t = frame[:3, 3].detach().float().cpu()
                    translations.append(t)
                    Kvec = None
                    if absolute_Ks_per_frame is not None:
                        Kf = absolute_Ks_per_frame[min(ci * chunk_size + mid,
                                                       len(absolute_Ks_per_frame) - 1)]
                        Kvec = Kf.detach().float().cpu().flatten().tolist()
                    poses.append({
                        "translation": t.tolist(),
                        "rotation": R.tolist(),
                        "intrinsics": Kvec,
                        "camera_forward": camera_forward_per_chunk[ci].detach().float().cpu().tolist(),
                    })
                camera_pose_per_chunk = poses
                # Translation scale = median pairwise step for SE(3) normalization.
                if len(translations) >= 2:
                    steps = [float(torch.norm(translations[i] - translations[i - 1]).item())
                             for i in range(1, len(translations))]
                    steps = sorted(s for s in steps if s > 1e-8)
                    if steps:
                        med = steps[len(steps) // 2]
                        self._sel_translation_scale = max(med, 1e-3)
            else:
                camera_forward_per_chunk = None
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            prev_c2ws_plucker_emb = None
            prev_x0 = None
            sink_chunk_count = int(math.ceil(self.sink_size / chunk_size)) if chunk_size > 0 else 0
            runtime_stats = {
                "motion_adaptive_enabled": self.enable_motion_adaptive_kv_eviction,
                "chunk_times": [],
                "context_tokens": [],
                "rescued_chunk_ids": set(),
                "evicted_chunk_ids": set(),
                "evicted_motion_scores": {},
                "evicted_motion_scores": {},
            }
            # Oracle collection: per-chunk attention-mass labels.
            if self.collect_oracle and self.enable_motion_adaptive_kv_eviction:
                oracle_reset()
                oracle_set(True, probe_every=self.oracle_probe_every)
            else:
                oracle_set(False)
            self._chunk_size = chunk_size
            # Reset per-chunk metadata each generation (not just for oracle). Without
            # this, batch_generate's reused pipe accumulates stale chunk_meta across
            # clips in the learned-selector path (inflated peak memory in the harness).
            self._chunk_meta = {}
            self._chunk_poses = {}
            if camera_pose_per_chunk is not None and len(camera_pose_per_chunk) >= 2:
                ts = [
                    float(torch.norm(
                        torch.tensor(camera_pose_per_chunk[i]["translation"])
                        - torch.tensor(camera_pose_per_chunk[i - 1]["translation"])
                    ).item())
                    for i in range(1, len(camera_pose_per_chunk))
                ]
                ts = sorted(s for s in ts if s > 1e-8)
                if ts:
                    self._sel_translation_scale = max(ts[len(ts) // 2], 1e-3)
            if camera_pose_per_chunk is not None:
                for ci, pose in enumerate(camera_pose_per_chunk):
                    self._chunk_poses[int(ci)] = pose
            if self.enable_motion_adaptive_kv_eviction:
                if max_attention_size is not None:
                    kv_attention_limit = max_attention_size
                elif self.local_attn_size > -1:
                    kv_attention_limit = frame_seqlen * self.local_attn_size
                else:
                    kv_attention_limit = None
            else:
                kv_attention_limit = kv_size if max_attention_size is None else max_attention_size
            chunk_iter = range(num_inference_chunk)
            if self._cr_fast_runtime:
                # CR: no tqdm (I/O + refresh overhead). Window keeps progress bar.
                pass
            else:
                chunk_iter = tqdm(chunk_iter, disable=(self.rank != 0))
            for chunk_id in chunk_iter:
                # Window: sync each chunk for accurate per-chunk timing.
                # CR: skip — overlaps GPU work; only end-to-end sync below.
                if (not self._cr_fast_runtime) and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                chunk_start_time = time.perf_counter()
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                current_chunk_tokens = int(current_latent.shape[1] * frame_seqlen)

                retained_history_tokens = self._get_retained_token_count(self_kv_cache)
                current_context_tokens = retained_history_tokens + current_chunk_tokens
                if kv_attention_limit is not None:
                    current_context_tokens = min(current_context_tokens, kv_attention_limit)
                runtime_stats["context_tokens"].append(int(current_context_tokens))

                if current_c2ws_plucker_emb is not None:
                    dit_cond_dict = {
                        "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                    }
                else:
                    dit_cond_dict = None

                if self.enable_motion_adaptive_kv_eviction and current_c2ws_plucker_emb is not None:
                    current_motion_score = compute_chunk_motion_score(
                        current_chunk=current_c2ws_plucker_emb,
                        previous_chunk=prev_c2ws_plucker_emb,
                    )
                elif self.enable_motion_adaptive_kv_eviction:
                    current_motion_score = float("inf") if chunk_id == 0 else 0.0

                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_attention_limit
                }

                # CR: never empty_cache in the hot loop (large stall).
                if offload_model and (not self._cr_fast_runtime):
                    torch.cuda.empty_cache()

                if self.enable_motion_adaptive_kv_eviction:
                    self._set_motion_adaptive_kv_runtime(
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                        motion_score=current_motion_score,
                        commit_current=False,
                        is_sink=(chunk_id < sink_chunk_count),
                    )
                    self._debug_log_dynamic_kv_state(
                        tag="after_set_runtime_sampling",
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                        extra={"motion_score": current_motion_score},
                    )

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)

                    noise_pred = self.model(
                        x=latent_model_input, t=timestep, **kwargs)[0]

                    if offload_model and (not self._cr_fast_runtime):
                        torch.cuda.empty_cache()

                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                if self.enable_motion_adaptive_kv_eviction:
                    if self.ma_kv_latent_rescue:
                        latent_motion_score = self._compute_latent_motion_score(x0, prev_x0)
                        if math.isfinite(latent_motion_score):
                            current_motion_score = max(current_motion_score, latent_motion_score)
                            if latent_motion_score >= self.ma_kv_latent_rescue_thr:
                                runtime_stats["rescued_chunk_ids"].add(chunk_id)
                    self._set_motion_adaptive_kv_runtime(
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                        motion_score=current_motion_score,
                        commit_current=True,
                        is_sink=(chunk_id < sink_chunk_count),
                    )
                    self._debug_log_dynamic_kv_state(
                        tag="before_context_forward",
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                        extra={"motion_score": current_motion_score},
                    )
                if self.enable_motion_adaptive_kv_eviction:
                    # SWTP: precompute token saliency for lazy SWTP at archive promotion.
                    swtp_saliency = None
                    swtp_grid = None
                    if self.enable_swtp or self.consolidation.tiers_enabled:
                        swtp_saliency, swtp_grid = self._compute_token_saliency(x0, prev_x0)
                    _, pending_kv_segments = self.model(
                        x=[x0],
                        t=timestep,
                        return_kv_segments=True,
                        **kwargs,
                    )
                    current_camera_forward = (
                        camera_forward_per_chunk[chunk_id]
                        if camera_forward_per_chunk is not None
                        else None
                    )
                    self._append_motion_adaptive_kv_segments(
                        kv_cache=self_kv_cache,
                        pending_segments=pending_kv_segments,
                        chunk_id=chunk_id,
                        motion_score=current_motion_score,
                        is_sink=(chunk_id < sink_chunk_count),
                        token_saliency=swtp_saliency,
                        token_grid=swtp_grid,
                        camera_forward=current_camera_forward,
                        runtime_stats=runtime_stats,
                    )
                elif self.enable_swtp:
                    # Compute per-token saliency from latent residual
                    token_saliency, token_grid = self._compute_token_saliency(x0, prev_x0)
                    _, pending_kv_segments = self.model(
                        x=[x0],
                        t=timestep,
                        return_kv_segments=True,
                        **kwargs,
                    )
                    self._append_swtp_kv_segments(
                        kv_cache=self_kv_cache,
                        pending_segments=pending_kv_segments,
                        chunk_id=chunk_id,
                        token_saliency=token_saliency,
                        token_grid=token_grid,
                        is_sink=(chunk_id < sink_chunk_count),
                    )
                    # Track stats: count of SWTP-applied vs full-stored chunks
                    if self_kv_cache and self_kv_cache[0]['segments']:
                        last_seg = self_kv_cache[0]['segments'][-1]
                        if last_seg.get('is_swtp', False):
                            runtime_stats.setdefault("swtp_applied_chunks", set()).add(chunk_id)
                        else:
                            runtime_stats.setdefault("swtp_fallback_chunks", set()).add(chunk_id)
                else:
                    self.model(x=[x0], t=timestep, **kwargs)
                if self.enable_motion_adaptive_kv_eviction:
                    self._debug_log_dynamic_kv_state(
                        tag="after_context_forward",
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                    )
                    # Leave headroom for the next chunk's KV during forward, so
                    # the attention call never has to fall back to sink-blind
                    # tail truncation in ``build_dynamic_kv_tensors``.
                    if kv_attention_limit is not None:
                        eviction_budget = max(0, kv_attention_limit - current_chunk_tokens)
                    else:
                        eviction_budget = None
                    evicted_segments = self._evict_motion_adaptive_kv_cache(
                        self_kv_cache, attention_budget=eviction_budget)
                    self._debug_log_dynamic_kv_state(
                        tag="after_evict",
                        kv_cache=self_kv_cache,
                        chunk_id=chunk_id,
                        extra={"evicted_chunk_ids": [segment["chunk_id"] for segment in evicted_segments]},
                    )
                    for segment in evicted_segments:
                        runtime_stats["evicted_chunk_ids"].add(segment["chunk_id"])
                        runtime_stats["evicted_motion_scores"][segment["chunk_id"]] = segment["motion_score"]
                    prev_c2ws_plucker_emb = current_c2ws_plucker_emb.detach().clone() if current_c2ws_plucker_emb is not None else None
                prev_x0 = x0.detach().clone()
                if (not self._cr_fast_runtime) and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                runtime_stats["chunk_times"].append(time.perf_counter() - chunk_start_time)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            # World-State CR: dump oracle records + per-chunk metadata for offline selector training.
            if self.collect_oracle and self.oracle_out is not None and self.rank == 0:
                self._dump_oracle(self.oracle_out)
            if self.collect_oracle:
                oracle_set(False)

            # Snapshot retained-segment metadata for the stats block below
            # BEFORE the KV cache is freed.
            if self.enable_motion_adaptive_kv_eviction and self_kv_cache:
                retained_segments = []
                for seg in self_kv_cache[0]['segments']:
                    retained_segments.append({
                        'motion_score': seg['motion_score'],
                        'chunk_id': seg['chunk_id'],
                        'is_sink': seg['is_sink'],
                        'token_count': int(seg.get('token_count', 0)),
                        'memory_tier': seg.get('memory_tier', 'L0'),
                        'is_swtp': bool(seg.get('is_swtp', False)),
                        'is_gist': bool(seg.get('is_gist', False)),
                    })
            else:
                retained_segments = []

            # Free KV cache before VAE decode. Critical for SWTP standalone
            # mode (no chunk eviction → KV grows to ~40 GB / GPU over 40 chunks).
            # Also benefits MoCE when offload_model is on.
            if self.enable_swtp or self.enable_motion_adaptive_kv_eviction:
                for layer_cache in self_kv_cache:
                    layer_cache['segments'].clear()
                    if 'packed_k' in layer_cache:
                        layer_cache['packed_k'] = None
                        layer_cache['packed_v'] = None
            del self_kv_cache
            del cross_kv_cache
            # One cache flush after freeing KV is enough; CR skips extra mid-run flushes.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if offload_model:
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            if (not self._cr_fast_runtime) and torch.cuda.is_available():
                torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # End-to-end sync kept for total wall time / peak memory (both paths).
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        total_generation_time = time.perf_counter() - generation_start_time
        peak_allocated_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        chunk_times = runtime_stats["chunk_times"]
        context_tokens = runtime_stats["context_tokens"]
        tail_start = len(chunk_times) // 2
        tail_chunk_times = chunk_times[tail_start:] if chunk_times else []
        # retained_segments was snapshotted before the KV cache was freed above.
        kept_motion_scores = [segment["motion_score"] for segment in retained_segments if not segment["is_sink"]]
        evicted_motion_scores = list(runtime_stats["evicted_motion_scores"].values())
        retained_chunks = len(retained_segments) if self.enable_motion_adaptive_kv_eviction else len(chunk_times)
        generation_stats = {
            "motion_adaptive_enabled": self.enable_motion_adaptive_kv_eviction,
            "cr_fast_runtime": bool(getattr(self, "_cr_fast_runtime", False)),
            "total_chunks": len(chunk_times),
            "retained_chunks": retained_chunks,
            "evicted_chunks": len(runtime_stats["evicted_chunk_ids"]),
            "retained_chunk_ratio": (retained_chunks / len(chunk_times)) if chunk_times else 0.0,
            "avg_motion_score_kept": (sum(kept_motion_scores) / len(kept_motion_scores)) if kept_motion_scores else None,
            "avg_motion_score_evicted": (sum(evicted_motion_scores) / len(evicted_motion_scores)) if evicted_motion_scores else None,
            "rescue_count": len(runtime_stats["rescued_chunk_ids"]),
            "avg_attention_context_tokens": (sum(context_tokens) / len(context_tokens)) if context_tokens else 0.0,
            "max_attention_context_tokens": max(context_tokens) if context_tokens else 0,
            "total_generation_time_s": self._reduce_max_across_ranks(total_generation_time),
            "avg_chunk_time_s": (sum(chunk_times) / len(chunk_times)) if chunk_times else 0.0,
            "tail_chunk_time_s": (sum(tail_chunk_times) / len(tail_chunk_times)) if tail_chunk_times else 0.0,
            "peak_memory_allocated_gb": self._reduce_max_across_ranks(peak_allocated_gb),
            "peak_memory_reserved_gb": self._reduce_max_across_ranks(peak_reserved_gb),
            "timesteps_index": list(getattr(self, "_last_timesteps_index", [])),
            "selector": self.selector,
            "feature_schema": (
                getattr(self, "_sel_schema_version", FEATURE_SCHEMA_VERSION)
                if self.selector == "learned" else None
            ),
            "translation_scale": float(self._sel_translation_scale),
            "p50_chunk_time_s": (
                float(sorted(chunk_times)[(len(chunk_times) - 1) // 2]) if chunk_times else 0.0
            ),
            "p95_chunk_time_s": (
                float(sorted(chunk_times)[min(len(chunk_times) - 1, int(0.95 * (len(chunk_times) - 1)))])
                if chunk_times else 0.0
            ),
            "consolidation": self.consolidation.mode,
            "revisit_coverage": runtime_stats.get("revisit_coverage"),
            "revisit_events": runtime_stats.get("revisit_events", 0),
            "tier_counts": {
                t: sum(1 for s in retained_segments if s.get("memory_tier") == t)
                for t in ("L0", "L1", "L2")
            } if retained_segments else None,
        }
        self.last_generation_stats = generation_stats
        self._log_generation_summary(generation_stats)

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache
