from typing import List, Optional
import numpy as np
import torch
import time
import copy

from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.visualize import process_video
import torch.nn.functional as F
from demo_utils.constant import ZERO_VAE_CACHE
from tqdm import tqdm


def _cr_reset(pipeline):
    planner = getattr(pipeline, "kv_retention", None)
    if planner is not None:
        planner.reset()


def _cr_observe(pipeline, start_frame, num_frames, conditional_dict):
    planner = getattr(pipeline, "kv_retention", None)
    if planner is not None and planner.enabled:
        planner.observe_block(
            start_frame,
            num_frames,
            mouse=conditional_dict.get("mouse_cond"),
            keyboard=conditional_dict.get("keyboard_cond"),
        )


def _rope_finish_block(pipeline, e2e=None) -> None:
    rope = getattr(getattr(pipeline, "generator", None), "model", None)
    rope = getattr(rope, "causal_rope", None) if rope is not None else None
    if rope is None or not getattr(rope, "profile", False):
        return
    ms = rope.finish_block()
    if e2e is not None and e2e.enabled and ms is not None:
        e2e.time_ms.setdefault("rope", []).append(float(ms))


def _gemm_finish_block(pipeline, e2e=None) -> None:
    model = getattr(getattr(pipeline, "generator", None), "model", None)
    timer = getattr(model, "gemm_timer", None) if model is not None else None
    if timer is None or not getattr(timer, "enabled", False):
        return
    parts = timer.finish_block()
    if e2e is not None and e2e.enabled:
        e2e.time_ms.setdefault("attn", []).append(float(parts.get("attn", 0.0)))
        e2e.time_ms.setdefault("ffn", []).append(float(parts.get("ffn", 0.0)))


def _rope_summary(pipeline) -> None:
    rope = getattr(getattr(pipeline, "generator", None), "model", None)
    rope = getattr(rope, "causal_rope", None) if rope is not None else None
    e2e = getattr(pipeline, "e2e", None)
    want = (rope is not None and getattr(rope, "profile", False)) or (
        e2e is not None and e2e.enabled
    )
    if not want:
        return
    if rope is not None and getattr(rope, "profile", False):
        print(f"[ROPE] {rope.summary()}", flush=True)
    timer = getattr(getattr(pipeline, "generator", None), "model", None)
    timer = getattr(timer, "gemm_timer", None) if timer is not None else None
    if timer is not None and getattr(timer, "enabled", False):
        print(f"[FFN] {timer.summary()}", flush=True)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        print(f"[DIT] peak_mem_mib={peak:.1f}", flush=True)


def _tich_configure(pipeline, conditional_dict=None, rollout_id=0):
    from wan.memory.cond_hoist import TICHState, attach_tich
    enabled = bool(getattr(pipeline, "cond_hoist_enabled", False))
    profile = bool(getattr(getattr(pipeline, "args", None), "cond_hoist_profile", False))
    state = TICHState(enabled=enabled, profile=profile, assert_counts=enabled)
    pipeline.tich = state
    attach_tich(pipeline.generator.model, state)
    if enabled:
        state.begin_rollout(int(rollout_id))
        vis = None if not isinstance(conditional_dict, dict) else conditional_dict.get("visual_context")
        if vis is not None:
            state.precompute_img_emb(pipeline.generator.model, vis)
        print(
            "[TICH] enable_cond_hoist=True instance=TICHState "
            f"profile={profile} conv_split=True img_emb=True",
            flush=True,
        )


def _tich_begin(pipeline, block_id, block_cond=None):
    state = getattr(pipeline, "tich", None)
    if state is None or not state.enabled:
        return
    if state.profile:
        state._block_t0 = time.perf_counter()
        if torch.cuda.is_available():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            state._block_ev0 = ev
        else:
            state._block_ev0 = None
    cond = block_cond.get("cond_concat") if isinstance(block_cond, dict) else None
    state.begin_block(
        block_id,
        model=pipeline.generator.model,
        cond_concat=cond,
    )


def _tich_end(pipeline, expected_forwards=None):
    state = getattr(pipeline, "tich", None)
    if state is None or not state.enabled:
        return
    if state.profile:
        if torch.cuda.is_available() and getattr(state, "_block_ev0", None) is not None:
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            ev1.synchronize()
            state.time_ms["block"].append(float(state._block_ev0.elapsed_time(ev1)))
        else:
            t0 = float(getattr(state, "_block_t0", time.perf_counter()))
            state.time_ms["block"].append((time.perf_counter() - t0) * 1000.0)
    if expected_forwards is None:
        expected_forwards = int(len(getattr(pipeline, "denoising_step_list", [])) + 1)
    state.end_block(expected_forwards=int(expected_forwards))


def _tich_summary(pipeline):
    state = getattr(pipeline, "tich", None)
    if state is None or not state.enabled:
        return
    state.end_rollout()
    print(f"[TICH] {state.summary()}", flush=True)


def _tich_generate(pipeline, **kwargs):
    state = getattr(pipeline, "tich", None)
    if state is not None and state.enabled and state.profile:
        return state.timed("denoise", lambda: pipeline.generator(**kwargs))
    return pipeline.generator(**kwargs)


def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        print()
        print('-'*30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print('-'*30)
        KEYBOARD_IDX = { 
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    
    new_cond = {}
    
    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame: current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            vae_decoder=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
            
        self.vae_decoder = vae_decoder
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        from wan.memory.kv_retention import attach_planner, build_planner
        self.kv_retention = build_planner(args)
        if self.kv_retention.enabled:
            attach_planner(self.generator.model, self.kv_retention)
            print(
                f"[CR] memory_policy={self.kv_retention.policy} "
                f"sink={self.kv_retention.sink_frames} recent={self.kv_retention.recent_frames}",
                flush=True,
            )
        self.cond_hoist_enabled = bool(getattr(args, "enable_cond_hoist", False))
        from demo_utils.e2e_profile import E2EProfile
        self.e2e = E2EProfile(enabled=bool(getattr(args, "e2e_profile", False)))
        if self.e2e.enabled:
            print("[E2E] per-block CUDA-event split dit/vae/other", flush=True)
        from wan.modules.causal_rope import attach_causal_rope
        attach_causal_rope(
            self.generator.model,
            mode=str(getattr(args, "rope_mode", "fp64")),
            profile=bool(getattr(args, "dit_profile", False)),
        )
        from wan.modules.ffn_gemm import attach_gemm_timer
        attach_gemm_timer(
            self.generator.model,
            enabled=bool(getattr(args, "dit_profile", False)),
        )
        self.dump_latents = str(getattr(args, "dump_latents", "") or "")

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent = None,
        return_latents = False,
        mode = 'universal',
        profile = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
        _cr_reset(self)
        _tich_configure(self, conditional_dict)
        dumped_blocks = [] if self.dump_latents else None
        e2e_or_rope = bool(getattr(self, "e2e", None) and self.e2e.enabled)
        _rope = getattr(self.generator.model, "causal_rope", None)
        if _rope is not None and _rope.profile:
            e2e_or_rope = True
        if torch.cuda.is_available() and e2e_or_rope:
            torch.cuda.reset_peak_memory_stats()
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                block_cond = cond_current(
                    conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode)
                _cr_observe(self, current_start_frame, self.num_frame_per_block, conditional_dict)
                _tich_begin(self, current_start_frame, block_cond)
                _tich_generate(
                    self,
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=block_cond,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                _tich_end(self, expected_forwards=1)
                current_start_frame += self.num_frame_per_block


        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
        for current_num_frames in tqdm(all_num_frames):

            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            block_cond = cond_current(
                conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode)
            _cr_observe(self, current_start_frame, current_num_frames, conditional_dict)
            _tich_begin(self, current_start_frame, block_cond)

            e2e = getattr(self, "e2e", None)
            e2e_on = e2e is not None and e2e.enabled
            if e2e_on and torch.cuda.is_available():
                _eb0 = torch.cuda.Event(enable_timing=True)
                _eb0.record()
                _ed0 = _eb0
            elif e2e_on:
                _eb0 = time.perf_counter()

            # Step 3.1: Spatial denoising loop
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = _tich_generate(
                        self,
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_cond,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = _tich_generate(
                        self,
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_cond,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            if dumped_blocks is not None:
                dumped_blocks.append(denoised_pred.detach().float().cpu())

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            _tich_generate(
                self,
                noisy_image_or_video=denoised_pred,
                conditional_dict=block_cond,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            _tich_end(self)

            if e2e_on and torch.cuda.is_available():
                _ed1 = torch.cuda.Event(enable_timing=True)
                _ed1.record()
            elif e2e_on:
                _ed1_cpu = time.perf_counter()
            _rope_finish_block(self, e2e if e2e_on else None)
            _gemm_finish_block(self, e2e if e2e_on else None)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

            denoised_pred = denoised_pred.transpose(1,2)
            if e2e_on:
                video, vae_cache = e2e.timed(
                    "vae",
                    lambda: self.vae_decoder(denoised_pred.half(), *vae_cache),
                )
            else:
                video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            if e2e_on:
                if torch.cuda.is_available():
                    _eb1 = torch.cuda.Event(enable_timing=True)
                    _eb1.record()
                    _eb1.synchronize()
                    e2e.time_ms["dit"].append(float(_ed0.elapsed_time(_ed1)))
                    e2e.time_ms["block"].append(float(_eb0.elapsed_time(_eb1)))
                else:
                    e2e.time_ms["dit"].append((_ed1_cpu - _eb0) * 1000.0)
                    e2e.time_ms["block"].append((time.perf_counter() - _eb0) * 1000.0)
                e2e.finish_block()

            if profile:
                torch.cuda.synchronize()
                diffusion_end.record()
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                print(f"diffusion_time: {diffusion_time}", flush=True)
                fps = video.shape[1]*1000/ diffusion_time
                print(f"  - FPS: {fps:.2f}")

        if getattr(self, "kv_retention", None) is not None and self.kv_retention.enabled:
            print(f"[CR] {self.kv_retention.summary()}", flush=True)
        _tich_summary(self)
        if getattr(self, "e2e", None) is not None and self.e2e.enabled:
            print(f"[E2E] {self.e2e.summary()}", flush=True)
        _rope_summary(self)
        if self.dump_latents:
            torch.save(
                {
                    "blocks": dumped_blocks,
                    "output": output.detach().float().cpu(),
                },
                self.dump_latents,
            )
            print(f"[DIT] dumped latents {self.dump_latents}", flush=True)
        if return_latents:
            return output
        else:
            return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache


class CausalInferenceStreamingPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            vae_decoder=None,
            generator=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880 # 1590 # HW/4

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        from wan.memory.kv_retention import attach_planner, build_planner
        self.kv_retention = build_planner(args)
        if self.kv_retention.enabled:
            attach_planner(self.generator.model, self.kv_retention)
            print(
                f"[CR] memory_policy={self.kv_retention.policy} "
                f"sink={self.kv_retention.sink_frames} recent={self.kv_retention.recent_frames}",
                flush=True,
            )
        self.cond_hoist_enabled = bool(getattr(args, "enable_cond_hoist", False))
        from demo_utils.e2e_profile import E2EProfile
        self.e2e = E2EProfile(enabled=bool(getattr(args, "e2e_profile", False)))
        if self.e2e.enabled:
            print("[E2E] per-block CUDA-event split dit/vae/other", flush=True)
        from wan.modules.causal_rope import attach_causal_rope
        attach_causal_rope(
            self.generator.model,
            mode=str(getattr(args, "rope_mode", "fp64")),
            profile=bool(getattr(args, "dit_profile", False)),
        )
        from wan.modules.ffn_gemm import attach_gemm_timer
        attach_gemm_timer(
            self.generator.model,
            enabled=bool(getattr(args, "dit_profile", False)),
        )
        self.dump_latents = str(getattr(args, "dump_latents", "") or "")

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        output_folder = None,
        name = None,
        mode = 'universal'
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        # Set up profiling if requested
        self.kv_cache1=self.kv_cache_keyboard=self.kv_cache_mouse=self.crossattn_cache=None
        _cr_reset(self)
        _tich_configure(self, conditional_dict)
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                block_cond = cond_current(
                    conditional_dict, current_start_frame, self.num_frame_per_block, replace=True)
                _cr_observe(self, current_start_frame, self.num_frame_per_block, conditional_dict)
                _tich_begin(self, current_start_frame, block_cond)
                _tich_generate(
                    self,
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=block_cond,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                _tich_end(self, expected_forwards=1)
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            current_actions = get_current_action(mode=mode)
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            _cr_observe(self, current_start_frame, current_num_frames, new_act)
            _tich_begin(self, current_start_frame, new_act)
            # Step 3.1: Spatial denoising loop

            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = _tich_generate(
                        self,
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = _tich_generate(
                        self,
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            _tich_generate(
                self,
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            _tich_end(self)

            # Step 3.4: update the start and end frame indices
            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            mouse_icon = 'assets/images/mouse.png'
            if mode != 'templerun':
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                )
            else:
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
                )
            process_video(video.astype(np.uint8), output_folder+f'/{name}_current.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            current_start_frame += current_num_frames

            if input("Continue? (Press `n` to break)").strip() == "n":
                break
                
        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        if mode != 'templerun':
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
            )
        else:
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
            )
        process_video(video.astype(np.uint8), output_folder+f'/{name}_icon.mp4', config, mouse_icon, mouse_scale=0.1, mode=mode)
        process_video(video.astype(np.uint8), output_folder+f'/{name}.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)

        if getattr(self, "kv_retention", None) is not None and self.kv_retention.enabled:
            print(f"[CR] {self.kv_retention.summary()}", flush=True)
        _tich_summary(self)
        if getattr(self, "e2e", None) is not None and self.e2e.enabled:
            print(f"[E2E] {self.e2e.summary()}", flush=True)

        if return_latents:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
