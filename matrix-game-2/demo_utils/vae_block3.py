from typing import List
from einops import rearrange
import torch
import torch.nn as nn

from wan.modules.vae import AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Upsample
from wan.modules.vae_cache import (
    feat_cache_ready,
    join_time,
    set_vae_prealloc,
    store_feat_cache,
)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.cache_t = 2

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -self.cache_t:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    store_feat_cache(feat_cache, idx, cache_x)
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    store_feat_cache(feat_cache, idx, cache_x)
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class VAEDecoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = VAEDecoder3d()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = 16
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
        # Graph is a no-go: no steady-state win, and replay used to return the
        # same ``_g_out`` buffer while ``forward`` appended aliases into
        # ``pieces``. CLI no longer exposes it. Clone remains if re-enabled.
        self.use_cuda_graph = False
        self.use_prealloc = True
        self._g = None
        self._g_in = None
        self._g_out = None
        self._g_warm = False
        self._g_failed = False
        self._scale = None

    def reset_cuda_graph(self):
        self._g = None
        self._g_in = None
        self._g_out = None
        self._g_warm = False
        self._g_failed = False

    def _decode_frame(self, x1: torch.Tensor, feat_cache):
        want_graph = (
            self.use_cuda_graph
            and (not self._g_failed)
            and x1.is_cuda
            and feat_cache_ready(feat_cache)
        )
        if not want_graph:
            return self.decoder(x1, feat_cache=feat_cache)
        if not self._g_warm:
            # One eager frame with a fully allocated cache so capture does not
            # allocate. This frame is still a real decode.
            self._g_warm = True
            return self.decoder(x1, feat_cache=feat_cache)
        if self._g is None:
            self._g_in = x1.clone()
            try:
                torch.cuda.synchronize()
                self._g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._g):
                    self._g_out, feat_cache = self.decoder(
                        self._g_in, feat_cache=feat_cache)
            except Exception as e:
                print(
                    f"[VAE] CUDA graph capture failed ({e}); "
                    "falling back to eager decode",
                    flush=True,
                )
                self._g = None
                self._g_failed = True
                self.use_cuda_graph = False
                return self.decoder(x1, feat_cache=feat_cache)
            # Independent storage: callers append this into ``pieces``.
            return self._g_out.clone(), feat_cache
        self._g_in.copy_(x1)
        self._g.replay()
        return self._g_out.clone(), feat_cache

    def forward(
            self,
            z: torch.Tensor,
            *feat_cache: List[torch.Tensor]
    ):
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        z = z.permute(0, 2, 1, 3, 4)
        feat_cache = list(feat_cache)

        device, dtype = z.device, z.dtype
        if (
            self._scale is None
            or self._scale[0].device != device
            or self._scale[0].dtype != dtype
        ):
            self._scale = (
                self.mean.to(device=device, dtype=dtype),
                (1.0 / self.std).to(device=device, dtype=dtype),
            )
        z = z / self._scale[1].view(1, self.z_dim, 1, 1, 1) + self._scale[0].view(
            1, self.z_dim, 1, 1, 1)
        iter_ = z.shape[2]
        x = self.conv2(z)
        if self.use_prealloc:
            pieces = []
            for i in range(iter_):
                piece, feat_cache = self._decode_frame(
                    x[:, :, i:i + 1, :, :], feat_cache)
                pieces.append(piece)
            out = join_time(pieces)
        else:
            out = None
            for i in range(iter_):
                piece, feat_cache = self._decode_frame(
                    x[:, :, i:i + 1, :, :], feat_cache)
                out = piece if out is None else torch.cat([out, piece], 2)
        out = out.float().clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        out = out.permute(0, 2, 1, 3, 4)
        return out, feat_cache


def configure_vae_decoder(decoder, *, legacy: bool = False, compile_vae: bool = False):
    """A/B/C stream config. CUDA Graph is not a CLI option."""
    decoder.use_cuda_graph = False
    decoder.use_prealloc = not legacy
    set_vae_prealloc(not legacy)
    if compile_vae:
        decoder.compile(mode="max-autotune-no-cudagraphs")
    kind = "legacy" if legacy else "prealloc"
    print(
        f"[VAE] stream_opt={kind} compile={bool(compile_vae)} cuda_graph=disabled",
        flush=True,
    )
    return decoder


class VAEDecoder3d(nn.Module):
    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.cache_t = 2
        self.decoder_conv_num = 32

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(
            self,
            x: torch.Tensor,
            feat_cache: List[torch.Tensor]
    ):
        feat_idx = [0]

        # conv1
        idx = feat_idx[0]
        cache_x = x[:, :, -self.cache_t:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device), cache_x
            ],
                dim=2)
        x = self.conv1(x, feat_cache[idx])
        store_feat_cache(feat_cache, idx, cache_x)
        feat_idx[0] += 1

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -self.cache_t:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                store_feat_cache(feat_cache, idx, cache_x)
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache
