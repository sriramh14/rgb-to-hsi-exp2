"""DiffIR RGB-to-HSI with MST++ spectral transformer blocks.

This file keeps the DiffIR training interface/backbone:
    Stage 1: TeacherPriorEncoder(rgb, hsi_gt) -> compact oracle prior -> G(rgb, prior)
    Stage 2: RGBConditionEncoder(rgb) + compact-prior diffusion -> G(rgb, sampled_prior)

The reconstruction generator G no longer uses DIRFormer attention blocks.  It keeps
DiffIR's encoder/decoder scaffold, PixelUnshuffle(4), skip connections, compact
prior conditioning, and RGB->HSI residual skip, but every reconstruction block is
replaced by an MST++-style MSAB spectral transformer block.

Expected tensors
----------------
RGB: [B, 3, H, W]
HSI: [B, num_bands, H, W]

H and W must be divisible by 32 because the generator uses PixelUnshuffle(4) and
three additional 2x downsampling stages.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out


@dataclass
class ModelConfig:
    num_bands: int = 31
    dim: int = 48
    num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    num_refinement_blocks: int = 4

    # Kept for checkpoint/config compatibility with the original DiffIR file.
    # The MST++ blocks derive their heads as block_dim // dim, matching MST++'s
    # fixed per-head spectral dimension convention.
    heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    ffn_expansion_factor: float = 2.66
    bias: bool = False
    layer_norm_type: str = "WithBias"

    prior_dim: int = 256
    n_encoder_res: int = 6
    n_denoise_res: int = 1
    timesteps: int = 4
    linear_start: float = 0.1
    linear_end: float = 0.99
    use_rgb_to_hsi_skip: bool = True

    # MST++ feed-forward expansion. Original MST++ uses mult=4.
    mst_ffn_mult: int = 4

    def to_dict(self) -> Dict:
        output = asdict(self)
        output["num_blocks"] = list(self.num_blocks)
        output["heads"] = list(self.heads)
        return output

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelConfig":
        values = dict(values)
        if "num_blocks" in values:
            values["num_blocks"] = tuple(values["num_blocks"])
        if "heads" in values:
            values["heads"] = tuple(values["heads"])
        return cls(**values)


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, kernel_size),
            nn.ReLU(inplace=True),
            default_conv(channels, channels, kernel_size),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """PixelShuffle upsampler used to reverse the initial PixelUnshuffle(4)."""

    def __init__(self, scale: int, n_feats: int, bias: bool = True):
        modules: List[nn.Module] = []
        if scale > 0 and (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                modules.extend(
                    [
                        default_conv(n_feats, 4 * n_feats, 3, bias=bias),
                        nn.PixelShuffle(2),
                    ]
                )
        elif scale == 3:
            modules.extend(
                [
                    default_conv(n_feats, 9 * n_feats, 3, bias=bias),
                    nn.PixelShuffle(3),
                ]
            )
        else:
            raise ValueError(f"Unsupported upsampling scale: {scale}")
        super().__init__(*modules)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# -----------------------------------------------------------------------------
# MST++ spectral transformer utilities
# -----------------------------------------------------------------------------


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"invalid mode {mode}")

    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class PriorFiLMBHWC(nn.Module):
    """Zero-initialized prior FiLM for [B,H,W,C] tensors.

    It starts as an identity transform, so the generator initially behaves like an
    unconditioned MST++ spectral transformer and then learns how the DiffIR prior
    should modulate every stage.
    """

    def __init__(self, channels: int, prior_dim: int):
        super().__init__()
        self.affine = nn.Linear(prior_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(prior).chunk(2, dim=1)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return x * (1.0 + gamma) + beta


class MSTSpectralMSA(nn.Module):
    """MST++ MS_MSA block: attention is computed across spectral/channel tokens."""

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be >= 1")
        if dim_head * heads != dim:
            raise ValueError(
                f"For MST++ MS_MSA in this generator, dim_head * heads must equal dim. "
                f"Got dim={dim}, dim_head={dim_head}, heads={heads}."
            )
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: [B,H,W,C], output: [B,H,W,C]."""
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (head d) -> b head n d", head=self.num_heads),
            (q_inp, k_inp, v_inp),
        )

        # Same channel/spectral attention pattern as MST++:
        # q,k,v: [B, heads, HW, dim_head] -> [B, heads, dim_head, HW]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 3, 1, 2).reshape(b, h * w, c)
        out_c = self.proj(out).view(b, h, w, c)

        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        out_p = out_p.permute(0, 2, 3, 1)
        return out_c + out_p


class MSTFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,H,W,C], output: [B,H,W,C]."""
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class PriorMSTBlock(nn.Module):
    """One prior-conditioned MST++ spectral transformer block."""

    def __init__(self, dim: int, dim_head: int, heads: int, prior_dim: int, ffn_mult: int):
        super().__init__()
        self.prior_attn = PriorFiLMBHWC(dim, prior_dim)
        self.attn = MSTSpectralMSA(dim=dim, dim_head=dim_head, heads=heads)
        self.norm = nn.LayerNorm(dim)
        self.prior_ffn = PriorFiLMBHWC(dim, prior_dim)
        self.ffn = MSTFeedForward(dim=dim, mult=ffn_mult)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prior_attn(x, prior))
        x = x + self.ffn(self.prior_ffn(self.norm(x), prior))
        return x


class PriorMSAB(nn.Module):
    """MST++ MSAB with DiffIR-style [x, prior] sequential interface."""

    def __init__(self, dim: int, base_dim: int, num_blocks: int, prior_dim: int, ffn_mult: int):
        super().__init__()
        if dim % base_dim != 0:
            raise ValueError(f"block dim={dim} must be divisible by base dim={base_dim}")
        heads = dim // base_dim
        dim_head = base_dim
        self.blocks = nn.ModuleList(
            [
                PriorMSTBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    prior_dim=prior_dim,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, values: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        x, prior = values
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x, prior)
        x = x.permute(0, 3, 1, 2)
        return [x, prior]


class MSTSpectralDiffIRGeneratorRGB2HSI(nn.Module):
    """DiffIR reconstruction generator with DIRFormer replaced by MST++ MSAB."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim

        self.input_transform = nn.PixelUnshuffle(4)
        self.patch_embed = OverlapPatchEmbed(3 * 16, dim, config.bias)

        def make_blocks(block_dim: int, count: int) -> PriorMSAB:
            return PriorMSAB(
                dim=block_dim,
                base_dim=dim,
                num_blocks=count,
                prior_dim=config.prior_dim,
                ffn_mult=config.mst_ffn_mult,
            )

        self.encoder_level1 = make_blocks(dim, config.num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(dim * 2, config.num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(dim * 4, config.num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = make_blocks(dim * 8, config.num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=config.bias)
        self.decoder_level3 = make_blocks(dim * 4, config.num_blocks[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=config.bias)
        self.decoder_level2 = make_blocks(dim * 2, config.num_blocks[1])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = make_blocks(dim * 2, config.num_blocks[0])
        self.refinement = make_blocks(dim * 2, config.num_refinement_blocks)

        self.tail = nn.Sequential(
            Upsampler(4, dim * 2, bias=True),
            default_conv(dim * 2, config.num_bands, 3),
        )

        self.rgb_to_hsi = (
            nn.Conv2d(3, config.num_bands, 1)
            if config.use_rgb_to_hsi_skip
            else None
        )
        self.apply(self._init_weights)
        self._zero_prior_films()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _zero_prior_films(self) -> None:
        # Keep prior modulation identity at initialization, even after apply().
        for module in self.modules():
            if isinstance(module, PriorFiLMBHWC):
                nn.init.zeros_(module.affine.weight)
                nn.init.zeros_(module.affine.bias)

    def forward(self, rgb: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")
        if prior.ndim != 2 or prior.shape[1] != self.config.prior_dim:
            raise ValueError(
                f"Expected prior [B,{self.config.prior_dim}], received {tuple(prior.shape)}"
            )
        h, w = rgb.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError("RGB height and width must be divisible by 32")

        x1 = self.patch_embed(self.input_transform(rgb))
        e1, _ = self.encoder_level1([x1, prior])
        e2, _ = self.encoder_level2([self.down1_2(e1), prior])
        e3, _ = self.encoder_level3([self.down2_3(e2), prior])
        latent, _ = self.latent([self.down3_4(e3), prior])

        d3 = self.up4_3(latent)
        d3 = self.reduce_chan_level3(torch.cat([d3, e3], dim=1))
        d3, _ = self.decoder_level3([d3, prior])

        d2 = self.up3_2(d3)
        d2 = self.reduce_chan_level2(torch.cat([d2, e2], dim=1))
        d2, _ = self.decoder_level2([d2, prior])

        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1, _ = self.decoder_level1([d1, prior])
        d1, _ = self.refinement([d1, prior])

        hsi_residual = self.tail(d1)
        if self.rgb_to_hsi is None:
            return hsi_residual
        return self.rgb_to_hsi(rgb) + hsi_residual


# Backward-compatible alias: anything expecting DIRformerRGB2HSI will now build
# the MST++ spectral-attention generator.
#DIRformerRGB2HSI = MSTSpectralDiffIRGeneratorRGB2HSI


# -----------------------------------------------------------------------------
# DiffIR prior encoders and compact-prior diffusion, kept unchanged in interface.
# -----------------------------------------------------------------------------


class PriorEncoderBase(nn.Module):
    def __init__(self, in_channels: int, config: ModelConfig):
        super().__init__()
        n_feats = 64
        modules: List[nn.Module] = [
            nn.Conv2d(in_channels, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        modules.extend([ResBlock(n_feats) for _ in range(config.n_encoder_res)])
        modules.extend(
            [
                nn.Conv2d(n_feats, n_feats * 2, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(n_feats * 2, n_feats * 2, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(n_feats * 2, n_feats * 4, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveAvgPool2d(1),
            ]
        )
        self.encoder = nn.Sequential(*modules)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, config.prior_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(config.prior_dim, config.prior_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(x).flatten(1))


class TeacherPriorEncoder(PriorEncoderBase):
    """Stage-1 oracle CPEN receiving both RGB and ground-truth HSI."""

    def __init__(self, config: ModelConfig):
        super().__init__(16 * (3 + config.num_bands), config)
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(4)

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        if rgb.shape[1] != 3:
            raise ValueError("TeacherPriorEncoder expects three RGB channels")
        if hsi.shape[1] != self.config.num_bands:
            raise ValueError(
                f"Expected {self.config.num_bands} HSI bands, got {hsi.shape[1]}"
            )
        if rgb.shape[-2:] != hsi.shape[-2:]:
            raise ValueError("RGB and HSI must have identical spatial dimensions")
        fused = torch.cat([self.unshuffle(rgb), self.unshuffle(hsi)], dim=1)
        return self.encode(fused)


class RGBConditionEncoder(PriorEncoderBase):
    """Stage-2 CPEN that forms the diffusion condition from RGB alone."""

    def __init__(self, config: ModelConfig):
        super().__init__(3 * 16, config)
        self.unshuffle = nn.PixelUnshuffle(4)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[1] != 3:
            raise ValueError("RGBConditionEncoder expects three RGB channels")
        return self.encode(self.unshuffle(rgb))


class ResidualMLP(nn.Module):
    """Named after the released DiffIR block; its public code applies a plain MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PriorDenoiser(nn.Module):
    """Small MLP denoiser retained from DiffIR, operating on the compact prior."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(config.prior_dim * 2 + 1, config.prior_dim),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        layers.extend([ResidualMLP(config.prior_dim) for _ in range(config.n_denoise_res)])
        self.net = nn.Sequential(*layers)
        self.max_period = float(config.timesteps * 10)

    def forward(
        self,
        noisy_prior: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        t = timestep.float().view(-1, 1) / self.max_period
        return self.net(torch.cat([condition, t, noisy_prior], dim=1))


def _extract(buffer: torch.Tensor, timestep: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timestep)
    return values.view(timestep.shape[0], *((1,) * (len(target_shape) - 1)))


class CompactPriorDiffusion(nn.Module):
    """Four-step x0-prediction diffusion over the compact reconstruction prior."""

    def __init__(self, config: ModelConfig, condition: RGBConditionEncoder, denoiser: PriorDenoiser):
        super().__init__()
        self.config = config
        self.condition = condition
        self.denoiser = denoiser

        betas = torch.linspace(
            math.sqrt(config.linear_start),
            math.sqrt(config.linear_end),
            config.timesteps,
            dtype=torch.float64,
        ).square().float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / torch.clamp(1.0 - alphas_cumprod, min=1e-20),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / torch.clamp(1.0 - alphas_cumprod, min=1e-20),
        )

    def q_sample(
        self,
        clean_prior: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_prior)
        return (
            _extract(self.sqrt_alphas_cumprod, timestep, clean_prior.shape) * clean_prior
            + _extract(self.sqrt_one_minus_alphas_cumprod, timestep, clean_prior.shape) * noise
        )

    def reverse_step(
        self,
        prior_t: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_x0 = self.denoiser(prior_t, timestep, condition)
        posterior_mean = (
            _extract(self.posterior_mean_coef1, timestep, prior_t.shape) * predicted_x0
            + _extract(self.posterior_mean_coef2, timestep, prior_t.shape) * prior_t
        )
        return posterior_mean, predicted_x0

    def forward_train(
        self,
        rgb: torch.Tensor,
        target_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch = rgb.shape[0]
        device = rgb.device
        final_t = torch.full(
            (batch,),
            self.config.timesteps - 1,
            device=device,
            dtype=torch.long,
        )
        prior = self.q_sample(target_prior, final_t)
        condition = self.condition(rgb)
        sequence: List[torch.Tensor] = []
        for step in reversed(range(self.config.timesteps)):
            timestep = torch.full((batch,), step, device=device, dtype=torch.long)
            prior, _ = self.reverse_step(prior, timestep, condition)
            sequence.append(prior)
        return prior, sequence

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = rgb.shape[0]
        device = rgb.device
        if initial_noise is None:
            prior = torch.randn(batch, self.config.prior_dim, device=device)
        else:
            expected = (batch, self.config.prior_dim)
            if tuple(initial_noise.shape) != expected:
                raise ValueError(f"Expected initial_noise {expected}, got {tuple(initial_noise.shape)}")
            prior = initial_noise
        condition = self.condition(rgb)
        for step in reversed(range(self.config.timesteps)):
            timestep = torch.full((batch,), step, device=device, dtype=torch.long)
            prior, _ = self.reverse_step(prior, timestep, condition)
        return prior


class DiffIRS1RGB2HSI(nn.Module):
    """Stage 1: ground-truth-assisted oracle prior + HSI reconstruction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.E = TeacherPriorEncoder(config)
        self.G = MSTSpectralDiffIRGeneratorRGB2HSI(config)

    def forward(self, rgb: torch.Tensor, hsi_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prior = self.E(rgb, hsi_gt)
        pred_hsi = self.G(rgb, prior)
        return pred_hsi, prior


class DiffIRS2RGB2HSI(nn.Module):
    """Stage 2: RGB-conditioned compact-prior diffusion + HSI reconstruction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.G = MSTSpectralDiffIRGeneratorRGB2HSI(config)
        condition = RGBConditionEncoder(config)
        denoiser = PriorDenoiser(config)
        self.diffusion = CompactPriorDiffusion(config, condition, denoiser)

    @property
    def condition(self) -> RGBConditionEncoder:
        return self.diffusion.condition

    @property
    def denoiser(self) -> PriorDenoiser:
        return self.diffusion.denoiser

    def initialize_generator_from_stage1(self, stage1: DiffIRS1RGB2HSI) -> None:
        self.G.load_state_dict(stage1.G.state_dict(), strict=True)

    def forward(
        self,
        rgb: torch.Tensor,
        target_prior: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ):
        if self.training:
            if target_prior is None:
                raise ValueError("Stage-2 training requires the frozen Stage-1 target prior")
            prior, prior_sequence = self.diffusion.forward_train(rgb, target_prior)
            pred_hsi = self.G(rgb, prior)
            return pred_hsi, prior_sequence

        prior = self.diffusion.sample(rgb, initial_noise=initial_noise)
        return self.G(rgb, prior)


def build_model(stage: int, config: ModelConfig) -> nn.Module:
    if stage == 1:
        return DiffIRS1RGB2HSI(config)
    if stage == 2:
        return DiffIRS2RGB2HSI(config)
    raise ValueError("stage must be 1 or 2")
