"""DiffIR RGB-to-HSI with strict MST++ backbone and spatial spectral-prior conditioning.

This file is based on the strict MST++ replacement version:

    RGB -> conv_in -> [MST U-Net stage] x 3 -> conv_out -> + feature skip

Main changes in this version:
    1. `use_prior_conditioning=True` by default.
    2. CPEN no longer uses AdaptiveAvgPool2d or Linear layers.
    3. CPEN outputs a spatial spectral prior map instead of a compact vector.
    4. Diffusion is applied to the spatial spectral prior map.
    5. MST blocks use convolutional spatial FiLM from the spectral prior map.
    6. Stage 1 passes the real oracle prior into G; it no longer passes zero prior.
    7. Stage 1 decomposes its prior into an RGB base and an HSI oracle residual.
    8. Stage 2 predicts only the residual around its RGB condition prior.
    9. The predicted prior returned to the training script remains the full prior.
    10. The generator is frozen in Stage 2 and remains in evaluation mode.

Public interface is kept:
    Stage 1: DiffIRS1RGB2HSI(rgb, hsi_gt) -> pred_hsi, teacher_prior
    Stage 2: DiffIRS2RGB2HSI(rgb, target_prior) -> pred_hsi, prior_sequence

Stage 2 internally models the residual:
    residual_target = teacher_prior - stop_gradient(rgb_condition_prior)
    predicted_prior = rgb_condition_prior + predicted_residual

Tensor shapes:
    RGB:        [B, 3, H, W]
    HSI:        [B, num_bands, H, W]
    prior map:  [B, num_bands, ceil(H/4), ceil(W/4)] by default

The reconstruction generator itself keeps the strict MST++ full-resolution
feature path; the low-resolution prior map is resized internally wherever it is
used for conditioning.
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # RGB -> 31-band HSI by default.
    num_bands: int = 31

    # Strict MST++ feature-space setting: conv_in maps RGB to 31 channels and
    # conv_out maps 31 channels to 31 bands; feature_skip is added afterward.
    dim: int = 31

    # Common MST++ setting: MST(dim=31, stage=2, num_blocks=[1,1,1]).
    num_blocks: Tuple[int, ...] = (1, 1, 1)
    mst_stages: int = 3
    mst_stage_depth: int = 2
    mst_ffn_mult: int = 4
    bias: bool = False

    # Reference MST_Plus_Plus pads to multiple of 8.
    pad_multiple: int = 8

    # Prior is active by default in this file.
    use_prior_conditioning: bool = True

    # Spectral-prior settings.
    # Lightweight prior encoders downsample by this factor using strided convs.
    prior_downsample_factor: int = 4
    prior_feat_dim: int = 32

    # Lightweight base/residual prior design.
    rgb_prior_hidden: int = 32
    oracle_residual_hidden: int = 32
    oracle_scale_init: float = 0.50
    oracle_scale_max: float = 1.00
    oracle_drop_prob: float = 0.05

    # Diffusion schedule. Cosine is recommended for a small number of steps
    # because its final cumulative signal approaches zero even at 25 steps.
    diffusion_schedule: str = "cosine"
    cosine_s: float = 0.008

    # Legacy fields retained for old config/checkpoint compatibility.
    # The lightweight fusion implementation does not use them.
    projection_hidden_dim: int = 64
    projection_res_blocks: int = 3
    fusion_compact_dim: int = 64
    fusion_hsi_drop_prob: float = 0.10

    use_spectral_prior_output_skip: bool = True
    spectral_prior_output_scale_init: float = 1.0

    # Diffusion settings. prior_dim is kept only for compatibility with older
    # config/checkpoints; the new prior is not [B, prior_dim].
    prior_dim: int = 256
    n_encoder_res: int = 6

    #Old value is 2
    n_denoise_res: int = 4

    #Change : Slightly less aggressive noise schedule
    timesteps: int = 25                           #was 4 originally
    linear_start: float = 1e-4
    linear_end: float = 2e-2

    # Kept only for checkpoint/config compatibility with earlier files.
    heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    ffn_expansion_factor: float = 2.66
    layer_norm_type: str = "WithBias"
    num_refinement_blocks: int = 0
    use_rgb_to_hsi_skip: bool = False

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


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


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


class ConvResBlock(nn.Module):
    """Residual conv block for CPEN and spatial-prior denoiser."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


def _pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, [0, pad_w, 0, pad_h], mode=mode), (pad_h, pad_w)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# MST++ initialization utilities
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


# -----------------------------------------------------------------------------
# MST++ spectral transformer blocks with spatial spectral-prior conditioning
# -----------------------------------------------------------------------------


class SpatialPriorFiLMBHWC(nn.Module):
    """Spatial FiLM for [B,H,W,C] tensors using a spectral prior map.

    x:     [B, H, W, C]
    prior: [B, num_bands, Hp, Wp]

    The prior map is resized to the feature resolution and converted to per-pixel
    gamma/beta using 1x1 convolution. The modulation starts as identity.
    """

    def __init__(self, prior_channels: int, channels: int):
        super().__init__()
        self.affine = nn.Conv2d(prior_channels, channels * 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[1:3]
        prior_resized = F.interpolate(prior, size=(h, w), mode="bicubic", align_corners=False)   #Mode was bilinear originally
        gamma_beta = self.affine(prior_resized).permute(0, 2, 3, 1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class MSTSpectralMSA(nn.Module):
    """MST++ MS_MSA block.

    Input/output layout: [B,H,W,C].
    Attention is computed across spectral/channel tokens following MST++:
        [B,HW,C] -> Q/K/V -> [B,heads,HW,dim_head]
        transpose -> [B,heads,dim_head,HW]
        attention = K @ Q^T
    """

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be >= 1")
        if dim_head * heads != dim:
            raise ValueError(
                f"dim_head * heads must equal dim. Got dim={dim}, "
                f"dim_head={dim_head}, heads={heads}."
            )

        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v = map(
            lambda t: rearrange(t, "b n (head d) -> b head n d", head=self.num_heads),
            (q_inp, k_inp, v_inp),
        )

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
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
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSTBlock(nn.Module):
    """One MST++ block with optional spatial spectral-prior conditioning."""

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        ffn_mult: int,
        prior_channels: int,
        use_prior_conditioning: bool,
    ):
        super().__init__()
        self.use_prior_conditioning = use_prior_conditioning
        self.attn = MSTSpectralMSA(dim=dim, dim_head=dim_head, heads=heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = MSTFeedForward(dim=dim, mult=ffn_mult)

        if use_prior_conditioning:
            self.prior_attn = SpatialPriorFiLMBHWC(prior_channels, dim)
            self.prior_ffn = SpatialPriorFiLMBHWC(prior_channels, dim)
        else:
            self.prior_attn = None
            self.prior_ffn = None

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        attn_in = x
        if self.use_prior_conditioning:
            if prior is None:
                raise ValueError("prior must be provided when use_prior_conditioning=True")
            attn_in = self.prior_attn(attn_in, prior)

        x = x + self.attn(attn_in)

        ffn_in = self.norm(x)
        if self.use_prior_conditioning:
            ffn_in = self.prior_ffn(ffn_in, prior)

        x = x + self.ffn(ffn_in)
        return x


class MSTMSAB(nn.Module):
    """MST++ MSAB with DiffIR-style [x, prior] interface."""

    def __init__(
        self,
        dim: int,
        base_dim: int,
        num_blocks: int,
        prior_channels: int,
        ffn_mult: int,
        use_prior_conditioning: bool,
    ):
        super().__init__()
        if dim % base_dim != 0:
            raise ValueError(f"block dim={dim} must be divisible by base_dim={base_dim}")
        heads = dim // base_dim
        dim_head = base_dim
        self.blocks = nn.ModuleList(
            [
                MSTBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ffn_mult=ffn_mult,
                    prior_channels=prior_channels,
                    use_prior_conditioning=use_prior_conditioning,
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


class MSTUNetStage(nn.Module):
    """One internal MST++ stage.

    Mirrors the reference MST module:
        embedding -> encoder MSAB/downsample -> bottleneck -> decoder/fusion/MSAB
        -> mapping -> feature-space residual add
    """

    def __init__(
        self,
        dim: int,
        base_dim: int,
        prior_channels: int,
        stage_depth: int,
        num_blocks: Tuple[int, ...],
        ffn_mult: int,
        bias: bool,
        use_prior_conditioning: bool,
    ):
        super().__init__()
        if stage_depth < 1:
            raise ValueError("stage_depth must be >= 1")
        if len(num_blocks) < stage_depth + 1:
            raise ValueError(
                f"num_blocks must contain at least stage_depth+1 entries. "
                f"Got len(num_blocks)={len(num_blocks)}, stage_depth={stage_depth}."
            )

        self.stage_depth = stage_depth
        self.embedding = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.encoder_layers = nn.ModuleList([])

        dim_stage = dim
        for i in range(stage_depth):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        MSTMSAB(
                            dim=dim_stage,
                            base_dim=base_dim,
                            num_blocks=num_blocks[i],
                            prior_channels=prior_channels,
                            ffn_mult=ffn_mult,
                            use_prior_conditioning=use_prior_conditioning,
                        ),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=bias),
                    ]
                )
            )
            dim_stage *= 2

        self.bottleneck = MSTMSAB(
            dim=dim_stage,
            base_dim=base_dim,
            num_blocks=num_blocks[stage_depth],
            prior_channels=prior_channels,
            ffn_mult=ffn_mult,
            use_prior_conditioning=use_prior_conditioning,
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(stage_depth):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_stage,
                            dim_stage // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                            bias=bias,
                        ),
                        nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=bias),
                        MSTMSAB(
                            dim=dim_stage // 2,
                            base_dim=base_dim,
                            num_blocks=num_blocks[stage_depth - 1 - i],
                            prior_channels=prior_channels,
                            ffn_mult=ffn_mult,
                            use_prior_conditioning=use_prior_conditioning,
                        ),
                    ]
                )
            )
            dim_stage //= 2

        self.mapping = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        identity = x
        fea = self.embedding(x)
        fea_encoder: List[torch.Tensor] = []

        for msab, downsample in self.encoder_layers:
            fea, _ = msab([fea, prior])
            fea_encoder.append(fea)
            fea = downsample(fea)

        fea, _ = self.bottleneck([fea, prior])

        for i, (upsample, fusion, msab) in enumerate(self.decoder_layers):
            fea = upsample(fea)
            skip = fea_encoder[self.stage_depth - 1 - i]
            fea = fusion(torch.cat([fea, skip], dim=1))
            fea, _ = msab([fea, prior])

        return identity + self.mapping(fea)


# -----------------------------------------------------------------------------
# Generator: strict MST++ backbone + active spectral prior
# -----------------------------------------------------------------------------


class MSTSpectralDiffIRGeneratorRGB2HSI(nn.Module):
    """RGB-to-HSI generator with strict MST++ backbone and spectral-prior FiLM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim

        if dim != config.num_bands:
            raise ValueError(
                "Strict MST++ feature skip requires config.dim == config.num_bands. "
                f"Got dim={dim}, num_bands={config.num_bands}. Set dim=31 for RGB->31-band HSI."
            )
        if config.mst_stages < 1:
            raise ValueError("config.mst_stages must be >= 1")
        if config.mst_stage_depth < 1:
            raise ValueError("config.mst_stage_depth must be >= 1")
        if len(config.num_blocks) < config.mst_stage_depth + 1:
            raise ValueError(
                f"config.num_blocks must have at least config.mst_stage_depth+1 entries. "
                f"Got num_blocks={config.num_blocks}, mst_stage_depth={config.mst_stage_depth}."
            )

        self.conv_in = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.body = nn.ModuleList(
            [
                MSTUNetStage(
                    dim=dim,
                    base_dim=dim,
                    prior_channels=config.num_bands,
                    stage_depth=config.mst_stage_depth,
                    num_blocks=config.num_blocks,
                    ffn_mult=config.mst_ffn_mult,
                    bias=config.bias,
                    use_prior_conditioning=config.use_prior_conditioning,
                )
                for _ in range(config.mst_stages)
            ]
        )
        self.conv_out = nn.Conv2d(dim, config.num_bands, 3, 1, 1, bias=False)

        if config.use_spectral_prior_output_skip:
            self.prior_output_scale = nn.Parameter(
                torch.tensor(float(config.spectral_prior_output_scale_init))
            )
        else:
            self.register_parameter("prior_output_scale", None)

        self.apply(self._init_weights)
        self._zero_prior_films()

    def _init_weights(self, m: nn.Module) -> None:
        # Same init policy as MST++ for transformer Linear/LayerNorm layers.
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _zero_prior_films(self) -> None:
        for module in self.modules():
            if isinstance(module, SpatialPriorFiLMBHWC):
                nn.init.zeros_(module.affine.weight)
                nn.init.zeros_(module.affine.bias)

    def _validate_prior(self, rgb: torch.Tensor, prior: torch.Tensor | None) -> None:
        if not self.config.use_prior_conditioning:
            return
        if prior is None:
            raise ValueError("prior is required when use_prior_conditioning=True")
        if prior.ndim != 4 or prior.shape[1] != self.config.num_bands:
            raise ValueError(
                f"Expected spectral prior [B,{self.config.num_bands},Hp,Wp], "
                f"received {tuple(prior.shape)}"
            )
        expected_h = _ceil_div(rgb.shape[-2], self.config.prior_downsample_factor)
        expected_w = _ceil_div(rgb.shape[-1], self.config.prior_downsample_factor)
        if prior.shape[-2:] != (expected_h, expected_w):
            raise ValueError(
                f"Expected prior spatial size {(expected_h, expected_w)} for RGB size "
                f"{tuple(rgb.shape[-2:])} and prior_downsample_factor="
                f"{self.config.prior_downsample_factor}; got {tuple(prior.shape[-2:])}."
            )

    def forward(self, rgb: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")
        self._validate_prior(rgb, prior)

        h_inp, w_inp = rgb.shape[-2:]
        rgb_pad, _ = _pad_to_multiple(rgb, self.config.pad_multiple, mode="reflect")

        x = self.conv_in(rgb_pad)
        feature_skip = x

        for stage in self.body:
            x = stage(x, prior)

        out = self.conv_out(x) + feature_skip

        if self.config.use_prior_conditioning and self.config.use_spectral_prior_output_skip:
            prior_full = F.interpolate(prior, size=out.shape[-2:], mode="bilinear", align_corners=False)
            out = out + self.prior_output_scale * prior_full

        return out[:, :, :h_inp, :w_inp]


# Backward-compatible alias if old training code imports this name.
DIRformerRGB2HSI = MSTSpectralDiffIRGeneratorRGB2HSI


# -----------------------------------------------------------------------------
# Spatial spectral-prior CPEN and diffusion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Lightweight decomposed prior encoders and residual diffusion
# -----------------------------------------------------------------------------


class RGBToSpectralProjection(nn.Module):
    """Small RGB-to-31-channel projection used by the RGB base-prior branch."""

    def __init__(self, num_bands: int = 31):
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        self.num_bands = num_bands
        self.projection = nn.Conv2d(
            3,
            num_bands,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "RGBToSpectralProjection expects [B,3,H,W], "
                f"received {tuple(rgb.shape)}"
            )
        return self.projection(rgb)


class LightweightPriorEncoder(nn.Module):
    """Two-stage learned downsampler that maps a full-resolution map to H/4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        downsample_factor: int = 4,
    ):
        super().__init__()
        if downsample_factor != 4:
            raise ValueError("LightweightPriorEncoder currently expects factor=4")
        if min(in_channels, out_channels, hidden_channels) < 1:
            raise ValueError("all channel counts must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        return self.encoder(x)


class RGBBasePriorEncoder(nn.Module):
    """RGB -> projected RGB -> compact RGB base prior."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rgb_projection = RGBToSpectralProjection(config.num_bands)
        self.prior_encoder = LightweightPriorEncoder(
            in_channels=config.num_bands,
            out_channels=config.num_bands,
            hidden_channels=config.rgb_prior_hidden,
            downsample_factor=config.prior_downsample_factor,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        return_projection: bool = False,
    ):
        projected_rgb = self.rgb_projection(rgb)
        rgb_prior = self.prior_encoder(projected_rgb)
        if return_projection:
            return projected_rgb, rgb_prior
        return rgb_prior


class OracleResidualEncoder(nn.Module):
    """Extract the HSI-only correction relative to the projected RGB.

    Input channels are [projected_rgb, hsi - projected_rgb]. The explicit error
    channel makes the branch focus on information missing from RGB rather than
    rebuilding the complete teacher prior independently.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = LightweightPriorEncoder(
            in_channels=config.num_bands * 2,
            out_channels=config.num_bands,
            hidden_channels=config.oracle_residual_hidden,
            downsample_factor=config.prior_downsample_factor,
        )

    def forward(
        self,
        projected_rgb: torch.Tensor,
        hsi: torch.Tensor,
    ) -> torch.Tensor:
        if projected_rgb.shape != hsi.shape:
            raise ValueError(
                "projected_rgb and hsi must have the same shape; got "
                f"{tuple(projected_rgb.shape)} and {tuple(hsi.shape)}"
            )
        spectral_error = hsi - projected_rgb
        oracle_input = torch.cat([projected_rgb, spectral_error], dim=1)
        return self.encoder(oracle_input)


class TeacherPriorEncoder(nn.Module):
    """Stage-1 teacher prior with explicit recoverable decomposition.

    teacher_prior = rgb_base_prior + oracle_scale * oracle_residual

    The RGB base is directly available to Stage 2. The oracle residual contains
    the extra information obtained from ground-truth HSI.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rgb_encoder = RGBBasePriorEncoder(config)
        self.oracle_encoder = OracleResidualEncoder(config)

        if not 0.0 < config.oracle_scale_init < config.oracle_scale_max:
            raise ValueError(
                "oracle_scale_init must lie strictly between 0 and oracle_scale_max"
            )
        ratio = config.oracle_scale_init / config.oracle_scale_max
        self.oracle_scale_logit = nn.Parameter(
            torch.logit(torch.tensor(ratio, dtype=torch.float32))
        )

    def get_oracle_scale(self) -> torch.Tensor:
        return self.config.oracle_scale_max * torch.sigmoid(
            self.oracle_scale_logit
        )

    def _apply_oracle_dropout(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        p = float(self.config.oracle_drop_prob)
        if not self.training or p <= 0.0:
            return residual
        if not 0.0 <= p < 1.0:
            raise ValueError("oracle_drop_prob must lie in [0,1)")
        keep = (
            torch.rand(
                residual.shape[0], 1, 1, 1,
                device=residual.device,
            ) >= p
        ).to(residual.dtype)
        return residual * keep

    def project_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_encoder.rgb_projection(rgb)

    def forward_rgb_only(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_encoder(rgb)

    def forward_components(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if hsi.ndim != 4 or hsi.shape[1] != self.config.num_bands:
            raise ValueError(
                f"Expected HSI [B,{self.config.num_bands},H,W], "
                f"got {tuple(hsi.shape)}"
            )
        if rgb.shape[0] != hsi.shape[0] or rgb.shape[-2:] != hsi.shape[-2:]:
            raise ValueError("RGB and HSI batch/spatial sizes must match")

        projected_rgb, rgb_prior = self.rgb_encoder(
            rgb,
            return_projection=True,
        )
        raw_oracle_residual = self.oracle_encoder(projected_rgb, hsi)
        oracle_residual = self._apply_oracle_dropout(raw_oracle_residual)
        oracle_scale = self.get_oracle_scale()
        scaled_oracle_residual = oracle_scale * oracle_residual
        teacher_prior = rgb_prior + scaled_oracle_residual

        return {
            "teacher_prior": teacher_prior,
            "rgb_prior": rgb_prior,
            "raw_oracle_residual": raw_oracle_residual,
            "oracle_residual": oracle_residual,
            "scaled_oracle_residual": scaled_oracle_residual,
            "projected_rgb": projected_rgb,
            "oracle_scale": oracle_scale,
        }

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        return self.forward_components(rgb, hsi)["teacher_prior"]


class RGBConditionEncoder(nn.Module):
    """Stage-2 RGB-only base prior used as an additional diffusion input."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rgb_encoder = RGBBasePriorEncoder(config)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_encoder(rgb)

    def initialize_from_teacher(self, teacher: TeacherPriorEncoder) -> None:
        self.rgb_encoder.load_state_dict(
            teacher.rgb_encoder.state_dict(),
            strict=True,
        )


class ResidualPriorDenoiser(nn.Module):
    """Predict the clean missing residual conditioned on the RGB base prior."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        bands = config.num_bands
        feat = max(config.prior_feat_dim, bands)

        # noisy residual + RGB condition prior + scalar timestep map
        self.stem = nn.Sequential(
            nn.Conv2d(bands * 2 + 1, feat, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            *[ConvResBlock(feat) for _ in range(config.n_denoise_res)]
        )
        self.head = nn.Conv2d(feat, bands, 3, padding=1)
        self.max_period = float(max(config.timesteps - 1, 1))

        # Start near zero residual prediction. The RGB condition already gives
        # a useful base prior, so early Stage-2 outputs stay stable.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        condition_prior: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_residual.shape != condition_prior.shape:
            raise ValueError(
                "noisy_residual and condition_prior must have identical shapes; "
                f"got {tuple(noisy_residual.shape)} and {tuple(condition_prior.shape)}"
            )
        b, _, h, w = noisy_residual.shape
        t_map = (
            timestep.float().view(-1, 1, 1, 1) / self.max_period
        ).expand(b, 1, h, w)
        x = torch.cat([condition_prior, t_map, noisy_residual], dim=1)
        return self.head(self.body(self.stem(x)))


def _extract(
    buffer: torch.Tensor,
    timestep: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    values = buffer.gather(0, timestep)
    return values.view(timestep.shape[0], *((1,) * (len(target_shape) - 1)))


def _cosine_betas(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from cumulative alpha values, clipped for stability."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-8, max=0.999).float()


class ResidualSpectralPriorDiffusion(nn.Module):
    """Conditional diffusion of only the missing teacher-prior residual.

    During training:
        condition = C(rgb)
        target_residual = teacher_prior - stop_gradient(condition)
        predicted_residual = D(q(target_residual, t), t, condition)
        predicted_prior = condition + predicted_residual

    Returning the full predicted prior keeps the original training-script API.
    """

    def __init__(
        self,
        config: ModelConfig,
        condition: RGBConditionEncoder,
        denoiser: ResidualPriorDenoiser,
    ):
        super().__init__()
        self.config = config
        self.condition = condition
        self.denoiser = denoiser

        schedule = config.diffusion_schedule.lower()
        if schedule == "cosine":
            betas = _cosine_betas(config.timesteps, config.cosine_s)
        elif schedule == "linear":
            betas = torch.linspace(
                config.linear_start,
                config.linear_end,
                config.timesteps,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                "diffusion_schedule must be either 'cosine' or 'linear'"
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]]
        )

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod),
        )
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
        clean_residual: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_residual)
        return (
            _extract(
                self.sqrt_alphas_cumprod,
                timestep,
                clean_residual.shape,
            ) * clean_residual
            + _extract(
                self.sqrt_one_minus_alphas_cumprod,
                timestep,
                clean_residual.shape,
            ) * noise
        )

    def reverse_step(
        self,
        residual_t: torch.Tensor,
        timestep: torch.Tensor,
        condition_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_residual_x0 = self.denoiser(
            residual_t,
            timestep,
            condition_prior,
        )
        posterior_mean = (
            _extract(
                self.posterior_mean_coef1,
                timestep,
                residual_t.shape,
            ) * predicted_residual_x0
            + _extract(
                self.posterior_mean_coef2,
                timestep,
                residual_t.shape,
            ) * residual_t
        )
        return posterior_mean, predicted_residual_x0

    def forward_train(
        self,
        rgb: torch.Tensor,
        target_prior: torch.Tensor,
        target_residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        if target_prior.ndim != 4:
            raise ValueError(
                "target_prior must be [B,C,H,W], got "
                f"{tuple(target_prior.shape)}"
            )

        condition_prior = self.condition(rgb)
        if condition_prior.shape != target_prior.shape:
            raise ValueError(
                "Condition and teacher prior shapes must match; got "
                f"{tuple(condition_prior.shape)} and {tuple(target_prior.shape)}"
            )

        # Default compatibility path: the residual is defined around the
        # current Stage-2 condition. Detaching the condition from the target
        # prevents the target itself from moving through this subtraction.
        if target_residual is None:
            target_residual = target_prior - condition_prior.detach()
        elif target_residual.shape != target_prior.shape:
            raise ValueError(
                "target_residual must match target_prior shape; got "
                f"{tuple(target_residual.shape)}"
            )

        batch = rgb.shape[0]
        timestep = torch.randint(
            0,
            self.config.timesteps,
            (batch,),
            device=rgb.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(target_residual)
        noisy_residual = self.q_sample(
            target_residual,
            timestep,
            noise,
        )
        predicted_residual = self.denoiser(
            noisy_residual,
            timestep,
            condition_prior,
        )
        predicted_prior = condition_prior + predicted_residual

        details = {
            "condition_prior": condition_prior,
            "target_residual": target_residual,
            "noisy_residual": noisy_residual,
            "predicted_residual": predicted_residual,
            "predicted_prior": predicted_prior,
            "timestep": timestep,
        }
        # Sequence contains full priors so old prior-loss code remains valid.
        return predicted_prior, [predicted_prior], details

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        initial_noise: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        condition_prior = self.condition(rgb)
        expected = tuple(condition_prior.shape)
        if initial_noise is None:
            residual = torch.randn_like(condition_prior)
        else:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"Expected initial_noise {expected}, got {tuple(initial_noise.shape)}"
                )
            residual = initial_noise

        for step in reversed(range(self.config.timesteps)):
            timestep = torch.full(
                (rgb.shape[0],),
                step,
                device=rgb.device,
                dtype=torch.long,
            )
            residual, _ = self.reverse_step(
                residual,
                timestep,
                condition_prior,
            )

        predicted_prior = condition_prior + residual
        if return_details:
            return predicted_prior, {
                "condition_prior": condition_prior,
                "predicted_residual": residual,
                "predicted_prior": predicted_prior,
            }
        return predicted_prior


# -----------------------------------------------------------------------------
# Stage wrappers
# -----------------------------------------------------------------------------


class DiffIRS1RGB2HSI(nn.Module):
    """Stage 1: RGB base prior + HSI oracle residual -> teacher prior."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.E = TeacherPriorEncoder(config)
        self.G = MSTSpectralDiffIRGeneratorRGB2HSI(config)

    def forward(
        self,
        rgb: torch.Tensor,
        hsi_gt: torch.Tensor,
        return_components: bool = False,
    ):
        components = self.E.forward_components(rgb, hsi_gt)
        teacher_prior = components["teacher_prior"]
        pred_hsi = self.G(rgb, teacher_prior)
        if return_components:
            return pred_hsi, teacher_prior, components
        return pred_hsi, teacher_prior


class DiffIRS2RGB2HSI(nn.Module):
    """Stage 2: RGB condition prior + diffusion-generated residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.G = MSTSpectralDiffIRGeneratorRGB2HSI(config)
        condition = RGBConditionEncoder(config)
        denoiser = ResidualPriorDenoiser(config)
        self.diffusion = ResidualSpectralPriorDiffusion(
            config,
            condition,
            denoiser,
        )
        self.freeze_generator()

    @property
    def condition(self) -> RGBConditionEncoder:
        return self.diffusion.condition

    @property
    def denoiser(self) -> ResidualPriorDenoiser:
        return self.diffusion.denoiser

    def freeze_generator(self) -> None:
        self.G.requires_grad_(False)
        self.G.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent train() would otherwise put the frozen generator in train mode.
        self.G.eval()
        return self

    def initialize_from_stage1(self, stage1: DiffIRS1RGB2HSI) -> None:
        self.G.load_state_dict(stage1.G.state_dict(), strict=True)
        self.condition.initialize_from_teacher(stage1.E)
        self.freeze_generator()

    # Old name retained for existing training scripts.
    def initialize_generator_from_stage1(
        self,
        stage1: DiffIRS1RGB2HSI,
    ) -> None:
        self.initialize_from_stage1(stage1)

    def forward(
        self,
        rgb: torch.Tensor,
        target_prior: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        target_residual: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        if self.training:
            if target_prior is None:
                raise ValueError(
                    "Stage-2 training requires the frozen Stage-1 teacher prior"
                )
            predicted_prior, prior_sequence, details = (
                self.diffusion.forward_train(
                    rgb,
                    target_prior,
                    target_residual=target_residual,
                )
            )
            pred_hsi = self.G(rgb, predicted_prior)
            if return_details:
                return pred_hsi, prior_sequence, details
            return pred_hsi, prior_sequence

        sampled = self.diffusion.sample(
            rgb,
            initial_noise=initial_noise,
            return_details=return_details,
        )
        if return_details:
            predicted_prior, details = sampled
            return self.G(rgb, predicted_prior), details
        predicted_prior = sampled
        return self.G(rgb, predicted_prior)


def build_model(stage: int, config: ModelConfig) -> nn.Module:
    if stage == 1:
        return DiffIRS1RGB2HSI(config)
    if stage == 2:
        return DiffIRS2RGB2HSI(config)
    raise ValueError("stage must be 1 or 2")
