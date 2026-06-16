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

Public interface is kept:
    Stage 1: DiffIRS1RGB2HSI(rgb, hsi_gt) -> pred_hsi, prior
    Stage 2: DiffIRS2RGB2HSI(rgb, target_prior) -> pred_hsi, prior_sequence

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
    # CPEN internally pads to this factor and PixelUnshuffles by this factor.
    prior_downsample_factor: int = 4
    prior_feat_dim: int = 64
    use_spectral_prior_output_skip: bool = True
    spectral_prior_output_scale_init: float = 1.0

    # Diffusion settings. prior_dim is kept only for compatibility with older
    # config/checkpoints; the new prior is not [B, prior_dim].
    prior_dim: int = 256
    n_encoder_res: int = 6

    #Old value is 2
    n_denoise_res: int = 4

    #Change : Slightly less aggressive noise schedule
    timesteps: int = 4
    linear_start: float = 1e-4
    linear_end: float = 0.05

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
        prior_resized = F.interpolate(prior, size=(h, w), mode="bilinear", align_corners=False)
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


class SpatialSpectralPriorEncoderBase(nn.Module):
    """CPEN without pooling or Linear layers.

    It preserves spatial layout and outputs a spectral prior map:
        [B, num_bands, ceil(H/4), ceil(W/4)] by default.
    """

    def __init__(self, in_channels: int, config: ModelConfig):
        super().__init__()
        feat = config.prior_feat_dim
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, feat, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        layers.extend([ResBlock(feat) for _ in range(config.n_encoder_res)])
        layers.extend(
            [
                nn.Conv2d(feat, feat * 2, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                ConvResBlock(feat * 2),
                nn.Conv2d(feat * 2, feat * 2, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                ConvResBlock(feat * 2),
                nn.Conv2d(feat * 2, feat * 4, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                ConvResBlock(feat * 4),
            ]
        )
        self.encoder = nn.Sequential(*layers)
        self.spectral_head = nn.Sequential(
            nn.Conv2d(feat * 4, feat * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat * 2, config.num_bands, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral_head(self.encoder(x))


class TeacherPriorEncoder(SpatialSpectralPriorEncoderBase):
    """Stage-1 oracle CPEN: RGB + GT HSI -> spectral prior map."""

    def __init__(self, config: ModelConfig):
        if config.prior_downsample_factor != 4:
            raise ValueError("This implementation currently expects prior_downsample_factor=4")
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

        rgb_pad, _ = _pad_to_multiple(rgb, self.config.prior_downsample_factor, mode="reflect")
        hsi_pad, _ = _pad_to_multiple(hsi, self.config.prior_downsample_factor, mode="reflect")
        fused = torch.cat([self.unshuffle(rgb_pad), self.unshuffle(hsi_pad)], dim=1)
        return self.encode(fused)


class RGBConditionEncoder(SpatialSpectralPriorEncoderBase):
    """Stage-2 CPEN: RGB -> RGB-conditioned spectral prior map."""

    def __init__(self, config: ModelConfig):
        if config.prior_downsample_factor != 4:
            raise ValueError("This implementation currently expects prior_downsample_factor=4")
        super().__init__(3 * 16, config)
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(4)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[1] != 3:
            raise ValueError("RGBConditionEncoder expects three RGB channels")
        rgb_pad, _ = _pad_to_multiple(rgb, self.config.prior_downsample_factor, mode="reflect")
        return self.encode(self.unshuffle(rgb_pad))


class SpatialPriorDenoiser(nn.Module):
    """Convolutional denoiser for spatial spectral-prior diffusion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        bands = config.num_bands
        feat = max(config.prior_feat_dim, bands)
        self.stem = nn.Sequential(
            nn.Conv2d(bands * 2 + 1, feat, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(*[ConvResBlock(feat) for _ in range(config.n_denoise_res)])
        self.head = nn.Conv2d(feat, bands, 3, padding=1)
        self.max_period = float(config.timesteps * 10)

    def forward(
        self,
        noisy_prior: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_prior.shape != condition.shape:
            raise ValueError(
                f"noisy_prior and condition must have identical shapes. "
                f"Got {tuple(noisy_prior.shape)} and {tuple(condition.shape)}."
            )
        b, _, h, w = noisy_prior.shape
        t = timestep.float().view(-1, 1, 1, 1) / self.max_period
        t_map = t.expand(b, 1, h, w)
        x = torch.cat([condition, t_map, noisy_prior], dim=1)
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)


def _extract(buffer: torch.Tensor, timestep: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timestep)
    return values.view(timestep.shape[0], *((1,) * (len(target_shape) - 1)))


class SpatialSpectralPriorDiffusion(nn.Module):
    """Four-step x0-prediction diffusion over the spatial spectral prior map."""

    def __init__(self, config: ModelConfig, condition: RGBConditionEncoder, denoiser: SpatialPriorDenoiser):
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
        condition = self.condition(rgb)
        if condition.shape != target_prior.shape:
            raise ValueError(
                f"Condition prior shape {tuple(condition.shape)} must match target prior shape "
                f"{tuple(target_prior.shape)}."
            )
        prior = self.q_sample(target_prior, final_t)
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
        condition = self.condition(rgb)
        batch, channels, h, w = condition.shape
        device = rgb.device
        if initial_noise is None:
            prior = torch.randn(batch, channels, h, w, device=device)
        else:
            expected = (batch, channels, h, w)
            if tuple(initial_noise.shape) != expected:
                raise ValueError(f"Expected initial_noise {expected}, got {tuple(initial_noise.shape)}")
            prior = initial_noise
        for step in reversed(range(self.config.timesteps)):
            timestep = torch.full((batch,), step, device=device, dtype=torch.long)
            prior, _ = self.reverse_step(prior, timestep, condition)
        return prior


# -----------------------------------------------------------------------------
# Stage wrappers
# -----------------------------------------------------------------------------


class DiffIRS1RGB2HSI(nn.Module):
    """Stage 1: RGB+GT-HSI oracle spectral prior + conditioned HSI reconstruction."""

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
    """Stage 2: RGB-conditioned spectral-prior diffusion + conditioned reconstruction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.G = MSTSpectralDiffIRGeneratorRGB2HSI(config)
        condition = RGBConditionEncoder(config)
        denoiser = SpatialPriorDenoiser(config)
        self.diffusion = SpatialSpectralPriorDiffusion(config, condition, denoiser)

    @property
    def condition(self) -> RGBConditionEncoder:
        return self.diffusion.condition

    @property
    def denoiser(self) -> SpatialPriorDenoiser:
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
