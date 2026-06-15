"""DiffIR adapted from RGB restoration to same-resolution RGB-to-HSI reconstruction.

This module intentionally has no BasicSR dependency.  The main changes from the
original DiffIR-RealSR code are marked with ``RGB2HSI CHANGE`` comments.

Expected tensors
----------------
RGB: [B, 3, H, W]
HSI: [B, num_bands, H, W]

The Stage-1 model uses the ground-truth HSI to extract an oracle compact prior.
The Stage-2 model learns to recover that prior from RGB alone with four-step
latent diffusion, then conditions the same DIRformer reconstruction network.

METAMER EXTENSION
-----------------
This version adds a local spatial-signature branch and a context-aware spectral
prototype refiner. The goal is to reduce metamer ambiguity by allowing similar
RGB values to map to different spectral signatures when their local spatial
surroundings differ.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class ModelConfig:
    num_bands: int = 31
    dim: int = 48
    num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    num_refinement_blocks: int = 4
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

    # RGB2HSI METAMER CHANGE: enable/disable local spatial context refinement.
    use_local_context_refiner: bool = True

    # RGB2HSI METAMER CHANGE: feature width of the local spatial signature branch.
    local_signature_dim: int = 64

    # RGB2HSI METAMER CHANGE: number of learned spectral/material prototypes.
    num_spectral_prototypes: int = 64

    # RGB2HSI METAMER CHANGE: keeps the context refiner stable at initialization.
    context_residual_scale: float = 0.1

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


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects one normalized dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects one normalized dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2D(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("layer_norm_type must be 'BiasFree' or 'WithBias'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float,
        bias: bool,
        prior_dim: int,
    ):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

        # RGB2HSI CHANGE: prior dimensionality is explicit and configurable.
        self.prior_affine = nn.Linear(prior_dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        _, channels, _, _ = x.shape
        affine = self.prior_affine(prior).view(-1, channels * 2, 1, 1)
        scale, shift = affine.chunk(2, dim=1)
        x = x * scale + shift
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool, prior_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # RGB2HSI CHANGE: prior dimensionality is explicit and configurable.
        self.prior_affine = nn.Linear(prior_dim, dim * 2, bias=False)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        _, channels, h, w = x.shape
        affine = self.prior_affine(prior).view(-1, channels * 2, 1, 1)
        scale, shift = affine.chunk(2, dim=1)
        x = x * scale + shift

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=h,
            w=w,
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
        prior_dim: int,
    ):
        super().__init__()
        self.norm1 = LayerNorm2D(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias, prior_dim)
        self.norm2 = LayerNorm2D(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, prior_dim)

    def forward(self, values: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        x, prior = values
        x = x + self.attn(self.norm1(x), prior)
        x = x + self.ffn(self.norm2(x), prior)
        return [x, prior]


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



class LocalSpatialSignature(nn.Module):
    """
    RGB2HSI METAMER CHANGE.

    Extracts a local spatial/material signature from RGB neighborhoods.

    Why this helps:
        A single RGB value can correspond to multiple spectra. The surrounding
        spatial pattern can indicate whether the pixel is likely grass, leaf,
        soil, sky, cloth, paint, plastic, etc. This branch gives the spectral
        decoder access to that local context.

    Output:
        local_signature: [B, hidden_dim, H, W]
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 5, padding=2),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 7, padding=3),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")
        f3 = self.branch3(rgb)
        f5 = self.branch5(rgb)
        f7 = self.branch7(rgb)
        return self.fuse(torch.cat([f3, f5, f7], dim=1))


class ContextAwareSpectralPrototypeRefiner(nn.Module):
    """
    RGB2HSI METAMER CHANGE.

    Refines an initial HSI estimate using:
        1. the RGB value,
        2. the preliminary predicted spectrum,
        3. the local spatial signature.

    It learns a dictionary of spectral prototypes P_k and predicts per-pixel
    prototype probabilities. This allows similar RGB colors to choose different
    spectral explanations depending on local context.

    Formula:
        S_final = S_initial + alpha * (sum_k p_k P_k + residual)

    Shapes:
        rgb:             [B, 3, H, W]
        initial_hsi:     [B, L, H, W]
        local_signature: [B, C, H, W]
        prototypes:      [K, L]
        weights:         [B, K, H, W]
    """

    def __init__(
        self,
        num_bands: int,
        local_signature_dim: int = 64,
        num_prototypes: int = 64,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_prototypes = num_prototypes
        self.residual_scale = residual_scale

        in_channels = 3 + num_bands + local_signature_dim

        # Learned material/spectral prototype dictionary.
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, num_bands) * 0.02)

        # Pixel-wise probability over prototypes.
        self.prototype_weight_head = nn.Sequential(
            nn.Conv2d(in_channels, local_signature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_signature_dim, num_prototypes, 1),
        )

        # Local residual spectral correction.
        self.residual_head = nn.Sequential(
            nn.Conv2d(in_channels, local_signature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_signature_dim, num_bands, 1),
        )

        # Stored for visualization/diagnostics. It is detached to avoid holding
        # the computation graph after forward passes.
        self.latest_prototype_weights: torch.Tensor | None = None

    def forward(
        self,
        rgb: torch.Tensor,
        initial_hsi: torch.Tensor,
        local_signature: torch.Tensor,
    ) -> torch.Tensor:
        if initial_hsi.shape[1] != self.num_bands:
            raise ValueError(
                f"Expected initial_hsi with {self.num_bands} bands, "
                f"received {initial_hsi.shape[1]}"
            )
        if rgb.shape[-2:] != initial_hsi.shape[-2:]:
            raise ValueError("RGB and initial_hsi must have the same spatial size")
        if local_signature.shape[-2:] != initial_hsi.shape[-2:]:
            raise ValueError("local_signature and initial_hsi must have the same spatial size")

        features = torch.cat([rgb, initial_hsi, local_signature], dim=1)
        logits = self.prototype_weight_head(features)
        weights = torch.softmax(logits, dim=1)
        self.latest_prototype_weights = weights.detach()

        prototype_hsi = torch.einsum("bkhw,kl->blhw", weights, self.prototypes)
        residual_hsi = self.residual_head(features)

        # The small residual scale protects the already strong DiffIR baseline
        # from being disrupted at the start of training.
        return initial_hsi + self.residual_scale * (prototype_hsi + residual_hsi)


class DIRformerRGB2HSI(nn.Module):
    """Prior-conditioned spatial encoder-decoder that predicts an HSI cube."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim

        # RGB2HSI CHANGE: same-resolution RGB-to-HSI always uses scale=1.
        # PixelUnshuffle(4) is an internal representation change, not output SR.
        self.input_transform = nn.PixelUnshuffle(4)
        self.patch_embed = OverlapPatchEmbed(3 * 16, dim, config.bias)

        def make_blocks(block_dim: int, block_heads: int, count: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    TransformerBlock(
                        block_dim,
                        block_heads,
                        config.ffn_expansion_factor,
                        config.bias,
                        config.layer_norm_type,
                        config.prior_dim,
                    )
                    for _ in range(count)
                ]
            )

        self.encoder_level1 = make_blocks(dim, config.heads[0], config.num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(dim * 2, config.heads[1], config.num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(dim * 4, config.heads[2], config.num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = make_blocks(dim * 8, config.heads[3], config.num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=config.bias)
        self.decoder_level3 = make_blocks(dim * 4, config.heads[2], config.num_blocks[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=config.bias)
        self.decoder_level2 = make_blocks(dim * 2, config.heads[1], config.num_blocks[1])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = make_blocks(dim * 2, config.heads[0], config.num_blocks[0])
        self.refinement = make_blocks(
            dim * 2,
            config.heads[0],
            config.num_refinement_blocks,
        )

        self.tail = nn.Sequential(
            Upsampler(4, dim * 2, bias=True),
            default_conv(dim * 2, config.num_bands, 3),
        )

        # RGB2HSI CHANGE: a direct 3-channel RGB residual cannot be added to an
        # L-band HSI. Learn a per-pixel 3 -> L spectral lifting instead.
        self.rgb_to_hsi = (
            nn.Conv2d(3, config.num_bands, 1)
            if config.use_rgb_to_hsi_skip
            else None
        )

        # RGB2HSI METAMER CHANGE:
        # Local spatial-signature branch + context-aware spectral prototype refiner.
        # These modules implement the hypothesis that local surroundings help
        # disambiguate metamers by selecting context-dependent spectral patterns.
        if config.use_local_context_refiner:
            self.local_signature = LocalSpatialSignature(
                in_channels=3,
                hidden_dim=config.local_signature_dim,
            )
            self.context_refiner = ContextAwareSpectralPrototypeRefiner(
                num_bands=config.num_bands,
                local_signature_dim=config.local_signature_dim,
                num_prototypes=config.num_spectral_prototypes,
                residual_scale=config.context_residual_scale,
            )
        else:
            self.local_signature = None
            self.context_refiner = None

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
            initial_hsi = hsi_residual
        else:
            initial_hsi = self.rgb_to_hsi(rgb) + hsi_residual

        # RGB2HSI METAMER CHANGE:
        # Refine the initial HSI using local RGB surroundings. This allows the
        # model to learn that similar RGB values can correspond to different
        # spectra when the local spatial/material signature is different.
        if self.context_refiner is not None:
            local_signature = self.local_signature(rgb)
            return self.context_refiner(
                rgb=rgb,
                initial_hsi=initial_hsi,
                local_signature=local_signature,
            )

        return initial_hsi


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
        # RGB2HSI CHANGE: PixelUnshuffle(4) gives 16*(3+L) fused channels.
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
        # Faithful to the released implementation: no explicit x + residual addition.
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

        # Match DiffIR's 'linear' schedule: linearly spaced square roots, squared.
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
        # DiffIR predicts the clean prior x0 at every step.
        predicted_x0 = self.denoiser(prior_t, timestep, condition)
        posterior_mean = (
            _extract(self.posterior_mean_coef1, timestep, prior_t.shape) * predicted_x0
            + _extract(self.posterior_mean_coef2, timestep, prior_t.shape) * prior_t
        )
        # Faithful to the released implementation: no additional posterior noise.
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
        self.G = DIRformerRGB2HSI(config)

    def forward(self, rgb: torch.Tensor, hsi_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prior = self.E(rgb, hsi_gt)
        pred_hsi = self.G(rgb, prior)
        return pred_hsi, prior


class DiffIRS2RGB2HSI(nn.Module):
    """Stage 2: RGB-conditioned compact-prior diffusion + HSI reconstruction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.G = DIRformerRGB2HSI(config)
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



def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    RGB2HSI METAMER CHANGE: optional regularizer.

    Encourages learned spectral prototypes to be different from one another.
    Add this in the training loop only if you want explicit prototype diversity.
    """
    normalized = F.normalize(prototypes, dim=1)
    similarity = normalized @ normalized.t()
    k = prototypes.shape[0]
    eye = torch.eye(k, device=prototypes.device, dtype=prototypes.dtype)
    off_diagonal = similarity * (1.0 - eye)
    return off_diagonal.pow(2).mean()


def prototype_entropy_loss(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    RGB2HSI METAMER CHANGE: optional regularizer.

    Minimizing this loss encourages each pixel to select a small number of
    prototypes instead of using all prototypes uniformly.
    """
    entropy = -weights * torch.log(weights + eps)
    return entropy.sum(dim=1).mean()


def build_model(stage: int, config: ModelConfig) -> nn.Module:
    if stage == 1:
        return DiffIRS1RGB2HSI(config)
    if stage == 2:
        return DiffIRS2RGB2HSI(config)
    raise ValueError("stage must be 1 or 2")
