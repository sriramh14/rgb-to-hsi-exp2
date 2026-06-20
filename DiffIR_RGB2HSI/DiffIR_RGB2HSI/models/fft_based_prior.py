"""RGB-to-HSI model with FFT cross-attention CPEN and per-block prior injection.

Stage 1
-------
RGB and ground-truth HSI are transformed with 2-D FFTs. RGB frequency tokens
query HSI frequency tokens through cross-attention. The resulting frequency
descriptor is fused with the RGB spatial context to produce a compact oracle
prior. That same prior is:
    1. fused with RGB before MST++;
    2. explicitly injected at the beginning of every MST++ transformer block.

Stage 2
-------
An RGB-conditioned deterministic DDIM model is trained only to reproduce the
frozen Stage-1 CPEN prior. The Stage-1 adapter, all per-block prior conditioners,
and MST++ remain frozen. They are used only for final HSI reconstruction.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out


# ============================================================================
# MST++ INITIALISATION UTILITIES
# ============================================================================

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


def variance_scaling_(
    tensor,
    scale=1.0,
    mode="fan_in",
    distribution="normal",
):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denominator = fan_in
    elif mode == "fan_out":
        denominator = fan_out
    elif mode == "fan_avg":
        denominator = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    variance = scale / denominator

    with torch.no_grad():
        if distribution == "truncated_normal":
            trunc_normal_(
                tensor,
                std=math.sqrt(variance) / 0.87962566103423978,
            )
        elif distribution == "normal":
            tensor.normal_(std=math.sqrt(variance))
        elif distribution == "uniform":
            bound = math.sqrt(3 * variance)
            tensor.uniform_(-bound, bound)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(
        tensor,
        mode="fan_in",
        distribution="truncated_normal",
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(
    in_channels,
    out_channels,
    kernel_size,
    bias=False,
    padding=1,
    stride=1,
):
    del padding
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
        stride=stride,
    )


def shift_back(inputs, step=2):
    """
    Retained from the original MST++ implementation.

    Input:
        [B, C, H, W_shifted]

    Output:
        [B, C, H, H]
    """
    bs, n_channels, rows, columns = inputs.shape
    del bs, columns

    down_sample = 256 // rows
    step = float(step) / float(down_sample * down_sample)
    out_columns = rows

    for channel_idx in range(n_channels):
        start = int(step * channel_idx)
        inputs[:, channel_idx, :, :out_columns] = inputs[
            :,
            channel_idx,
            :,
            start : start + out_columns,
        ]

    return inputs[:, :, :, :out_columns]


# ============================================================================
# ORIGINAL MST++ MODULES
# ============================================================================

class MS_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
    ):
        super().__init__()

        self.num_heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(
            dim,
            dim_head * heads,
            bias=False,
        )
        self.to_k = nn.Linear(
            dim,
            dim_head * heads,
            bias=False,
        )
        self.to_v = nn.Linear(
            dim,
            dim_head * heads,
            bias=False,
        )

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(
            dim_head * heads,
            dim,
            bias=True,
        )

        self.pos_emb = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=dim,
            ),
            GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=dim,
            ),
        )

        self.dim = dim

    def forward(self, x_in):
        """
        Input:
            x_in: [B, H, W, C]

        Output:
            out: [B, H, W, C]
        """
        batch, height, width, channels = x_in.shape

        x = x_in.reshape(
            batch,
            height * width,
            channels,
        )

        q_input = self.to_q(x)
        k_input = self.to_k(x)
        v_input = self.to_v(x)

        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                "b n (h d) -> b h n d",
                h=self.num_heads,
            ),
            (q_input, k_input, v_input),
        )

        # [B, heads, HW, dim_head] -> [B, heads, dim_head, HW]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # Spectral/channel attention:
        # [B, heads, dim_head, HW] @ [B, heads, HW, dim_head]
        # -> [B, heads, dim_head, dim_head]
        attention = k @ q.transpose(-2, -1)
        attention = attention * self.rescale
        attention = attention.softmax(dim=-1)

        # [B, heads, dim_head, dim_head] @
        # [B, heads, dim_head, HW]
        output = attention @ v

        output = output.permute(0, 3, 1, 2)
        output = output.reshape(
            batch,
            height * width,
            self.num_heads * self.dim_head,
        )

        output_channels = self.proj(output).view(
            batch,
            height,
            width,
            channels,
        )

        output_position = self.pos_emb(
            v_input.reshape(
                batch,
                height,
                width,
                channels,
            ).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        return output_channels + output_position


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                dim,
                dim * mult,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            GELU(),
            nn.Conv2d(
                dim * mult,
                dim * mult,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=dim * mult,
            ),
            GELU(),
            nn.Conv2d(
                dim * mult,
                dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        """
        Input:
            x: [B, H, W, C]

        Output:
            [B, H, W, C]
        """
        output = self.net(
            x.permute(0, 3, 1, 2)
        )

        return output.permute(0, 2, 3, 1)



class PriorAdditiveConditioning(nn.Module):
    """
    Adds the compact prior at the beginning of one transformer block.

    x_conditioned =
        x + sigmoid(gate) * max_delta * tanh(W(LayerNorm(prior)))

    Every transformer block owns an independent projection and gate. Thus the
    same global prior can be interpreted differently at 31, 62, and 124
    feature channels and by different cascaded MST stages.
    """

    def __init__(
        self,
        prior_dim,
        feature_dim,
        max_delta=0.10,
        initial_gate_logit=-2.1972246,
    ):
        super().__init__()

        self.prior_dim = prior_dim
        self.feature_dim = feature_dim
        self.max_delta = float(max_delta)

        self.prior_norm = nn.LayerNorm(prior_dim)
        self.projection = nn.Linear(
            prior_dim,
            feature_dim,
            bias=True,
        )

        # Small non-zero weights keep the initial perturbation weak while
        # allowing gradients to reach CPEN from the first optimization step.
        trunc_normal_(
            self.projection.weight,
            std=0.02,
        )
        nn.init.zeros_(self.projection.bias)

        self.gate_logit = nn.Parameter(
            torch.tensor(float(initial_gate_logit))
        )

    def forward(self, x, prior):
        if prior is None:
            return x

        if prior.ndim != 2:
            raise ValueError(
                f"Expected prior [B,{self.prior_dim}], "
                f"got {tuple(prior.shape)}."
            )

        if prior.shape[0] != x.shape[0]:
            raise ValueError(
                "Feature and prior batch sizes must match."
            )

        if prior.shape[1] != self.prior_dim:
            raise ValueError(
                f"Expected prior dimension {self.prior_dim}, "
                f"got {prior.shape[1]}."
            )

        prior_bias = self.max_delta * torch.tanh(
            self.projection(
                self.prior_norm(prior)
            )
        )
        prior_bias = prior_bias[:, None, None, :]
        gate = torch.sigmoid(self.gate_logit)

        return x + gate * prior_bias


class MSAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
        prior_dim=256,
        prior_max_delta=0.10,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PriorAdditiveConditioning(
                            prior_dim=prior_dim,
                            feature_dim=dim,
                            max_delta=prior_max_delta,
                        ),
                        MS_MSA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim=dim),
                        ),
                    ]
                )
            )

    def forward(self, x, prior=None):
        """
        Inputs:
            x:     [B, C, H, W]
            prior: [B, prior_dim]

        Output:
            [B, C, H, W]

        The prior is explicitly added once at the beginning of every
        attention-plus-feed-forward transformer block.
        """
        x = x.permute(0, 2, 3, 1)

        for prior_add, attention, feed_forward in self.blocks:
            x = prior_add(x, prior)
            x = attention(x) + x
            x = feed_forward(x) + x

        return x.permute(0, 3, 1, 2)


class MST(nn.Module):
    """
    One U-shaped Single-stage Spectral-wise Transformer.

    For the default stage=2:
        encoder level 1
        encoder level 2
        bottleneck
        decoder level 1
        decoder level 2
    """

    def __init__(
        self,
        in_dim=31,
        out_dim=31,
        dim=31,
        stage=2,
        num_blocks=(2, 4, 4),
        prior_dim=256,
        prior_max_delta=0.10,
    ):
        super().__init__()

        self.dim = dim
        self.stage = stage

        if len(num_blocks) != stage + 1:
            raise ValueError(
                "num_blocks must contain one value per encoder level "
                "plus one bottleneck value."
            )

        # Input projection
        self.embedding = nn.Conv2d(
            in_dim,
            self.dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim

        for level in range(stage):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        MSAB(
                            dim=dim_stage,
                            num_blocks=num_blocks[level],
                            dim_head=dim,
                            heads=dim_stage // dim,
                            prior_dim=prior_dim,
                            prior_max_delta=prior_max_delta,
                        ),
                        nn.Conv2d(
                            dim_stage,
                            dim_stage * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
            )
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage,
            dim_head=dim,
            heads=dim_stage // dim,
            num_blocks=num_blocks[-1],
            prior_dim=prior_dim,
            prior_max_delta=prior_max_delta,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])

        for level in range(stage):
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
                        ),
                        nn.Conv2d(
                            dim_stage,
                            dim_stage // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        MSAB(
                            dim=dim_stage // 2,
                            num_blocks=num_blocks[stage - 1 - level],
                            dim_head=dim,
                            heads=(dim_stage // 2) // dim,
                            prior_dim=prior_dim,
                            prior_max_delta=prior_max_delta,
                        ),
                    ]
                )
            )
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(
            self.dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, prior=None):
        """
        Inputs:
            x:     [B, C, H, W]
            prior: [B, prior_dim]

        Output:
            [B, C, H, W]
        """
        # Embedding
        feature = self.embedding(x)

        # Encoder
        encoder_features = []

        for transformer_block, downsample in self.encoder_layers:
            feature = transformer_block(feature, prior)
            encoder_features.append(feature)
            feature = downsample(feature)

        # Bottleneck
        feature = self.bottleneck(feature, prior)

        # Decoder
        for level, (
            upsample,
            fusion,
            transformer_block,
        ) in enumerate(self.decoder_layers):
            feature = upsample(feature)

            skip = encoder_features[
                self.stage - 1 - level
            ]

            feature = fusion(
                torch.cat(
                    [feature, skip],
                    dim=1,
                )
            )

            feature = transformer_block(feature, prior)

        # Residual mapping
        return self.mapping(feature) + x


class MST_Plus_Plus(nn.Module):
    """
    Original cascaded multi-stage MST++.

    The default stage=3 creates three complete MST/SST stages.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=3,
        prior_dim=256,
        prior_max_delta=0.10,
    ):
        super().__init__()

        if n_feat != 31:
            raise ValueError(
                "This supplied MST++ configuration expects n_feat=31."
            )

        if out_channels != n_feat:
            raise ValueError(
                "out_channels must equal n_feat because the final residual "
                "adds the n_feat input feature to the output."
            )

        self.stage = stage

        self.conv_in = nn.Conv2d(
            in_channels,
            n_feat,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        modules_body = [
            MST(
                in_dim=n_feat,
                out_dim=n_feat,
                dim=31,
                stage=2,
                num_blocks=(1, 1, 1),
                prior_dim=prior_dim,
                prior_max_delta=prior_max_delta,
            )
            for _ in range(stage)
        ]

        self.body = nn.ModuleList(modules_body)

        self.conv_out = nn.Conv2d(
            n_feat,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x, prior=None):
        """
        Inputs:
            x:     [B, 3, H, W]
            prior: [B, prior_dim] or None

        Output:
            HSI: [B, 31, H, W]
        """
        _, _, input_height, input_width = x.shape

        pad_base_height = 8
        pad_base_width = 8

        pad_height = (
            pad_base_height
            - input_height % pad_base_height
        ) % pad_base_height

        pad_width = (
            pad_base_width
            - input_width % pad_base_width
        ) % pad_base_width

        x = F.pad(
            x,
            [0, pad_width, 0, pad_height],
            mode="reflect",
        )

        x = self.conv_in(x)

        output = x
        for mst_stage in self.body:
            output = mst_stage(
                output,
                prior,
            )

        output = self.conv_out(output)
        output = output + x

        return output[
            :,
            :,
            :input_height,
            :input_width,
        ]


# ============================================================================
# RGB-HSI CPEN
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual convolutional block used by CPEN and the Stage-2 predictor."""

    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.body(x)


class SpectralMixingBlock(nn.Module):
    """
    Mixes the 31 HSI wavelength channels independently at each spatial point.
    """

    def __init__(
        self,
        hsi_channels=31,
        hidden_channels=64,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                hsi_channels,
                hidden_channels,
                kernel_size=1,
            ),
            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
            nn.Conv2d(
                hidden_channels,
                hsi_channels,
                kernel_size=1,
            ),
        )

    def forward(self, hsi):
        return hsi + self.body(hsi)


class CPEN(nn.Module):
    """
    Shared CPEN backbone used by both stages.

    This base class owns all RGB-side and compact-vector encoding layers:

        RGB -> PixelUnshuffle -> RGB stem -> residual body
            -> context encoder -> global vector

    Child classes only define how the feature entering the shared residual body
    is constructed:

        Stage1CPEN: fused RGB + ground-truth HSI feature
        Stage2CPEN: RGB feature only
    """

    def __init__(
        self,
        rgb_channels=3,
        n_feats=64,
        n_encoder_res=6,
        output_dim=256,
        unshuffle_factor=4,
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.n_feats = n_feats
        self.output_dim = output_dim
        self.unshuffle_factor = unshuffle_factor

        rgb_unshuffled_channels = (
            rgb_channels * unshuffle_factor ** 2
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(
            unshuffle_factor
        )

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(
                rgb_unshuffled_channels,
                n_feats,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.encoder_body = nn.Sequential(
            *[
                ResidualBlock(n_feats)
                for _ in range(n_encoder_res)
            ]
        )

        self.context_encoder = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                n_feats * 2,
                n_feats * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                n_feats * 2,
                n_feats * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(n_feats * 4, output_dim),
        )

    def _validate_rgb(self, rgb):
        if rgb.ndim != 4:
            raise ValueError(
                f"Expected RGB [B,{self.rgb_channels},H,W], "
                f"got {tuple(rgb.shape)}."
            )

        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f"Expected {self.rgb_channels} RGB channels, "
                f"got {rgb.shape[1]}."
            )

        height, width = rgb.shape[-2:]
        factor = self.unshuffle_factor

        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"RGB height/width must be divisible by {factor}; "
                f"got {(height, width)}."
            )

    def encode_rgb(self, rgb):
        """Return the shared RGB feature map [B,n_feats,H/f,W/f]."""
        self._validate_rgb(rgb)
        return self.rgb_encoder(
            self.pixel_unshuffle(rgb)
        )

    def encode_context(self, feature):
        """Convert an n_feats feature map into the shared global context."""
        feature = self.encoder_body(feature)
        return self.context_encoder(feature).flatten(1)

    def encode_compact(self, feature):
        """Convert an n_feats feature map into the compact output vector."""
        return self.mlp(
            self.encode_context(feature)
        )

    def copy_shared_from(
        self,
        source_cpen,
        copy_deeper_blocks=False,
    ):
        """Initialize shared RGB layers from another CPEN child instance."""
        self.rgb_encoder.load_state_dict(
            source_cpen.rgb_encoder.state_dict(),
            strict=True,
        )

        if copy_deeper_blocks:
            self.encoder_body.load_state_dict(
                source_cpen.encoder_body.state_dict(),
                strict=True,
            )
            self.context_encoder.load_state_dict(
                source_cpen.context_encoder.state_dict(),
                strict=True,
            )


class FFTFeatureEncoder(nn.Module):
    """
    Encodes a channel-wise 2-D FFT into a compact frequency feature grid.

    The representation concatenates log magnitude and signed-log real and
    imaginary components. Real/imaginary components preserve phase-related
    structure without the wrap discontinuity of an explicit phase angle.
    """

    def __init__(
        self,
        input_channels,
        feature_dim=64,
        frequency_grid=(16, 9),
    ):
        super().__init__()

        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.frequency_grid = tuple(frequency_grid)

        groups = max(1, math.gcd(8, feature_dim))

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels * 3,
                feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, feature_dim),
            nn.GELU(),
            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(self.frequency_grid),
        )

    @staticmethod
    def signed_log(value):
        return torch.sign(value) * torch.log1p(
            torch.abs(value)
        )

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError(
                "FFTFeatureEncoder expects [B,C,H,W], "
                f"got {tuple(image.shape)}."
            )

        if image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, "
                f"got {image.shape[1]}."
            )

        output_dtype = image.dtype

        # Evaluate FFT in float32 for mixed-precision stability.
        spectrum = torch.fft.rfft2(
            image.float(),
            dim=(-2, -1),
            norm="ortho",
        )

        fft_feature = torch.cat(
            [
                torch.log1p(torch.abs(spectrum)),
                self.signed_log(spectrum.real),
                self.signed_log(spectrum.imag),
            ],
            dim=1,
        ).to(dtype=output_dtype)

        return self.encoder(fft_feature)


class FFTCrossAttentionBlock(nn.Module):
    """RGB FFT tokens query richer HSI FFT key/value tokens."""

    def __init__(
        self,
        dim,
        heads=4,
        expansion=2,
        dropout=0.0,
        initial_gate_logit=-1.3862944,
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by heads={heads}."
            )

        self.rgb_norm = nn.LayerNorm(dim)
        self.hsi_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_gate_logit = nn.Parameter(
            torch.tensor(float(initial_gate_logit))
        )

        hidden_dim = dim * expansion
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, rgb_tokens, hsi_tokens):
        retrieved_hsi, _ = self.cross_attention(
            query=self.rgb_norm(rgb_tokens),
            key=self.hsi_norm(hsi_tokens),
            value=self.hsi_norm(hsi_tokens),
            need_weights=False,
        )

        rgb_tokens = (
            rgb_tokens
            + torch.sigmoid(self.attention_gate_logit)
            * retrieved_hsi
        )

        return (
            rgb_tokens
            + self.ffn(
                self.ffn_norm(rgb_tokens)
            )
        )


class Stage1CPEN(CPEN):
    """
    Oracle CPEN using RGB-to-HSI FFT cross-attention.

    Frequency coordinates are not treated as spatial image coordinates. The
    cross-attended frequency tokens are globally pooled and fused with a
    global RGB spatial context before the 256-D oracle prior is produced.
    """

    def __init__(
        self,
        rgb_channels=3,
        hsi_channels=31,
        n_feats=64,
        n_encoder_res=6,
        prior_dim=256,
        unshuffle_factor=4,
        frequency_grid=(16, 9),
        frequency_heads=4,
        frequency_blocks=2,
        frequency_dropout=0.0,
    ):
        super().__init__(
            rgb_channels=rgb_channels,
            n_feats=n_feats,
            n_encoder_res=n_encoder_res,
            output_dim=prior_dim,
            unshuffle_factor=unshuffle_factor,
        )

        if n_feats % frequency_heads != 0:
            raise ValueError(
                "n_feats must be divisible by frequency_heads."
            )

        self.hsi_channels = hsi_channels
        self.prior_dim = prior_dim
        self.frequency_grid = tuple(frequency_grid)

        self.rgb_fft_encoder = FFTFeatureEncoder(
            input_channels=rgb_channels,
            feature_dim=n_feats,
            frequency_grid=frequency_grid,
        )
        self.hsi_fft_encoder = FFTFeatureEncoder(
            input_channels=hsi_channels,
            feature_dim=n_feats,
            frequency_grid=frequency_grid,
        )

        number_of_tokens = (
            self.frequency_grid[0]
            * self.frequency_grid[1]
        )
        self.frequency_position = nn.Parameter(
            torch.zeros(1, number_of_tokens, n_feats)
        )
        trunc_normal_(self.frequency_position, std=0.02)

        self.frequency_cross_blocks = nn.ModuleList(
            [
                FFTCrossAttentionBlock(
                    dim=n_feats,
                    heads=frequency_heads,
                    dropout=frequency_dropout,
                )
                for _ in range(frequency_blocks)
            ]
        )

        # Mean captures the overall frequency distribution; max preserves
        # sparse strong responses such as edges and repeating textures.
        self.frequency_pool = nn.Sequential(
            nn.LayerNorm(n_feats * 2),
            nn.Linear(n_feats * 2, n_feats * 4),
            nn.GELU(),
        )

        self.spatial_context_norm = nn.LayerNorm(n_feats * 4)
        self.frequency_context_norm = nn.LayerNorm(n_feats * 4)

        # A vector-valued gate prevents HSI frequency content from completely
        # replacing the RGB-observable context.
        self.context_gate = nn.Sequential(
            nn.Linear(n_feats * 8, n_feats * 4),
            nn.Sigmoid(),
        )
        self.context_refinement = nn.Sequential(
            nn.LayerNorm(n_feats * 4),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.GELU(),
            nn.Linear(n_feats * 4, n_feats * 4),
        )

    def _validate_inputs(self, rgb, gt_hsi):
        self._validate_rgb(rgb)

        if gt_hsi.ndim != 4:
            raise ValueError(
                f"Expected HSI [B,{self.hsi_channels},H,W], "
                f"got {tuple(gt_hsi.shape)}."
            )

        if gt_hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f"Expected {self.hsi_channels} HSI channels, "
                f"got {gt_hsi.shape[1]}."
            )

        if rgb.shape[0] != gt_hsi.shape[0]:
            raise ValueError(
                "RGB and HSI batch sizes must match."
            )

        if rgb.shape[-2:] != gt_hsi.shape[-2:]:
            raise ValueError(
                "RGB and HSI spatial dimensions must match."
            )

    @staticmethod
    def to_tokens(feature):
        return feature.flatten(2).transpose(1, 2)

    def forward(self, rgb, gt_hsi):
        self._validate_inputs(rgb, gt_hsi)

        # RGB-only spatial anchor: Stage 2 must ultimately predict this prior
        # using RGB as its only observed input.
        rgb_spatial_feature = self.encode_rgb(rgb)
        spatial_context = self.spatial_context_norm(
            self.encode_context(rgb_spatial_feature)
        )

        rgb_tokens = (
            self.to_tokens(self.rgb_fft_encoder(rgb))
            + self.frequency_position
        )
        hsi_tokens = (
            self.to_tokens(self.hsi_fft_encoder(gt_hsi))
            + self.frequency_position
        )

        for cross_block in self.frequency_cross_blocks:
            rgb_tokens = cross_block(
                rgb_tokens,
                hsi_tokens,
            )

        frequency_context = self.frequency_pool(
            torch.cat(
                [
                    rgb_tokens.mean(dim=1),
                    rgb_tokens.amax(dim=1),
                ],
                dim=-1,
            )
        )
        frequency_context = self.frequency_context_norm(
            frequency_context
        )

        gate = self.context_gate(
            torch.cat(
                [spatial_context, frequency_context],
                dim=-1,
            )
        )

        fused_context = (
            spatial_context
            + gate * frequency_context
        )
        fused_context = (
            fused_context
            + self.context_refinement(fused_context)
        )

        prior = self.mlp(fused_context)
        return prior, [prior]


class Stage2CPEN(CPEN):
    """
    RGB-only CPEN child used as the condition encoder in Stage 2.

    It does not directly replace the diffusion model. It supplies the RGB
    condition vector required by the DDIM prior denoiser.
    """

    def __init__(
        self,
        rgb_channels=3,
        n_feats=64,
        n_res_blocks=6,
        condition_dim=256,
        unshuffle_factor=4,
    ):
        super().__init__(
            rgb_channels=rgb_channels,
            n_feats=n_feats,
            n_encoder_res=n_res_blocks,
            output_dim=condition_dim,
            unshuffle_factor=unshuffle_factor,
        )

        self.condition_dim = condition_dim

    def initialise_from_cpen(
        self,
        cpen,
        copy_deeper_blocks=False,
    ):
        self.copy_shared_from(
            cpen,
            copy_deeper_blocks=copy_deeper_blocks,
        )

    def forward(self, rgb):
        rgb_feature = self.encode_rgb(rgb)
        return self.encode_compact(rgb_feature)


# Compatibility aliases used by earlier files and training scripts.
RGBConditionEncoder = Stage2CPEN
RGBPriorPredictor = Stage2CPEN



# ============================================================================
# DETERMINISTIC VECTOR DDIM
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for integer diffusion timesteps."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(half, 1)
        )

        arguments = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)],
            dim=-1,
        )

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class PriorMLPResidualBlock(nn.Module):
    """Residual MLP block used by the compact prior denoiser."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class PriorDenoiser(nn.Module):
    """
    Predicts epsilon for a noisy compact prior vector.

    Inputs:
        noisy_prior: [B, prior_dim]
        timestep:    [B]
        condition:   [B, condition_dim]

    Output:
        predicted_noise: [B, prior_dim]
    """

    def __init__(
        self,
        prior_dim=256,
        condition_dim=256,
        hidden_dim=512,
        time_dim=128,
        num_blocks=5,
        dropout=0.0,
    ):
        super().__init__()

        self.prior_dim = prior_dim
        self.condition_dim = condition_dim

        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_projection = nn.Linear(
            prior_dim + condition_dim + hidden_dim,
            hidden_dim,
        )

        self.blocks = nn.Sequential(
            *[
                PriorMLPResidualBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, prior_dim),
        )

    def forward(self, noisy_prior, timestep, condition):
        if noisy_prior.ndim != 2:
            raise ValueError(
                f"Expected noisy prior [B,{self.prior_dim}], "
                f"got {tuple(noisy_prior.shape)}."
            )

        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition [B,{self.condition_dim}], "
                f"got {tuple(condition.shape)}."
            )

        if timestep.ndim != 1:
            raise ValueError(
                f"Expected timestep [B], got {tuple(timestep.shape)}."
            )

        time_feature = self.time_encoder(timestep)

        feature = torch.cat(
            [noisy_prior, condition, time_feature],
            dim=1,
        )

        feature = self.input_projection(feature)
        feature = self.blocks(feature)

        return self.output(feature)


def _extract_schedule_value(values, timesteps, reference):
    """Collects one schedule value per batch item and reshapes for a vector."""
    gathered = values.gather(0, timesteps)
    return gathered.view(reference.shape[0], 1)


class DeterministicPriorDDIM(nn.Module):
    """
    DDIM-style diffusion over the compact prior vector.

    Important properties
    --------------------
    * Only the prior vector is diffused; no HSI image is diffused.
    * Reverse sampling uses eta=0.
    * No random variance term is inserted at reverse steps.
    * Inference defaults to one fixed registered initial-noise vector, making
      repeated inference deterministic for the same input.
    * During Stage-2 training, the complete reverse trajectory remains in the
      computation graph, enabling joint optimization with MST++.
    """

    def __init__(
        self,
        denoiser,
        prior_dim=256,
        train_timesteps=100,
        sample_steps=8,
        beta_schedule="cosine",
        beta_start=1e-4,
        beta_end=2e-2,
        cosine_s=0.008,
        deterministic_seed=0,
        prior_scale=1.0,
    ):
        super().__init__()

        if train_timesteps < 2:
            raise ValueError("train_timesteps must be at least 2.")

        if sample_steps < 1 or sample_steps > train_timesteps:
            raise ValueError(
                "sample_steps must lie in [1, train_timesteps]."
            )

        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive.")

        self.denoiser = denoiser
        self.prior_dim = prior_dim
        self.train_timesteps = train_timesteps
        self.sample_steps = sample_steps
        self.prior_scale = float(prior_scale)

        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start,
                beta_end,
                train_timesteps,
                dtype=torch.float64,
            )
        elif beta_schedule == "cosine":
            steps = train_timesteps + 1
            x = torch.linspace(
                0,
                train_timesteps,
                steps,
                dtype=torch.float64,
            )
            alpha_bar = torch.cos(
                (
                    (x / train_timesteps + cosine_s)
                    / (1 + cosine_s)
                )
                * math.pi
                * 0.5
            ).pow(2)
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError(
                f"Unknown beta_schedule: {beta_schedule}"
            )

        betas = betas.float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

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

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(deterministic_seed))
        fixed_noise = torch.randn(
            1,
            prior_dim,
            generator=generator,
        )
        self.register_buffer(
            "fixed_initial_noise",
            fixed_noise,
        )

    def _sampling_times(self, device):
        """
        Returns descending DDIM timesteps and their previous timestep.

        Example for four steps:
            times      = [99, 66, 33, 0]
            times_prev = [66, 33, 0, -1]
        """
        times = torch.linspace(
            self.train_timesteps - 1,
            0,
            self.sample_steps,
            device=device,
        ).round().long()

        # Preserve descending order while removing any accidental duplicates.
        unique_times = []
        for value in times.tolist():
            if not unique_times or value != unique_times[-1]:
                unique_times.append(value)

        times = torch.tensor(
            unique_times,
            device=device,
            dtype=torch.long,
        )

        times_prev = torch.cat(
            [
                times[1:],
                torch.full(
                    (1,),
                    -1,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )

        return times, times_prev

    def normalize_prior(self, prior):
        return prior / self.prior_scale

    def denormalize_prior(self, prior):
        return prior * self.prior_scale

    def q_sample(self, clean_prior, timestep, noise=None):
        """Forward diffusion q(z_t | z_0)."""
        if noise is None:
            noise = torch.randn_like(clean_prior)

        sqrt_alpha = _extract_schedule_value(
            self.sqrt_alphas_cumprod,
            timestep,
            clean_prior,
        )
        sqrt_one_minus_alpha = _extract_schedule_value(
            self.sqrt_one_minus_alphas_cumprod,
            timestep,
            clean_prior,
        )

        return (
            sqrt_alpha * clean_prior
            + sqrt_one_minus_alpha * noise
        )

    def predict_clean_from_noise(
        self,
        noisy_prior,
        timestep,
        predicted_noise,
    ):
        alpha_bar = _extract_schedule_value(
            self.alphas_cumprod,
            timestep,
            noisy_prior,
        )

        return (
            noisy_prior
            - torch.sqrt(1.0 - alpha_bar)
            * predicted_noise
        ) / torch.sqrt(alpha_bar.clamp_min(1e-8))

    def ddim_step(
        self,
        noisy_prior,
        timestep,
        previous_timestep,
        condition,
    ):
        """
        One deterministic DDIM step with eta=0.

        z_{t-1} =
            sqrt(alpha_bar_{t-1}) * predicted_z0
            + sqrt(1-alpha_bar_{t-1}) * predicted_epsilon

        There is deliberately no sigma_t * random_noise term.
        """
        predicted_noise = self.denoiser(
            noisy_prior,
            timestep,
            condition,
        )

        predicted_clean = self.predict_clean_from_noise(
            noisy_prior,
            timestep,
            predicted_noise,
        )

        if previous_timestep < 0:
            previous_prior = predicted_clean
        else:
            previous_batch_timestep = torch.full_like(
                timestep,
                previous_timestep,
            )

            previous_alpha_bar = _extract_schedule_value(
                self.alphas_cumprod,
                previous_batch_timestep,
                noisy_prior,
            )

            previous_prior = (
                torch.sqrt(previous_alpha_bar)
                * predicted_clean
                + torch.sqrt(1.0 - previous_alpha_bar)
                * predicted_noise
            )

        return previous_prior, predicted_clean, predicted_noise

    def reverse_from_latent(
        self,
        initial_latent,
        condition,
        return_trajectory=False,
    ):
        """
        Runs the complete eta=0 DDIM trajectory.

        This method is differentiable and is used for joint Stage-2 training.
        """
        latent = initial_latent
        trajectory = []

        times, previous_times = self._sampling_times(
            initial_latent.device
        )

        for current_time, previous_time in zip(
            times.tolist(),
            previous_times.tolist(),
        ):
            timestep = torch.full(
                (latent.shape[0],),
                current_time,
                device=latent.device,
                dtype=torch.long,
            )

            latent, predicted_clean, predicted_noise = self.ddim_step(
                latent,
                timestep,
                previous_time,
                condition,
            )

            if return_trajectory:
                trajectory.append(
                    {
                        "timestep": current_time,
                        "latent": latent,
                        "predicted_clean": predicted_clean,
                        "predicted_noise": predicted_noise,
                    }
                )

        return latent, trajectory

    def training_forward(
        self,
        teacher_prior,
        condition,
        return_trajectory=False,
    ):
        """
        Stage-2 training path.

        1. A conventional random-timestep epsilon objective trains the denoiser.
        2. The teacher prior is noised at the last timestep.
        3. The full deterministic DDIM reverse path estimates the prior.
        4. The estimated prior remains connected to MST++ for joint training.
        """
        normalized_teacher = self.normalize_prior(
            teacher_prior
        )

        batch_size = teacher_prior.shape[0]
        device = teacher_prior.device

        # Random-timestep denoising objective.
        random_timestep = torch.randint(
            low=0,
            high=self.train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        target_noise = torch.randn_like(
            normalized_teacher
        )

        random_noisy_prior = self.q_sample(
            normalized_teacher,
            random_timestep,
            noise=target_noise,
        )

        predicted_noise = self.denoiser(
            random_noisy_prior,
            random_timestep,
            condition,
        )

        diffusion_noise_loss = F.mse_loss(
            predicted_noise,
            target_noise,
        )

        # Joint full-trajectory prior estimation.
        final_timestep = torch.full(
            (batch_size,),
            self.train_timesteps - 1,
            device=device,
            dtype=torch.long,
        )

        trajectory_noise = torch.randn_like(
            normalized_teacher
        )

        initial_latent = self.q_sample(
            normalized_teacher,
            final_timestep,
            noise=trajectory_noise,
        )

        normalized_prediction, trajectory = (
            self.reverse_from_latent(
                initial_latent,
                condition,
                return_trajectory=return_trajectory,
            )
        )

        predicted_prior = self.denormalize_prior(
            normalized_prediction
        )

        return {
            "predicted_prior": predicted_prior,
            "diffusion_noise_loss": diffusion_noise_loss,
            "trajectory": trajectory,
            "initial_latent": initial_latent,
        }

    @torch.no_grad()
    def sample(
        self,
        condition,
        initial_noise=None,
        return_trajectory=False,
    ):
        """
        Deterministic inference.

        When initial_noise is omitted, the same registered noise vector is
        expanded for every sample. Since DDIM uses eta=0, repeated inference
        for the same RGB input yields the same prior.
        """
        batch_size = condition.shape[0]

        if initial_noise is None:
            initial_noise = self.fixed_initial_noise.expand(
                batch_size,
                -1,
            ).clone()
        else:
            if initial_noise.shape != (
                batch_size,
                self.prior_dim,
            ):
                raise ValueError(
                    "initial_noise must have shape "
                    f"[{batch_size},{self.prior_dim}], "
                    f"got {tuple(initial_noise.shape)}."
                )

        normalized_prediction, trajectory = (
            self.reverse_from_latent(
                initial_noise,
                condition,
                return_trajectory=return_trajectory,
            )
        )

        return {
            "predicted_prior": self.denormalize_prior(
                normalized_prediction
            ),
            "trajectory": trajectory,
            "initial_latent": initial_noise,
        }


# ============================================================================
# PRIOR-TO-RGB INPUT ADAPTER
# ============================================================================

class PriorToRGBAdapter(nn.Module):
    """
    Converts [B, prior_dim] into a small spatial RGB residual.

    The modified MST++ input is:

        conditioned_rgb = rgb + alpha * delta_rgb

    where:
        delta_rgb is bounded by max_delta through tanh;
        alpha is learned and starts close to zero;
        the last residual convolution starts from zero.

    Consequently, training begins with:
        conditioned_rgb == rgb
    """

    def __init__(
        self,
        prior_dim=256,
        rgb_channels=3,
        prior_channels=16,
        hidden_channels=32,
        max_delta=0.10,
        initial_gate_logit=-4.0,
    ):
        super().__init__()

        if max_delta <= 0:
            raise ValueError(
                "max_delta must be positive."
            )

        self.prior_dim = prior_dim
        self.rgb_channels = rgb_channels
        self.max_delta = float(max_delta)

        self.prior_projection = nn.Sequential(
            nn.Linear(
                prior_dim,
                prior_channels,
            ),
            nn.GELU(),
            nn.Linear(
                prior_channels,
                prior_channels,
            ),
        )

        self.residual_generator = nn.Sequential(
            nn.Conv2d(
                rgb_channels + prior_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                rgb_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        nn.init.zeros_(
            self.residual_generator[-1].weight
        )
        nn.init.zeros_(
            self.residual_generator[-1].bias
        )

        self.gate_logit = nn.Parameter(
            torch.tensor(float(initial_gate_logit))
        )

    def forward(self, rgb, prior):
        if rgb.ndim != 4:
            raise ValueError(
                f"Expected RGB [B,C,H,W], got {tuple(rgb.shape)}."
            )

        if prior.ndim != 2:
            raise ValueError(
                f"Expected prior [B,{self.prior_dim}], "
                f"got {tuple(prior.shape)}."
            )

        if rgb.shape[0] != prior.shape[0]:
            raise ValueError(
                "RGB and prior batch sizes must match."
            )

        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f"Expected {self.rgb_channels} RGB channels, "
                f"got {rgb.shape[1]}."
            )

        if prior.shape[1] != self.prior_dim:
            raise ValueError(
                f"Expected prior dimension {self.prior_dim}, "
                f"got {prior.shape[1]}."
            )

        _, _, height, width = rgb.shape

        prior_map = self.prior_projection(prior)
        prior_map = prior_map[:, :, None, None].expand(
            -1,
            -1,
            height,
            width,
        )

        fusion_input = torch.cat(
            [rgb, prior_map],
            dim=1,
        )

        delta_rgb = self.max_delta * torch.tanh(
            self.residual_generator(
                fusion_input
            )
        )

        gate = torch.sigmoid(
            self.gate_logit
        )

        conditioned_rgb = rgb + gate * delta_rgb

        return conditioned_rgb, delta_rgb, gate


# ============================================================================
# STAGE 1
# ============================================================================

class Stage1InputPriorMSTPP(nn.Module):
    """
    Stage-1 oracle-prior reconstruction model.

    RGB + GT HSI -> FFT CPEN -> input adapter -> MST++, with the same prior injected at every transformer block.
    """

    def __init__(
        self,
        cpen=None,
        mstpp=None,
        rgb_channels=3,
        hsi_channels=31,
        cpen_feats=64,
        cpen_res_blocks=6,
        prior_dim=256,
        unshuffle_factor=4,
        mst_stages=3,
        prior_channels=16,
        adapter_hidden_channels=32,
        max_delta=0.10,
        transformer_prior_max_delta=0.10,
        frequency_grid=(16, 9),
        frequency_heads=4,
        frequency_blocks=2,
        frequency_dropout=0.0,
    ):
        super().__init__()

        self.prior_dim = prior_dim

        self.cpen = (
            cpen
            if cpen is not None
            else Stage1CPEN(
                rgb_channels=rgb_channels,
                hsi_channels=hsi_channels,
                n_feats=cpen_feats,
                n_encoder_res=cpen_res_blocks,
                prior_dim=prior_dim,
                unshuffle_factor=unshuffle_factor,
                frequency_grid=frequency_grid,
                frequency_heads=frequency_heads,
                frequency_blocks=frequency_blocks,
                frequency_dropout=frequency_dropout,
            )
        )

        self.mstpp = (
            mstpp
            if mstpp is not None
            else MST_Plus_Plus(
                in_channels=rgb_channels,
                out_channels=hsi_channels,
                n_feat=hsi_channels,
                stage=mst_stages,
                prior_dim=prior_dim,
                prior_max_delta=transformer_prior_max_delta,
            )
        )

        self.input_adapter = PriorToRGBAdapter(
            prior_dim=prior_dim,
            rgb_channels=rgb_channels,
            prior_channels=prior_channels,
            hidden_channels=adapter_hidden_channels,
            max_delta=max_delta,
        )

    @staticmethod
    def _unpack_cpen_output(
        cpen_output: Any,
    ) -> Tuple[torch.Tensor, list]:
        if isinstance(cpen_output, (tuple, list)):
            prior = cpen_output[0]

            if len(cpen_output) > 1:
                prior_list = cpen_output[1]
            else:
                prior_list = [prior]
        else:
            prior = cpen_output
            prior_list = [prior]

        return prior, prior_list

    def forward(self, rgb, gt_hsi):
        oracle_prior, prior_list = (
            self._unpack_cpen_output(
                self.cpen(rgb, gt_hsi)
            )
        )

        (
            conditioned_rgb,
            prior_rgb_residual,
            prior_gate,
        ) = self.input_adapter(
            rgb,
            oracle_prior,
        )

        predicted_hsi = self.mstpp(
            conditioned_rgb,
            oracle_prior,
        )

        return {
            "pred_hsi": predicted_hsi,
            "prior": oracle_prior,
            "prior_list": prior_list,
            "conditioned_rgb": conditioned_rgb,
            "prior_rgb_residual": prior_rgb_residual,
            "prior_gate": prior_gate,
        }




# ============================================================================
# STAGE 2: PURE CPEN-PRIOR DISTILLATION
# ============================================================================

class Stage2CPENPriorDiffusion(nn.Module):
    """
    Predicts the frozen Stage-1 CPEN output from RGB using deterministic DDIM.

    Trainable Stage-2 modules
    -------------------------
    * condition_encoder
    * prior_diffusion

    Frozen Stage-1 modules
    ----------------------
    * teacher_cpen
    * input_adapter
    * mstpp, including every per-transformer prior-fusion layer

    Training
    --------
        model.train()
        output = model(rgb, gt_hsi)

    This returns the predicted and teacher priors plus the diffusion loss.
    HSI reconstruction is skipped by default during training.

    Inference
    ---------
        model.eval()
        output = model(rgb)

    In evaluation mode, HSI reconstruction is performed by default with the
    frozen Stage-1 adapter and MST++.

    To predict only the prior at inference:
        output = model(rgb, reconstruct_hsi=False)
    """

    def __init__(
        self,
        condition_encoder,
        prior_diffusion,
        teacher_cpen,
        input_adapter,
        mstpp,
    ):
        super().__init__()

        self.condition_encoder = condition_encoder
        self.prior_diffusion = prior_diffusion

        self.teacher_cpen = teacher_cpen
        self.input_adapter = input_adapter
        self.mstpp = mstpp

        self._freeze_stage1_modules()

    def _freeze_stage1_modules(self):
        """Permanently freeze all modules learned in Stage 1."""
        for module in (
            self.teacher_cpen,
            self.input_adapter,
            self.mstpp,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _extract_prior(cpen_output):
        if isinstance(cpen_output, (tuple, list)):
            return cpen_output[0]
        return cpen_output

    def train(self, mode=True):
        """
        Train only the RGB condition encoder and DDIM prior diffusion module.
        """
        super().train(mode)

        # model.train() recursively switches every child to training mode.
        # Restore the frozen Stage-1 modules to evaluation mode.
        self.teacher_cpen.eval()
        self.input_adapter.eval()
        self.mstpp.eval()

        self.condition_encoder.train(mode)
        self.prior_diffusion.train(mode)

        return self

    def diffusion_parameters(self):
        """
        Returns exactly the parameters that should be optimized in Stage 2.
        """
        for parameter in self.condition_encoder.parameters():
            if parameter.requires_grad:
                yield parameter

        for parameter in self.prior_diffusion.parameters():
            if parameter.requires_grad:
                yield parameter

    @torch.no_grad()
    def extract_teacher_prior(self, rgb, gt_hsi):
        """
        Computes the target CPEN vector:
            z_target = frozen_CPEN(rgb, gt_hsi)
        """
        return self._extract_prior(
            self.teacher_cpen(
                rgb,
                gt_hsi,
            )
        )

    def predict_prior(
        self,
        rgb,
        target_prior=None,
        initial_noise=None,
        return_trajectory=False,
    ):
        """
        Predicts the compact CPEN prior.

        During training, target_prior must be supplied and the method returns:
            predicted_prior
            diffusion_noise_loss

        During evaluation, deterministic DDIM sampling starts from the fixed
        registered latent unless initial_noise is explicitly supplied.
        """
        condition = self.condition_encoder(rgb)

        if self.training:
            if target_prior is None:
                raise ValueError(
                    "Training requires target_prior from the frozen CPEN."
                )

            diffusion_output = self.prior_diffusion.training_forward(
                teacher_prior=target_prior,
                condition=condition,
                return_trajectory=return_trajectory,
            )

            diffusion_noise_loss = diffusion_output[
                "diffusion_noise_loss"
            ]
        else:
            diffusion_output = self.prior_diffusion.sample(
                condition=condition,
                initial_noise=initial_noise,
                return_trajectory=return_trajectory,
            )

            diffusion_noise_loss = None

        return {
            "predicted_prior": diffusion_output[
                "predicted_prior"
            ],
            "condition": condition,
            "diffusion_noise_loss": diffusion_noise_loss,
            "prior_trajectory": diffusion_output["trajectory"],
            "initial_prior_latent": diffusion_output[
                "initial_latent"
            ],
        }

    @torch.no_grad()
    def reconstruct_hsi(self, rgb, predicted_prior):
        """
        Uses the frozen Stage-1 reconstruction path.

        No gradients are allowed through this function because Stage 2 is
        defined strictly as CPEN-prior prediction.
        """
        (
            conditioned_rgb,
            prior_rgb_residual,
            prior_gate,
        ) = self.input_adapter(
            rgb,
            predicted_prior,
        )

        predicted_hsi = self.mstpp(
            conditioned_rgb,
            predicted_prior,
        )

        return {
            "pred_hsi": predicted_hsi,
            "conditioned_rgb": conditioned_rgb,
            "prior_rgb_residual": prior_rgb_residual,
            "prior_gate": prior_gate,
        }

    def forward(
        self,
        rgb,
        gt_hsi=None,
        initial_noise=None,
        reconstruct_hsi=None,
        return_trajectory=False,
    ):
        """
        Training:
            output = model(rgb, gt_hsi)
            reconstruct_hsi defaults to False.

        Evaluation:
            output = model(rgb)
            reconstruct_hsi defaults to True.
        """
        if reconstruct_hsi is None:
            reconstruct_hsi = not self.training

        teacher_prior = None

        if self.training:
            if gt_hsi is None:
                raise ValueError(
                    "Stage-2 training requires gt_hsi to obtain the frozen "
                    "CPEN target prior."
                )

            teacher_prior = self.extract_teacher_prior(
                rgb,
                gt_hsi,
            )

        prior_output = self.predict_prior(
            rgb=rgb,
            target_prior=teacher_prior,
            initial_noise=initial_noise,
            return_trajectory=return_trajectory,
        )

        output = {
            **prior_output,
            "teacher_prior": teacher_prior,
            "pred_hsi": None,
            "conditioned_rgb": None,
            "prior_rgb_residual": None,
            "prior_gate": None,
        }

        if reconstruct_hsi:
            reconstruction_output = self.reconstruct_hsi(
                rgb,
                prior_output["predicted_prior"],
            )
            output.update(reconstruction_output)

        return output


# Compatibility aliases.
Stage2DDIMPriorMSTPP = Stage2CPENPriorDiffusion
Stage2InputPriorMSTPP = Stage2CPENPriorDiffusion


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_stage1_model(
    rgb_channels=3,
    hsi_channels=31,
    cpen_feats=64,
    cpen_res_blocks=6,
    prior_dim=256,
    unshuffle_factor=4,
    mst_stages=3,
    prior_channels=16,
    adapter_hidden_channels=32,
    max_delta=0.10,
    transformer_prior_max_delta=0.10,
    frequency_grid=(16, 9),
    frequency_heads=4,
    frequency_blocks=2,
    frequency_dropout=0.0,
):
    """Constructs Stage 1: CPEN + prior-conditioned MST++."""

    return Stage1InputPriorMSTPP(
        cpen=None,
        mstpp=None,
        rgb_channels=rgb_channels,
        hsi_channels=hsi_channels,
        cpen_feats=cpen_feats,
        cpen_res_blocks=cpen_res_blocks,
        prior_dim=prior_dim,
        unshuffle_factor=unshuffle_factor,
        mst_stages=mst_stages,
        prior_channels=prior_channels,
        adapter_hidden_channels=adapter_hidden_channels,
        max_delta=max_delta,
        transformer_prior_max_delta=transformer_prior_max_delta,
        frequency_grid=frequency_grid,
        frequency_heads=frequency_heads,
        frequency_blocks=frequency_blocks,
        frequency_dropout=frequency_dropout,
    )


def build_stage2_from_stage1(
    stage1_model,
    condition_encoder=None,
    denoiser=None,
    rgb_channels=3,
    prior_dim=256,
    condition_dim=256,
    condition_feats=64,
    condition_res_blocks=6,
    unshuffle_factor=4,
    denoiser_hidden_dim=512,
    denoiser_time_dim=128,
    denoiser_blocks=5,
    denoiser_dropout=0.0,
    train_timesteps=100,
    sample_steps=8,
    beta_schedule="cosine",
    beta_start=1e-4,
    beta_end=2e-2,
    deterministic_seed=0,
    prior_scale=1.0,
    initialise_condition_from_cpen=True,
    copy_deeper_condition_blocks=False,
    freeze_mstpp=True,
):
    """
    Builds pure-prior-distillation Stage 2 from a trained Stage-1 model.

    `freeze_mstpp` is retained for compatibility but must remain True because
    Stage 2 is explicitly defined to train only the prior diffusion subsystem.
    """
    if not freeze_mstpp:
        raise ValueError(
            "Stage 2 is prior-only distillation; MST++ must remain frozen."
        )

    teacher_cpen = copy.deepcopy(
        stage1_model.cpen
    )
    input_adapter = copy.deepcopy(
        stage1_model.input_adapter
    )
    mstpp = copy.deepcopy(
        stage1_model.mstpp
    )

    if condition_encoder is None:
        condition_encoder = Stage2CPEN(
            rgb_channels=rgb_channels,
            n_feats=condition_feats,
            n_res_blocks=condition_res_blocks,
            condition_dim=condition_dim,
            unshuffle_factor=unshuffle_factor,
        )

    if initialise_condition_from_cpen:
        condition_encoder.initialise_from_cpen(
            teacher_cpen,
            copy_deeper_blocks=copy_deeper_condition_blocks,
        )

    if denoiser is None:
        denoiser = PriorDenoiser(
            prior_dim=prior_dim,
            condition_dim=condition_dim,
            hidden_dim=denoiser_hidden_dim,
            time_dim=denoiser_time_dim,
            num_blocks=denoiser_blocks,
            dropout=denoiser_dropout,
        )

    prior_diffusion = DeterministicPriorDDIM(
        denoiser=denoiser,
        prior_dim=prior_dim,
        train_timesteps=train_timesteps,
        sample_steps=sample_steps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        deterministic_seed=deterministic_seed,
        prior_scale=prior_scale,
    )

    return Stage2CPENPriorDiffusion(
        condition_encoder=condition_encoder,
        prior_diffusion=prior_diffusion,
        teacher_cpen=teacher_cpen,
        input_adapter=input_adapter,
        mstpp=mstpp,
    )







