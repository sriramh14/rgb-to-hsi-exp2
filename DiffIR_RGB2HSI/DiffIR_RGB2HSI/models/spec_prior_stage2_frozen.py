"""Frozen-generator Stage-2 variant for the spatial-prior MST++/DiffIR model.

Place this file beside ``models/spec_prior_model.py`` as
``models/spec_prior_frozen_model.py``.

Stage 1 is unchanged.  In Stage 2, the complete pretrained reconstruction
network ``G`` (including all MST++ transformer blocks, convolutional paths,
FiLM adapters, and output layers) is frozen permanently.  Gradients are still
allowed to propagate through ``G`` to the predicted prior, so the RGB
condition encoder and prior denoiser can be trained by both prior supervision
and the final HSI reconstruction loss.

The default public interface remains compatible with the original training
code:

    Stage 1: model(rgb, hsi_gt) -> pred_hsi, oracle_prior
    Stage 2: model(rgb, target_prior=...) -> pred_hsi, prior_sequence

The minimally modified training script can additionally request the RGB
condition prior:

    pred_hsi, prior_sequence, condition_prior = model(
        rgb, target_prior=target_prior, return_aux=True
    )

For Stage 2, ``prior_sequence`` contains the denoiser's x0 estimates at every
reverse step.  This makes deep supervision of the complete diffusion trajectory
well defined; the final entry is the final clean-prior estimate.
"""

from __future__ import annotations

from typing import List

import torch

from .spec_prior_model import (
    ModelConfig,
    DiffIRS1RGB2HSI,
    DiffIRS2RGB2HSI as _OriginalDiffIRS2RGB2HSI,
)


class DiffIRS2RGB2HSI(_OriginalDiffIRS2RGB2HSI):
    """Stage 2 with a permanently frozen pretrained MST++ generator."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.freeze_generator()

    def freeze_generator(self) -> None:
        """Freeze every parameter in the reconstruction generator ``G``."""
        for parameter in self.G.parameters():
            parameter.requires_grad_(False)
        self.G.eval()

    def initialize_generator_from_stage1(self, stage1: DiffIRS1RGB2HSI) -> None:
        """Copy Stage-1 generator weights, then enforce permanent freezing."""
        super().initialize_generator_from_stage1(stage1)
        self.freeze_generator()

    def train(self, mode: bool = True):
        """Train the prior branch while always keeping ``G`` in eval mode."""
        super().train(mode)
        self.G.eval()
        return self

    def forward(
        self,
        rgb: torch.Tensor,
        target_prior: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if not self.training:
            # Preserve the original Stage-2 evaluation/sampling behaviour.
            return super().forward(
                rgb,
                target_prior=target_prior,
                initial_noise=initial_noise,
            )

        if target_prior is None:
            raise ValueError("Stage-2 training requires the frozen Stage-1 target prior")

        batch = rgb.shape[0]
        device = rgb.device

        # Compute the RGB-conditioned prior exactly once so it can receive
        # direct supervision without repeating the condition encoder forward.
        condition_prior = self.diffusion.condition(rgb)
        if condition_prior.shape != target_prior.shape:
            raise ValueError(
                f"Condition prior shape {tuple(condition_prior.shape)} must match "
                f"target prior shape {tuple(target_prior.shape)}."
            )

        final_t = torch.full(
            (batch,),
            self.config.timesteps - 1,
            device=device,
            dtype=torch.long,
        )
        prior_t = self.diffusion.q_sample(target_prior, final_t)

        # Supervise the clean-prior (x0) prediction made at every reverse step,
        # rather than supervising only intermediate posterior means.
        x0_sequence: List[torch.Tensor] = []
        for step in reversed(range(self.config.timesteps)):
            timestep = torch.full(
                (batch,),
                step,
                device=device,
                dtype=torch.long,
            )
            prior_t, predicted_x0 = self.diffusion.reverse_step(
                prior_t,
                timestep,
                condition_prior,
            )
            x0_sequence.append(predicted_x0)

        # Do not wrap this call in torch.no_grad().  G has no trainable
        # parameters, but reconstruction gradients must pass through G to the
        # predicted prior and therefore to the prior network.
        pred_hsi = self.G(rgb, prior_t)

        if return_aux:
            return pred_hsi, x0_sequence, condition_prior
        return pred_hsi, x0_sequence


def build_model(stage: int, config: ModelConfig) -> torch.nn.Module:
    if stage == 1:
        return DiffIRS1RGB2HSI(config)
    if stage == 2:
        return DiffIRS2RGB2HSI(config)
    raise ValueError("stage must be 1 or 2")


__all__ = [
    "ModelConfig",
    "DiffIRS1RGB2HSI",
    "DiffIRS2RGB2HSI",
    "build_model",
]
