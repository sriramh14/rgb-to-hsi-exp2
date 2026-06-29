#!/usr/bin/env python3
"""
Simple RGB-only visualization for a resumed Stage-1 DiffIR RGB-to-HSI model.

The model is forced to use HSI scale/modulation = 0.
No HSI tensor is passed into the model.

Example:
    python visualize_rgb_only.py \
        --checkpoint checkpoints/diffir_rgb2hsi_stage1_hsi_annealed_latest.pth \
        --rgb sample.png \
        --output-dir rgb_only_result
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.spec_prior_with_hsi_weight import DiffIRS1RGB2HSI, ModelConfig
checkpoint_dir = "/kaggle/working/rgb-to-hsi-exp2/DiffIR_RGB2HSI/DiffIR_RGB2HSI/checkpoints/diffir_rgb2hsi_stage1_hsi_annealed_best.pth"
rgb = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB/ARAD_1K_0001.jpg"
output_dir = "/kaggle/working/rgb-to-hsi-exp2/DiffIR_RGB2HSI/DiffIR_RGB2HSI/checkpoints"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    #parser.add_argument("--checkpoint", type=Path, required=True)
    #parser.add_argument("--rgb", type=Path, required=True)
    #parser.add_argument("--output-dir", type=Path, default=Path("rgb_only_result"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict.")

    return checkpoint


def load_rgb(path: Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    rgb_np = np.asarray(image, dtype=np.float32) / 255.0

    rgb_tensor = (
        torch.from_numpy(rgb_np.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device)
    )

    return rgb_tensor, rgb_np


def make_pseudo_rgb(hsi: np.ndarray) -> np.ndarray:
    bands = hsi.shape[0]
    indices = [25, 15, 5] if bands >= 26 else [bands - 1, bands // 2, 0]

    pseudo_rgb = np.stack([hsi[index] for index in indices], axis=-1)

    for channel in range(3):
        values = pseudo_rgb[..., channel]
        minimum = float(values.min())
        maximum = float(values.max())

        if maximum > minimum:
            pseudo_rgb[..., channel] = (values - minimum) / (maximum - minimum)
        else:
            pseudo_rgb[..., channel] = 0.0

    return np.clip(pseudo_rgb, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    #args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(Path(checkpoint_dir), device)

    if int(checkpoint.get("stage", 1)) != 1:
        raise ValueError("This script expects a Stage-1 checkpoint.")

    config = ModelConfig.from_dict(checkpoint["model_config"])

    model = DiffIRS1RGB2HSI(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    # No HSI is used as model input.
    model.set_hsi_scale(0.0)
    model.eval()

    rgb_tensor, rgb_np = load_rgb(rgb, device)

    with torch.inference_mode():
        pred_hsi, prior = model.forward_rgb_only(rgb_tensor)

    pred_hsi_np = pred_hsi[0].float().cpu().numpy()
    prior_np = prior[0].float().cpu().numpy()

    np.save(Path(output_dir) / "predicted_hsi.npy", pred_hsi_np)
    np.save(Path(output_dir) / "predicted_prior.npy", prior_np)

    pseudo_rgb = make_pseudo_rgb(pred_hsi_np)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_np)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(pseudo_rgb)
    axes[1].set_title("Predicted HSI pseudo-RGB")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(
        path(output_dir) / "visualization.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"HSI scale: {model.get_hsi_scale():.1f}")
    print(f"Predicted HSI shape: {pred_hsi_np.shape}")
    print(f"Prior shape: {prior_np.shape}")
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
