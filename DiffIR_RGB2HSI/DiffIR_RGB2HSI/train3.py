#!/usr/bin/env python3
"""Single-file training/evaluation entry point for DiffIR RGB-to-HSI.

Edit the CONFIG section and run:

    python train.py

Stages
------
STAGE = 1: train the oracle-prior Stage-1 model using RGB + GT HSI.
STAGE = 2: load/freeze Stage 1 and train the compact-prior diffusion model.
MODE  = "eval": evaluate the selected checkpoint.

This version is intentionally defensive: it accepts tuple/list batches such as
``(rgb, hsi)`` as well as dictionary batches such as ``{"rgb": rgb, "hsi": hsi}``.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from dataset.random_arad_loader import load_random_arad1k_samples
from loss import compute_metrics, prior_kd_loss, prior_l1_loss, reconstruction_loss

#Change this to metamer_aware_model or spec_prior_model
from models.spec_prior_model import DiffIRS1RGB2HSI, DiffIRS2RGB2HSI, ModelConfig


# ==================================================
# CONFIG
# ==================================================

MODE = "train"                 # "train" or "eval"
STAGE = 1                       # 1: Stage-1 oracle spectral prior, 2: Stage-2 spectral-prior diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234

# Dataset configuration. This repository's dataset_loader.py expects the
# HuggingFace ARAD/NTIRE folder names under DATA_ROOT:
#   DATA_ROOT/NTIRE2020_Train_Spectral/*.mat
#   DATA_ROOT/NTIRE2020_Train_RealWorld/*.jpg
DATA_ROOT = "data"
HSI_KEY = "cube"
DOWNLOAD_DATA = True
TRAIN_IMAGES = 200
TOTAL_IMAGES = 230
EVAL_RANDOM_IMAGES = 50
EVAL_RANDOM_TOTAL_IMAGES = 1000

BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"

NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# Reconstruction loss used by both stages: "mrae", "l1", or "mse".
RECONSTRUCTION_LOSS = "mrae"
MRAE_EPS = 1e-6

# Stage-2 spatial spectral-prior supervision.
LAMBDA_PRIOR_L1 = 1.0
LAMBDA_PRIOR_KD = 0.0
KD_TEMPERATURE = 0.15

# Validation MRAE controls LR scheduling, best checkpoint, and early stopping.
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Architecture.
NUM_BANDS = 31
DIM = 31
NUM_BLOCKS = (1, 1, 1)
MST_STAGES = 3
MST_STAGE_DEPTH = 2
MST_FFN_MULT = 4
PAD_MULTIPLE = 8
BIAS = False

# Prior conditioning is active in the spectral-prior model.
USE_PRIOR_CONDITIONING = True
PRIOR_DOWNSAMPLE_FACTOR = 4
PRIOR_FEAT_DIM = 64
USE_SPECTRAL_PRIOR_OUTPUT_SKIP = True
SPECTRAL_PRIOR_OUTPUT_SCALE_INIT = 1.0

# Kept for compatibility with model configs/checkpoints.
NUM_REFINEMENT_BLOCKS = 0
HEADS = (1, 2, 4, 8)
FFN_EXPANSION_FACTOR = 2.66
LAYER_NORM_TYPE = "WithBias"
PRIOR_DIM = 256
N_ENCODER_RES = 6
USE_RGB_TO_HSI_SKIP = False

# Stage-2 diffusion.
N_DENOISE_RES = 4               #original value was 2
DIFFUSION_TIMESTEPS = 4

#Original schedule was 0.1 to 0.99
LINEAR_START = 1e-4            
LINEAR_END = 0.05

CHECKPOINT_DIR = Path("checkpoints")
STAGE1_BEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_best.pth"
STAGE1_BEST_LOSS_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_best_loss.pth"
STAGE1_LATEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_latest.pth"
STAGE2_BEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_best.pth"
STAGE2_BEST_LOSS_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_best_loss.pth"
STAGE2_LATEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_latest.pth"

TEACHER_CHECKPOINT = STAGE1_BEST_PATH
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# COMMAND LINE OVERRIDES
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DiffIR RGB-to-HSI"
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=STAGE,
        help="Training stage: 1 for oracle prior training, 2 for diffusion prior training.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default=MODE,
        help="Run mode: train or eval.",
    )

    return parser.parse_args()


args = parse_args()

STAGE = args.stage
MODE = args.mode

# ==================================================
# REPRODUCIBILITY
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ==================================================
# SMALL UTILITIES
# ==================================================


def make_model_config() -> ModelConfig:
    return ModelConfig(
        num_bands=NUM_BANDS,
        dim=DIM,
        num_blocks=NUM_BLOCKS,
        mst_stages=MST_STAGES,
        mst_stage_depth=MST_STAGE_DEPTH,
        mst_ffn_mult=MST_FFN_MULT,
        bias=BIAS,
        pad_multiple=PAD_MULTIPLE,
        use_prior_conditioning=USE_PRIOR_CONDITIONING,
        prior_downsample_factor=PRIOR_DOWNSAMPLE_FACTOR,
        prior_feat_dim=PRIOR_FEAT_DIM,
        use_spectral_prior_output_skip=USE_SPECTRAL_PRIOR_OUTPUT_SKIP,
        spectral_prior_output_scale_init=SPECTRAL_PRIOR_OUTPUT_SCALE_INIT,
        prior_dim=PRIOR_DIM,
        n_encoder_res=N_ENCODER_RES,
        n_denoise_res=N_DENOISE_RES,
        timesteps=DIFFUSION_TIMESTEPS,
        linear_start=LINEAR_START,
        linear_end=LINEAR_END,
        # Compatibility fields retained in the model config.
        heads=HEADS,
        ffn_expansion_factor=FFN_EXPANSION_FACTOR,
        layer_norm_type=LAYER_NORM_TYPE,
        num_refinement_blocks=NUM_REFINEMENT_BLOCKS,
        use_rgb_to_hsi_skip=USE_RGB_TO_HSI_SKIP,
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
    """Normalize all supported dataset outputs to ``rgb, hsi, name, orig_hw``.

    Supported formats:
      - (rgb, hsi)
      - [rgb, hsi]
      - (rgb, hsi, name)
      - (rgb, hsi, name, orig_hw)
      - {"rgb": rgb, "hsi": hsi}
      - {"lq": rgb, "gt": hsi}
    """
    name = None
    orig_hw = None

    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
        name = batch.get("name", batch.get("filename"))
        orig_hw = batch.get("orig_hw", batch.get("original_hw"))
        if rgb is None or hsi is None:
            raise KeyError(
                "Dictionary batch must contain ('rgb','hsi') or ('lq','gt'). "
                f"Available keys: {list(batch.keys())}"
            )
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("List/tuple batch must contain at least [rgb, hsi].")
        rgb = batch[0]
        hsi = batch[1]
        if len(batch) >= 3:
            name = batch[2]
        if len(batch) >= 4:
            orig_hw = batch[3]
    else:
        raise TypeError(
            "Expected a dict, list, or tuple batch, but received "
            f"{type(batch).__name__}."
        )

    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError(
            "After DataLoader collation, RGB and HSI must be tensors. "
            f"Received rgb={type(rgb).__name__}, hsi={type(hsi).__name__}."
        )

    if rgb.ndim != 4 or hsi.ndim != 4:
        raise ValueError(
            "Expected batched tensors rgb=[B,3,H,W] and hsi=[B,L,H,W], "
            f"got rgb={tuple(rgb.shape)}, hsi={tuple(hsi.shape)}."
        )

    if rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB to have 3 channels, got {rgb.shape[1]}.")

    return rgb, hsi, name, orig_hw


def make_orig_hw_tensor(
    orig_hw: Optional[Any],
    hsi: torch.Tensor,
) -> torch.Tensor:
    """Return [B,2] original height/width tensor even if dataset omitted it."""
    batch_size = hsi.shape[0]
    if orig_hw is None:
        return torch.tensor(
            [[hsi.shape[-2], hsi.shape[-1]]] * batch_size,
            dtype=torch.long,
        )

    if torch.is_tensor(orig_hw):
        value = orig_hw.detach().cpu()
    else:
        value = torch.as_tensor(orig_hw, dtype=torch.long)

    if value.ndim == 1:
        if value.numel() == 2:
            value = value.view(1, 2).repeat(batch_size, 1)
        else:
            raise ValueError(f"orig_hw must contain H,W, got shape {tuple(value.shape)}")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"orig_hw must have shape [B,2], got {tuple(value.shape)}")
    if value.shape[0] != batch_size:
        if value.shape[0] == 1:
            value = value.repeat(batch_size, 1)
        else:
            raise ValueError(
                f"orig_hw batch size {value.shape[0]} does not match tensor batch {batch_size}."
            )
    return value.long()


def crop_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
    sample_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_h = int(orig_hw[sample_index, 0].item())
    original_w = int(orig_hw[sample_index, 1].item())
    return (
        pred[sample_index : sample_index + 1, :, :original_h, :original_w],
        target[sample_index : sample_index + 1, :, :original_h, :original_w],
    )


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ==================================================
# DATA
# ==================================================


def make_dataloaders(device: torch.device) -> Tuple[Optional[DataLoader], DataLoader]:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader: Optional[DataLoader] = None
    if MODE == "train":
        train_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=True,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty. Check DATA_ROOT and ARAD file pairing.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda" and PIN_MEMORY),
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=generator,
            drop_last=False,
        )

    if MODE == "eval":
        val_dataset, _ = load_random_arad1k_samples(
            root_dir=DATA_ROOT,
            num_samples=EVAL_RANDOM_IMAGES,
            seed=VAL_SEED,
            total_images=EVAL_RANDOM_TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
    else:
        val_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=False,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            # Do not redownload during validation if training already prepared files.
            download=False,
        )
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check TRAIN_IMAGES/TOTAL_IMAGES and file pairing.")
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader


# ==================================================
# CHECKPOINTS
# ==================================================


def checkpoint_paths(stage: int) -> Tuple[Path, Path, Path]:
    if stage == 1:
        return STAGE1_BEST_PATH, STAGE1_BEST_LOSS_PATH, STAGE1_LATEST_PATH
    if stage == 2:
        return STAGE2_BEST_PATH, STAGE2_BEST_LOSS_PATH, STAGE2_LATEST_PATH
    raise ValueError("STAGE must be 1 or 2")


def save_checkpoint(
    path: Path,
    *,
    stage: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    config: ModelConfig,
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    payload = {
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": config.to_dict(),
        "best_val_mrae": best_val_mrae,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"{path} is not a complete DiffIR RGB-to-HSI checkpoint. "
            "Expected a dictionary containing at least a 'model' state dict."
        )
    return checkpoint


def load_stage1_teacher(
    path: Union[str, Path],
    device: torch.device,
) -> Tuple[DiffIRS1RGB2HSI, ModelConfig]:
    checkpoint = load_checkpoint(path, device)
    if int(checkpoint.get("stage", -1)) != 1:
        raise ValueError(
            f"Teacher checkpoint must be Stage 1, found stage={checkpoint.get('stage')}."
        )
    teacher_config = ModelConfig.from_dict(checkpoint["model_config"])
    teacher = DiffIRS1RGB2HSI(teacher_config).to(device)
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, teacher_config


# ==================================================
# MODEL CONSTRUCTION
# ==================================================


def build_training_models(
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[DiffIRS1RGB2HSI], ModelConfig]:
    current_config = make_model_config()
    if STAGE == 1:
        return DiffIRS1RGB2HSI(current_config).to(device), None, current_config

    if STAGE != 2:
        raise ValueError("STAGE must be 1 or 2")

    teacher, teacher_config = load_stage1_teacher(TEACHER_CHECKPOINT, device)
    stage2_config = ModelConfig.from_dict(teacher_config.to_dict())
    stage2_config.n_denoise_res = N_DENOISE_RES
    stage2_config.timesteps = DIFFUSION_TIMESTEPS
    stage2_config.linear_start = LINEAR_START
    stage2_config.linear_end = LINEAR_END

    model = DiffIRS2RGB2HSI(stage2_config).to(device)
    model.initialize_generator_from_stage1(teacher)
    return model, teacher, stage2_config


def build_evaluation_model(
    checkpoint: Dict,
    device: torch.device,
) -> Tuple[torch.nn.Module, ModelConfig]:
    checkpoint_stage = int(checkpoint.get("stage", -1))
    if checkpoint_stage != STAGE:
        raise ValueError(
            f"Evaluation checkpoint is Stage {checkpoint_stage}, but STAGE={STAGE}."
        )
    config = ModelConfig.from_dict(checkpoint["model_config"])
    if STAGE == 1:
        model: torch.nn.Module = DiffIRS1RGB2HSI(config)
    elif STAGE == 2:
        model = DiffIRS2RGB2HSI(config)
    else:
        raise ValueError("STAGE must be 1 or 2")
    model = model.to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


# ==================================================
# VALIDATION
# ==================================================


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mrae": 0.0, "rmse": 0.0, "psnr": 0.0, "sam": 0.0, "ssim": 0.0}
    count = 0

    eval_generator = None
    if STAGE == 2:
        try:
            eval_generator = torch.Generator(device=device)
        except TypeError:
            eval_generator = torch.Generator()
        eval_generator.manual_seed(VAL_SEED)

    for batch in val_loader:
        rgb, hsi, _, orig_hw = unpack_batch(batch)
        orig_hw_tensor = make_orig_hw_tensor(orig_hw, hsi)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = hsi.to(device, non_blocking=(device.type == "cuda"))

        if STAGE == 1:
            if not isinstance(model, DiffIRS1RGB2HSI):
                raise TypeError("STAGE=1 validation requires DiffIRS1RGB2HSI")
            pred_hsi, _ = model(rgb, hsi)
        else:
            if not isinstance(model, DiffIRS2RGB2HSI):
                raise TypeError("STAGE=2 validation requires DiffIRS2RGB2HSI")
            prior_h = ceil_div(rgb.shape[-2], model.config.prior_downsample_factor)
            prior_w = ceil_div(rgb.shape[-1], model.config.prior_downsample_factor)
            noise_kwargs = {
                "size": (rgb.shape[0], model.config.num_bands, prior_h, prior_w),
                "device": device,
            }
            if eval_generator is not None:
                noise_kwargs["generator"] = eval_generator
            try:
                initial_noise = torch.randn(**noise_kwargs)
            except TypeError:
                noise_kwargs.pop("generator", None)
                initial_noise = torch.randn(**noise_kwargs)
            pred_hsi = model(rgb, initial_noise=initial_noise)

        for sample_index in range(rgb.shape[0]):
            sample_pred, sample_hsi = crop_sample(pred_hsi, hsi, orig_hw_tensor, sample_index)
            sample_loss = reconstruction_loss(
                sample_pred,
                sample_hsi,
                loss_type=RECONSTRUCTION_LOSS,
                mrae_eps=MRAE_EPS,
            )
            sample_metrics = compute_metrics(sample_pred, sample_hsi, mrae_eps=MRAE_EPS)
            totals["loss"] += float(sample_loss.item())
            for metric_name in ("mrae", "rmse", "psnr", "sam", "ssim"):
                totals[metric_name] += float(sample_metrics[metric_name])
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


def train() -> None:
    if STAGE not in (1, 2):
        raise ValueError("STAGE must be 1 or 2")
    set_seed(SEED)
    device = torch.device(DEVICE)
    train_loader, val_loader = make_dataloaders(device)
    if train_loader is None:
        raise RuntimeError("Training requested but train_loader is None")

    model, teacher, config = build_training_models(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        resume = load_checkpoint(RESUME_CHECKPOINT, device)
        if int(resume.get("stage", -1)) != STAGE:
            raise ValueError("RESUME_CHECKPOINT stage does not match STAGE")
        resume_config = ModelConfig.from_dict(resume["model_config"])
        if resume_config.to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint architecture differs from current CONFIG")
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            lr_scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_val_mrae = float(resume.get("best_val_mrae", math.inf))
        best_val_loss = float(resume.get("best_val_loss", math.inf))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        print(f"Resumed Stage {STAGE} from epoch {start_epoch}")

    best_path, best_loss_path, latest_path = checkpoint_paths(STAGE)

    print(f"Device: {device}")
    print(f"Stage: {STAGE}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch format is normalized with unpack_batch(); tuple/list/dict batches are supported.")
    print(f"Model configuration: {config.to_dict()}")
    if STAGE == 2:
        print(f"Loaded frozen Stage-1 teacher: {TEACHER_CHECKPOINT}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_total = 0.0
        running_reconstruction = 0.0
        running_prior_l1 = 0.0
        running_prior_kd = 0.0
        train_count = 0

        for batch in train_loader:
            rgb, hsi, _, _ = unpack_batch(batch)
            rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
            hsi = hsi.to(device, non_blocking=(device.type == "cuda"))
            batch_size = rgb.shape[0]
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                if STAGE == 1:
                    if not isinstance(model, DiffIRS1RGB2HSI):
                        raise TypeError("STAGE=1 training requires DiffIRS1RGB2HSI")
                    pred_hsi, _ = model(rgb, hsi)
                    rec_loss = reconstruction_loss(
                        pred_hsi,
                        hsi,
                        loss_type=RECONSTRUCTION_LOSS,
                        mrae_eps=MRAE_EPS,
                    )
                    prior_l1 = rec_loss.new_zeros(())
                    prior_kd = rec_loss.new_zeros(())
                    total_loss = rec_loss
                else:
                    if teacher is None or not isinstance(model, DiffIRS2RGB2HSI):
                        raise TypeError("STAGE=2 training requires teacher and DiffIRS2RGB2HSI")
                    with torch.no_grad():
                        target_prior = teacher.E(rgb, hsi)
                    pred_hsi, prior_sequence = model(rgb, target_prior=target_prior)
                    predicted_prior = prior_sequence[-1]
                    rec_loss = reconstruction_loss(
                        pred_hsi,
                        hsi,
                        loss_type=RECONSTRUCTION_LOSS,
                        mrae_eps=MRAE_EPS,
                    )
                    #Old code uncomment if needed
                    #prior_l1 = prior_l1_loss(predicted_prior, target_prior)
                    prior_l1 = sum(
                        prior_l1_loss(p, target_prior) for p in prior_sequence
                    ) / len(prior_sequence)
                    prior_kd = prior_kd_loss(
                        predicted_prior,
                        target_prior,
                        temperature=KD_TEMPERATURE,
                    )
                    total_loss = (
                        rec_loss
                        + LAMBDA_PRIOR_L1 * prior_l1
                        + LAMBDA_PRIOR_KD * prior_kd
                    )

            scaler.scale(total_loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_total += float(total_loss.item()) * batch_size
            running_reconstruction += float(rec_loss.item()) * batch_size
            running_prior_l1 += float(prior_l1.item()) * batch_size
            running_prior_kd += float(prior_kd.item()) * batch_size
            train_count += batch_size

        train_total = running_total / max(train_count, 1)
        train_rec = running_reconstruction / max(train_count, 1)
        train_prior_l1 = running_prior_l1 / max(train_count, 1)
        train_prior_kd = running_prior_kd / max(train_count, 1)

        val_results = validate(model, val_loader, device)
        lr_scheduler.step(val_results["mrae"])
        current_lr = optimizer.param_groups[0]["lr"]

        if STAGE == 1:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} "
                f"| Train Loss {train_total:.6f} "
                f"| Val Loss {val_results['loss']:.6f} "
                f"| Val MRAE {val_results['mrae']:.6f} "
                f"| Val RMSE {val_results['rmse']:.6f} "
                f"| Val SAM {val_results['sam']:.4f} "
                f"| Val PSNR {val_results['psnr']:.4f} "
                f"| Val SSIM {val_results['ssim']:.6f} "
                f"| LR {current_lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} "
                f"| Train Total {train_total:.6f} "
                f"| Train Reconstruction {train_rec:.6f} "
                f"| Train Prior L1 {train_prior_l1:.6f} "
                f"| Train Prior KD {train_prior_kd:.6f} "
                f"| Val Loss {val_results['loss']:.6f} "
                f"| Val MRAE {val_results['mrae']:.6f} "
                f"| Val RMSE {val_results['rmse']:.6f} "
                f"| Val SAM {val_results['sam']:.4f} "
                f"| Val PSNR {val_results['psnr']:.4f} "
                f"| Val SSIM {val_results['ssim']:.6f} "
                f"| LR {current_lr:.2e}"
            )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            save_checkpoint(
                best_loss_path,
                stage=STAGE,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )

        if val_results["mrae"] < best_val_mrae:
            best_val_mrae = val_results["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                stage=STAGE,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            print(f"Saved best Stage-{STAGE} model (Val MRAE: {best_val_mrae:.6f})")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation MRAE improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

        save_checkpoint(
            latest_path,
            stage=STAGE,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            config=config,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered. Best validation MRAE: {best_val_mrae:.6f}")
            break


# ==================================================
# EVALUATION
# ==================================================


def evaluate() -> None:
    if STAGE not in (1, 2):
        raise ValueError("STAGE must be 1 or 2")
    set_seed(SEED)
    device = torch.device(DEVICE)
    _, val_loader = make_dataloaders(device)
    default_path, _, _ = checkpoint_paths(STAGE)
    selected_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else default_path
    checkpoint = load_checkpoint(selected_path, device)
    model, config = build_evaluation_model(checkpoint, device)
    results = validate(model, val_loader, device)

    print(f"Evaluated checkpoint: {selected_path}")
    print(f"Model configuration: {config.to_dict()}")
    print(
        f"MRAE {results['mrae']:.6f} "
        f"| RMSE {results['rmse']:.6f} "
        f"| SAM {results['sam']:.4f} "
        f"| PSNR {results['psnr']:.4f} "
        f"| SSIM {results['ssim']:.6f}"
    )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    if MODE == "train":
        train()
    elif MODE == "eval":
        evaluate()
    else:
        raise ValueError("MODE must be 'train' or 'eval'")


if __name__ == "__main__":
    main()
