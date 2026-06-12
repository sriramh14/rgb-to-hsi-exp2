#!/usr/bin/env python3
"""Train or evaluate DiffIR for ARAD1K RGB-to-HSI reconstruction.

This is intentionally a single, configuration-at-the-top entry point, similar
in style to the user's earlier repositories. Edit the CONFIG section and run:

    python main.py

Workflow
--------
1. Set STAGE = 1 and train the oracle-prior DiffIR model.
2. Set STAGE = 2. The script loads the best Stage-1 checkpoint, freezes the
   Stage-1 teacher, initializes the Stage-2 DIRformer from Stage 1, and trains
   compact-prior diffusion.
3. Set MODE = "eval" to evaluate the selected stage checkpoint.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ARADDataset
from models import DiffIRS1RGB2HSI, DiffIRS2RGB2HSI, ModelConfig


# ==================================================
# CONFIG
# ==================================================

# MAIN STYLE CHANGE: no command-line parser is required. Change values here.
MODE = "train"                 # "train" or "eval"
STAGE = 1                       # 1: oracle-prior pretraining, 2: diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234

# ARAD1K root containing Train_RGB, Train_Spec, and split_txt.
DATA_ROOT = "../dataset"
HSI_KEY = "cube"
NUM_BANDS = 31
HSI_SCALE = 1.0                 # Keep 1.0 for standard ARAD1K cubes in [0, 1].
CLIP_HSI_ON_LOAD = False

PATCH_SIZE = 64                 # Must be divisible by 32 for DIRformer.
STRIDE = 32                     # Use 8 only when you intentionally want many patches.
BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
DATA_CACHE_SIZE = 2

NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# Reconstruction loss used by both stages.
RECONSTRUCTION_LOSS = "mrae"   # "mrae", "l1", or "mse"
MRAE_EPS = 1e-6

# Stage-2 prior supervision.
LAMBDA_PRIOR_L1 = 1.0
LAMBDA_PRIOR_KD = 0.0
KD_TEMPERATURE = 0.15

# Validation MRAE controls LR scheduling, best-model selection, and stopping.
EARLY_STOPPING_PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7

# DiffIR architecture. These defaults follow the provided S1 architecture.
DIM = 48
NUM_BLOCKS = (4, 6, 6, 8)
NUM_REFINEMENT_BLOCKS = 4
HEADS = (1, 2, 4, 8)
FFN_EXPANSION_FACTOR = 2.66
BIAS = False
LAYER_NORM_TYPE = "WithBias"   # "BiasFree" or "WithBias"
PRIOR_DIM = 256
N_ENCODER_RES = 6
USE_RGB_TO_HSI_SKIP = True

# Compact-prior diffusion configuration used in Stage 2.
N_DENOISE_RES = 1
DIFFUSION_TIMESTEPS = 4
LINEAR_START = 0.1
LINEAR_END = 0.99

CHECKPOINT_DIR = Path("checkpoints")
STAGE1_BEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_best.pth"
STAGE1_BEST_LOSS_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_best_loss.pth"
STAGE1_LATEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage1_latest.pth"
STAGE2_BEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_best.pth"
STAGE2_BEST_LOSS_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_best_loss.pth"
STAGE2_LATEST_PATH = CHECKPOINT_DIR / "diffir_rgb2hsi_stage2_latest.pth"

# Stage 2 obtains its oracle prior targets from this frozen Stage-1 model.
TEACHER_CHECKPOINT = STAGE1_BEST_PATH

# Optional complete training-state checkpoint. Use None to start normally.
RESUME_CHECKPOINT: Optional[Path] = None

# Used only when MODE = "eval". None selects the best checkpoint for STAGE.
EVAL_CHECKPOINT: Optional[Path] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ==================================================
# MODEL CONFIGURATION
# ==================================================


def make_model_config() -> ModelConfig:
    return ModelConfig(
        num_bands=NUM_BANDS,
        dim=DIM,
        num_blocks=NUM_BLOCKS,
        num_refinement_blocks=NUM_REFINEMENT_BLOCKS,
        heads=HEADS,
        ffn_expansion_factor=FFN_EXPANSION_FACTOR,
        bias=BIAS,
        layer_norm_type=LAYER_NORM_TYPE,
        prior_dim=PRIOR_DIM,
        n_encoder_res=N_ENCODER_RES,
        n_denoise_res=N_DENOISE_RES,
        timesteps=DIFFUSION_TIMESTEPS,
        linear_start=LINEAR_START,
        linear_end=LINEAR_END,
        use_rgb_to_hsi_skip=USE_RGB_TO_HSI_SKIP,
    )


def verify_patch_size() -> None:
    if PATCH_SIZE <= 0:
        raise ValueError("PATCH_SIZE must be positive")
    if PATCH_SIZE % 32 != 0:
        raise ValueError(
            f"PATCH_SIZE={PATCH_SIZE} is invalid. DIRformer requires a multiple of 32."
        )
    if STRIDE <= 0:
        raise ValueError("STRIDE must be positive")


# ==================================================
# DATA
# ==================================================


def make_dataloaders(device: torch.device) -> Tuple[Optional[DataLoader], DataLoader]:
    # MAIN STYLE CHANGE: ARAD1K split files are selected inside ARAD1KDataset.
    val_dataset = ARAD1KDataset(
        data_root=DATA_ROOT,
        split="valid",
        num_bands=NUM_BANDS,
        hsi_key=HSI_KEY,
        hsi_scale=HSI_SCALE,
        clip_hsi=CLIP_HSI_ON_LOAD,
        required_multiple=32,
        cache_size=DATA_CACHE_SIZE,
    )

    pin_memory = device.type == "cuda"

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=NUM_WORKERS > 0,
    )

    if MODE == "eval":
        return None, val_loader

    train_dataset = ARAD1KDataset(
        data_root=DATA_ROOT,
        split="train",
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        augment=True,
        num_bands=NUM_BANDS,
        hsi_key=HSI_KEY,
        hsi_scale=HSI_SCALE,
        clip_hsi=CLIP_HSI_ON_LOAD,
        required_multiple=32,
        cache_size=DATA_CACHE_SIZE,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader


# ==================================================
# LOSSES AND METRICS
# ==================================================


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if RECONSTRUCTION_LOSS == "mrae":
        denominator = torch.clamp(target.abs(), min=MRAE_EPS)
        return torch.mean(torch.abs(pred - target) / denominator)
    if RECONSTRUCTION_LOSS == "l1":
        return F.l1_loss(pred, target)
    if RECONSTRUCTION_LOSS == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(
        "RECONSTRUCTION_LOSS must be 'mrae', 'l1', or 'mse', "
        f"not {RECONSTRUCTION_LOSS!r}"
    )


def prior_kd_loss(
    student_prior: torch.Tensor,
    teacher_prior: torch.Tensor,
) -> torch.Tensor:
    student_log_prob = F.log_softmax(
        student_prior / KD_TEMPERATURE,
        dim=1,
    )
    teacher_prob = F.softmax(
        teacher_prior.detach() / KD_TEMPERATURE,
        dim=1,
    )
    return F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    )


def prepare_metric_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Clamp only when targets already follow the standard [0, 1] range."""
    target_min = float(target.detach().amin().item())
    target_max = float(target.detach().amax().item())

    if target_min >= -1e-6 and target_max <= 1.0 + 1e-6:
        return pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0), 1.0

    data_range = max(target_max - target_min, MRAE_EPS)
    return pred, target, data_range


def spectral_angle_mapper(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_spectra = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target_spectra = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    numerator = (pred_spectra * target_spectra).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(pred_spectra, dim=1)
        * torch.linalg.vector_norm(target_spectra, dim=1)
    )
    cosine = torch.clamp(
        numerator / torch.clamp(denominator, min=MRAE_EPS),
        -1.0,
        1.0,
    )
    return torch.rad2deg(torch.acos(cosine)).mean()


def hyperspectral_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
) -> torch.Tensor:
    """Local SSIM averaged over batches, bands, and spatial locations."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred = (
        F.avg_pool2d(pred.square(), kernel_size=3, stride=1, padding=1)
        - mu_pred_sq
    )
    sigma_target = (
        F.avg_pool2d(target.square(), kernel_size=3, stride=1, padding=1)
        - mu_target_sq
    )
    sigma_cross = (
        F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1)
        - mu_cross
    )

    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (sigma_pred + sigma_target + c2)
    )
    return (numerator / torch.clamp(denominator, min=MRAE_EPS)).mean()


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    pred, target, data_range = prepare_metric_tensors(pred, target)
    error = pred - target

    batch_mrae = torch.mean(
        torch.abs(error) / torch.clamp(target.abs(), min=MRAE_EPS)
    )
    batch_rmse = torch.sqrt(torch.mean(error.square()))
    batch_psnr = 10.0 * torch.log10(
        torch.tensor(data_range**2, device=pred.device)
        / torch.clamp(torch.mean(error.square()), min=MRAE_EPS)
    )
    batch_sam = spectral_angle_mapper(pred, target)
    batch_ssim = hyperspectral_ssim(pred, target, data_range)

    return {
        "mrae": float(batch_mrae.item()),
        "rmse": float(batch_rmse.item()),
        "psnr": float(batch_psnr.item()),
        "sam": float(batch_sam.item()),
        "ssim": float(batch_ssim.item()),
    }


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


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"{path} is not a complete DiffIR RGB-to-HSI checkpoint. "
            "Expected a dictionary containing a 'model' state dict."
        )
    return checkpoint


def load_stage1_teacher(
    path: Path,
    device: torch.device,
) -> Tuple[DiffIRS1RGB2HSI, ModelConfig]:
    checkpoint = load_checkpoint(path, device)
    if int(checkpoint.get("stage", -1)) != 1:
        raise ValueError(
            f"Teacher checkpoint must be Stage 1, found stage={checkpoint.get('stage')}"
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

    teacher, teacher_config = load_stage1_teacher(
        Path(TEACHER_CHECKPOINT),
        device,
    )

    # MAIN STYLE CHANGE: Stage 2 automatically inherits all reconstruction
    # architecture settings from Stage 1. Only diffusion-specific fields are
    # taken from the current CONFIG section.
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


def crop_to_original_size(
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


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }
    count = 0

    # MAIN STYLE CHANGE: recreate the same Stage-2 Gaussian sequence every
    # epoch so validation MRAE is comparable and suitable for early stopping.
    eval_generator = None
    if STAGE == 2:
        eval_generator = torch.Generator(device=device)
        eval_generator.manual_seed(VAL_SEED)

    for batch in val_loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        hsi = batch["hsi"].to(device, non_blocking=True)
        orig_hw = batch["orig_hw"]

        if STAGE == 1:
            assert isinstance(model, DiffIRS1RGB2HSI)
            pred_hsi, _ = model(rgb, hsi)
        else:
            assert isinstance(model, DiffIRS2RGB2HSI)
            initial_noise = torch.randn(
                rgb.shape[0],
                model.config.prior_dim,
                generator=eval_generator,
                device=device,
            )
            pred_hsi = model(rgb, initial_noise=initial_noise)

        for sample_index in range(rgb.shape[0]):
            sample_pred, sample_hsi = crop_to_original_size(
                pred_hsi,
                hsi,
                orig_hw,
                sample_index,
            )
            sample_loss = reconstruction_loss(sample_pred, sample_hsi)
            sample_metrics = compute_metrics(sample_pred, sample_hsi)

            totals["loss"] += float(sample_loss.item())
            for metric_name in ("mrae", "rmse", "psnr", "sam", "ssim"):
                totals[metric_name] += sample_metrics[metric_name]
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader is empty")

    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


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


def train() -> None:
    verify_patch_size()
    set_seed(SEED)

    device = torch.device(DEVICE)
    pin_memory = device.type == "cuda"

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
        resume = load_checkpoint(Path(RESUME_CHECKPOINT), device)
        if int(resume.get("stage", -1)) != STAGE:
            raise ValueError("RESUME_CHECKPOINT stage does not match STAGE")

        resume_config = ModelConfig.from_dict(resume["model_config"])
        if resume_config.to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint architecture differs from CONFIG")

        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        lr_scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume["epoch"]) + 1
        best_val_mrae = float(resume.get("best_val_mrae", math.inf))
        best_val_loss = float(resume.get("best_val_loss", math.inf))
        epochs_without_improvement = int(
            resume.get("epochs_without_improvement", 0)
        )
        print(f"Resumed Stage {STAGE} from epoch {start_epoch}")

    best_path, best_loss_path, latest_path = checkpoint_paths(STAGE)

    print(f"Device: {device}")
    print(f"Stage: {STAGE}")
    print(f"Training samples/patches: {len(train_loader.dataset)}")
    print(f"Validation scenes: {len(val_loader.dataset)}")
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
            rgb = batch["rgb"].to(
                device,
                non_blocking=pin_memory,
            )
            hsi = batch["hsi"].to(
                device,
                non_blocking=pin_memory,
            )
            batch_size = rgb.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                if STAGE == 1:
                    assert isinstance(model, DiffIRS1RGB2HSI)
                    pred_hsi, _ = model(rgb, hsi)
                    rec_loss = reconstruction_loss(pred_hsi, hsi)
                    prior_l1 = rec_loss.new_zeros(())
                    prior_kd = rec_loss.new_zeros(())
                    total_loss = rec_loss
                else:
                    assert teacher is not None
                    assert isinstance(model, DiffIRS2RGB2HSI)

                    with torch.no_grad():
                        target_prior = teacher.E(rgb, hsi)

                    pred_hsi, prior_sequence = model(
                        rgb,
                        target_prior=target_prior,
                    )
                    predicted_prior = prior_sequence[-1]

                    rec_loss = reconstruction_loss(pred_hsi, hsi)
                    prior_l1 = F.l1_loss(
                        predicted_prior,
                        target_prior.detach(),
                    )
                    prior_kd = prior_kd_loss(
                        predicted_prior,
                        target_prior,
                    )

                    total_loss = (
                        rec_loss
                        + LAMBDA_PRIOR_L1 * prior_l1
                        + LAMBDA_PRIOR_KD * prior_kd
                    )

            scaler.scale(total_loss).backward()

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )

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

        # Diagnostic checkpoint based on reconstruction loss.
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

        # Primary checkpoint criterion: full reconstructed HSI MRAE.
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

            print(
                f"Saved best Stage-{STAGE} model "
                f"(Val MRAE: {best_val_mrae:.6f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                "No validation MRAE improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

        # Save complete state every epoch for straightforward continuation.
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
            print(
                "Early stopping triggered. "
                f"Best validation MRAE: {best_val_mrae:.6f}"
            )
            break


# ==================================================
# EVALUATION
# ==================================================


def evaluate() -> None:
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
