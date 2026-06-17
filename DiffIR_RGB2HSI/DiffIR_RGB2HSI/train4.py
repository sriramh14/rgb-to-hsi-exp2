#!/usr/bin/env python3
"""Train, validate, or test the combined MST++/DiffIR RGB-to-HSI model.

Only two command-line arguments are exposed:

    python main.py --mode train --stage 1
    python main.py --mode train --stage 2
    python main.py --mode val   --stage 2
    python main.py --mode test  --stage 2

The data path, split ratios, optimizer settings, and architecture settings live
in the CONFIG section. The dataset may be outside this cloned repository.
Set ``ARAD_DATA_ROOT`` in the environment or edit ``DATA_ROOT`` below.

The train/validation/test indices are generated once, saved to disk, and reused
for both stages. Training samples use the dataset's training view, while
validation and test samples use its non-training/evaluation view.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset_loader import ARADDataset
from loss import compute_metrics, prior_kd_loss, prior_l1_loss, reconstruction_loss

#Change this to metamer_aware_model or spec_prior_model
from models.spec_prior_model import DiffIRS1RGB2HSI, DiffIRS2RGB2HSI, ModelConfig


# ==================================================
# CONFIG
# ==================================================

MODE = "train"                 # "train", "val", or "test"; "eval" aliases "test"
STAGE = 1                       # 1: oracle spectral prior, 2: diffusion prior

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234
SPLIT_SEED = 42

# Resolve all repository-owned paths from this file rather than from the shell's
# current working directory. This keeps checkpoint/split locations stable.
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# DATASET PATH AND TRAIN/VAL/TEST SPLIT
# -----------------------------------------------------------------------------
# The dataset can be anywhere on disk. Priority:
#   1. ARAD_DATA_ROOT environment variable
#   2. DATA_ROOT below
#
# Absolute example:
#   DATA_ROOT = Path("/kaggle/input/arad1k")
# Relative example (resolved from this main.py file):
#   DATA_ROOT = Path("../dataset")
#
# DATA_ROOT may be written either as a string or as a pathlib.Path.
# Example for Kaggle:
#   DATA_ROOT = "/kaggle/input/datasets/sriramhari14/ntire-2022"
#
# ARADDataset historically expects these canonical directory names:
#   NTIRE2020_Train_Spectral
#   NTIRE2020_Train_RealWorld
# This script can also discover a separate HSI/spectral folder and RGB folder
# directly below DATA_ROOT and expose them to the existing loader through local
# symbolic links. The external Kaggle input directory is never modified.
#DATA_ROOT: Union[str, Path] = os.environ.get("ARAD_DATA_ROOT", "../data")
DATA_ROOT = "/kaggle/input/datasets/sriramhari14/ntire-2022"
HSI_KEY = "cube"
DOWNLOAD_DATA = False           # The user already has the dataset locally.

# None means infer the usable image count from the two dataset directories.
# Set an integer to intentionally use only the first N paired samples.
TOTAL_IMAGES: Optional[int] = None

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SPLIT_DIR = PROJECT_ROOT / "splits"
SPLIT_FILE = SPLIT_DIR / "arad_train_val_test_split.pth"

BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
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
LAMBDA_PRIOR_KD = 1e-4              #original value is zero
KD_TEMPERATURE = 0.15

# Validation MRAE controls LR scheduling, best checkpoint, and early stopping.
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 4
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
DIFFUSION_TIMESTEPS = 10        #original value was 4

#Original schedule was 0.1 to 0.99
LINEAR_START = 0.1            
LINEAR_END = 0.99

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
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
        choices=["train", "val", "test", "eval"],
        default=MODE,
        help="Run mode: train, val, or test. eval is accepted as an alias for test.",
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


CANONICAL_HSI_DIR = "NTIRE2020_Train_Spectral"
CANONICAL_RGB_DIR = "NTIRE2020_Train_RealWorld"
DATASET_ADAPTER_ROOT = PROJECT_ROOT / ".dataset_adapter"


def resolve_data_root() -> Path:
    """Resolve DATA_ROOT while accepting either a string or pathlib.Path."""
    root = Path(DATA_ROOT).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    else:
        root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist or is not a directory: {root}"
        )
    return root


def _count_files(directory: Path, suffixes: set[str]) -> int:
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _choose_data_directory(
    root: Path,
    *,
    suffixes: set[str],
    preferred_words: Tuple[str, ...],
    kind: str,
) -> Path:
    """Find a direct child directory containing the requested file type."""
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        count = _count_files(child, suffixes)
        if count == 0:
            continue
        name = child.name.lower()
        keyword_score = sum(word in name for word in preferred_words)
        candidates.append((keyword_score, count, child))

    if not candidates:
        child_names = sorted(child.name for child in root.iterdir() if child.is_dir())
        raise FileNotFoundError(
            f"Could not locate the {kind} directory under {root}. "
            f"Direct child directories are: {child_names}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_path = candidates[0]

    # Avoid silently selecting between equally plausible directories.
    equally_ranked = [
        path for score, count, path in candidates
        if score == best_score and count == candidates[0][1]
    ]
    if len(equally_ranked) > 1:
        raise RuntimeError(
            f"Multiple possible {kind} directories were found: "
            f"{[str(path) for path in equally_ranked]}. "
            "Rename the intended folders to include 'hsi'/'spectral' and 'rgb'."
        )
    return best_path


def prepare_dataset_root(root: Path) -> Path:
    """Return a root compatible with the existing ARADDataset loader.

    When the external dataset already uses the canonical NTIRE2020 folder names,
    it is used directly. Otherwise, separate HSI and RGB child directories are
    discovered and linked into a small writable adapter directory in the repo.
    """
    canonical_hsi = root / CANONICAL_HSI_DIR
    canonical_rgb = root / CANONICAL_RGB_DIR
    if canonical_hsi.is_dir() and canonical_rgb.is_dir():
        return root

    hsi_dir = _choose_data_directory(
        root,
        suffixes={".mat"},
        preferred_words=("hsi", "spectral", "spec", "hyper"),
        kind="HSI/spectral",
    )
    rgb_dir = _choose_data_directory(
        root,
        suffixes={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"},
        preferred_words=("rgb", "realworld", "image"),
        kind="RGB",
    )

    DATASET_ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    links = {
        DATASET_ADAPTER_ROOT / CANONICAL_HSI_DIR: hsi_dir,
        DATASET_ADAPTER_ROOT / CANONICAL_RGB_DIR: rgb_dir,
    }
    for link, target in links.items():
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(
                f"Dataset adapter path already exists and is not a symlink: {link}"
            )
        link.symlink_to(target, target_is_directory=True)

    print(f"External dataset root: {root}")
    print(f"Detected HSI directory: {hsi_dir}")
    print(f"Detected RGB directory: {rgb_dir}")
    print(f"Dataset adapter root: {DATASET_ADAPTER_ROOT}")
    return DATASET_ADAPTER_ROOT


def infer_total_images(root: Path) -> int:
    """Infer a safe upper bound from the local spectral and RGB file counts."""
    spectral_dir = root / CANONICAL_HSI_DIR
    rgb_dir = root / CANONICAL_RGB_DIR

    if not spectral_dir.is_dir():
        raise FileNotFoundError(f"Spectral directory not found: {spectral_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    spectral_count = sum(
        1 for path in spectral_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mat"
    )
    rgb_count = sum(
        1 for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if spectral_count == 0:
        raise RuntimeError(f"No .mat files found in {spectral_dir}")
    if rgb_count == 0:
        raise RuntimeError(f"No RGB image files found in {rgb_dir}")

    available = min(spectral_count, rgb_count)
    if spectral_count != rgb_count:
        print(
            "Warning: spectral/RGB file counts differ "
            f"({spectral_count} vs {rgb_count}); using at most {available} samples."
        )

    if TOTAL_IMAGES is None:
        return available
    if TOTAL_IMAGES <= 0:
        raise ValueError("TOTAL_IMAGES must be positive or None")
    if TOTAL_IMAGES > available:
        raise ValueError(
            f"TOTAL_IMAGES={TOTAL_IMAGES}, but only {available} paired candidates are available."
        )
    return int(TOTAL_IMAGES)


def compute_split_lengths(total: int) -> Tuple[int, int, int]:
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0, "
            f"but received {ratio_sum:.8f}."
        )
    if total < 3:
        raise ValueError("At least three samples are required for train/val/test splitting")

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    # Keep every split non-empty for very small test datasets.
    if train_count == 0:
        train_count = 1
    if val_count == 0:
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("The configured split ratios leave no training samples")

    return train_count, val_count, test_count


def load_or_create_split_indices(total: int) -> Dict[str, list]:
    """Create one deterministic split manifest and reuse it across both stages."""
    train_count, val_count, test_count = compute_split_lengths(total)
    expected_meta = {
        "total": total,
        "split_seed": SPLIT_SEED,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
    }

    if SPLIT_FILE.exists():
        try:
            saved = torch.load(SPLIT_FILE, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(SPLIT_FILE, map_location="cpu")
        if isinstance(saved, dict) and all(saved.get(k) == v for k, v in expected_meta.items()):
            indices = saved.get("indices")
            if isinstance(indices, dict):
                merged = indices.get("train", []) + indices.get("val", []) + indices.get("test", [])
                if len(merged) == total and len(set(merged)) == total:
                    return indices
        print(f"Existing split manifest is incompatible and will be regenerated: {SPLIT_FILE}")

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_end = train_count
    val_end = train_count + val_count
    indices = {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({**expected_meta, "indices": indices}, SPLIT_FILE)
    return indices


def make_dataloaders(
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build reproducible train, validation, and test loaders.

    Two dataset pools are constructed deliberately:
      * ``train_pool`` uses ``train=True`` so existing training transforms remain active.
      * ``eval_pool`` uses ``train=False`` so validation/test do not use training transforms.

    Both pools cover the same complete ordered sample list, and Subset applies the
    same saved split indices to the appropriate view.
    """
    set_seed(SEED)
    external_data_root = resolve_data_root()
    data_root = prepare_dataset_root(external_data_root)
    total_images = infer_total_images(data_root)
    split_indices = load_or_create_split_indices(total_images)

    # train=True with train_images=total_images exposes the complete training view.
    train_pool = ARADDataset(
        root_dir=str(data_root),
        train=True,
        train_images=total_images,
        total_images=total_images,
        cube_key=HSI_KEY,
        download=DOWNLOAD_DATA,
    )

    # train=False with train_images=0 exposes the complete non-training view.
    eval_pool = ARADDataset(
        root_dir=str(data_root),
        train=False,
        train_images=0,
        total_images=total_images,
        cube_key=HSI_KEY,
        download=False,
    )

    if len(train_pool) != total_images:
        raise RuntimeError(
            f"Training-view dataset length is {len(train_pool)}, expected {total_images}. "
            "Check ARADDataset's train_images/total_images slicing behavior."
        )
    if len(eval_pool) != total_images:
        raise RuntimeError(
            f"Evaluation-view dataset length is {len(eval_pool)}, expected {total_images}. "
            "Check ARADDataset's train_images/total_images slicing behavior."
        )

    train_dataset = Subset(train_pool, split_indices["train"])
    val_dataset = Subset(eval_pool, split_indices["val"])
    test_dataset = Subset(eval_pool, split_indices["test"])

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda" and PIN_MEMORY,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        "drop_last": False,
    }

    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"Dataset root: {data_root}")
    print(f"Split manifest: {SPLIT_FILE}")
    print(
        f"Dataset split: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


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
    train_loader, val_loader, test_loader = make_dataloaders(device)
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
    print(f"Reserved test samples: {len(test_loader.dataset)}")
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
                    prior_l1 = prior_l1_loss(predicted_prior, target_prior)

                    #This didn't work well
                    #prior_l1 = sum(
                        #prior_l1_loss(p, target_prior) for p in prior_sequence
                    #) / len(prior_sequence)

                    #New attempt to fix (also didn't work)
                    #final_prior_l1 = prior_l1_loss(prior_sequence[-1], target_prior)

                    #aux_prior_l1 = sum(
                       # prior_l1_loss(p, target_prior) for p in prior_sequence[:-1]
                    #) / max(1, len(prior_sequence) - 1)
                    
                   # prior_l1 = final_prior_l1 + 0.25 * aux_prior_l1

                    
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


def evaluate(split_name: str) -> None:
    if STAGE not in (1, 2):
        raise ValueError("STAGE must be 1 or 2")
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'")

    set_seed(SEED)
    device = torch.device(DEVICE)
    _, val_loader, test_loader = make_dataloaders(device)
    evaluation_loader = val_loader if split_name == "val" else test_loader

    default_path, _, _ = checkpoint_paths(STAGE)
    selected_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else default_path
    checkpoint = load_checkpoint(selected_path, device)
    model, config = build_evaluation_model(checkpoint, device)
    results = validate(model, evaluation_loader, device)

    print(f"Evaluated split: {split_name}")
    print(f"Evaluated samples: {len(evaluation_loader.dataset)}")
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
    elif MODE == "val":
        evaluate("val")
    elif MODE in {"test", "eval"}:
        # Keep eval as a backwards-compatible alias for held-out test evaluation.
        evaluate("test")
    else:
        raise ValueError("MODE must be 'train', 'val', or 'test'")


if __name__ == "__main__":
    main()
