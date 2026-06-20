#!/usr/bin/env python3
"""Train, validate, or test CPEN + deterministic-DDIM prior + MST++.

Commands
--------
    python train_cpen_mstpp_ddim.py --mode train --stage 1
    python train_cpen_mstpp_ddim.py --mode train --stage 2
    python train_cpen_mstpp_ddim.py --mode val   --stage 2
    python train_cpen_mstpp_ddim.py --mode test  --stage 2

Stage 1
-------
RGB + ground-truth HSI -> CPEN oracle prior -> frozen-compatible input adapter
-> three-stage MST++ -> reconstructed HSI.

All Stage-1 components are trained jointly.

Stage 2
-------
RGB -> RGB condition encoder -> deterministic DDIM compact-prior diffusion
-> Stage-1 prior-to-RGB adapter -> Stage-1 MST++ -> reconstructed HSI.

Only the diffusion subsystem is trained in Stage 2:
    * RGB condition encoder
    * compact-prior DDIM denoiser

The following Stage-1 components are copied and frozen:
    * oracle CPEN teacher
    * prior-to-RGB adapter
    * complete three-stage MST++ backbone

The reconstruction loss is still used in Stage 2. Although MST++ and the
adapter are frozen, gradients pass through them into the predicted prior and
therefore train the diffusion subsystem to generate priors that improve the
final HSI reconstruction.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, Dataset, Subset

from loss import compute_metrics, prior_l1_loss, reconstruction_loss

# Can be spec_prior_alternate or fft_based_prior
from models.spec_prior_alternate import (
    Stage1InputPriorMSTPP,
    Stage2DDIMPriorMSTPP,
    build_stage1_model,
    build_stage2_from_stage1,
)


# ==================================================
# CONFIG
# ==================================================

MODE = "train"
STAGE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SPLIT_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

DATA_ROOT: Union[str, Path] = os.environ.get(
    "ARAD_DATA_ROOT",
    "/kaggle/input/datasets/sriramhari14/ntire-2022",
)
HSI_KEY = "cube"
TOTAL_IMAGES: Optional[int] = None

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SPLIT_DIR = PROJECT_ROOT / "splits"
SPLIT_FILE = SPLIT_DIR / "arad_train_val_test_split.pth"

VALIDATE_DATASET_FILES = True
SKIP_INVALID_PAIRS = True
DATASET_VALIDATION_CACHE = SPLIT_DIR / "ntire_valid_pairs_cache.pth"
DATASET_VALIDATION_PROGRESS = 100

# Full-resolution, single-GPU training. Batch size one is safer for the
# three-stage MST++ backbone and full-resolution ARAD/NTIRE images.
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = DEVICE == "cuda"

PROGRESS_EVERY_N_BATCHES = 30
TRAIN_METRICS_EVERY_N_BATCHES = PROGRESS_EVERY_N_BATCHES

# -----------------------------------------------------------------------------
# OPTIMIZATION
# -----------------------------------------------------------------------------

NUM_EPOCHS = 50
LR_STAGE1 = 2e-4
LR_STAGE2 = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# This works for both stages, but substantially increases runtime. The default
# is false because Stage 2 already freezes the large MST++ backbone.
USE_GRADIENT_CHECKPOINTING = False

RECONSTRUCTION_LOSS = "mrae"
MRAE_EPS = 1e-6

# Stage-2 objective:
#   L = reconstruction + lambda_prior * prior_L1
#       + lambda_diffusion * epsilon_MSE
LAMBDA_PRIOR_L1 = 0.10
LAMBDA_DIFFUSION = 0.10

EARLY_STOPPING_PATIENCE = 20
MIN_LR = 1e-7
COSINE_T_MAX = NUM_EPOCHS

# -----------------------------------------------------------------------------
# ARCHITECTURE
# -----------------------------------------------------------------------------

NUM_BANDS = 31
MST_STAGES = 3
PAD_MULTIPLE = 8

PRIOR_DIM = 256
CPEN_FEATS = 64
N_ENCODER_RES = 6
PRIOR_UNSHUFFLE_FACTOR = 4

PRIOR_ADAPTER_CHANNELS = 16
PRIOR_ADAPTER_HIDDEN = 32
PRIOR_MAX_RGB_DELTA = 0.10

# Stage-2 RGB condition encoder. It is part of the diffusion subsystem and is
# trainable because the diffusion denoiser requires an RGB-dependent condition.
CONDITION_DIM = 256
CONDITION_FEATS = 64
CONDITION_RES_BLOCKS = 6
INITIALISE_CONDITION_FROM_CPEN = True
COPY_DEEPER_CONDITION_BLOCKS = False

# Compact-vector deterministic DDIM.
DENOISER_HIDDEN_DIM = 512
DENOISER_TIME_DIM = 128
DENOISER_BLOCKS = 5
DENOISER_DROPOUT = 0.0

DIFFUSION_TRAIN_TIMESTEPS = 200
DDIM_SAMPLE_STEPS = 10
BETA_SCHEDULE = "cosine"
BETA_START = 1e-4
BETA_END = 2e-2
DETERMINISTIC_DDIM_SEED = 0

# Scale used internally before diffusing the CPEN prior. Keep at 1.0 initially.
# It can later be adjusted if Stage-1 prior magnitudes are far above or below 1.
PRIOR_SCALE = 1.0

# Stage-2 freezing policy requested by the user.
FREEZE_STAGE2_MSTPP = True
FREEZE_STAGE2_ADAPTER = True
FREEZE_STAGE2_TEACHER_CPEN = True

# -----------------------------------------------------------------------------
# CHECKPOINTS
# -----------------------------------------------------------------------------

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

STAGE1_BEST_PATH = CHECKPOINT_DIR / "cpen_mstpp_stage1_best.pth"
STAGE1_BEST_LOSS_PATH = CHECKPOINT_DIR / "cpen_mstpp_stage1_best_loss.pth"
STAGE1_LATEST_PATH = CHECKPOINT_DIR / "cpen_mstpp_stage1_latest.pth"

STAGE2_BEST_PATH = CHECKPOINT_DIR / "cpen_mstpp_ddim_stage2_best.pth"
STAGE2_BEST_LOSS_PATH = CHECKPOINT_DIR / "cpen_mstpp_ddim_stage2_best_loss.pth"
STAGE2_LATEST_PATH = CHECKPOINT_DIR / "cpen_mstpp_ddim_stage2_latest.pth"

TEACHER_CHECKPOINT = STAGE1_BEST_PATH
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# COMMAND-LINE OVERRIDES
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CPEN + DDIM prior + MST++ for RGB-to-HSI"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=STAGE,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "test", "eval"],
        default=MODE,
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
# MODEL CONFIGURATION DICTIONARY
# ==================================================

def make_model_config() -> Dict[str, Any]:
    """Plain dictionary saved in checkpoints to reconstruct either stage."""
    return {
        "rgb_channels": 3,
        "hsi_channels": NUM_BANDS,
        "cpen_feats": CPEN_FEATS,
        "cpen_res_blocks": N_ENCODER_RES,
        "prior_dim": PRIOR_DIM,
        "unshuffle_factor": PRIOR_UNSHUFFLE_FACTOR,
        "mst_stages": MST_STAGES,
        "prior_channels": PRIOR_ADAPTER_CHANNELS,
        "adapter_hidden_channels": PRIOR_ADAPTER_HIDDEN,
        "max_delta": PRIOR_MAX_RGB_DELTA,
        "condition_dim": CONDITION_DIM,
        "condition_feats": CONDITION_FEATS,
        "condition_res_blocks": CONDITION_RES_BLOCKS,
        "denoiser_hidden_dim": DENOISER_HIDDEN_DIM,
        "denoiser_time_dim": DENOISER_TIME_DIM,
        "denoiser_blocks": DENOISER_BLOCKS,
        "denoiser_dropout": DENOISER_DROPOUT,
        "train_timesteps": DIFFUSION_TRAIN_TIMESTEPS,
        "sample_steps": DDIM_SAMPLE_STEPS,
        "beta_schedule": BETA_SCHEDULE,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "deterministic_seed": DETERMINISTIC_DDIM_SEED,
        "prior_scale": PRIOR_SCALE,
        "initialise_condition_from_cpen": INITIALISE_CONDITION_FROM_CPEN,
        "copy_deeper_condition_blocks": COPY_DEEPER_CONDITION_BLOCKS,
        "freeze_mstpp": FREEZE_STAGE2_MSTPP,
        "freeze_adapter": FREEZE_STAGE2_ADAPTER,
        "freeze_teacher_cpen": FREEZE_STAGE2_TEACHER_CPEN,
    }


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


TRAIN_DISPLAY_METRICS = ("mrae", "rmse", "psnr", "sam", "ssim")


@torch.no_grad()
def compute_batch_display_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-image training/validation metrics without retaining gradients.

    Metrics are evaluated only over the original image extent, so any temporary
    padding added by the full-resolution collate function is excluded.
    """
    totals = {name: 0.0 for name in TRAIN_DISPLAY_METRICS}
    batch_size = pred.shape[0]

    for sample_index in range(batch_size):
        sample_pred, sample_target = crop_sample(
            pred.detach(), target.detach(), orig_hw, sample_index
        )
        # Metric implementations are numerically safer in float32 when AMP is
        # enabled for the model forward pass.
        metrics = compute_metrics(
            sample_pred.float(),
            sample_target.float(),
            mrae_eps=MRAE_EPS,
        )
        for name in TRAIN_DISPLAY_METRICS:
            totals[name] += float(metrics[name])

    return {name: value / max(batch_size, 1) for name, value in totals.items()}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ==================================================
# DATA
# ==================================================

RGB_DIRECTORY_NAMES = ("Train_RGB", "train_rgb", "RGB", "rgb")
HSI_DIRECTORY_NAMES = (
    "Train_spectral",
    "Train_Spectral",
    "train_spectral",
    "Train_HSI",
    "train_hsi",
    "HSI",
    "hsi",
)
RGB_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HSI_SUFFIXES = {".mat"}

# Leave this as None for the usual ARAD/NTIRE floating-point cubes in [0, 1].
# Set a fixed divisor only when your MAT files use a known encoded range, e.g.
# 65535.0 for uint16-like floating-point values.
HSI_VALUE_SCALE: Optional[float] = None
CLIP_INPUTS_TO_UNIT_RANGE = True


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


def _find_named_directory(root: Path, names: Tuple[str, ...], kind: str) -> Path:
    """Find a known dataset directory directly or one level below DATA_ROOT."""
    wanted = {name.lower() for name in names}

    direct = [path for path in root.iterdir() if path.is_dir()]
    for path in direct:
        if path.name.lower() in wanted:
            return path

    # Some Kaggle datasets add one wrapper directory around the actual data.
    for wrapper in direct:
        try:
            children = [path for path in wrapper.iterdir() if path.is_dir()]
        except PermissionError:
            continue
        for path in children:
            if path.name.lower() in wanted:
                return path

    raise FileNotFoundError(
        f"Could not find the {kind} directory under {root}. "
        f"Expected one of {list(names)}. Direct children are "
        f"{sorted(path.name for path in direct)}."
    )


def _collect_files(directory: Path, suffixes: set[str]) -> list[Path]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {sorted(suffixes)} found under {directory}"
        )
    return files


def _normalized_pair_key(path: Path) -> str:
    """Normalize common RGB/HSI suffixes while preserving the sample identity."""
    key = path.stem.lower().strip()
    removable_suffixes = (
        "_spectral", "-spectral", "_spectrum", "-spectrum",
        "_realworld", "-realworld", "_clean", "-clean",
        "_rgb", "-rgb", "_hsi", "-hsi", "_hyperspectral",
    )
    changed = True
    while changed:
        changed = False
        for suffix in removable_suffixes:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                changed = True
                break
    return key


def _build_unique_map(files: list[Path], kind: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in files:
        key = _normalized_pair_key(path)
        if key in result:
            raise RuntimeError(
                f"Duplicate normalized {kind} key '{key}' for:\n"
                f"  {result[key]}\n  {path}\n"
                "Rename one of the files or make the pairing rule more specific."
            )
        result[key] = path
    return result


def discover_paired_samples(root: Path) -> list[Tuple[Path, Path, str]]:
    """Return deterministic ``(rgb_path, hsi_path, sample_name)`` pairs."""
    rgb_dir = _find_named_directory(root, RGB_DIRECTORY_NAMES, "RGB")
    hsi_dir = _find_named_directory(root, HSI_DIRECTORY_NAMES, "HSI/spectral")
    rgb_files = _collect_files(rgb_dir, RGB_SUFFIXES)
    hsi_files = _collect_files(hsi_dir, HSI_SUFFIXES)

    rgb_map = _build_unique_map(rgb_files, "RGB")
    hsi_map = _build_unique_map(hsi_files, "HSI")
    common_keys = sorted(set(rgb_map).intersection(hsi_map))

    if not common_keys:
        rgb_examples = [path.name for path in rgb_files[:5]]
        hsi_examples = [path.name for path in hsi_files[:5]]
        raise RuntimeError(
            "No RGB/HSI pairs could be matched by filename stem. "
            f"RGB examples: {rgb_examples}; HSI examples: {hsi_examples}."
        )

    unmatched_rgb = sorted(set(rgb_map).difference(hsi_map))
    unmatched_hsi = sorted(set(hsi_map).difference(rgb_map))
    if unmatched_rgb or unmatched_hsi:
        print(
            "Warning: ignoring unmatched files: "
            f"RGB={len(unmatched_rgb)}, HSI={len(unmatched_hsi)}"
        )
        if unmatched_rgb:
            print(f"First unmatched RGB keys: {unmatched_rgb[:5]}")
        if unmatched_hsi:
            print(f"First unmatched HSI keys: {unmatched_hsi[:5]}")

    pairs = [(rgb_map[key], hsi_map[key], key) for key in common_keys]
    if TOTAL_IMAGES is not None:
        if TOTAL_IMAGES <= 0:
            raise ValueError("TOTAL_IMAGES must be positive or None")
        if TOTAL_IMAGES > len(pairs):
            raise ValueError(
                f"TOTAL_IMAGES={TOTAL_IMAGES}, but only {len(pairs)} matched pairs exist."
            )
        pairs = pairs[: int(TOTAL_IMAGES)]

    print(f"Dataset root: {root}")
    print(f"RGB directory: {rgb_dir}")
    print(f"HSI directory: {hsi_dir}")
    print(f"RGB files found: {len(rgb_files)}")
    print(f"HSI files found: {len(hsi_files)}")
    print(f"Matched RGB-HSI pairs: {len(pairs)}")
    return pairs



def _pair_files_fingerprint(pairs: list[Tuple[Path, Path, str]]) -> str:
    """Fingerprint paired files using paths, sizes, and modification times."""
    import hashlib

    records = []
    for rgb_path, hsi_path, name in pairs:
        rgb_stat = rgb_path.stat()
        hsi_stat = hsi_path.stat()
        records.append(
            f"{name}|{rgb_path.resolve()}|{rgb_stat.st_size}|{rgb_stat.st_mtime_ns}|"
            f"{hsi_path.resolve()}|{hsi_stat.st_size}|{hsi_stat.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _shape_contains_valid_cube(
    shape: Tuple[int, ...],
    rgb_hw: Tuple[int, int],
) -> bool:
    """Check whether a 3-D MAT/HDF5 shape can represent the paired HSI cube."""
    if len(shape) != 3 or NUM_BANDS not in shape:
        return False

    rgb_h, rgb_w = rgb_hw
    for spectral_axis, size in enumerate(shape):
        if size != NUM_BANDS:
            continue
        spatial = tuple(shape[i] for i in range(3) if i != spectral_axis)
        if spatial == (rgb_h, rgb_w) or spatial == (rgb_w, rgb_h):
            return True
    return False


def _inspect_hdf5_cube(
    path: Path,
    cube_key: str,
    rgb_hw: Tuple[int, int],
) -> None:
    """Open an HDF5 MAT file and inspect metadata without loading the cube."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path} is a MATLAB v7.3/HDF5 file, but h5py is unavailable."
        ) from exc

    with h5py.File(path, "r") as handle:
        candidates: list[Tuple[str, Tuple[int, ...]]] = []

        if cube_key in handle and isinstance(handle[cube_key], h5py.Dataset):
            dataset = handle[cube_key]
            candidates.append((cube_key, tuple(int(v) for v in dataset.shape)))
        else:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    candidates.append((name, tuple(int(v) for v in obj.shape)))

            handle.visititems(visitor)

        if not candidates:
            raise KeyError(
                f"No 3-D dataset found in MATLAB v7.3 file {path}; "
                f"requested key '{cube_key}'."
            )

        compatible = [
            (name, shape)
            for name, shape in candidates
            if _shape_contains_valid_cube(shape, rgb_hw)
        ]
        if not compatible:
            raise ValueError(
                f"No {NUM_BANDS}-band cube in {path.name} matches RGB size {rgb_hw}. "
                f"HDF5 datasets: {candidates}."
            )


def _inspect_mat_cube(
    path: Path,
    cube_key: str,
    rgb_hw: Tuple[int, int],
) -> None:
    """Validate MAT metadata without reading the complete spectral array."""
    try:
        from scipy.io import whosmat
    except ImportError:
        _inspect_hdf5_cube(path, cube_key, rgb_hw)
        return

    try:
        metadata = whosmat(path)
    except (NotImplementedError, ValueError, OSError):
        # MATLAB v7.3 files are HDF5 containers. h5py opening also detects the
        # common truncated-file condition before an epoch begins.
        _inspect_hdf5_cube(path, cube_key, rgb_hw)
        return

    candidates = [
        (name, tuple(int(v) for v in shape))
        for name, shape, _class_name in metadata
        if len(shape) == 3
    ]
    if not candidates:
        raise KeyError(f"No 3-D numeric cube metadata found in {path}.")

    preferred = [item for item in candidates if item[0] == cube_key]
    checked = preferred if preferred else candidates
    if not any(_shape_contains_valid_cube(shape, rgb_hw) for _, shape in checked):
        raise ValueError(
            f"No {NUM_BANDS}-band cube in {path.name} matches RGB size {rgb_hw}. "
            f"MAT arrays: {candidates}."
        )


def _validate_one_pair(rgb_path: Path, hsi_path: Path, cube_key: str) -> None:
    """Validate that RGB is readable and HSI container metadata is intact."""
    from PIL import Image

    with Image.open(rgb_path) as image:
        rgb_w, rgb_h = image.size
        image.verify()

    _inspect_mat_cube(hsi_path, cube_key, (rgb_h, rgb_w))


def validate_paired_samples(
    pairs: list[Tuple[Path, Path, str]],
) -> list[Tuple[Path, Path, str]]:
    """Exclude unreadable/corrupt pairs before deterministic splitting."""
    if not VALIDATE_DATASET_FILES:
        return pairs
    if not pairs:
        raise RuntimeError("No paired files are available for dataset validation.")

    fingerprint = _pair_files_fingerprint(pairs)
    pair_by_name = {name: (rgb, hsi, name) for rgb, hsi, name in pairs}

    if DATASET_VALIDATION_CACHE.exists():
        try:
            try:
                cached = torch.load(
                    DATASET_VALIDATION_CACHE,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(DATASET_VALIDATION_CACHE, map_location="cpu")

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("valid_names"), list)
            ):
                valid_names = cached["valid_names"]
                valid_pairs = [pair_by_name[name] for name in valid_names if name in pair_by_name]
                invalid = cached.get("invalid", [])
                print(
                    f"Dataset integrity cache: {DATASET_VALIDATION_CACHE} "
                    f"({len(valid_pairs)} valid, {len(invalid)} invalid)"
                )
                for item in invalid:
                    print(
                        f"Skipping cached invalid pair {item.get('name', '?')}: "
                        f"{item.get('error', 'unknown error')}"
                    )
                if valid_pairs:
                    return valid_pairs
        except Exception as exc:
            print(f"Ignoring unreadable dataset validation cache: {exc}")

    print(f"Validating {len(pairs)} RGB-HSI pairs before splitting...", flush=True)
    valid_pairs: list[Tuple[Path, Path, str]] = []
    invalid: list[Dict[str, str]] = []

    for index, (rgb_path, hsi_path, name) in enumerate(pairs, start=1):
        try:
            _validate_one_pair(rgb_path, hsi_path, HSI_KEY)
            valid_pairs.append((rgb_path, hsi_path, name))
        except Exception as exc:
            record = {
                "name": name,
                "rgb": str(rgb_path),
                "hsi": str(hsi_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            invalid.append(record)
            print(
                f"Invalid pair excluded: {name}\n"
                f"  RGB: {rgb_path}\n"
                f"  HSI: {hsi_path}\n"
                f"  Reason: {record['error']}",
                flush=True,
            )
            if not SKIP_INVALID_PAIRS:
                raise RuntimeError(
                    f"Dataset validation failed for {name}: {record['error']}"
                ) from exc

        if index % DATASET_VALIDATION_PROGRESS == 0 or index == len(pairs):
            print(
                f"Dataset validation {index}/{len(pairs)} "
                f"| valid {len(valid_pairs)} | invalid {len(invalid)}",
                flush=True,
            )

    if not valid_pairs:
        raise RuntimeError("Every paired sample failed the dataset integrity scan.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_names": [name for _, _, name in valid_pairs],
            "invalid": invalid,
        },
        DATASET_VALIDATION_CACHE,
    )

    print(
        f"Usable paired samples after integrity scan: {len(valid_pairs)} "
        f"(excluded {len(invalid)})"
    )
    return valid_pairs


def _select_mat_array(mapping: Dict[str, Any], cube_key: str, path: Path) -> np.ndarray:
    if cube_key in mapping:
        value = mapping[cube_key]
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value

    candidates = [
        (key, value)
        for key, value in mapping.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if len(candidates) == 1:
        return candidates[0][1]
    if not candidates:
        raise KeyError(
            f"No 3-D numeric cube found in {path}. Requested key '{cube_key}'. "
            f"Available keys: {[key for key in mapping if not key.startswith('__')]}"
        )
    raise KeyError(
        f"Multiple 3-D arrays found in {path}: {[key for key, _ in candidates]}. "
        f"Set HSI_KEY to the correct key."
    )


def _load_hdf5_cube(path: Path, cube_key: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path} is a MATLAB v7.3/HDF5 file, but h5py is unavailable."
        ) from exc

    with h5py.File(path, "r") as handle:
        if cube_key in handle and isinstance(handle[cube_key], h5py.Dataset):
            return np.asarray(handle[cube_key])

        candidates: list[Tuple[str, np.ndarray]] = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                candidates.append((name, np.asarray(obj)))

        handle.visititems(visitor)
        if len(candidates) == 1:
            return candidates[0][1]
        if not candidates:
            raise KeyError(
                f"No 3-D dataset found in MATLAB v7.3 file {path}; "
                f"requested key '{cube_key}'."
            )
        raise KeyError(
            f"Multiple 3-D datasets found in {path}: {[name for name, _ in candidates]}. "
            f"Set HSI_KEY to the correct key."
        )


def load_hsi_cube(path: Path, cube_key: str, rgb_hw: Tuple[int, int]) -> np.ndarray:
    """Load a MAT cube and return float32 CHW aligned with the RGB image."""
    try:
        from scipy.io import loadmat
        try:
            mapping = loadmat(path)
            raw = _select_mat_array(mapping, cube_key, path)
        except (NotImplementedError, ValueError, OSError):
            try:
                raw = _load_hdf5_cube(path, cube_key)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read HSI file {path}: {type(exc).__name__}: {exc}"
                ) from exc
    except ImportError:
        raw = _load_hdf5_cube(path, cube_key)

    original_dtype = raw.dtype
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3-D HSI cube in {path}, got shape {raw.shape}")

    candidates: list[np.ndarray] = []
    for spectral_axis, size in enumerate(raw.shape):
        if size != NUM_BANDS:
            continue
        if spectral_axis == 0:
            chw = raw
        elif spectral_axis == 1:
            chw = np.transpose(raw, (1, 0, 2))
        else:
            chw = np.transpose(raw, (2, 0, 1))
        candidates.append(chw)

    if not candidates:
        raise ValueError(
            f"Could not find a {NUM_BANDS}-band axis in {path}; cube shape is {raw.shape}."
        )

    rgb_h, rgb_w = rgb_hw
    aligned: Optional[np.ndarray] = None
    for chw in candidates:
        if tuple(chw.shape[1:]) == (rgb_h, rgb_w):
            aligned = chw
            break
        # h5py frequently exposes MATLAB dimensions in reversed spatial order.
        if tuple(chw.shape[1:]) == (rgb_w, rgb_h):
            aligned = np.transpose(chw, (0, 2, 1))
            break

    if aligned is None:
        candidate_shapes = [tuple(value.shape) for value in candidates]
        raise ValueError(
            f"Spatial mismatch for sample {path.name}: RGB is {(rgb_h, rgb_w)}, "
            f"HSI CHW candidates are {candidate_shapes}."
        )

    cube = np.asarray(aligned, dtype=np.float32)
    if np.issubdtype(original_dtype, np.integer):
        cube /= float(np.iinfo(original_dtype).max)
    elif HSI_VALUE_SCALE is not None:
        if HSI_VALUE_SCALE <= 0:
            raise ValueError("HSI_VALUE_SCALE must be positive")
        cube /= float(HSI_VALUE_SCALE)

    if not np.isfinite(cube).all():
        cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)

    if HSI_VALUE_SCALE is None and not np.issubdtype(original_dtype, np.integer):
        cube_max = float(cube.max())
        cube_min = float(cube.min())
        if cube_max > 1.5 or cube_min < -0.1:
            raise ValueError(
                f"HSI values in {path.name} fall outside the expected [0,1] range "
                f"(min={cube_min:.6g}, max={cube_max:.6g}). "
                "Set HSI_VALUE_SCALE in the CONFIG section when the files use a known scale."
            )

    if CLIP_INPUTS_TO_UNIT_RANGE:
        cube = np.clip(cube, 0.0, 1.0)
    return np.ascontiguousarray(cube)


class NTIREPairedDataset(Dataset):
    """Paired NTIRE RGB/HSI dataset independent of the old ARADDataset class."""

    def __init__(
        self,
        pairs: list[Tuple[Path, Path, str]],
        *,
        training: bool,
        cube_key: str,
    ) -> None:
        self.pairs = pairs
        self.training = training
        self.cube_key = cube_key

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from PIL import Image

        rgb_path, hsi_path, name = self.pairs[index]
        with Image.open(rgb_path) as image:
            rgb_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

        original_h, original_w = rgb_array.shape[:2]
        hsi_array = load_hsi_cube(
            hsi_path,
            cube_key=self.cube_key,
            rgb_hw=(original_h, original_w),
        )

        rgb = torch.from_numpy(np.ascontiguousarray(rgb_array.transpose(2, 0, 1)))
        hsi = torch.from_numpy(hsi_array)

        # Paired augmentations that preserve H and W, hence remain batch-safe.
        if self.training:
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=(-1,))
                hsi = torch.flip(hsi, dims=(-1,))
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=(-2,))
                hsi = torch.flip(hsi, dims=(-2,))

        return {
            "rgb": rgb.contiguous(),
            "hsi": hsi.contiguous(),
            "name": name,
            "orig_hw": torch.tensor([original_h, original_w], dtype=torch.long),
        }


def _safe_pad(tensor: torch.Tensor, pad_right: int, pad_bottom: int) -> torch.Tensor:
    if pad_right == 0 and pad_bottom == 0:
        return tensor
    mode = "reflect"
    if tensor.shape[-2] <= pad_bottom or tensor.shape[-1] <= pad_right:
        mode = "replicate"
    return F.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)


def collate_paired_batch(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch to a shared multiple without changing original-size metadata."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    max_h = max(int(sample["rgb"].shape[-2]) for sample in samples)
    max_w = max(int(sample["rgb"].shape[-1]) for sample in samples)
    padded_h = ceil_div(max_h, PAD_MULTIPLE) * PAD_MULTIPLE
    padded_w = ceil_div(max_w, PAD_MULTIPLE) * PAD_MULTIPLE

    rgb_batch = []
    hsi_batch = []
    names = []
    original_sizes = []
    for sample in samples:
        rgb = sample["rgb"]
        hsi = sample["hsi"]
        if rgb.shape[-2:] != hsi.shape[-2:]:
            raise ValueError(
                f"RGB/HSI size mismatch during collation for {sample['name']}: "
                f"{tuple(rgb.shape)} vs {tuple(hsi.shape)}"
            )
        pad_bottom = padded_h - rgb.shape[-2]
        pad_right = padded_w - rgb.shape[-1]
        rgb_batch.append(_safe_pad(rgb, pad_right, pad_bottom))
        hsi_batch.append(_safe_pad(hsi, pad_right, pad_bottom))
        names.append(sample["name"])
        original_sizes.append(sample["orig_hw"])

    return {
        "rgb": torch.stack(rgb_batch, dim=0),
        "hsi": torch.stack(hsi_batch, dim=0),
        "name": names,
        "orig_hw": torch.stack(original_sizes, dim=0),
    }


def compute_split_lengths(total: int) -> Tuple[int, int, int]:
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0, "
            f"but received {ratio_sum:.8f}."
        )
    if total < 3:
        raise ValueError("At least three samples are required for train/val/test splitting")

    # Round the two declared split sizes and assign the remainder to test.
    # For 900 files this gives 720/90/90; after excluding one corrupt file it
    # gives 719/90/90 instead of unnecessarily shrinking validation to 89.
    train_count = int(round(total * TRAIN_RATIO))
    val_count = int(round(total * VAL_RATIO))
    test_count = total - train_count - val_count

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


def _dataset_fingerprint(pairs: list[Tuple[Path, Path, str]]) -> str:
    import hashlib

    payload = "\n".join(
        f"{name}|{rgb.name}|{hsi.name}" for rgb, hsi, name in pairs
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_create_split_indices(
    pairs: list[Tuple[Path, Path, str]],
) -> Dict[str, list]:
    """Create one deterministic split manifest and reuse it across both stages."""
    total = len(pairs)
    train_count, val_count, test_count = compute_split_lengths(total)
    expected_meta = {
        "total": total,
        "split_seed": SPLIT_SEED,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "dataset_fingerprint": _dataset_fingerprint(pairs),
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
    """Build custom paired NTIRE train, validation, and test loaders."""
    set_seed(SEED)
    data_root = resolve_data_root()
    pairs = discover_paired_samples(data_root)
    pairs = validate_paired_samples(pairs)
    split_indices = load_or_create_split_indices(pairs)

    train_pool = NTIREPairedDataset(pairs, training=True, cube_key=HSI_KEY)
    eval_pool = NTIREPairedDataset(pairs, training=False, cube_key=HSI_KEY)

    train_dataset = Subset(train_pool, split_indices["train"])
    val_dataset = Subset(eval_pool, split_indices["val"])
    test_dataset = Subset(eval_pool, split_indices["test"])

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda" and PIN_MEMORY,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        "drop_last": False,
        "collate_fn": collate_paired_batch,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

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
        return (
            STAGE1_BEST_PATH,
            STAGE1_BEST_LOSS_PATH,
            STAGE1_LATEST_PATH,
        )
    if stage == 2:
        return (
            STAGE2_BEST_PATH,
            STAGE2_BEST_LOSS_PATH,
            STAGE2_LATEST_PATH,
        )
    raise ValueError("stage must be 1 or 2")


def save_checkpoint(
    path: Path,
    *,
    stage: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    config: Dict[str, Any],
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    payload = {
        "stage": int(stage),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "model_config": dict(config),
        "best_val_mrae": float(best_val_mrae),
        "best_val_loss": float(best_val_loss),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "scheduler_name": (
            type(scheduler).__name__
            if scheduler is not None
            else None
        ),
        "trainable_parameters": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Union[str, Path],
    device: torch.device,
) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    if (
        not isinstance(checkpoint, dict)
        or "model" not in checkpoint
        or "model_config" not in checkpoint
    ):
        raise ValueError(
            f"{path} is not a complete checkpoint."
        )

    return checkpoint


# ==================================================
# MODEL CONSTRUCTION AND FREEZING
# ==================================================

def build_stage1_from_config(
    config: Dict[str, Any],
) -> Stage1InputPriorMSTPP:
    return build_stage1_model(
        rgb_channels=config["rgb_channels"],
        hsi_channels=config["hsi_channels"],
        cpen_feats=config["cpen_feats"],
        cpen_res_blocks=config["cpen_res_blocks"],
        prior_dim=config["prior_dim"],
        unshuffle_factor=config["unshuffle_factor"],
        mst_stages=config["mst_stages"],
        prior_channels=config["prior_channels"],
        adapter_hidden_channels=config[
            "adapter_hidden_channels"
        ],
        max_delta=config["max_delta"],
    )


def build_stage2_from_config(
    stage1_model: Stage1InputPriorMSTPP,
    config: Dict[str, Any],
) -> Stage2DDIMPriorMSTPP:
    return build_stage2_from_stage1(
        stage1_model=stage1_model,
        rgb_channels=config["rgb_channels"],
        prior_dim=config["prior_dim"],
        condition_dim=config["condition_dim"],
        condition_feats=config["condition_feats"],
        condition_res_blocks=config[
            "condition_res_blocks"
        ],
        unshuffle_factor=config["unshuffle_factor"],
        denoiser_hidden_dim=config[
            "denoiser_hidden_dim"
        ],
        denoiser_time_dim=config[
            "denoiser_time_dim"
        ],
        denoiser_blocks=config["denoiser_blocks"],
        denoiser_dropout=config["denoiser_dropout"],
        train_timesteps=config["train_timesteps"],
        sample_steps=config["sample_steps"],
        beta_schedule=config["beta_schedule"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        deterministic_seed=config[
            "deterministic_seed"
        ],
        prior_scale=config["prior_scale"],
        initialise_condition_from_cpen=config[
            "initialise_condition_from_cpen"
        ],
        copy_deeper_condition_blocks=config[
            "copy_deeper_condition_blocks"
        ],
        freeze_mstpp=True,
    )


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def configure_stage2_trainability(
    model: Stage2DDIMPriorMSTPP,
) -> None:
    """
    Freeze every Stage-1 component and train only the diffusion subsystem.

    Trainable:
        model.condition_encoder
        model.prior_diffusion

    Frozen:
        model.teacher_cpen
        model.input_adapter
        model.mstpp
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if model.teacher_cpen is not None:
        freeze_module(model.teacher_cpen)

    freeze_module(model.input_adapter)
    freeze_module(model.mstpp)

    # The RGB condition encoder is the conditional branch of the diffusion
    # model, so it must learn alongside the denoiser.
    unfreeze_module(model.condition_encoder)
    unfreeze_module(model.prior_diffusion)

    model.condition_encoder.train()
    model.prior_diffusion.train()


def enforce_stage2_modes(
    model: Stage2DDIMPriorMSTPP,
) -> None:
    """
    Call after model.train(); model.train() recursively changes all child
    modules to training mode, so frozen Stage-1 components are reset to eval.
    """
    if model.teacher_cpen is not None:
        model.teacher_cpen.eval()

    model.input_adapter.eval()
    model.mstpp.eval()

    model.condition_encoder.train()
    model.prior_diffusion.train()


def count_parameters(
    model: torch.nn.Module,
) -> Tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def trainable_parameters(
    model: torch.nn.Module,
) -> Iterable[torch.nn.Parameter]:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    return parameters


def load_stage1_teacher(
    path: Union[str, Path],
    device: torch.device,
) -> Tuple[Stage1InputPriorMSTPP, Dict[str, Any]]:
    checkpoint = load_checkpoint(path, device)

    if int(checkpoint.get("stage", -1)) != 1:
        raise ValueError(
            "The teacher checkpoint must be Stage 1."
        )

    config = dict(checkpoint["model_config"])
    stage1 = build_stage1_from_config(config).to(device)
    stage1.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    stage1.eval()

    for parameter in stage1.parameters():
        parameter.requires_grad_(False)

    return stage1, config


def build_training_model(
    device: torch.device,
) -> Tuple[
    torch.nn.Module,
    Dict[str, Any],
]:
    if STAGE == 1:
        config = make_model_config()
        model = build_stage1_from_config(config).to(device)
        return model, config

    if STAGE != 2:
        raise ValueError("STAGE must be 1 or 2.")

    stage1, teacher_config = load_stage1_teacher(
        TEACHER_CHECKPOINT,
        device,
    )

    # Preserve Stage-1 architecture parameters but use current DDIM settings.
    config = dict(teacher_config)
    current = make_model_config()

    for key in (
        "condition_dim",
        "condition_feats",
        "condition_res_blocks",
        "denoiser_hidden_dim",
        "denoiser_time_dim",
        "denoiser_blocks",
        "denoiser_dropout",
        "train_timesteps",
        "sample_steps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "deterministic_seed",
        "prior_scale",
        "initialise_condition_from_cpen",
        "copy_deeper_condition_blocks",
        "freeze_mstpp",
        "freeze_adapter",
        "freeze_teacher_cpen",
    ):
        config[key] = current[key]

    model = build_stage2_from_config(
        stage1,
        config,
    ).to(device)

    configure_stage2_trainability(model)

    return model, config


def build_evaluation_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint_stage = int(
        checkpoint.get("stage", -1)
    )

    if checkpoint_stage != STAGE:
        raise ValueError(
            f"Checkpoint stage is {checkpoint_stage}, "
            f"but STAGE={STAGE}."
        )

    config = dict(checkpoint["model_config"])

    if checkpoint_stage == 1:
        model: torch.nn.Module = (
            build_stage1_from_config(config)
        )
    else:
        # A structural Stage-1 model is sufficient because the complete
        # teacher, adapter, MST++, DDIM schedule, and condition encoder are
        # overwritten by the Stage-2 state dictionary.
        stage1_skeleton = build_stage1_from_config(
            config
        )
        model = build_stage2_from_config(
            stage1_skeleton,
            config,
        )
        configure_stage2_trainability(model)

    model = model.to(device)
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    return model, config


# ==================================================
# VALIDATION
# ==================================================

@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    split_name: str = "Validation",
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
    total_batches = len(val_loader)
    split_start = time.perf_counter()

    print(
        f"{split_name} started: "
        f"{total_batches} full-resolution batches",
        flush=True,
    )

    for batch_index, batch in enumerate(val_loader):
        batch_start = time.perf_counter()

        rgb, hsi, _, orig_hw = unpack_batch(batch)
        orig_hw_tensor = make_orig_hw_tensor(
            orig_hw,
            hsi,
        )

        rgb = rgb.to(
            device,
            non_blocking=(device.type == "cuda"),
        )
        hsi = hsi.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        if STAGE == 1:
            if not isinstance(
                model,
                Stage1InputPriorMSTPP,
            ):
                raise TypeError(
                    "Stage-1 validation requires "
                    "Stage1InputPriorMSTPP."
                )

            output = model(rgb, hsi)
        else:
            if not isinstance(
                model,
                Stage2DDIMPriorMSTPP,
            ):
                raise TypeError(
                    "Stage-2 validation requires "
                    "Stage2DDIMPriorMSTPP."
                )

            # No external random tensor is supplied. The model uses its fixed
            # registered initial latent and eta=0 DDIM reverse trajectory.
            output = model(rgb)

        pred_hsi = output["pred_hsi"]

        for sample_index in range(rgb.shape[0]):
            sample_pred, sample_hsi = crop_sample(
                pred_hsi,
                hsi,
                orig_hw_tensor,
                sample_index,
            )

            sample_loss = reconstruction_loss(
                sample_pred,
                sample_hsi,
                loss_type=RECONSTRUCTION_LOSS,
                mrae_eps=MRAE_EPS,
            )

            metrics = compute_metrics(
                sample_pred,
                sample_hsi,
                mrae_eps=MRAE_EPS,
            )

            totals["loss"] += float(
                sample_loss.item()
            )

            for name in (
                "mrae",
                "rmse",
                "psnr",
                "sam",
                "ssim",
            ):
                totals[name] += float(metrics[name])

            count += 1

        completed = batch_index + 1

        if (
            completed % PROGRESS_EVERY_N_BATCHES == 0
            or completed == total_batches
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            running = {
                name: totals[name] / max(count, 1)
                for name in TRAIN_DISPLAY_METRICS
            }

            batch_seconds = (
                time.perf_counter() - batch_start
            )
            elapsed_minutes = (
                time.perf_counter() - split_start
            ) / 60.0

            print(
                f"{split_name} batch "
                f"{completed}/{total_batches} "
                f"| MRAE {running['mrae']:.6f} "
                f"| RMSE {running['rmse']:.6f} "
                f"| PSNR {running['psnr']:.4f} "
                f"| SAM {running['sam']:.4f} "
                f"| SSIM {running['ssim']:.6f} "
                f"| {batch_seconds:.1f}s/batch "
                f"| elapsed {elapsed_minutes:.1f} min",
                flush=True,
            )

    if count == 0:
        raise RuntimeError(
            f"{split_name} loader is empty."
        )

    return {
        name: value / count
        for name, value in totals.items()
    }


# ==================================================
# TRAINING
# ==================================================

def train() -> None:
    if STAGE not in (1, 2):
        raise ValueError("STAGE must be 1 or 2.")

    set_seed(SEED)
    device = torch.device(DEVICE)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()

    train_loader, val_loader, test_loader = (
        make_dataloaders(device)
    )

    model, config = build_training_model(device)

    learning_rate = (
        LR_STAGE1
        if STAGE == 1
        else LR_STAGE2
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=COSINE_T_MAX,
        eta_min=MIN_LR,
    )

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        resume = load_checkpoint(
            RESUME_CHECKPOINT,
            device,
        )

        if int(resume.get("stage", -1)) != STAGE:
            raise ValueError(
                "Resume checkpoint stage does not "
                "match STAGE."
            )

        if dict(resume["model_config"]) != config:
            raise ValueError(
                "Resume checkpoint architecture differs "
                "from the current configuration."
            )

        model.load_state_dict(
            resume["model"],
            strict=True,
        )

        if "optimizer" in resume:
            optimizer.load_state_dict(
                resume["optimizer"]
            )

        if "scheduler" in resume:
            scheduler.load_state_dict(
                resume["scheduler"]
            )

        start_epoch = int(
            resume.get("epoch", 0)
        ) + 1

        best_val_mrae = float(
            resume.get(
                "best_val_mrae",
                math.inf,
            )
        )
        best_val_loss = float(
            resume.get(
                "best_val_loss",
                math.inf,
            )
        )
        epochs_without_improvement = int(
            resume.get(
                "epochs_without_improvement",
                0,
            )
        )

        print(
            f"Resumed Stage {STAGE} from "
            f"epoch {start_epoch}."
        )

    best_path, best_loss_path, latest_path = (
        checkpoint_paths(STAGE)
    )

    total_parameters, trainable_count = (
        count_parameters(model)
    )

    print("Execution mode: single GPU")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(f"Stage: {STAGE}")
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    if STAGE == 2:
        print(
            "Stage-2 trainable modules: "
            "condition_encoder + prior_diffusion"
        )
        print(
            "Stage-2 frozen modules: "
            "teacher_cpen + input_adapter + mstpp"
        )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )
    print(
        f"Reserved test samples: "
        f"{len(test_loader.dataset)}"
    )
    print(
        f"Gradient checkpointing: "
        f"{USE_GRADIENT_CHECKPOINTING}"
    )
    print(
        f"Model configuration: {config}"
    )

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS} started "
            f"({len(train_loader)} batches)",
            flush=True,
        )

        model.train()

        if STAGE == 2:
            if not isinstance(
                model,
                Stage2DDIMPriorMSTPP,
            ):
                raise TypeError(
                    "Stage 2 requires "
                    "Stage2DDIMPriorMSTPP."
                )
            enforce_stage2_modes(model)

        running_total = 0.0
        running_reconstruction = 0.0
        running_prior_l1 = 0.0
        running_diffusion = 0.0

        running_metrics = {
            name: 0.0
            for name in TRAIN_DISPLAY_METRICS
        }

        metric_sample_count = 0
        train_count = 0

        num_batches = len(train_loader)
        previous_batch_end = time.perf_counter()

        for batch_index, batch in enumerate(
            train_loader
        ):
            batch_start = time.perf_counter()
            data_wait_seconds = (
                batch_start - previous_batch_end
            )

            rgb, hsi, names, orig_hw = (
                unpack_batch(batch)
            )

            orig_hw_tensor = make_orig_hw_tensor(
                orig_hw,
                hsi,
            )

            if batch_index == 0:
                print(
                    "First batch "
                    f"| RGB {tuple(rgb.shape)} "
                    f"| HSI {tuple(hsi.shape)} "
                    f"| sample {names}",
                    flush=True,
                )

            rgb = rgb.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )
            hsi = hsi.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            batch_size = rgb.shape[0]
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                if STAGE == 1:
                    if not isinstance(
                        model,
                        Stage1InputPriorMSTPP,
                    ):
                        raise TypeError(
                            "Stage 1 requires "
                            "Stage1InputPriorMSTPP."
                        )

                    if USE_GRADIENT_CHECKPOINTING:
                        def stage1_forward(
                            rgb_input,
                            hsi_input,
                        ):
                            return model(
                                rgb_input,
                                hsi_input,
                            )["pred_hsi"]

                        pred_hsi = activation_checkpoint(
                            stage1_forward,
                            rgb,
                            hsi,
                            use_reentrant=False,
                        )
                    else:
                        output = model(rgb, hsi)
                        pred_hsi = output[
                            "pred_hsi"
                        ]

                    rec_loss = reconstruction_loss(
                        pred_hsi,
                        hsi,
                        loss_type=RECONSTRUCTION_LOSS,
                        mrae_eps=MRAE_EPS,
                    )

                    prior_l1 = rec_loss.new_zeros(())
                    diffusion_loss = (
                        rec_loss.new_zeros(())
                    )
                    total_loss = rec_loss

                else:
                    if not isinstance(
                        model,
                        Stage2DDIMPriorMSTPP,
                    ):
                        raise TypeError(
                            "Stage 2 requires "
                            "Stage2DDIMPriorMSTPP."
                        )

                    if USE_GRADIENT_CHECKPOINTING:
                        def stage2_forward(
                            rgb_input,
                            hsi_input,
                        ):
                            result = model(
                                rgb_input,
                                hsi_input,
                            )
                            return (
                                result["pred_hsi"],
                                result[
                                    "predicted_prior"
                                ],
                                result[
                                    "teacher_prior"
                                ],
                                result[
                                    "diffusion_noise_loss"
                                ],
                            )

                        (
                            pred_hsi,
                            predicted_prior,
                            teacher_prior,
                            diffusion_loss,
                        ) = activation_checkpoint(
                            stage2_forward,
                            rgb,
                            hsi,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                    else:
                        output = model(
                            rgb,
                            hsi,
                            reconstruct_hsi=True,
                        )

                        pred_hsi = output[
                            "pred_hsi"
                        ]
                        predicted_prior = output[
                            "predicted_prior"
                        ]
                        teacher_prior = output[
                            "teacher_prior"
                        ]
                        diffusion_loss = output[
                            "diffusion_noise_loss"
                        ]

                    if teacher_prior is None:
                        raise RuntimeError(
                            "Stage-2 training did not "
                            "produce a teacher prior."
                        )

                    if diffusion_loss is None:
                        raise RuntimeError(
                            "Stage-2 training did not "
                            "produce a diffusion loss."
                        )
                    print(pred_hsi)
                    rec_loss = reconstruction_loss(
                        pred_hsi,
                        hsi,
                        loss_type=RECONSTRUCTION_LOSS,
                        mrae_eps=MRAE_EPS,
                    )

                    prior_l1 = prior_l1_loss(
                        predicted_prior,
                        teacher_prior.detach(),
                    )

                    total_loss = (
                        rec_loss
                        + LAMBDA_PRIOR_L1
                        * prior_l1
                        + LAMBDA_DIFFUSION
                        * diffusion_loss
                    )

            scaler.scale(total_loss).backward()

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    list(trainable_parameters(model)),
                    max_norm=GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

            running_total += (
                float(total_loss.item())
                * batch_size
            )
            running_reconstruction += (
                float(rec_loss.item())
                * batch_size
            )
            running_prior_l1 += (
                float(prior_l1.item())
                * batch_size
            )
            running_diffusion += (
                float(diffusion_loss.item())
                * batch_size
            )
            train_count += batch_size

            completed = batch_index + 1

            should_measure = (
                completed
                % TRAIN_METRICS_EVERY_N_BATCHES
                == 0
                or completed == num_batches
            )

            if should_measure:
                batch_metrics = (
                    compute_batch_display_metrics(
                        pred_hsi,
                        hsi,
                        orig_hw_tensor,
                    )
                )

                for name in TRAIN_DISPLAY_METRICS:
                    running_metrics[name] += (
                        batch_metrics[name]
                        * batch_size
                    )

                metric_sample_count += batch_size

            should_log = (
                completed
                % PROGRESS_EVERY_N_BATCHES
                == 0
                or completed == num_batches
            )

            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                    allocated_gb = (
                        torch.cuda.memory_allocated(
                            device
                        )
                        / (1024 ** 3)
                    )
                    reserved_gb = (
                        torch.cuda.memory_reserved(
                            device
                        )
                        / (1024 ** 3)
                    )

                    memory_text = (
                        f" | CUDA "
                        f"{allocated_gb:.2f}/"
                        f"{reserved_gb:.2f} GiB "
                        "alloc/reserved"
                    )
                else:
                    memory_text = ""

                average_metrics = {
                    name: running_metrics[name]
                    / max(metric_sample_count, 1)
                    for name in TRAIN_DISPLAY_METRICS
                }

                if RECONSTRUCTION_LOSS == "mrae":
                    average_metrics["mrae"] = (
                        running_reconstruction
                        / max(train_count, 1)
                    )

                batch_seconds = (
                    time.perf_counter()
                    - batch_start
                )

                stage2_text = ""

                if STAGE == 2:
                    stage2_text = (
                        " | Prior L1 "
                        f"{running_prior_l1 / max(train_count, 1):.6f}"
                        " | Diffusion "
                        f"{running_diffusion / max(train_count, 1):.6f}"
                    )

                print(
                    f"Epoch {epoch} batch "
                    f"{completed}/{num_batches} "
                    f"| MRAE "
                    f"{average_metrics['mrae']:.6f} "
                    f"| RMSE "
                    f"{average_metrics['rmse']:.6f} "
                    f"| PSNR "
                    f"{average_metrics['psnr']:.4f} "
                    f"| SAM "
                    f"{average_metrics['sam']:.4f} "
                    f"| SSIM "
                    f"{average_metrics['ssim']:.6f}"
                    f"{stage2_text} "
                    f"| data {data_wait_seconds:.2f}s "
                    f"| step {batch_seconds:.1f}s"
                    f"{memory_text}",
                    flush=True,
                )

            previous_batch_end = time.perf_counter()

        train_total = (
            running_total / max(train_count, 1)
        )
        train_rec = (
            running_reconstruction
            / max(train_count, 1)
        )
        train_prior_l1 = (
            running_prior_l1
            / max(train_count, 1)
        )
        train_diffusion = (
            running_diffusion
            / max(train_count, 1)
        )

        train_metrics = {
            name: running_metrics[name]
            / max(metric_sample_count, 1)
            for name in TRAIN_DISPLAY_METRICS
        }

        if RECONSTRUCTION_LOSS == "mrae":
            train_metrics["mrae"] = train_rec

        if device.type == "cuda":
            peak_gb = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024 ** 3)
            )
            print(
                f"Epoch {epoch} peak CUDA "
                f"allocation: {peak_gb:.2f} GiB"
            )
            torch.cuda.reset_peak_memory_stats(
                device
            )

        val_results = validate(
            model,
            val_loader,
            device,
            split_name="Validation",
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]
        scheduler.step()
        next_lr = optimizer.param_groups[0][
            "lr"
        ]

        epoch_minutes = (
            time.perf_counter() - epoch_start
        ) / 60.0

        if STAGE == 1:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} "
                f"| Train MRAE "
                f"{train_metrics['mrae']:.6f} "
                f"| Val MRAE "
                f"{val_results['mrae']:.6f} "
                f"| Val RMSE "
                f"{val_results['rmse']:.6f} "
                f"| Val PSNR "
                f"{val_results['psnr']:.4f} "
                f"| Val SAM "
                f"{val_results['sam']:.4f} "
                f"| Val SSIM "
                f"{val_results['ssim']:.6f} "
                f"| LR {current_lr:.2e} "
                f"| Next LR {next_lr:.2e} "
                f"| {epoch_minutes:.1f} min"
            )
        else:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} "
                f"| Train Total "
                f"{train_total:.6f} "
                f"| Train Reconstruction "
                f"{train_rec:.6f} "
                f"| Train Prior L1 "
                f"{train_prior_l1:.6f} "
                f"| Train Diffusion "
                f"{train_diffusion:.6f} "
                f"| Val MRAE "
                f"{val_results['mrae']:.6f} "
                f"| Val RMSE "
                f"{val_results['rmse']:.6f} "
                f"| Val PSNR "
                f"{val_results['psnr']:.4f} "
                f"| Val SAM "
                f"{val_results['sam']:.4f} "
                f"| Val SSIM "
                f"{val_results['ssim']:.6f} "
                f"| LR {current_lr:.2e} "
                f"| Next LR {next_lr:.2e} "
                f"| {epoch_minutes:.1f} min"
            )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]

            save_checkpoint(
                best_loss_path,
                stage=STAGE,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
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
                scheduler=scheduler,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=0,
            )

            print(
                f"Saved best Stage-{STAGE} model "
                f"(Val MRAE "
                f"{best_val_mrae:.6f})."
            )
        else:
            epochs_without_improvement += 1

            print(
                "No validation MRAE improvement "
                f"for {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

        save_checkpoint(
            latest_path,
            stage=STAGE,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping triggered. "
                f"Best validation MRAE: "
                f"{best_val_mrae:.6f}"
            )
            break


# ==================================================
# EVALUATION
# ==================================================

def resolve_evaluation_checkpoint(
    stage: int,
) -> Path:
    if EVAL_CHECKPOINT is not None:
        selected = Path(EVAL_CHECKPOINT)

        if not selected.exists():
            raise FileNotFoundError(
                f"EVAL_CHECKPOINT does not exist: "
                f"{selected}"
            )

        return selected

    best_path, best_loss_path, latest_path = (
        checkpoint_paths(stage)
    )

    for candidate in (
        best_path,
        best_loss_path,
        latest_path,
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No Stage-{stage} checkpoint exists."
    )


def evaluate(split_name: str) -> None:
    if STAGE not in (1, 2):
        raise ValueError("STAGE must be 1 or 2.")

    if split_name not in {"val", "test"}:
        raise ValueError(
            "split_name must be 'val' or 'test'."
        )

    set_seed(SEED)
    device = torch.device(DEVICE)

    _, val_loader, test_loader = (
        make_dataloaders(device)
    )

    loader = (
        val_loader
        if split_name == "val"
        else test_loader
    )

    selected_path = resolve_evaluation_checkpoint(
        STAGE
    )

    checkpoint = load_checkpoint(
        selected_path,
        device,
    )

    model, config = build_evaluation_model(
        checkpoint,
        device,
    )

    results = validate(
        model,
        loader,
        device,
        split_name=split_name.capitalize(),
    )

    print(f"Evaluated split: {split_name}")
    print(
        f"Evaluated samples: "
        f"{len(loader.dataset)}"
    )
    print(
        f"Evaluated checkpoint: "
        f"{selected_path}"
    )
    print(f"Model configuration: {config}")
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
        evaluate("test")
    else:
        raise ValueError(
            "MODE must be train, val, test, or eval."
        )


if __name__ == "__main__":
    main()
