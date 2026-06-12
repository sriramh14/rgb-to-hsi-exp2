"""Paired RGB-HSI dataset for the standalone DiffIR RGB-to-HSI baseline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".h5", ".hdf5"}


def _index_by_stem(root: str | Path, extensions: Iterable[str]) -> Dict[str, Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    extensions = {suffix.lower() for suffix in extensions}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    if not files:
        raise FileNotFoundError(f"No supported files found in {root}")

    indexed: Dict[str, Path] = {}
    for path in sorted(files):
        key = path.stem
        if key in indexed:
            raise ValueError(
                f"Duplicate file stem '{key}' in {root}: {indexed[key]} and {path}. "
                "Use unique names for paired matching."
            )
        indexed[key] = path
    return indexed


def _select_3d_array(container: Dict, explicit_key: Optional[str]) -> np.ndarray:
    if explicit_key:
        if explicit_key not in container:
            raise KeyError(f"Requested HSI key '{explicit_key}' not found. Available keys: {list(container)}")
        array = np.asarray(container[explicit_key])
        if array.ndim != 3:
            raise ValueError(f"HSI key '{explicit_key}' must contain a 3D array, got {array.shape}")
        return array

    candidates = []
    for key, value in container.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 3 and np.issubdtype(array.dtype, np.number):
            candidates.append((key, array))

    if not candidates:
        raise ValueError("Could not find a numeric 3D HSI array in the file")
    if len(candidates) > 1:
        candidates.sort(key=lambda pair: pair[1].size, reverse=True)
    return candidates[0][1]


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        rgb = np.load(path)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))

    rgb = np.asarray(rgb)
    if rgb.ndim != 3:
        raise ValueError(f"RGB file must be 3D, got {rgb.shape} from {path}")
    if rgb.shape[-1] == 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    elif rgb.shape[0] != 3:
        raise ValueError(f"Cannot infer RGB channel axis for {path}: {rgb.shape}")

    rgb = rgb.astype(np.float32, copy=False)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return np.ascontiguousarray(rgb)


def load_hsi(path: Path, num_bands: int, hsi_key: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        hsi = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as values:
            hsi = _select_3d_array(dict(values), hsi_key)
    elif suffix == ".mat":
        try:
            hsi = _select_3d_array(loadmat(path), hsi_key)
        except NotImplementedError:
            with h5py.File(path, "r") as values:
                arrays = {key: np.asarray(values[key]) for key in values.keys()}
            hsi = _select_3d_array(arrays, hsi_key)
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as values:
            arrays = {key: np.asarray(values[key]) for key in values.keys()}
        hsi = _select_3d_array(arrays, hsi_key)
    else:
        raise ValueError(f"Unsupported HSI extension: {path.suffix}")

    hsi = np.asarray(hsi)
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3D, got {hsi.shape} from {path}")

    # Convert HWC or common MATLAB transpositions to CHW.
    if hsi.shape[0] == num_bands:
        pass
    elif hsi.shape[-1] == num_bands:
        hsi = np.transpose(hsi, (2, 0, 1))
    elif hsi.shape[1] == num_bands:
        hsi = np.transpose(hsi, (1, 0, 2))
    else:
        raise ValueError(
            f"No axis of {path} has num_bands={num_bands}; array shape is {hsi.shape}. "
            "Set --num-bands correctly or preprocess the cube."
        )

    return np.ascontiguousarray(hsi.astype(np.float32, copy=False))


def _random_crop(rgb: np.ndarray, hsi: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    _, h, w = rgb.shape
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image size {(h, w)} is smaller than patch_size={patch_size}. "
            "Reduce --patch-size or preprocess the data."
        )
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        rgb[:, top : top + patch_size, left : left + patch_size],
        hsi[:, top : top + patch_size, left : left + patch_size],
    )


def _augment(rgb: np.ndarray, hsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        rgb = rgb[:, :, ::-1]
        hsi = hsi[:, :, ::-1]
    if random.random() < 0.5:
        rgb = rgb[:, ::-1, :]
        hsi = hsi[:, ::-1, :]
    rotations = random.randint(0, 3)
    if rotations:
        rgb = np.rot90(rgb, rotations, axes=(1, 2))
        hsi = np.rot90(hsi, rotations, axes=(1, 2))
    return np.ascontiguousarray(rgb), np.ascontiguousarray(hsi)


class RGBHSIDataset(Dataset):
    """Pairs RGB and HSI files by identical filename stem."""

    def __init__(
        self,
        rgb_dir: str | Path,
        hsi_dir: str | Path,
        num_bands: int,
        patch_size: Optional[int] = None,
        training: bool = False,
        hsi_key: Optional[str] = None,
        hsi_scale: float = 1.0,
        clip_hsi: bool = False,
    ):
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if patch_size is not None and patch_size % 32 != 0:
            raise ValueError("patch_size must be divisible by 32 for DIRformer")
        if hsi_scale <= 0:
            raise ValueError("hsi_scale must be positive")

        rgb_files = _index_by_stem(rgb_dir, RGB_EXTENSIONS)
        hsi_files = _index_by_stem(hsi_dir, HSI_EXTENSIONS)
        common = sorted(set(rgb_files) & set(hsi_files))
        if not common:
            raise ValueError(
                "No RGB-HSI pairs have matching filename stems. "
                f"RGB examples: {list(rgb_files)[:3]}, HSI examples: {list(hsi_files)[:3]}"
            )

        missing_hsi = sorted(set(rgb_files) - set(hsi_files))
        missing_rgb = sorted(set(hsi_files) - set(rgb_files))
        if missing_hsi or missing_rgb:
            print(
                f"Warning: using {len(common)} matched pairs; "
                f"unmatched RGB={len(missing_hsi)}, unmatched HSI={len(missing_rgb)}"
            )

        self.pairs = [(rgb_files[key], hsi_files[key], key) for key in common]
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.training = training
        self.hsi_key = hsi_key
        self.hsi_scale = hsi_scale
        self.clip_hsi = clip_hsi

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        rgb_path, hsi_path, name = self.pairs[index]
        rgb = load_rgb(rgb_path)
        hsi = load_hsi(hsi_path, self.num_bands, self.hsi_key) / self.hsi_scale

        if rgb.shape[-2:] != hsi.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch for '{name}': RGB {rgb.shape[-2:]}, HSI {hsi.shape[-2:]}"
            )

        if self.clip_hsi:
            hsi = np.clip(hsi, 0.0, 1.0)

        if self.training and self.patch_size is not None:
            rgb, hsi = _random_crop(rgb, hsi, self.patch_size)
            rgb, hsi = _augment(rgb, hsi)

        return {
            "rgb": torch.from_numpy(rgb).float(),
            "hsi": torch.from_numpy(hsi).float(),
            "name": name,
        }
