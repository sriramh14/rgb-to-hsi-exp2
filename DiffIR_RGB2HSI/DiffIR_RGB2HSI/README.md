# DiffIR-RGB2HSI

A small, standalone adaptation of **DiffIR** for paired, same-resolution RGB-to-hyperspectral reconstruction.

The repository keeps the original two-stage idea:

1. **Stage 1 (oracle teacher):** extract a compact prior from paired RGB and ground-truth HSI, then reconstruct HSI with a prior-conditioned DIRformer.
2. **Stage 2 (deployable model):** predict the compact prior from RGB alone using short latent diffusion, then reconstruct HSI with the same DIRformer.

Unlike `DiffIR-RealSR`, this counterpart:

- does not synthesize RealSR degradations;
- does not use JPEG, blur, sharpening, or GAN training;
- uses paired RGB and HSI files directly;
- fixes the spatial scale to 1;
- predicts `L` spectral bands rather than three RGB channels;
- replaces the invalid RGB residual addition with a learned `3 -> L` projection;
- has one entry point, `train.py`, for Stage 1, Stage 2, resume, and evaluation;
- has no BasicSR dependency.

Every architectural adaptation is marked with `RGB2HSI CHANGE` in the model file.

## File structure

```text
DiffIR_RGB2HSI/
├── train.py                       # single train/eval script
├── data.py                        # paired RGB-HSI loader
├── requirements.txt
├── README.md
└── models/
    ├── __init__.py
    └── diffir_rgb2hsi.py          # Stage 1, Stage 2, DIRformer, diffusion
```

## Data layout

RGB and HSI files are paired by identical filename stem.

```text
dataset/
├── train_rgb/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── train_hsi/
│   ├── 0001.mat
│   ├── 0002.mat
│   └── ...
├── val_rgb/
│   ├── 0901.png
│   └── ...
└── val_hsi/
    ├── 0901.mat
    └── ...
```

Supported RGB formats:

```text
.png .jpg .jpeg .bmp .tif .tiff .npy
```

Supported HSI formats:

```text
.mat .npy .npz .h5 .hdf5
```

HSI arrays may be stored as `[H,W,L]` or `[L,H,W]`. Use `--hsi-key` when a MAT/HDF5 file contains more than one 3D array.

The model expects normalized floating-point HSI. Use `--hsi-scale 65535` for uint16 cubes, or another dataset-specific divisor. `--clip-hsi` optionally clips targets to `[0,1]`.

## Installation

```bash
pip install -r requirements.txt
```

## Stage 1 training

Example for 31-band HSI:

```bash
python train.py \
  --stage 1 \
  --train-rgb-dir /path/dataset/train_rgb \
  --train-hsi-dir /path/dataset/train_hsi \
  --val-rgb-dir /path/dataset/val_rgb \
  --val-hsi-dir /path/dataset/val_hsi \
  --num-bands 31 \
  --patch-size 64 \
  --batch-size 8 \
  --epochs 100 \
  --lr 2e-4 \
  --loss mrae \
  --amp \
  --out-dir ./exp/diffir_rgb2hsi
```

Best checkpoint:

```text
./exp/diffir_rgb2hsi/stage1/best_stage1.pth
```

For the larger RealSR-style x1 configuration, add:

```bash
--dim 64 \
--num-blocks 13,1,1,1 \
--num-refinement-blocks 13 \
--heads 1,2,4,8 \
--ffn-expansion-factor 2.2 \
--layer-norm-type BiasFree \
--n-encoder-res 9
```

## Stage 2 training

Stage 2 requires the trained Stage-1 checkpoint. The Stage-1 CPEN is frozen and supplies the target compact prior.

```bash
python train.py \
  --stage 2 \
  --teacher-checkpoint ./exp/diffir_rgb2hsi/stage1/best_stage1.pth \
  --train-rgb-dir /path/dataset/train_rgb \
  --train-hsi-dir /path/dataset/train_hsi \
  --val-rgb-dir /path/dataset/val_rgb \
  --val-hsi-dir /path/dataset/val_hsi \
  --patch-size 64 \
  --batch-size 8 \
  --epochs 100 \
  --lr 2e-4 \
  --loss mrae \
  --lambda-prior 1.0 \
  --timesteps 4 \
  --amp \
  --out-dir ./exp/diffir_rgb2hsi
```

Best checkpoint:

```text
./exp/diffir_rgb2hsi/stage2/best_stage2.pth
```

At Stage-2 validation and inference, only RGB is passed to the model. A fixed `--eval-seed` makes the Gaussian prior initialization repeatable.

## Resume training

```bash
python train.py \
  --stage 2 \
  --teacher-checkpoint ./exp/diffir_rgb2hsi/stage1/best_stage1.pth \
  --resume ./exp/diffir_rgb2hsi/stage2/latest_stage2.pth \
  --train-rgb-dir /path/dataset/train_rgb \
  --train-hsi-dir /path/dataset/train_hsi \
  --val-rgb-dir /path/dataset/val_rgb \
  --val-hsi-dir /path/dataset/val_hsi \
  --epochs 200
```

## Evaluation

```bash
python train.py \
  --mode eval \
  --stage 2 \
  --checkpoint ./exp/diffir_rgb2hsi/stage2/best_stage2.pth \
  --val-rgb-dir /path/dataset/val_rgb \
  --val-hsi-dir /path/dataset/val_hsi \
  --clip-output-eval \
  --out-dir ./exp/diffir_rgb2hsi
```

Metrics written to the log and JSON file:

- MRAE
- RMSE
- PSNR
- SAM in degrees

## Important shape requirements

The internal architecture starts with `PixelUnshuffle(4)` and then performs three spatial downsampling operations. Training patches and full validation images must therefore have height and width divisible by 32.

Recommended patch sizes:

```text
64, 128, or 256
```

## Stage-2 objective

The implemented Stage-2 loss is:

```text
L_total = L_reconstruction
        + lambda_prior * L1(predicted_prior, teacher_prior)
        + lambda_kd * KL(predicted_prior, teacher_prior)
```

`lambda_kd` defaults to zero because the released DiffIR training code optimizes the absolute prior-matching term while only logging the KL term. Set `--lambda-kd` explicitly to include it.

## Output checkpoints

```text
stage1/
├── best_stage1.pth
├── latest_stage1.pth
├── model_stage1_epoch_010_iter_XXXXXX.pth
├── arguments.json
└── train.log

stage2/
├── best_stage2.pth
├── latest_stage2.pth
├── model_stage2_epoch_010_iter_XXXXXX.pth
├── arguments.json
└── train.log
```

## Attribution

This code is an RGB-to-HSI research adaptation of the public DiffIR implementation. Preserve the original project attribution and comply with its license when redistributing derived code.
