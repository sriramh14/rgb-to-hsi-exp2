#!/usr/bin/env python3
"""Single training/evaluation entry point for DiffIR RGB-to-HSI.

Examples
--------
Stage 1:
    python train.py --stage 1 --train-rgb-dir ... --train-hsi-dir ... \
        --val-rgb-dir ... --val-hsi-dir ... --num-bands 31

Stage 2:
    python train.py --stage 2 --teacher-checkpoint exp/stage1/best_stage1.pth \
        --train-rgb-dir ... --train-hsi-dir ... --val-rgb-dir ... --val-hsi-dir ...

Evaluation:
    python train.py --mode eval --stage 2 --checkpoint exp/stage2/best_stage2.pth \
        --val-rgb-dir ... --val-hsi-dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import RGBHSIDataset
from models import DiffIRS1RGB2HSI, DiffIRS2RGB2HSI, ModelConfig, build_model


def parse_int_tuple(value: str, length: int = 4) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != length:
        raise argparse.ArgumentTypeError(f"Expected {length} comma-separated integers, got '{value}'")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the two-stage DiffIR baseline for RGB-to-HSI reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)

    parser.add_argument("--train-rgb-dir", type=str)
    parser.add_argument("--train-hsi-dir", type=str)
    parser.add_argument("--val-rgb-dir", type=str, required=True)
    parser.add_argument("--val-hsi-dir", type=str, required=True)
    parser.add_argument("--hsi-key", type=str, default=None)
    parser.add_argument(
        "--hsi-scale",
        type=float,
        default=1.0,
        help="Divide loaded HSI values by this constant, e.g. 65535 for uint16 cubes",
    )
    parser.add_argument("--clip-hsi", action="store_true", help="Clip loaded target HSI to [0,1]")

    parser.add_argument("--num-bands", type=int, default=31)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=["mrae", "l1", "mse"], default="mrae")
    parser.add_argument("--mrae-eps", type=float, default=1e-6)
    parser.add_argument("--lambda-prior", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num-blocks", type=lambda value: parse_int_tuple(value, 4), default=(4, 6, 6, 8))
    parser.add_argument("--num-refinement-blocks", type=int, default=4)
    parser.add_argument("--heads", type=lambda value: parse_int_tuple(value, 4), default=(1, 2, 4, 8))
    parser.add_argument("--ffn-expansion-factor", type=float, default=2.66)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--layer-norm-type", choices=["BiasFree", "WithBias"], default="WithBias")
    parser.add_argument("--prior-dim", type=int, default=256)
    parser.add_argument("--n-encoder-res", type=int, default=6)
    parser.add_argument("--n-denoise-res", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--linear-start", type=float, default=0.1)
    parser.add_argument("--linear-end", type=float, default=0.99)
    parser.add_argument("--no-rgb-to-hsi-skip", action="store_true")

    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint used by --mode eval")
    parser.add_argument("--out-dir", type=str, default="./exp/diffir_rgb2hsi")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-output-eval", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DiffIR_RGB2HSI")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(out_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        num_bands=args.num_bands,
        dim=args.dim,
        num_blocks=tuple(args.num_blocks),
        num_refinement_blocks=args.num_refinement_blocks,
        heads=tuple(args.heads),
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        layer_norm_type=args.layer_norm_type,
        prior_dim=args.prior_dim,
        n_encoder_res=args.n_encoder_res,
        n_denoise_res=args.n_denoise_res,
        timesteps=args.timesteps,
        linear_start=args.linear_start,
        linear_end=args.linear_end,
        use_rgb_to_hsi_skip=not args.no_rgb_to_hsi_skip,
    )


def load_raw_checkpoint(path: str | Path, device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def config_from_checkpoint(checkpoint: Dict) -> ModelConfig:
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_config'")
    return ModelConfig.from_dict(checkpoint["model_config"])


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, name: str, eps: float) -> torch.Tensor:
    if name == "mrae":
        return torch.mean(torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps))
    if name == "l1":
        return F.l1_loss(pred, target)
    if name == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown loss: {name}")


def kd_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log = F.log_softmax(student / temperature, dim=1)
    teacher_prob = F.softmax(teacher.detach() / temperature, dim=1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean")


@torch.no_grad()
def batch_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float) -> Dict[str, float]:
    error = pred - target
    mrae = torch.mean(torch.abs(error) / torch.clamp(torch.abs(target), min=eps))
    mse_per_sample = error.square().flatten(1).mean(dim=1)
    rmse = torch.sqrt(mse_per_sample).mean()
    psnr = (10.0 * torch.log10(1.0 / torch.clamp(mse_per_sample, min=eps))).mean()

    # Spectral angle mapper: average angle over all valid pixels.
    pred_spectra = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target_spectra = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    dot = (pred_spectra * target_spectra).sum(dim=1)
    denom = torch.linalg.vector_norm(pred_spectra, dim=1) * torch.linalg.vector_norm(target_spectra, dim=1)
    cosine = torch.clamp(dot / torch.clamp(denom, min=eps), -1.0, 1.0)
    sam = torch.rad2deg(torch.acos(cosine)).mean()

    return {
        "mrae": float(mrae.item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
        "sam": float(sam.item()),
    }


def make_dataloaders(args: argparse.Namespace, device: torch.device):
    val_dataset = RGBHSIDataset(
        args.val_rgb_dir,
        args.val_hsi_dir,
        num_bands=args.num_bands,
        patch_size=None,
        training=False,
        hsi_key=args.hsi_key,
        hsi_scale=args.hsi_scale,
        clip_hsi=args.clip_hsi,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    if args.mode == "eval":
        return None, val_loader

    if not args.train_rgb_dir or not args.train_hsi_dir:
        raise ValueError("--train-rgb-dir and --train-hsi-dir are required in train mode")
    train_dataset = RGBHSIDataset(
        args.train_rgb_dir,
        args.train_hsi_dir,
        num_bands=args.num_bands,
        patch_size=args.patch_size,
        training=True,
        hsi_key=args.hsi_key,
        hsi_scale=args.hsi_scale,
        clip_hsi=args.clip_hsi,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader


def load_teacher(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[DiffIRS1RGB2HSI, ModelConfig]:
    checkpoint = load_raw_checkpoint(checkpoint_path, device)
    if checkpoint.get("stage") != 1:
        raise ValueError(f"Teacher checkpoint must be Stage 1, got stage={checkpoint.get('stage')}")
    config = config_from_checkpoint(checkpoint)
    teacher = DiffIRS1RGB2HSI(config).to(device)
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, config


def save_checkpoint(
    path: Path,
    stage: int,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: ModelConfig,
    args: argparse.Namespace,
    best_mrae: float,
) -> None:
    payload = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": config.to_dict(),
        "args": vars(args),
        "best_mrae": best_mrae,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    stage: int,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals = {"mrae": 0.0, "rmse": 0.0, "psnr": 0.0, "sam": 0.0}
    count = 0

    # Stage-2 starts from random Gaussian prior. Use a fixed generator for stable validation.
    eval_generator = None
    if stage == 2:
        eval_generator = torch.Generator(device=device)
        eval_generator.manual_seed(args.eval_seed)

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        hsi = batch["hsi"].to(device, non_blocking=True)

        if rgb.shape[-2] % 32 != 0 or rgb.shape[-1] % 32 != 0:
            raise ValueError(
                f"Validation image '{batch['name']}' has size {tuple(rgb.shape[-2:])}; "
                "height and width must be divisible by 32."
            )

        if stage == 1:
            pred, _ = model(rgb, hsi)
        else:
            assert isinstance(model, DiffIRS2RGB2HSI)
            initial_noise = torch.randn(
                rgb.shape[0],
                model.config.prior_dim,
                generator=eval_generator,
                device=device,
            )
            pred = model(rgb, initial_noise=initial_noise)

        if args.clip_output_eval:
            pred = pred.clamp(0.0, 1.0)

        metrics = batch_metrics(pred, hsi, args.mrae_eps)
        batch_size = rgb.shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.out_dir) / f"stage{args.stage}"
    logger = configure_logging(out_dir)

    teacher = None
    if args.stage == 2:
        if not args.teacher_checkpoint:
            raise ValueError("Stage 2 requires --teacher-checkpoint from trained Stage 1")
        teacher, teacher_config = load_teacher(args.teacher_checkpoint, device)

        if args.resume:
            # A resumed Stage-2 checkpoint defines its own diffusion configuration.
            resume_metadata = load_raw_checkpoint(args.resume, device)
            config = config_from_checkpoint(resume_metadata)
        else:
            # Preserve the Stage-1 DIRformer/prior dimensions, while allowing Stage 2
            # to choose its denoiser depth and diffusion schedule from the CLI.
            config = ModelConfig.from_dict(teacher_config.to_dict())
            config.n_denoise_res = args.n_denoise_res
            config.timesteps = args.timesteps
            config.linear_start = args.linear_start
            config.linear_end = args.linear_end

        generator_fields = (
            "num_bands",
            "dim",
            "num_blocks",
            "num_refinement_blocks",
            "heads",
            "ffn_expansion_factor",
            "bias",
            "layer_norm_type",
            "prior_dim",
            "use_rgb_to_hsi_skip",
        )
        mismatched = [
            field
            for field in generator_fields
            if getattr(config, field) != getattr(teacher_config, field)
        ]
        if mismatched:
            raise ValueError(
                "Stage-2 generator configuration must match Stage 1. "
                f"Mismatched fields: {mismatched}"
            )

        args.num_bands = config.num_bands
        logger.info("Loaded frozen Stage-1 teacher from %s", args.teacher_checkpoint)
    else:
        config = config_from_args(args)

    with (out_dir / "arguments.json").open("w", encoding="utf-8") as file:
        json.dump({"args": vars(args), "model_config": config.to_dict()}, file, indent=2)

    train_loader, val_loader = make_dataloaders(args, device)
    assert train_loader is not None

    model = build_model(args.stage, config).to(device)
    if args.stage == 2 and not args.resume:
        assert teacher is not None
        assert isinstance(model, DiffIRS2RGB2HSI)
        model.initialize_generator_from_stage1(teacher)
        logger.info("Initialized Stage-2 DIRformer generator from Stage 1")

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)
    amp_enabled = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # Compatibility with older PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    best_mrae = math.inf
    if args.resume:
        checkpoint = load_raw_checkpoint(args.resume, device)
        if checkpoint.get("stage") != args.stage:
            raise ValueError("Resume checkpoint stage does not match --stage")
        loaded_config = config_from_checkpoint(checkpoint)
        if loaded_config.to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint model configuration differs from current configuration")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_mrae = float(checkpoint.get("best_mrae", math.inf))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    logger.info("Stage %d model configuration: %s", args.stage, config.to_dict())
    logger.info("Training pairs: %d | Validation pairs: %d", len(train_loader.dataset), len(val_loader.dataset))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_rec = 0.0
        running_prior = 0.0
        running_kd = 0.0
        seen = 0

        for iteration, batch in enumerate(train_loader, start=1):
            rgb = batch["rgb"].to(device, non_blocking=True)
            hsi = batch["hsi"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if args.stage == 1:
                    pred_hsi, _ = model(rgb, hsi)
                    loss_rec = reconstruction_loss(pred_hsi, hsi, args.loss, args.mrae_eps)
                    loss_prior = pred_hsi.new_zeros(())
                    loss_kd = pred_hsi.new_zeros(())
                    loss_total = loss_rec
                else:
                    assert teacher is not None
                    assert isinstance(model, DiffIRS2RGB2HSI)
                    with torch.no_grad():
                        target_prior = teacher.E(rgb, hsi)
                    pred_hsi, prior_sequence = model(rgb, target_prior=target_prior)
                    predicted_prior = prior_sequence[-1]
                    loss_rec = reconstruction_loss(pred_hsi, hsi, args.loss, args.mrae_eps)
                    loss_prior = F.l1_loss(predicted_prior, target_prior)
                    loss_kd = kd_loss(predicted_prior, target_prior, args.kd_temperature)
                    loss_total = (
                        loss_rec
                        + args.lambda_prior * loss_prior
                        + args.lambda_kd * loss_kd
                    )

            scaler.scale(loss_total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            batch_size = rgb.shape[0]
            seen += batch_size
            running_total += float(loss_total.item()) * batch_size
            running_rec += float(loss_rec.item()) * batch_size
            running_prior += float(loss_prior.item()) * batch_size
            running_kd += float(loss_kd.item()) * batch_size

            if iteration % args.print_freq == 0 or iteration == len(train_loader):
                logger.info(
                    "Iter %06d | Epoch %03d/%03d | LR %.3e | "
                    "Total %.6f | Recon %.6f | Prior %.6f | KD %.6f",
                    global_step,
                    epoch,
                    args.epochs,
                    optimizer.param_groups[0]["lr"],
                    running_total / max(seen, 1),
                    running_rec / max(seen, 1),
                    running_prior / max(seen, 1),
                    running_kd / max(seen, 1),
                )

        scheduler.step()
        metrics = validate(model, args.stage, val_loader, device, args)
        logger.info(
            "Epoch %03d validation | MRAE %.6f | RMSE %.6f | PSNR %.4f | SAM %.4f",
            epoch,
            metrics["mrae"],
            metrics["rmse"],
            metrics["psnr"],
            metrics["sam"],
        )

        latest_path = out_dir / f"latest_stage{args.stage}.pth"
        save_checkpoint(
            latest_path,
            args.stage,
            epoch,
            global_step,
            model,
            optimizer,
            scheduler,
            config,
            args,
            min(best_mrae, metrics["mrae"]),
        )

        if metrics["mrae"] < best_mrae:
            best_mrae = metrics["mrae"]
            best_path = out_dir / f"best_stage{args.stage}.pth"
            shutil.copy2(latest_path, best_path)
            logger.info("Saved new best checkpoint: %s", best_path)

        if epoch % args.save_every == 0:
            periodic_path = out_dir / f"model_stage{args.stage}_epoch_{epoch:03d}_iter_{global_step:06d}.pth"
            shutil.copy2(latest_path, periodic_path)


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint or args.resume
    if not checkpoint_path:
        raise ValueError("Evaluation requires --checkpoint")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    checkpoint = load_raw_checkpoint(checkpoint_path, device)
    checkpoint_stage = int(checkpoint.get("stage", -1))
    if checkpoint_stage != args.stage:
        raise ValueError(f"Checkpoint stage={checkpoint_stage} does not match --stage={args.stage}")

    config = config_from_checkpoint(checkpoint)
    args.num_bands = config.num_bands
    set_seed(args.seed)
    out_dir = Path(args.out_dir) / f"stage{args.stage}_eval"
    logger = configure_logging(out_dir)

    _, val_loader = make_dataloaders(args, device)
    model = build_model(args.stage, config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = validate(model, args.stage, val_loader, device, args)
    logger.info(
        "Evaluation | MRAE %.6f | RMSE %.6f | PSNR %.4f | SAM %.4f",
        metrics["mrae"],
        metrics["rmse"],
        metrics["psnr"],
        metrics["sam"],
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
