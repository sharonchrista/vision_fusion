"""
Master training entry point for LiquidFusionNet.

Single GPU:
    python train.py --config configs/config.yaml

4× T4 DDP (torchrun):
    torchrun --nproc_per_node=4 train.py --config configs/config.yaml

Resume from checkpoint:
    torchrun --nproc_per_node=4 train.py \\
        --config configs/config.yaml \\
        --resume results/checkpoints/best_epoch010.pt
"""
import argparse
import os

import torch
import torch.distributed as dist

from src.utils.config import load_config
from src.utils.logger import setup_logger, close_logger, get_logger
from src.utils.reproducibility import seed_everything
from src.training.dataset import build_dataloader
from src.training.trainer import Trainer


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main(cfg_path: str, resume: str | None = None) -> None:
    cfg = load_config(cfg_path)

    # Detect DDP environment (set by torchrun)
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        setup_ddp(rank, world_size)

    is_master = (rank == 0)

    if is_master:
        setup_logger(
            log_dir=cfg.paths.logs,
            use_wandb=cfg.logging.use_wandb,
            wandb_project=cfg.logging.project,
            run_name=f"{cfg.project.name}_v{cfg.project.version}",
        )

    log = get_logger()
    seed_everything(cfg.project.seed)

    if is_master:
        log.info(f"LiquidFusionNet — T-ITS training")
        log.info(f"  GPUs: {world_size}  |  batch/GPU: {cfg.training.batch_size}  "
                 f"|  effective batch: {cfg.training.batch_size * world_size}")
        log.info(f"  Precision: {cfg.training.precision}  "
                 f"|  Epochs: {cfg.training.epochs}")

    # ── Data ──────────────────────────────────────────────────
    train_loader = build_dataloader(cfg, "train", rank, world_size)
    val_loader   = build_dataloader(cfg, "val",   rank, world_size)

    if is_master:
        log.info(f"  Train batches: {len(train_loader)} | "
                 f"Val batches: {len(val_loader)}")

    # ── Trainer ───────────────────────────────────────────────
    trainer = Trainer(cfg, rank=rank, world_size=world_size)

    if resume:
        trainer.load_checkpoint(resume)

    # ── Fit ───────────────────────────────────────────────────
    try:
        trainer.fit(train_loader, val_loader)
    finally:
        cleanup_ddp()
        if is_master:
            close_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiquidFusionNet training")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.resume)
