"""
scripts/train.py

Main training entry point. Supports single-GPU and multi-GPU (DDP).

Single GPU:
    python scripts/train.py

4× T4 GPUs (torchrun):
    torchrun --nproc_per_node=4 scripts/train.py

With config override:
    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/default.yaml \
        training.lr=1e-3 training.batch_size=8
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.utils.config  import load_config, save_config, get_run_name
from src.utils.seed    import set_seed
from src.data.dataset  import build_dataloaders
from src.training.trainer import Trainer
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml", type=Path)
    parser.add_argument("--resume",   default=None, type=Path,
                        help="Path to checkpoint to resume from")
    parser.add_argument("overrides",  nargs="*",
                        help="OmegaConf dot-notation overrides")
    args = parser.parse_args()

    # ── DDP setup ────────────────────────────────────────────
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # ── Config ───────────────────────────────────────────────
    cfg = load_config(args.config, args.overrides)
    run_name = get_run_name(cfg)
    OmegaConf.update(cfg, "run_name", run_name, merge=True)

    if rank == 0:
        save_config(cfg, Path(cfg.paths.logs) / run_name)
        console.rule(f"[bold blue]{cfg.project.name} — {run_name}[/bold blue]")
        console.print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.project.seed)

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Trainer ──────────────────────────────────────────────
    trainer = Trainer(cfg, rank=local_rank, world_size=world_size)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ────────────────────────────────────────────────
    trainer.train(train_loader, val_loader)

    # ── Final test evaluation ─────────────────────────────────
    if rank == 0:
        console.print("\n[bold]Running final test evaluation …[/bold]")
        test_metrics = trainer._val_epoch(test_loader)
        console.print(f"[green]Test metrics:[/green] {test_metrics}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
