"""
Trainer for LiquidFusionNet.

Features
--------
- DistributedDataParallel (DDP) across 4× T4 GPUs via NCCL
- FP16 mixed precision via torch.cuda.amp
- Cosine LR schedule with linear warmup
- Encoder freeze / unfreeze strategy
- Top-1 checkpoint saving by val loss
- TensorBoard + optional W&B logging
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.liquidfusionnet import LiquidFusionNet
from src.training.losses import MultiTaskLoss
from src.utils.logger import get_logger, log_scalars
from src.utils.reproducibility import seed_everything

log = get_logger()


class Trainer:
    """
    Usage (single GPU):
        trainer = Trainer(cfg)
        trainer.fit(train_loader, val_loader)

    Usage (DDP — launched via torchrun):
        trainer = Trainer(cfg, rank=int(os.environ["LOCAL_RANK"]), world_size=4)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(self, cfg, rank: int = 0, world_size: int = 1) -> None:
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.is_master  = (rank == 0)
        self.device     = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

        seed_everything(cfg.project.seed + rank)

        # ── Model ────────────────────────────────────────────
        self.model = LiquidFusionNet(cfg).to(self.device)
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,   # NCP sparse wiring
            )

        # ── Loss ─────────────────────────────────────────────
        self.criterion = MultiTaskLoss(cfg).to(self.device)

        # ── Optimiser (model + Kendall σ params) ─────────────
        opt_cfg = cfg.training.optimizer
        all_params = (list(self.model.parameters()) +
                      list(self.criterion.parameters()))
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

        # ── LR schedule ──────────────────────────────────────
        self.scheduler = self._build_cosine_schedule(
            cfg.training.epochs,
            cfg.training.scheduler.warmup_epochs,
            cfg.training.scheduler.min_lr,
            opt_cfg.lr,
        )

        # ── AMP scaler ───────────────────────────────────────
        use_amp = (cfg.training.precision == "fp16")
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # ── State ────────────────────────────────────────────
        self.epoch        = 0
        self.global_step  = 0
        self.best_val     = float("inf")
        self.best_epoch   = -1
        self.val_history  = []
        self._patience    = getattr(getattr(cfg, "training", cfg), "patience", 15)
        self._no_improve  = 0
        self.ckpt_dir     = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── LR schedule ──────────────────────────────────────────────────────────

    def _build_cosine_schedule(self, total_epochs: int, warmup: int,
                               min_lr: float, base_lr: float):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── Encoder freeze ────────────────────────────────────────────────────────

    def _raw_model(self) -> LiquidFusionNet:
        return self.model.module if self.world_size > 1 else self.model

    def _apply_freeze_strategy(self, epoch: int) -> None:
        freeze_epochs = self.cfg.training.freeze_encoders_epochs
        if epoch == 0:
            self._raw_model().freeze_encoders()
            if self.is_master:
                log.info(f"Encoders FROZEN for first {freeze_epochs} epochs.")
        elif epoch == freeze_epochs:
            self._raw_model().unfreeze_encoders()
            if self.is_master:
                log.info(f"Encoders UNFROZEN at epoch {epoch}.")

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(self.device, non_blocking=True)
                          if isinstance(vv, torch.Tensor) else vv
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    # ── Train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, loader) -> dict:
        self.model.train()
        self.criterion.train()
        running = {"loss_total": 0.0, "loss_det": 0.0,
                   "loss_trk": 0.0, "loss_traj": 0.0}
        n = 0

        for batch_and_targets in loader:
            # dataset returns (batch, targets); targets may be {} for unlabelled data
            if isinstance(batch_and_targets, (list, tuple)) and len(batch_and_targets) == 2:
                batch, targets = batch_and_targets
            else:
                batch, targets = batch_and_targets, {}
            batch   = self._to_device(batch)
            targets = self._to_device(targets)

            # ── Missing-view dropout (reviewer Fix 4) ────────────────────────
            # Randomly zero each modality with p=0.15 during training,
            # forcing the model to be robust to missing sensors.
            if self.model.training:
                import random
                for mod in ("camera", "radar", "lidar", "imu_gps"):
                    if mod in batch and random.random() < 0.15:
                        batch[mod] = torch.zeros_like(batch[mod])
            # ─────────────────────────────────────────────────────────────────

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                loss, breakdown = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in running:
                running[k] += breakdown.get(k, 0.0)
            n += 1
            self.global_step += 1

            # Step-level logging (master only)
            if (self.is_master and
                    self.global_step % self.cfg.logging.log_every_n_steps == 0):
                log_scalars({f"train_step/{k}": v for k, v in breakdown.items()},
                            self.global_step)

        return {k: v / max(n, 1) for k, v in running.items()}

    # ── Val epoch ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader) -> dict:
        self.model.eval()
        running = {"loss_total": 0.0, "loss_det": 0.0,
                   "loss_trk": 0.0, "loss_traj": 0.0}
        n = 0

        for batch_and_targets in loader:
            if isinstance(batch_and_targets, (list, tuple)) and len(batch_and_targets) == 2:
                batch, targets = batch_and_targets
            else:
                batch, targets = batch_and_targets, {}
            batch   = self._to_device(batch)
            targets = self._to_device(targets)
            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                _, breakdown = self.criterion(outputs, targets)
            for k in running:
                running[k] += breakdown.get(k, 0.0)
            n += 1

        return {k: v / max(n, 1) for k, v in running.items()}

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_metrics: dict,
                         tag: str = "best") -> None:
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "model":       self._raw_model().state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self.scaler.state_dict(),
            "criterion":   self.criterion.state_dict(),
            "val_metrics": val_metrics,
            "cfg":         vars(self.cfg),
        }
        path = self.ckpt_dir / f"{tag}_epoch{epoch:03d}.pt"
        torch.save(ckpt, path)
        log.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._raw_model().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.criterion.load_state_dict(ckpt["criterion"])
        self.epoch       = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        log.info(f"Resumed from {path} (epoch {self.epoch})")

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader) -> None:
        cfg = self.cfg.training

        if self.is_master:
            param_counts = self._raw_model().count_parameters()
            log.info("Model parameter counts:")
            for k, v in param_counts.items():
                log.info(f"  {k:<22s}: {v:>10,}")

        for epoch in range(self.epoch, cfg.epochs):
            self.epoch = epoch

            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self._apply_freeze_strategy(epoch)

            # ── Train ──
            train_metrics = self._train_epoch(train_loader)
            self.scheduler.step()

            if self.is_master:
                log_scalars({f"train/{k}": v for k, v in train_metrics.items()},
                            epoch)
                log.info(
                    f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                    f"loss={train_metrics['loss_total']:.4f} | "
                    f"det={train_metrics['loss_det']:.4f} | "
                    f"trk={train_metrics['loss_trk']:.4f} | "
                    f"traj={train_metrics['loss_traj']:.4f} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Validate ──
            if (epoch + 1) % cfg.val_every_n_epochs == 0:
                val_metrics = self._val_epoch(val_loader)

                if self.is_master:
                    log_scalars({f"val/{k}": v for k, v in val_metrics.items()},
                                epoch)
                    log.info(
                        f"  VAL | loss={val_metrics['loss_total']:.4f} | "
                        f"det={val_metrics['loss_det']:.4f} | "
                        f"trk={val_metrics['loss_trk']:.4f} | "
                        f"traj={val_metrics['loss_traj']:.4f}"
                    )

                    self.val_history.append([epoch, float(val_metrics["loss_total"])])
                    if val_metrics["loss_total"] < self.best_val:
                        self.best_val    = val_metrics["loss_total"]
                        self.best_epoch  = epoch
                        self._no_improve = 0
                        self._save_checkpoint(epoch, val_metrics, tag="best")
                    else:
                        self._no_improve += 1
                        log.info(f"  No improvement: {self._no_improve}/{self._patience} patience checks")

            # ── Early stopping ──
            stop_flag = torch.tensor(
                1 if (self.is_master and self._no_improve >= self._patience) else 0,
                device=self.device
            )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(stop_flag, src=0)
            if stop_flag.item():
                if self.is_master:
                    log.info(f"Early stopping at epoch {epoch+1}: best val={self.best_val:.4f} at epoch {self.best_epoch+1}.")
                break

            # ── Periodic checkpoint ──
            if self.is_master and (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, {}, tag=f"epoch{epoch+1:03d}")
