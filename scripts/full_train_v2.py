"""
LiquidFusionNet v2 — full retraining with:
  - 9 classes (4 seen + 5 rare)
  - ZSD head for rare classes
  - Diffusion-augmented training data (200 synthetic samples)
  - Kendall multi-task loss (det + trk + traj + zsd)
"""
import sys, torch
sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.reproducibility import seed_everything
from src.training.dataset import build_dataloader
from src.training.trainer import Trainer

cfg = load_config("configs/config.yaml")

# ── overrides for v2 ──────────────────────────────────────────
cfg.model.camera_pretrained       = True
cfg.dataset.imu_gps.imu_channels  = 9
cfg.dataset.imu_gps.gps_channels  = 0
cfg.dataset.num_classes           = 9
cfg.dataset.classes               = [
    "car", "motorcycle", "trimotorcycle", "scooter",   # seen
    "pedestrian", "unknown1", "bus", "truck", "van",   # rare (ZSD)
]
cfg.training.batch_size           = 2
cfg.training.epochs               = 80
cfg.training.freeze_encoders_epochs = 10
cfg.training.val_every_n_epochs   = 2
cfg.training.save_every_n_epochs  = 10
cfg.logging.use_wandb             = False
cfg.paths.checkpoints             = "results/checkpoints_v2"

import pathlib
pathlib.Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)

setup_logger("results/logs")
seed_everything(cfg.project.seed)

train_loader = build_dataloader(cfg, "train", rank=0, world_size=1)
val_loader   = build_dataloader(cfg, "val",   rank=0, world_size=1)

print(f"Train samples : {len(train_loader.dataset)}", flush=True)
print(f"Val samples   : {len(val_loader.dataset)}",   flush=True)
print(f"GPU           : {torch.cuda.get_device_name(0)}", flush=True)
print(f"Classes       : {cfg.dataset.num_classes} "
      f"(4 seen + 5 rare/ZSD)", flush=True)
print(f"Augmented data: 200 synthetic rare-class samples", flush=True)

trainer = Trainer(cfg, rank=0, world_size=1)
trainer.fit(train_loader, val_loader)
print("v2 TRAINING COMPLETE.", flush=True)
