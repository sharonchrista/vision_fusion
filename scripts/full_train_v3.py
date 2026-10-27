"""
LiquidFusionNet v3 — Final model with all 4 fixes:
  Fix 1: nuScenes + TIAND joint training (22,079 combined objects)
  Fix 2: Scene-aware diffusion augmentation (2000 samples)
  Fix 3: Point-MAE style LiDAR encoder (improved architecture)
  Fix 4: Full temporal window training
"""
import sys, torch, pathlib
sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.reproducibility import seed_everything
from src.training.joint_dataset import build_joint_dataloader
from src.training.losses import MultiTaskLoss
from src.models.temporal_liquidfusionnet import TemporalLiquidFusionNet
from src.training.trainer import Trainer

cfg = load_config("configs/config.yaml")
cfg.model.camera_pretrained       = True
cfg.dataset.imu_gps.imu_channels  = 9
cfg.dataset.imu_gps.gps_channels  = 0
cfg.dataset.num_classes           = 9
cfg.dataset.classes               = [
    "car","motorcycle","trimotorcycle","scooter",
    "pedestrian","unknown1","bus","truck","van",
]
cfg.training.batch_size           = 2
cfg.training.epochs               = 50
cfg.training.freeze_encoders_epochs = 8
cfg.training.val_every_n_epochs   = 2
cfg.training.save_every_n_epochs  = 10
cfg.logging.use_wandb             = False
cfg.training.precision            = "fp32"
cfg.training.optimizer.lr         = 5e-5
cfg.training.grad_clip            = 0.3
cfg.paths.checkpoints             = "results/checkpoints_v3"
pathlib.Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)

setup_logger("results/logs")
seed_everything(cfg.project.seed)

# ── joint dataloaders ─────────────────────────────────────────
train_loader = build_joint_dataloader(
    cfg, "train", use_weighted_sampler=True)
val_loader   = build_joint_dataloader(
    cfg, "val",   use_weighted_sampler=False)

print(f"\nTrain samples : {len(train_loader.dataset)}", flush=True)
print(f"Val samples   : {len(val_loader.dataset)}",    flush=True)
print(f"GPU           : {torch.cuda.get_device_name(0)}", flush=True)

# ── model ─────────────────────────────────────────────────────
model = TemporalLiquidFusionNet(cfg, use_pretrained_lidar=True)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters    : {total:,}", flush=True)

# ── trainer ───────────────────────────────────────────────────
class V3Trainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, rank=0, world_size=1)
        self.model = model.to(self.device)
        opt_cfg    = cfg.training.optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) +
            list(self.criterion.parameters()),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

    def _raw_model(self):
        return self.model

    def _apply_freeze_strategy(self, epoch):
        fe = self.cfg.training.freeze_encoders_epochs
        if epoch == 0:
            self._raw_model().freeze_encoders()
            from src.utils.logger import get_logger
            get_logger().info(f"Encoders FROZEN for {fe} epochs.")
        elif epoch == fe:
            self._raw_model().unfreeze_encoders()
            from src.utils.logger import get_logger
            get_logger().info(f"Encoders UNFROZEN at epoch {epoch}.")

trainer = V3Trainer(cfg, model)
trainer.fit(train_loader, val_loader)
print("v3 TRAINING COMPLETE.", flush=True)
