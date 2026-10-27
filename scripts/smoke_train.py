import sys, torch
sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.reproducibility import seed_everything
from src.training.dataset import build_dataloader
from src.training.trainer import Trainer

cfg = load_config("configs/config.yaml")
cfg.model.camera_pretrained      = True
cfg.dataset.imu_gps.imu_channels = 9
cfg.dataset.imu_gps.gps_channels = 0
cfg.training.batch_size          = 2
cfg.training.epochs              = 5
cfg.training.freeze_encoders_epochs = 2
cfg.training.val_every_n_epochs  = 1
cfg.training.save_every_n_epochs = 5
cfg.logging.use_wandb            = False

setup_logger(cfg.paths.logs)
seed_everything(cfg.project.seed)

train_loader = build_dataloader(cfg, "train", rank=0, world_size=1)
val_loader   = build_dataloader(cfg, "val",   rank=0, world_size=1)

print(f"Train: {len(train_loader.dataset)} samples | "
      f"Val: {len(val_loader.dataset)} samples", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

trainer = Trainer(cfg, rank=0, world_size=1)
trainer.fit(train_loader, val_loader)
print("DONE — 5-epoch smoke run complete.", flush=True)
