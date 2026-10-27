"""
LiquidFusionNet v3 — 3-GPU DDP training.
Launch with:
  torchrun --nproc_per_node=3 --master_port=29602 scripts/full_train_v3_ddp.py
"""
import sys, torch, os, pathlib, numpy as np
sys.path.insert(0, "/home/sharon/vision_fusion")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger, log_scalars
from src.utils.reproducibility import seed_everything
from src.training.joint_dataset import build_joint_dataloader
from src.training.losses import MultiTaskLoss
from src.models.temporal_liquidfusionnet import TemporalLiquidFusionNet

# ── DDP init ──────────────────────────────────────────────────
rank       = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
try:
    dist.init_process_group(backend="nccl")
except Exception as e:
    print(f"DDP init failed: {e}")
    raise
torch.cuda.set_device(rank)
device    = torch.device(f"cuda:{rank}")
is_master = (rank == 0)

# ── config ────────────────────────────────────────────────────
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
cfg.training.epochs               = 80
cfg.training.precision            = "fp32"
cfg.training.freeze_encoders_epochs = 10
cfg.training.val_every_n_epochs   = 2
cfg.training.save_every_n_epochs  = 10
cfg.training.grad_clip            = 0.5
cfg.training.optimizer.lr         = 1e-4
cfg.logging.use_wandb             = False
cfg.paths.checkpoints             = "results/checkpoints_v3"
pathlib.Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)

if is_master:
    setup_logger("results/logs")
seed_everything(cfg.project.seed + rank)
log = get_logger()

# ── data ──────────────────────────────────────────────────────
train_loader = build_joint_dataloader(
    cfg, "train", rank=rank, world_size=world_size,
    use_weighted_sampler=False)
val_loader = build_joint_dataloader(
    cfg, "val", rank=rank, world_size=world_size,
    use_weighted_sampler=False)

if is_master:
    log.info(f"Train: {len(train_loader.dataset)} | "
             f"Val: {len(val_loader.dataset)} | "
             f"World: {world_size} | "
             f"Effective batch: {cfg.training.batch_size * world_size}")

# ── model ─────────────────────────────────────────────────────
model = TemporalLiquidFusionNet(cfg, use_pretrained_lidar=True).to(device)
model = DDP(model, device_ids=[rank], output_device=rank,
            find_unused_parameters=True)

criterion = MultiTaskLoss(cfg).to(device)

opt = torch.optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()),
    lr=cfg.training.optimizer.lr,
    weight_decay=cfg.training.optimizer.weight_decay,
    betas=tuple(cfg.training.optimizer.betas),
)

# cosine schedule
def lr_lambda(epoch):
    wu = cfg.training.scheduler.warmup_epochs
    if epoch < wu:
        return float(epoch+1) / max(wu, 1)
    progress = (epoch-wu) / max(cfg.training.epochs-wu, 1)
    cosine   = 0.5*(1.0+np.cos(np.pi*progress))
    min_lr   = cfg.training.scheduler.min_lr
    base_lr  = cfg.training.optimizer.lr
    return min_lr/base_lr + (1.0-min_lr/base_lr)*cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler    = GradScaler(enabled=False)   # FP32 — no scaler needed

# ── helpers ───────────────────────────────────────────────────
def move(obj, dev):
    if isinstance(obj, torch.Tensor): return obj.to(dev)
    if isinstance(obj, dict):
        return {k: move(v,dev) for k,v in obj.items()}
    return obj

def raw_model():
    return model.module

# ── freeze strategy ───────────────────────────────────────────
def apply_freeze(epoch):
    fe = cfg.training.freeze_encoders_epochs
    if epoch == 0:
        raw_model().freeze_encoders()
        if is_master:
            log.info(f"Encoders FROZEN for {fe} epochs.")
    elif epoch == fe:
        raw_model().unfreeze_encoders()
        if is_master:
            log.info(f"Encoders UNFROZEN at epoch {epoch}.")
        # lower encoder LR on unfreeze
        current_lr = opt.param_groups[0]["lr"]
        enc_ids = set()
        for m in [raw_model().cam_enc, raw_model().rad_enc,
                  raw_model().lid_enc, raw_model().imu_enc]:
            for p in m.parameters():
                enc_ids.add(id(p))
        for pg in opt.param_groups:
            if any(id(p) in enc_ids for p in pg["params"]):
                pg["lr"] = current_lr * 0.1
        if is_master:
            log.info(f"  Encoder LR -> {current_lr*0.1:.2e}")

# ── train epoch ───────────────────────────────────────────────
def train_epoch(loader):
    model.train(); criterion.train()
    running = {"loss_total":0.,"loss_det":0.,
               "loss_trk":0.,"loss_traj":0.,"loss_zsd":0.}
    n = 0
    for batch in loader:
        batch = move(batch, device)
        opt.zero_grad(set_to_none=True)
        outputs, _ = model(batch)
        loss, bd   = criterion(outputs, batch.get("targets",{}))

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            if is_master:
                log.warning("NaN loss — skipping batch")
            continue

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.grad_clip)
        if torch.isnan(norm) or norm > 1e4:
            if is_master:
                log.warning(f"NaN grad (norm={norm:.1f}) — skipping")
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()
        for k in running:
            running[k] += bd.get(k, 0.)
        n += 1
    return {k: v/max(n,1) for k,v in running.items()}

# ── val epoch ─────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(loader):
    model.eval()
    running = {"loss_total":0.,"loss_det":0.,
               "loss_trk":0.,"loss_traj":0.,"loss_zsd":0.}
    n = 0
    for batch in loader:
        batch = move(batch, device)
        outputs, _ = model(batch)
        _, bd = criterion(outputs, batch.get("targets",{}))
        for k in running:
            running[k] += bd.get(k, 0.)
        n += 1
    return {k: v/max(n,1) for k,v in running.items()}

# ── checkpoint ────────────────────────────────────────────────
best_val    = float("inf")
es_counter  = 0
ES_PATIENCE = 15

def save_ckpt(epoch, metrics, tag="best"):
    ckpt = {
        "epoch":       epoch,
        "model":       raw_model().state_dict(),
        "optimizer":   opt.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "val_metrics": metrics,
    }
    path = pathlib.Path(cfg.paths.checkpoints) / \
           f"{tag}_epoch{epoch:03d}.pt"
    torch.save(ckpt, path)
    log.info(f"Checkpoint -> {path}")

# ── resume from checkpoint ────────────────────────────────────
start_epoch = 0
best_val    = float("inf")
from pathlib import Path as _Path
ckpts = sorted(_Path(cfg.paths.checkpoints).glob("best_*.pt"))
if ckpts:
    ckpt_path = ckpts[-1]
    if is_master:
        log.info(f"Resuming from {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # strict=False: skip mismatched LiDAR encoder keys
    # (tokeniser dimensions changed with pretrained backbone)
    # Filter out mismatched LiDAR encoder keys before loading
    ckpt_sd  = ckpt["model"]
    model_sd = raw_model().state_dict()
    filtered = {k: v for k, v in ckpt_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped  = [k for k in ckpt_sd if k not in filtered]
    model_sd.update(filtered)
    raw_model().load_state_dict(model_sd)
    if is_master:
        log.info(f"  Loaded {len(filtered)}/{len(ckpt_sd)} keys "
                 f"({len(skipped)} skipped — LiDAR encoder reset "
                 f"to pretrained Point-MAE weights)")
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt["val_metrics"].get("loss_total",
                                          float("inf"))
    if is_master:
        log.info(f"Resumed at epoch {start_epoch}, "
                 f"best_val={best_val:.4f}")

# ── main loop ─────────────────────────────────────────────────
if is_master:
    log.info(f"Starting training: {cfg.training.epochs} epochs")

for epoch in range(start_epoch, cfg.training.epochs):
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    apply_freeze(epoch)

    tm = train_epoch(train_loader)
    scheduler.step()

    if is_master:
        log.info(
            f"Epoch {epoch+1:3d}/{cfg.training.epochs} | "
            f"loss={tm['loss_total']:.4f} | "
            f"det={tm['loss_det']:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

    if (epoch+1) % cfg.training.val_every_n_epochs == 0:
        vm = val_epoch(val_loader)
        if is_master:
            log.info(
                f"  VAL | loss={vm['loss_total']:.4f} | "
                f"det={vm['loss_det']:.4f}"
            )
            if vm["loss_total"] < best_val:
                best_val = vm["loss_total"]
                save_ckpt(epoch, vm, tag="best")
                es_counter = 0
            else:
                es_counter += 1
                log.info(f"  ES: {es_counter}/{ES_PATIENCE}")
            if es_counter >= ES_PATIENCE:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    if is_master and (epoch+1) % cfg.training.save_every_n_epochs == 0:
        save_ckpt(epoch, {}, tag=f"epoch{epoch+1:03d}")

if is_master:
    log.info("v3 3-GPU TRAINING COMPLETE.")

dist.destroy_process_group()
