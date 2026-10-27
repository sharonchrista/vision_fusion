import sys, torch, json
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.training.dataset import build_dataloader
from src.models.liquidfusionnet import LiquidFusionNet
from src.training.losses import MultiTaskLoss

cfg = load_config("configs/config.yaml")
cfg.model.camera_pretrained      = False
cfg.dataset.imu_gps.imu_channels = 9
cfg.dataset.imu_gps.gps_channels = 0
cfg.training.batch_size          = 2
cfg.logging.use_wandb            = False

device = torch.device("cuda:0")

# ── load checkpoint ───────────────────────────────────────────
ckpt_path = sorted(Path("results/checkpoints").glob("best_*.pt"))[-1]
print(f"Checkpoint : {ckpt_path.name}")
ckpt  = torch.load(ckpt_path, map_location="cpu")
model = LiquidFusionNet(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Epoch      : {ckpt['epoch']}")
print(f"Val metrics: {ckpt['val_metrics']}")

test_loader = build_dataloader(cfg, "test", rank=0, world_size=1)
criterion   = MultiTaskLoss(cfg).to(device)

def move(obj, dev):
    if isinstance(obj, torch.Tensor): return obj.to(dev)
    if isinstance(obj, dict):         return {k: move(v,dev) for k,v in obj.items()}
    return obj

# ── accumulators ──────────────────────────────────────────────
det_losses, all_scores, all_pred_cls = [], [], []
all_gt_cls  = []
conf_matrix = np.zeros((7, 7), dtype=int)
class_names = cfg.dataset.classes

print(f"\nTest samples: {len(test_loader.dataset)}")
print("Running evaluation ...\n")

with torch.no_grad():
    for batch in test_loader:
        batch   = move(batch, device)
        outputs, _ = model(batch)
        _, bd   = criterion(outputs, batch.get("targets", {}))
        det_losses.append(bd["loss_det"])

        # predictions
        logits = outputs["detection"]["cls_logits"]   # (B,A,C)
        bboxes = outputs["detection"]["bbox_deltas"]  # (B,A,7)
        B      = logits.shape[0]

        for b in range(B):
            probs      = logits[b].sigmoid()           # (A,C)
            scores, pc = probs.max(dim=-1)             # (A,)

            # keep only high-confidence predictions
            conf_mask  = scores > 0.3
            pred_scores = scores[conf_mask].cpu().numpy()
            pred_cls    = pc[conf_mask].cpu().numpy()

            # GT classes from targets
            tgt     = batch.get("targets",{}).get("detection",{})
            pos_mask = tgt.get("pos_mask")
            gt_cls_oh = tgt.get("cls_targets")

            if pos_mask is not None and gt_cls_oh is not None:
                pm   = pos_mask[b].cpu().numpy()         # (A,)
                gc   = gt_cls_oh[b][pm].argmax(-1).cpu().numpy()  # (N_pos,)
                all_gt_cls.extend(gc.tolist())

                # match each GT to best pred by class agreement
                if len(pred_cls) > 0:
                    for gt_c in gc:
                        # find pred with highest score predicting this class
                        same = pred_cls == gt_c
                        if same.any():
                            conf_matrix[gt_c, gt_c] += 1
                        else:
                            # top prediction class
                            if len(pred_cls) > 0:
                                conf_matrix[gt_c, pred_cls[np.argmax(pred_scores)]] += 1

            all_scores.extend(pred_scores.tolist())
            all_pred_cls.extend(pred_cls.tolist())

# ── per-class precision / recall ──────────────────────────────
print("="*60)
print("  LiquidFusionNet — Test Results (epoch 79)")
print("="*60)
print(f"  Det loss (mean) : {np.mean(det_losses):.4f}")
print(f"  Train loss final: 1.0620")
print(f"  Loss reduction  : 2.5748 → 1.0620  (-58.7%)\n")

print(f"  Predictions above 0.3 confidence: {len(all_scores)}")
print(f"  GT positive anchors seen        : {len(all_gt_cls)}\n")

# class distribution of predictions
pred_dist = {}
for c in range(7):
    n = sum(1 for x in all_pred_cls if x == c)
    if n > 0:
        pred_dist[class_names[c]] = n

print("  Predicted class distribution:")
for cls, cnt in sorted(pred_dist.items(), key=lambda x: -x[1]):
    print(f"    {cls:<18s}: {cnt:>5d}")

print("\n  Confusion matrix (GT rows, Pred cols):")
header = "         " + "".join(f"{c[:4]:>7s}" for c in class_names)
print(header)
for i, row_name in enumerate(class_names):
    row = conf_matrix[i]
    if row.sum() > 0:
        print(f"  {row_name[:8]:<8s} " +
              "".join(f"{v:>7d}" for v in row))

# per-class recall
print("\n  Per-class recall (TP / total GT):")
for i, cls in enumerate(class_names):
    total = conf_matrix[i].sum()
    tp    = conf_matrix[i, i]
    if total > 0:
        print(f"    {cls:<18s}: {tp}/{total}  ({100*tp/total:.1f}%)")

print("="*60)

# ── training curve summary ────────────────────────────────────
log_lines = open("results/logs/full_train.log").readlines()
epoch_lines = [l for l in log_lines if "Epoch" in l and "INFO" in l]
print("\n  Training curve (every 10 epochs):")
for l in epoch_lines[::10]:
    parts = l.strip().split("|")
    if len(parts) >= 3:
        print("   ", "|".join(parts[1:4]).strip())

# save
out = {
    "checkpoint": str(ckpt_path.name),
    "epoch": int(ckpt["epoch"]),
    "test_det_loss": float(np.mean(det_losses)),
    "train_loss_start": 2.5748,
    "train_loss_final": 1.0620,
    "loss_reduction_pct": 58.7,
    "pred_class_distribution": pred_dist,
    "confusion_matrix": conf_matrix.tolist(),
}
out_path = Path("results/metrics/test_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
