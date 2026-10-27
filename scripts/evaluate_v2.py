"""
LiquidFusionNet v2 evaluation — full 9-class results.
Covers: detection recall, ZSD rare-class performance,
        confusion matrix, and T-ITS paper table generation.
"""
import sys, torch, json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.training.dataset import build_dataloader
from src.models.liquidfusionnet import LiquidFusionNet
from src.training.losses import MultiTaskLoss

# ── config ────────────────────────────────────────────────────
cfg = load_config("configs/config.yaml")
cfg.model.camera_pretrained      = False
cfg.dataset.imu_gps.imu_channels = 9
cfg.dataset.imu_gps.gps_channels = 0
cfg.dataset.num_classes          = 9
cfg.dataset.classes              = [
    "car", "motorcycle", "trimotorcycle", "scooter",   # seen
    "pedestrian", "unknown1", "bus", "truck", "van",   # rare
]
cfg.training.batch_size          = 2
cfg.paths.checkpoints            = "results/checkpoints_v2"
cfg.logging.use_wandb            = False

SEEN_CLASSES = [0, 1, 2, 3]
RARE_CLASSES = [4, 5, 6, 7, 8]
CLASS_NAMES  = cfg.dataset.classes
NUM_CLASSES  = 9

device = torch.device("cuda:0")


def move(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: move(v, dev) for k, v in obj.items()}
    return obj


# ── load best checkpoint ──────────────────────────────────────
ckpt_dir  = Path(cfg.paths.checkpoints)
ckpt_path = sorted(ckpt_dir.glob("best_*.pt"))[-1]
print(f"{'='*60}")
print(f"  LiquidFusionNet v2 — Evaluation")
print(f"{'='*60}")
print(f"  Checkpoint : {ckpt_path.name}")

ckpt  = torch.load(ckpt_path, map_location="cpu")
model = LiquidFusionNet(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Epoch      : {ckpt['epoch']}")
print(f"  Val loss   : {ckpt['val_metrics'].get('loss_total', 'N/A'):.4f}")
print(f"  Val det    : {ckpt['val_metrics'].get('loss_det',   'N/A'):.4f}")

# ── data ──────────────────────────────────────────────────────
test_loader = build_dataloader(cfg, "test", rank=0, world_size=1)
crit        = MultiTaskLoss(cfg).to(device)
print(f"  Test samples: {len(test_loader.dataset)}\n")

# ── accumulators ──────────────────────────────────────────────
conf_matrix   = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
gt_counts     = defaultdict(int)
pred_counts   = defaultdict(int)
det_losses    = []
zsd_losses    = []
all_det_scores = []   # for mAP proxy

print("Running evaluation ...")
with torch.no_grad():
    for batch in test_loader:
        batch   = move(batch, device)
        outputs, _ = model(batch)
        _, bd   = crit(outputs, batch.get("targets", {}))
        det_losses.append(bd.get("loss_det",  0.0))
        zsd_losses.append(bd.get("loss_zsd",  0.0))

        B = outputs["detection"]["cls_logits"].shape[0]

        # ── detection head predictions (seen classes) ─────────
        det_logits = outputs["detection"]["cls_logits"]   # (B,A,C)
        det_scores, det_cls = det_logits.sigmoid().max(-1) # (B,A)

        # ── ZSD head predictions (all 9 classes) ─────────────
        zsd_logits = outputs["zero_shot"]["zsd_logits"]   # (B,9)
        zsd_cls    = zsd_logits.argmax(-1).cpu().numpy()  # (B,)
        zsd_scores = zsd_logits.softmax(-1).max(-1).values.cpu().numpy()

        # ── ground truth ──────────────────────────────────────
        tgt     = batch.get("targets", {}).get("detection", {})
        pm      = tgt.get("pos_mask")     # (B,A)
        gt_oh   = tgt.get("cls_targets")  # (B,A,C)

        for b in range(B):
            # ZSD prediction for this sample
            pred_c = int(zsd_cls[b])
            pred_counts[CLASS_NAMES[pred_c]] += 1

            if pm is not None and gt_oh is not None:
                mask = pm[b].cpu().numpy()       # (A,)
                if mask.any():
                    gt_labels = gt_oh[b][mask].argmax(-1).cpu().numpy()
                    for gl in gt_labels:
                        gt_counts[CLASS_NAMES[int(gl)]] += 1
                        conf_matrix[int(gl), pred_c] += 1

# ── per-class metrics ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  DETECTION RESULTS — LiquidFusionNet v2")
print(f"{'='*60}")
print(f"  Det loss (mean) : {np.mean(det_losses):.4f}")
print(f"  ZSD loss (mean) : {np.mean(zsd_losses):.4f}")

print(f"\n  {'Class':<18s} {'GT':>6} {'TP':>6} {'Recall':>8}  Type")
print(f"  {'-'*50}")

results = {}
for i, cls in enumerate(CLASS_NAMES):
    total = int(conf_matrix[i].sum())
    tp    = int(conf_matrix[i, i])
    recall = 100.0 * tp / total if total > 0 else None
    ctype  = "SEEN" if i in SEEN_CLASSES else "RARE/ZSD"
    recall_str = f"{recall:.1f}%" if recall is not None else "  n/a "
    print(f"  {cls:<18s} {total:>6} {tp:>6} {recall_str:>8}  {ctype}")
    results[cls] = {
        "gt_count": total,
        "tp":       tp,
        "recall":   recall,
        "type":     ctype,
    }

# ── seen vs rare summary ──────────────────────────────────────
seen_recalls = [results[CLASS_NAMES[i]]["recall"]
                for i in SEEN_CLASSES
                if results[CLASS_NAMES[i]]["recall"] is not None]
rare_recalls = [results[CLASS_NAMES[i]]["recall"]
                for i in RARE_CLASSES
                if results[CLASS_NAMES[i]]["recall"] is not None]

print(f"\n  {'─'*50}")
if seen_recalls:
    print(f"  Seen class mean recall  : {np.mean(seen_recalls):.1f}%")
if rare_recalls:
    print(f"  Rare class mean recall  : {np.mean(rare_recalls):.1f}%")
    print(f"  (ZSD + diffusion augmentation)")

# ── confusion matrix ──────────────────────────────────────────
print(f"\n  Confusion matrix (GT rows × Pred cols):")
header = f"  {'':16s}" + "".join(f"{c[:5]:>7s}" for c in CLASS_NAMES)
print(header)
for i, row_name in enumerate(CLASS_NAMES):
    if conf_matrix[i].sum() > 0:
        row = "".join(f"{v:>7d}" for v in conf_matrix[i])
        print(f"  {row_name[:16]:<16s}{row}")

# ── T-ITS paper table ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TABLE III — v1 vs v2 Comparison (T-ITS)")
print(f"{'='*60}")
print(f"  {'Class':<18s} {'v1 Recall':>12} {'v2 Recall':>12}  {'Δ':>8}")
print(f"  {'-'*55}")
v1_recalls = {
    "car": 100.0, "motorcycle": None, "trimotorcycle": None,
    "scooter": None, "pedestrian": 0.0, "unknown1": None,
    "bus": None, "truck": 0.0, "van": None,
}
for cls in CLASS_NAMES:
    v1 = v1_recalls.get(cls)
    v2 = results[cls]["recall"]
    v1_str = f"{v1:.1f}%" if v1 is not None else "  —"
    v2_str = f"{v2:.1f}%" if v2 is not None else "  —"
    if v1 is not None and v2 is not None:
        delta = v2 - v1
        d_str = f"{delta:+.1f}%"
    else:
        d_str = "  —"
    print(f"  {cls:<18s} {v1_str:>12} {v2_str:>12}  {d_str:>8}")
print(f"{'='*60}")

# ── save results ──────────────────────────────────────────────
out = {
    "checkpoint":      str(ckpt_path.name),
    "epoch":           int(ckpt["epoch"]),
    "det_loss_mean":   float(np.mean(det_losses)),
    "zsd_loss_mean":   float(np.mean(zsd_losses)),
    "per_class":       results,
    "confusion_matrix": conf_matrix.tolist(),
    "seen_mean_recall": float(np.mean(seen_recalls)) if seen_recalls else None,
    "rare_mean_recall": float(np.mean(rare_recalls)) if rare_recalls else None,
}
out_path = Path("results/metrics/test_results_v2.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")

# ── update figures ────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    classes_with_gt = [c for c in CLASS_NAMES
                       if results[c]["gt_count"] > 0]
    v1_vals = [v1_recalls.get(c, 0) or 0 for c in classes_with_gt]
    v2_vals = [results[c]["recall"] or 0  for c in classes_with_gt]

    x  = np.arange(len(classes_with_gt))
    w  = 0.35
    b1 = ax.bar(x - w/2, v1_vals, w, label="v1 baseline",
                color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x + w/2, v2_vals, w, label="v2 (+diffusion+ZSD)",
                color="#1565C0", edgecolor="white")

    for bar, val in zip(list(b1)+list(b2), v1_vals+v2_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{val:.0f}%", ha="center",
                    va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes_with_gt, rotation=15,
                       ha="right", fontsize=9)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("LiquidFusionNet v1 vs v2 — Per-class Detection Recall\n"
                 "(TIAND Dataset, Hyderabad, India)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # shade rare class region
    rare_with_gt = [i for i, c in enumerate(classes_with_gt)
                    if CLASS_NAMES.index(c) in RARE_CLASSES]
    if rare_with_gt:
        ax.axvspan(min(rare_with_gt)-0.5, max(rare_with_gt)+0.5,
                   alpha=0.08, color="orange",
                   label="Rare classes (ZSD)")
        ax.text(np.mean(rare_with_gt), 108,
                "← Rare classes (ZSD + Diffusion aug) →",
                ha="center", fontsize=8, color="darkorange")

    plt.tight_layout()
    out_fig = Path("results/figures/v1_vs_v2_recall.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"  Figure saved -> {out_fig}")
except Exception as e:
    print(f"  Figure skipped: {e}")
