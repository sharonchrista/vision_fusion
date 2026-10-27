"""
LiquidFusionNet v3 — Final evaluation.
Compares v1, v2, v3 side by side for T-ITS Table III.
"""
import sys, torch, json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/sharon/vision_fusion")

from src.utils.config import load_config
from src.training.joint_dataset import build_joint_dataloader
from src.models.temporal_liquidfusionnet import TemporalLiquidFusionNet
from src.training.losses import MultiTaskLoss

cfg = load_config("configs/config.yaml")
cfg.model.camera_pretrained      = False
cfg.dataset.imu_gps.imu_channels = 9
cfg.dataset.imu_gps.gps_channels = 0
cfg.dataset.num_classes          = 9
cfg.dataset.classes              = [
    "car","motorcycle","trimotorcycle","scooter",
    "pedestrian","unknown1","bus","truck","van",
]
cfg.training.batch_size          = 2
cfg.paths.checkpoints            = "results/checkpoints_v3"

CLASS_NAMES  = cfg.dataset.classes
NUM_CLASSES  = 9
SEEN_CLASSES = [0,1,2,3]
RARE_CLASSES = [4,5,6,7,8]
device       = torch.device("cuda:0")

# ── load best checkpoint ──────────────────────────────────────
ckpt_dir   = Path(cfg.paths.checkpoints)
ckpts      = sorted(ckpt_dir.glob("best_*.pt")) or \
             sorted(ckpt_dir.glob("epoch*.pt"))
ckpt_path  = ckpts[-1]
print(f"{'='*62}")
print(f"  LiquidFusionNet v3 — Final Evaluation")
print(f"{'='*62}")
print(f"  Checkpoint : {ckpt_path.name}")

ckpt  = torch.load(ckpt_path, map_location="cpu")
model = TemporalLiquidFusionNet(cfg,
                                use_pretrained_lidar=True).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Epoch      : {ckpt['epoch']}")
vm = ckpt.get("val_metrics", {})
print(f"  Val loss   : {vm.get('loss_total', 'N/A')}")

# ── test data (TIAND only — target domain) ───────────────────
test_loader = build_joint_dataloader(
    cfg, "test", use_weighted_sampler=False)
crit = MultiTaskLoss(cfg).to(device)
print(f"  Test samples: {len(test_loader.dataset)}")

def move(obj, dev):
    if isinstance(obj, torch.Tensor): return obj.to(dev)
    if isinstance(obj, dict):
        return {k: move(v,dev) for k,v in obj.items()}
    return obj

# ── evaluation ────────────────────────────────────────────────
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
det_losses, zsd_losses = [], []

print("\nRunning evaluation ...")
with torch.no_grad():
    for batch in test_loader:
        batch = move(batch, device)
        outputs, _ = model(batch)
        _, bd = crit(outputs, batch.get("targets",{}))
        det_losses.append(bd.get("loss_det",  0.0))
        zsd_losses.append(bd.get("loss_zsd",  0.0))

        # ZSD predictions — full 9-class
        zsd_logits = outputs["zero_shot"]["zsd_logits"]
        pred_cls   = zsd_logits.argmax(-1).cpu().numpy()

        # det head predictions — seen classes
        det_logits = outputs["detection"]["cls_logits"]
        det_scores, det_pred = det_logits.sigmoid().max(-1)

        tgt  = batch.get("targets",{}).get("detection",{})
        pm   = tgt.get("pos_mask")
        gt_c = tgt.get("cls_targets")

        B = pred_cls.shape[0]
        for b in range(B):
            zp = int(pred_cls[b])
            if pm is not None and gt_c is not None:
                mask = pm[b].cpu().numpy()
                if mask.any():
                    gt_labels = gt_c[b][mask].argmax(-1).cpu().numpy()
                    for gl in gt_labels:
                        conf_matrix[int(gl), zp] += 1

# ── results ───────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  DETECTION RESULTS — LiquidFusionNet v3")
print(f"{'='*62}")
print(f"  Det loss (mean) : {np.mean(det_losses):.4f}")
print(f"  ZSD loss (mean) : {np.mean(zsd_losses):.4f}")

results = {}
print(f"\n  {'Class':<18s} {'GT':>6} {'TP':>6} "
      f"{'Recall':>8}  {'Type':<10s}")
print(f"  {'-'*55}")

for i, cls in enumerate(CLASS_NAMES):
    total  = int(conf_matrix[i].sum())
    tp     = int(conf_matrix[i,i])
    recall = 100.0*tp/total if total > 0 else None
    ctype  = "SEEN" if i in SEEN_CLASSES else "RARE/ZSD"
    rstr   = f"{recall:.1f}%" if recall is not None else "  n/a"
    print(f"  {cls:<18s} {total:>6} {tp:>6} {rstr:>8}  {ctype}")
    results[cls] = {"gt":total,"tp":tp,"recall":recall,"type":ctype}

# summary
seen_r = [results[CLASS_NAMES[i]]["recall"]
          for i in SEEN_CLASSES
          if results[CLASS_NAMES[i]]["recall"] is not None]
rare_r = [results[CLASS_NAMES[i]]["recall"]
          for i in RARE_CLASSES
          if results[CLASS_NAMES[i]]["recall"] is not None]

print(f"\n  {'─'*55}")
if seen_r: print(f"  Seen mean recall : {np.mean(seen_r):.1f}%")
if rare_r: print(f"  Rare mean recall : {np.mean(rare_r):.1f}%")

# ── T-ITS comparison table ────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TABLE III — v1 vs v2 vs v3 (T-ITS paper)")
print(f"{'='*70}")
print(f"  {'Class':<18s} {'v1':>8} {'v2':>8} {'v3':>8}  Notes")
print(f"  {'-'*65}")

v1 = {"car":100.0,"two_wheeler":51.9,"pedestrian":0.0,"truck":0.0}
v2 = {"car":81.4,"unknown1":10.0}

for cls in CLASS_NAMES:
    v1r = v1.get(cls, None)
    v2r = v2.get(cls, None)
    v3r = results[cls]["recall"]
    v1s = f"{v1r:.0f}%" if v1r is not None else "—"
    v2s = f"{v2r:.0f}%" if v2r is not None else "—"
    v3s = f"{v3r:.1f}%" if v3r is not None else "n/a"
    # improvement flag
    flag = ""
    if v3r is not None and v3r > 0 and cls in ["pedestrian","bus","truck","van"]:
        flag = " ← IMPROVED"
    print(f"  {cls:<18s} {v1s:>8} {v2s:>8} {v3s:>8} {flag}")

print(f"{'='*70}")

# save
out = {
    "checkpoint": str(ckpt_path.name),
    "epoch": int(ckpt["epoch"]),
    "det_loss_mean": float(np.mean(det_losses)),
    "zsd_loss_mean": float(np.mean(zsd_losses)),
    "per_class": results,
    "confusion_matrix": conf_matrix.tolist(),
    "seen_mean_recall": float(np.mean(seen_r)) if seen_r else None,
    "rare_mean_recall": float(np.mean(rare_r)) if rare_r else None,
}
Path("results/metrics").mkdir(exist_ok=True)
with open("results/metrics/test_results_v3.json","w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> results/metrics/test_results_v3.json")

# generate final figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.patch.set_facecolor("white")

    # Panel 1: v1 vs v2 vs v3 recall
    ax = axes[0]
    cls_with_gt = [c for c in CLASS_NAMES
                   if results[c]["gt"] > 0]
    x  = np.arange(len(cls_with_gt))
    w  = 0.25

    v1_vals = []
    v2_vals = []
    v3_vals = []
    for c in cls_with_gt:
        v1_vals.append(v1.get(c, 0) or 0)
        v2_vals.append(v2.get(c, 0) or 0)
        v3_vals.append(results[c]["recall"] or 0)

    ax.bar(x-w,   v1_vals, w, label="v1 baseline",
           color="#BBDEFB", edgecolor="white")
    ax.bar(x,     v2_vals, w, label="v2 (+ZSD+aug)",
           color="#64B5F6", edgecolor="white")
    ax.bar(x+w,   v3_vals, w, label="v3 (+nuScenes+temporal)",
           color="#1565C0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cls_with_gt, rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Per-class Detection Recall\nv1 → v2 → v3",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # shade rare classes
    rare_idx = [i for i,c in enumerate(cls_with_gt)
                if CLASS_NAMES.index(c) in RARE_CLASSES]
    if rare_idx:
        ax.axvspan(min(rare_idx)-0.5, max(rare_idx)+0.5,
                   alpha=0.07, color="orange")
        ax.text(np.mean(rare_idx), 108,
                "Rare classes (ZSD)",
                ha="center", fontsize=8, color="darkorange")

    # Panel 2: training curve comparison
    ax2 = axes[1]
    import re
    def parse_epochs(log_path):
        ep, ls = [], []
        if not Path(log_path).exists():
            return ep, ls
        for line in open(log_path):
            m = re.search(
                r"Epoch\s+(\d+)/\d+.*det=([\d.]+)", line)
            if m and "VAL" not in line and "INFO" in line:
                ep.append(int(m.group(1)))
                ls.append(float(m.group(2)))
        return ep, ls

    e1,l1 = parse_epochs("results/logs/full_train.log")
    e2,l2 = parse_epochs("results/logs/full_train_v2.log")
    e3,l3 = parse_epochs("results/logs/full_train_v3.log")

    if e1: ax2.plot(e1, l1, "b-",  lw=1.5,
                    label="v1 (TIAND only)")
    if e2: ax2.plot(e2, l2, "r-",  lw=1.5,
                    label="v2 (+ZSD+aug)")
    if e3: ax2.plot(e3, l3, "g-",  lw=2,
                    label="v3 (+nuScenes)")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Detection Loss (Focal)")
    ax2.set_title("Training Convergence\nAll Versions",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "LiquidFusionNet — Progressive Improvement "
        "v1 → v2 → v3",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_fig = Path("results/figures/final_comparison.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"  Figure -> {out_fig}")
except Exception as e:
    print(f"  Figure skipped: {e}")
