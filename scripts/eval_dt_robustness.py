"""
eval_dt_robustness.py — Reviewer Fix 3: Δt perturbation robustness evaluation.

Loads the best full_model checkpoint, corrupts delta_t values with increasing
noise levels, and measures detection degradation (mAP@0.5).

Usage:
    PYTHONPATH=/home/sharon/vision_fusion \
      python scripts/eval_dt_robustness.py \
        --shards   data/shards/nuscenes_mini_labeled/val \
        --ckpt_dir results/checkpoints/ablations_v2 \
        --out      results/dt_robustness.json

Outputs:
    results/dt_robustness.json   — mAP@0.5 at each noise level
    results/dt_robustness.png    — plot for the paper (Fig. robustness)
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.liquidfusionnet import LiquidFusionNet
from src.training.dataset import HDF5ShardDataset
from src.evaluation.evaluator import DetectionEvaluator

CLASS_NAMES = ["Car", "Two_wheeler", "Pedestrian",
               "Auto_rickshaw", "Truck", "Bus", "Animal"]
NUM_CLASSES  = 7

# ── Noise levels to sweep ────────────────────────────────────────────────────
# Each entry: (label_for_plot, noise_std_seconds, dropout_prob)
# noise_std: Gaussian jitter added to delta_t (seconds)
# dropout_p: probability of zeroing a delta_t channel entirely (missing sensor)
NOISE_LEVELS = [
    ("Clean (σ=0)",      0.000, 0.00),
    ("σ=0.01s",          0.010, 0.00),
    ("σ=0.05s",          0.050, 0.00),
    ("σ=0.10s",          0.100, 0.00),
    ("σ=0.20s",          0.200, 0.00),
    ("σ=0.50s",          0.500, 0.00),
    ("Drop p=0.10",      0.000, 0.10),
    ("Drop p=0.25",      0.000, 0.25),
    ("Drop p=0.50",      0.000, 0.50),
    ("σ=0.10+Drop 0.25", 0.100, 0.25),
]



# BEV bbox denormalisation (inverse of dataset._norm_bbox)
_BBOX_MEAN  = [1500.0, 1197.0, 1.5,  3.0,  1.5,  1.8,  0.0]
_BBOX_SCALE = [ 100.0,  100.0, 3.0, 10.0, 10.0,  3.0,  3.14]

def _denorm_bbox(t):
    import torch
    mean  = torch.tensor(_BBOX_MEAN,  dtype=t.dtype, device=t.device)
    scale = torch.tensor(_BBOX_SCALE, dtype=t.dtype, device=t.device)
    return t * scale + mean


def corrupt_delta_t(delta_t: dict, noise_std: float, dropout_p: float,
                    device: torch.device) -> dict:
    """
    Apply Gaussian noise and/or random dropout to delta_t dict.
    delta_t keys: cam, rad, lid, imu — each (B,) tensor.
    """
    out = {}
    for k, v in delta_t.items():
        v = v.clone()
        if noise_std > 0:
            v = v + torch.randn_like(v) * noise_std
            v = v.clamp(min=1e-3)          # delta_t must be positive
        if dropout_p > 0:
            mask = torch.bernoulli(
                torch.full_like(v, dropout_p)
            ).bool()
            v[mask] = 0.0
        out[k] = v
    return out


def decode_detections(cls_logits, bbox_deltas, score_thresh=0.05):
    B = cls_logits.shape[0]
    scores_all = cls_logits.softmax(-1)
    max_scores, labels = scores_all.max(-1)
    pb, ps, pl = [], [], []
    for b in range(B):
        keep = max_scores[b] > score_thresh
        s = max_scores[b][keep]
        l = labels[b][keep]
        d = bbox_deltas[b][keep]
        cx, cy = d[:, 0], d[:, 1]
        w, h = d[:, 4], d[:, 3]
        boxes = torch.stack([cx, cy, w, h], dim=1)
        pb.append(boxes.cpu())
        ps.append(s.cpu())
        pl.append(l.cpu())
    return pb, ps, pl


def gt_to_bev(gt_bbox, gt_cls, gt_mask):
    B = gt_bbox.shape[0]
    gb, gl = [], []
    for b in range(B):
        valid = gt_mask[b].bool()
        d = gt_bbox[b][valid]
        c = gt_cls[b][valid]
        l = c.argmax(-1)
        cx, cy = d[:, 0], d[:, 1]
        w, h = d[:, 4], d[:, 3]
        boxes = torch.stack([cx, cy, w, h], dim=1)
        gb.append(boxes.cpu())
        gl.append(l.cpu())
    return gb, gl


def find_best_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = list(ckpt_dir.rglob("full_model/**/best_epoch*.pt"))
    if candidates:
        return sorted(candidates)[-1]
    candidates = list(ckpt_dir.rglob("full_model/**/epoch*.pt"))
    if candidates:
        return sorted(candidates)[-1]
    return None


def evaluate_at_noise(model, val_loader, noise_std, dropout_p, device):
    ev = DetectionEvaluator(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
    model.eval()
    with torch.no_grad():
        for batch, targets in val_loader:
            batch_d = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_d[k] = v.to(device)
                elif isinstance(v, dict):
                    batch_d[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor)
                                  else vv for kk, vv in v.items()}
                else:
                    batch_d[k] = v

            # Corrupt delta_t
            if "delta_t" in batch_d:
                batch_d["delta_t"] = corrupt_delta_t(
                    batch_d["delta_t"], noise_std, dropout_p, device
                )

            out, _ = model(batch_d)
            cls_logits  = out["detection"]["cls_logits"]
            bbox_deltas = out["detection"]["bbox_deltas"]

            gt_bbox = targets["detection"]["bbox_targets"].to(device)
            gt_cls  = targets["detection"]["cls_targets"].to(device)
            gt_mask = targets["detection"]["pos_mask"].to(device)

            pb, ps, pl = decode_detections(cls_logits, bbox_deltas)
            gb, gl     = gt_to_bev(gt_bbox, gt_cls, gt_mask)

            for b in range(len(pb)):
                ev.update(pb[b].numpy(), ps[b].numpy(), pl[b].numpy(),
                          gb[b].numpy(), gl[b].numpy(), iou_thresh=0.5)

    r = ev.compute(iou_thresh=0.5)
    return r["mAP"], r["per_class_AP"]


def plot_results(results: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    labels  = [r["label"] for r in results]
    maps    = [r["mAP@0.5"] for r in results]
    clean   = maps[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ["#2196F3" if m >= clean * 0.95 else
               "#FF9800" if m >= clean * 0.85 else
               "#F44336" for m in maps]

    bars = ax.bar(range(len(labels)), maps, color=colors, edgecolor="white",
                  linewidth=0.5)
    ax.axhline(clean, color="#2196F3", linestyle="--", linewidth=1.2,
               label=f"Clean baseline ({clean:.4f})")
    ax.axhline(clean * 0.95, color="#FF9800", linestyle=":", linewidth=1.0,
               label="−5% threshold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Δt Perturbation Robustness — LiquidFusionNet (full_model)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(maps) * 1.15 + 0.001)

    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(maps) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards",   default="data/shards/nuscenes_mini_labeled/val")
    ap.add_argument("--ckpt_dir", default="results/checkpoints/ablations_v2")
    ap.add_argument("--cfg",      default="configs/default.yaml")
    ap.add_argument("--out",      default="results/dt_robustness.json")
    ap.add_argument("--batch",    type=int, default=1)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device   = torch.device(args.device)
    ckpt_dir = ROOT / args.ckpt_dir
    out_json = ROOT / args.out
    out_png  = out_json.with_suffix(".png")

    cfg      = OmegaConf.load(ROOT / args.cfg)
    ckpt     = find_best_checkpoint(ckpt_dir)
    if ckpt is None:
        print(f"ERROR: no full_model checkpoint found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Checkpoint: {ckpt}")

    # Build model
    full_cfg = copy.deepcopy(cfg)
    full_cfg.ablation_modalities  = ["camera", "lidar", "radar", "imu"]
    full_cfg.ablation_use_ltc     = True
    full_cfg.ablation_inject_dt   = True

    model = LiquidFusionNet(full_cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    sd    = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Val loader
    val_ds     = HDF5ShardDataset(str(ROOT / args.shards), cfg, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch,
                            shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    results = []
    for label, noise_std, dropout_p in NOISE_LEVELS:
        print(f"  [{label}] ", end="", flush=True)
        mAP, per_class = evaluate_at_noise(
            model, val_loader, noise_std, dropout_p, device
        )
        print(f"mAP@0.5 = {mAP:.4f}")
        results.append({
            "label":       label,
            "noise_std_s": noise_std,
            "dropout_p":   dropout_p,
            "mAP@0.5":     mAP,
            "per_class":   per_class,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")

    # Print table
    clean = results[0]["mAP@0.5"]
    print(f"\n{'Perturbation':<25} {'mAP@0.5':>9} {'Δ vs clean':>12}")
    print("-" * 50)
    for r in results:
        delta = r["mAP@0.5"] - clean
        print(f"{r['label']:<25} {r['mAP@0.5']:9.4f} {delta:+12.4f}")

    plot_results(results, out_png)


if __name__ == "__main__":
    main()