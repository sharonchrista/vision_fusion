"""
evaluate_ablations.py — Run DetectionEvaluator on each ablation variant's
best checkpoint and produce Table 3 for the T-ITS paper.

Usage (on tyrone-hpc):
    PYTHONPATH=/home/sharon/vision_fusion \
      python scripts/evaluate_ablations.py \
        --shards   data/shards/nuscenes_mini_labeled/val \
        --ckpt_dir results/checkpoints/ablations_v2 \
        --out      results/ablation_eval.json

Outputs:
    results/ablation_eval.json  — per-variant mAP@0.5 / mAP@0.75 + per-class AP
    stdout table for copy-paste into Table 3
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # /home/sharon/vision_fusion
sys.path.insert(0, str(ROOT))

from src.models.liquidfusionnet import LiquidFusionNet
from src.training.dataset import HDF5ShardDataset
from src.evaluation.evaluator import DetectionEvaluator

# ── T-ITS class names ────────────────────────────────────────────────────────
CLASS_NAMES = ["Car", "Two_wheeler", "Pedestrian",
               "Auto_rickshaw", "Truck", "Bus", "Animal"]
NUM_CLASSES = 7

# ── Ablation variants (must match run_ablations_v2.py) ───────────────────────
ABLATION_VARIANTS = [
    {"name": "camera_only",        "modalities": ["camera"],
     "use_ltc": True,  "inject_dt": True},
    {"name": "camera_lidar",       "modalities": ["camera", "lidar"],
     "use_ltc": True,  "inject_dt": True},
    {"name": "camera_lidar_radar", "modalities": ["camera", "lidar", "radar"],
     "use_ltc": True,  "inject_dt": True},
    {"name": "full_no_delta_t",    "modalities": ["camera", "lidar", "radar", "imu"],
     "use_ltc": True,  "inject_dt": False},
    {"name": "full_lstm",          "modalities": ["camera", "lidar", "radar", "imu"],
     "use_ltc": False, "inject_dt": True},
    {"name": "full_model",         "modalities": ["camera", "lidar", "radar", "imu"],
     "use_ltc": True,  "inject_dt": True},
]

# ── Decode DetectionHead output → boxes / scores / labels ───────────────────

# BEV bbox denormalisation (inverse of dataset._norm_bbox)
_BBOX_MEAN  = [1500.0, 1197.0, 1.5,  3.0,  1.5,  1.8,  0.0]
_BBOX_SCALE = [ 100.0,  100.0, 3.0, 10.0, 10.0,  3.0,  3.14]

def _denorm_bbox(t):
    import torch
    mean  = torch.tensor(_BBOX_MEAN,  dtype=t.dtype, device=t.device)
    scale = torch.tensor(_BBOX_SCALE, dtype=t.dtype, device=t.device)
    return t * scale + mean


def decode_detections(
    cls_logits: torch.Tensor,   # (B, A, C)
    bbox_deltas: torch.Tensor,  # (B, A, 7)
    score_thresh: float = 0.05,
):
    """
    Trivial decoder: treat anchor index as spatial position, return
    top-scoring box per anchor above threshold.

    Returns lists (length B) of:
        pred_boxes  : (N, 4)  — x1,y1,x2,y2  BEV (metre scale, placeholder)
        pred_scores : (N,)
        pred_labels : (N,)   int
    """
    B = cls_logits.shape[0]
    scores_all = cls_logits.softmax(-1)           # (B, A, C)
    max_scores, labels = scores_all.max(-1)       # (B, A)

    pred_boxes_list, pred_scores_list, pred_labels_list = [], [], []
    for b in range(B):
        keep = max_scores[b] > score_thresh       # (A,)
        s = max_scores[b][keep]
        l = labels[b][keep]
        d = bbox_deltas[b][keep]                  # (N, 7)

        # Keep cx, cy, w, h format expected by box_iou_2d
        cx, cy = d[:, 0], d[:, 1]
        w, h   = d[:, 4], d[:, 3]
        boxes = torch.stack([cx, cy, w, h], dim=1)

        pred_boxes_list.append(boxes.cpu())
        pred_scores_list.append(s.cpu())
        pred_labels_list.append(l.cpu())

    return pred_boxes_list, pred_scores_list, pred_labels_list


def gt_to_bev_boxes(gt_bbox: torch.Tensor, gt_cls: torch.Tensor,
                    gt_mask: torch.Tensor):
    """
    Convert HDF5 gt tensors → DetectionEvaluator format.

    gt_bbox : (B, A, 7)  — cx, cy, cz, l, w, h, yaw
    gt_cls  : (B, A, C)  — one-hot or soft label (argmax → class)
    gt_mask : (B, A)     — 1 if valid anchor

    Returns lists (length B):
        gt_boxes  : (M, 4)
        gt_labels : (M,)
    """
    B = gt_bbox.shape[0]
    gt_boxes_list, gt_labels_list = [], []
    for b in range(B):
        valid = gt_mask[b].bool()
        d = gt_bbox[b][valid]         # (M, 7)
        c = gt_cls[b][valid]          # (M, C)
        l = c.argmax(-1)              # (M,)

        cx, cy = d[:, 0], d[:, 1]
        w, h   = d[:, 4], d[:, 3]
        boxes = torch.stack([cx, cy, w, h], dim=1)

        gt_boxes_list.append(boxes.cpu())
        gt_labels_list.append(l.cpu())

    return gt_boxes_list, gt_labels_list


# ── Find best checkpoint for a variant ──────────────────────────────────────
def find_best_checkpoint(ckpt_dir: Path, variant_name: str) -> Path | None:
    """
    Search:  ckpt_dir / variant_name / seed* / run* / best_epoch*.pt
    Falls back to latest epoch*.pt if no best_ found.
    Returns None if nothing found.
    """
    pattern_best = list(ckpt_dir.rglob(f"{variant_name}/**/best_epoch*.pt"))
    if pattern_best:
        return sorted(pattern_best)[-1]   # highest epoch number

    pattern_any = list(ckpt_dir.rglob(f"{variant_name}/**/epoch*.pt"))
    if pattern_any:
        return sorted(pattern_any)[-1]

    return None


# ── Evaluate one variant ─────────────────────────────────────────────────────
def evaluate_variant(
    variant: dict,
    base_cfg,
    ckpt_path: Path,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.ablation_modalities  = variant["modalities"]
    cfg.ablation_use_ltc     = variant["use_ltc"]
    cfg.ablation_inject_dt   = variant["inject_dt"]

    model = LiquidFusionNet(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd, strict=False)
    model.eval()

    eval50  = DetectionEvaluator(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
    eval75  = DetectionEvaluator(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)

    with torch.no_grad():
        for batch, targets in val_loader:
            # Move modality tensors to device
            batch_d = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else
                    {kk: vv.to(device) for kk, vv in v.items()}
                    if isinstance(v, dict) else v)
                for k, v in batch.items()
            }

            out, _ = model(batch_d)
            cls_logits  = out["detection"]["cls_logits"]    # (B, A, C)
            bbox_deltas = out["detection"]["bbox_deltas"]   # (B, A, 7)

            # Ground truth
            gt_bbox = targets["detection"]["bbox_targets"].to(device)   # (B, A, 7)
            gt_cls  = targets["detection"]["cls_targets"].to(device)    # (B, A, C)
            gt_mask = targets["detection"]["pos_mask"].to(device)       # (B, A)

            pb, ps, pl = decode_detections(cls_logits, bbox_deltas)
            gb, gl     = gt_to_bev_boxes(gt_bbox, gt_cls, gt_mask)

            for b in range(len(pb)):
                eval50.update(pb[b], ps[b], pl[b], gb[b], gl[b], iou_thresh=0.50)
                eval75.update(pb[b], ps[b], pl[b], gb[b], gl[b], iou_thresh=0.75)

    r50 = eval50.compute(iou_thresh=0.50)
    r75 = eval75.compute(iou_thresh=0.75)

    result = {
        "variant":   variant["name"],
        "ckpt":      str(ckpt_path),
        "mAP@0.5":   r50["mAP"],
        "mAP@0.75":  r75["mAP"],
        "AP@0.5_per_class":  r50["per_class_AP"],
        "AP@0.75_per_class": r75["per_class_AP"],
    }
    return result


# ── Pretty-print Table 3 ─────────────────────────────────────────────────────
def print_table(results: list[dict]):
    hdr = f"{'Variant':<25} {'mAP@0.5':>9} {'mAP@0.75':>9}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in results:
        print(f"{r['variant']:<25} {r['mAP@0.5']:9.4f} {r['mAP@0.75']:9.4f}")
    print("=" * len(hdr))

    # Per-class breakdown for full_model
    full = next((r for r in results if r["variant"] == "full_model"), None)
    if full:
        print("\nPer-class AP@0.5 (full_model):")
        for cls, ap in full["AP@0.5_per_class"].items():
            print(f"  {cls:<18} {ap:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards",   default="data/shards/nuscenes_mini_labeled/val")
    ap.add_argument("--ckpt_dir", default="results/checkpoints/ablations_v2")
    ap.add_argument("--cfg",      default="configs/liquidfusionnet.yaml")
    ap.add_argument("--out",      default="results/ablation_eval.json")
    ap.add_argument("--batch",    type=int, default=1)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device    = torch.device(args.device)
    ckpt_dir  = ROOT / args.ckpt_dir
    shards    = ROOT / args.shards
    out_path  = ROOT / args.out
    cfg_path  = ROOT / args.cfg

    if not cfg_path.exists():
        # fall back to any yaml in configs/
        candidates = list((ROOT / "configs").glob("*.yaml"))
        if not candidates:
            print("ERROR: no config yaml found under configs/", file=sys.stderr)
            sys.exit(1)
        cfg_path = candidates[0]
        print(f"Using config: {cfg_path}")

    base_cfg = OmegaConf.load(cfg_path)

    # Val dataset — no augmentation
    val_ds = HDF5ShardDataset(str(shards), base_cfg, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch,
                            shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    results = []
    for variant in ABLATION_VARIANTS:
        ckpt = find_best_checkpoint(ckpt_dir, variant["name"])
        if ckpt is None:
            print(f"[SKIP] {variant['name']}: no checkpoint found in {ckpt_dir}")
            results.append({
                "variant":   variant["name"],
                "ckpt":      None,
                "mAP@0.5":   None,
                "mAP@0.75":  None,
            })
            continue

        print(f"[EVAL] {variant['name']}  ← {ckpt.name}")
        try:
            r = evaluate_variant(variant, base_cfg, ckpt, val_loader, device)
            results.append(r)
            print(f"       mAP@0.5={r['mAP@0.5']:.4f}  mAP@0.75={r['mAP@0.75']:.4f}")
        except Exception as exc:
            print(f"       ERROR: {exc}")
            results.append({"variant": variant["name"], "ckpt": str(ckpt),
                            "error": str(exc)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print table for any completed variants
    done = [r for r in results if r.get("mAP@0.5") is not None]
    if done:
        print_table(done)


if __name__ == "__main__":
    main()