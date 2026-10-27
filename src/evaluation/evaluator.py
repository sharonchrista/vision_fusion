"""
Evaluation for LiquidFusionNet — T-ITS metric suite.

DetectionEvaluator   — mAP@0.5, mAP@0.75, per-class AP
TrackingEvaluator    — MOTA, MOTP, IDF1, ID switches
TrajectoryEvaluator  — ADE, FDE at 1s/2s/3s, miss rate
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger()


# ── IoU helpers ───────────────────────────────────────────────────────────────

def box_iou_2d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise 2-D IoU (BEV) for axis-aligned boxes.
    boxes: (N, 4) — cx, cy, w, h
    Returns: (N, M) IoU matrix
    """
    def to_corners(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

    a = to_corners(boxes_a)   # (N, 4)
    b = to_corners(boxes_b)   # (M, 4)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * \
                 np.maximum(0, inter_y2 - inter_y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union, 1e-6)


# ── Detection evaluator ───────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    Accumulates predictions and GT across batches, then computes
    mAP@0.5 and mAP@0.75 per class.
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        # per-class lists of (score, is_tp) tuples, sorted by score desc
        self._preds: dict[int, list] = defaultdict(list)
        self._n_gt:  dict[int, int]  = defaultdict(int)

    def update(self, pred_boxes: np.ndarray, pred_scores: np.ndarray,
               pred_labels: np.ndarray,
               gt_boxes:    np.ndarray, gt_labels: np.ndarray,
               iou_thresh:  float = 0.5) -> None:
        """
        pred_boxes:  (N, 4) cx,cy,w,h
        pred_scores: (N,)
        pred_labels: (N,) int class index
        gt_boxes:    (M, 4)
        gt_labels:   (M,) int
        """
        for cls in range(self.num_classes):
            gt_mask   = (gt_labels == cls)
            pred_mask = (pred_labels == cls)
            n_gt = gt_mask.sum()
            self._n_gt[cls] += int(n_gt)

            if pred_mask.sum() == 0:
                continue

            p_boxes  = pred_boxes[pred_mask]
            p_scores = pred_scores[pred_mask]
            g_boxes  = gt_boxes[gt_mask] if n_gt > 0 else np.empty((0, 4))

            if n_gt == 0:
                for score in p_scores:
                    self._preds[cls].append((float(score), 0))
                continue

            iou_mat  = box_iou_2d(p_boxes, g_boxes)   # (N_pred, N_gt)
            gt_used  = np.zeros(n_gt, dtype=bool)

            # Sort predictions by score descending
            order = np.argsort(-p_scores)
            for idx in order:
                best_iou = -1.0
                best_gt  = -1
                for j in range(n_gt):
                    if not gt_used[j] and iou_mat[idx, j] > best_iou:
                        best_iou = iou_mat[idx, j]
                        best_gt  = j
                is_tp = 1 if best_iou >= iou_thresh else 0
                if is_tp and best_gt >= 0:
                    gt_used[best_gt] = True
                self._preds[cls].append((float(p_scores[idx]), is_tp))

    def _ap_from_pr(self, preds: list) -> float:
        if not preds:
            return 0.0
        preds_sorted = sorted(preds, key=lambda x: -x[0])
        tp_cum = np.cumsum([p[1] for p in preds_sorted])
        fp_cum = np.cumsum([1 - p[1] for p in preds_sorted])
        n_gt   = sum(p[1] for p in preds_sorted)
        if n_gt == 0:
            return 0.0
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall    = tp_cum / (n_gt + 1e-9)
        # VOC 2012 interpolation
        ap = 0.0
        for thr in np.arange(0, 1.01, 0.1):
            mask = recall >= thr
            ap  += precision[mask].max() if mask.any() else 0.0
        return ap / 11.0

    def compute(self, iou_thresh: float = 0.5) -> dict:
        per_class_ap = {}
        for cls in range(self.num_classes):
            per_class_ap[self.class_names[cls]] = self._ap_from_pr(
                self._preds[cls]
            )
        mAP = float(np.mean(list(per_class_ap.values())))
        return {"mAP": mAP, "per_class_AP": per_class_ap,
                "iou_thresh": iou_thresh}


# ── Tracking evaluator ────────────────────────────────────────────────────────

class TrackingEvaluator:
    """
    Lightweight MOTA / MOTP / IDF1 accumulator.
    Full evaluation: use TrackEval toolkit on saved prediction files.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tp = self._fp = self._fn = self._id_sw = 0
        self._dist_sum = 0.0
        self._tp_dist  = 0

    def update(self, n_tp: int, n_fp: int, n_fn: int,
               id_switches: int, match_distances: list[float]) -> None:
        self._tp     += n_tp
        self._fp     += n_fp
        self._fn     += n_fn
        self._id_sw  += id_switches
        self._dist_sum += sum(match_distances)
        self._tp_dist  += len(match_distances)

    def compute(self) -> dict:
        total_gt = self._tp + self._fn
        mota = 1.0 - (self._fp + self._fn + self._id_sw) / max(total_gt, 1)
        motp = self._dist_sum / max(self._tp_dist, 1)
        return {
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "ID_switches": self._id_sw,
            "TP": self._tp,
            "FP": self._fp,
            "FN": self._fn,
        }


# ── Trajectory evaluator ──────────────────────────────────────────────────────

class TrajectoryEvaluator:
    """
    ADE / FDE at 1s, 2s, 3s horizons.
    miss rate at 2m threshold.
    """

    def __init__(self, hz: int = 10,
                 horizons_s: list[float] | None = None,
                 miss_threshold_m: float = 2.0):
        self.hz         = hz
        self.horizons   = horizons_s or [1.0, 2.0, 3.0]
        self.miss_thr   = miss_threshold_m
        self.reset()

    def reset(self):
        self._ade: dict[str, list] = {f"{h}s": [] for h in self.horizons}
        self._fde: dict[str, list] = {f"{h}s": [] for h in self.horizons}
        self._miss: list[int]      = []

    def update(self, pred_xy: np.ndarray, gt_xy: np.ndarray) -> None:
        """
        pred_xy : (T, 2) — predicted future positions (best mode)
        gt_xy   : (T, 2) — ground truth future positions
        """
        T = min(pred_xy.shape[0], gt_xy.shape[0])
        for h in self.horizons:
            t_idx = min(int(h * self.hz) - 1, T - 1)
            disp  = np.linalg.norm(pred_xy[:t_idx+1] - gt_xy[:t_idx+1], axis=1)
            self._ade[f"{h}s"].append(float(disp.mean()))
            self._fde[f"{h}s"].append(float(disp[-1]))
            self._miss.append(1 if disp[-1] > self.miss_thr else 0)

    def compute(self) -> dict:
        result = {}
        for h in self.horizons:
            key = f"{h}s"
            result[f"ADE_{key}"] = float(np.mean(self._ade[key])) \
                if self._ade[key] else 0.0
            result[f"FDE_{key}"] = float(np.mean(self._fde[key])) \
                if self._fde[key] else 0.0
        result["miss_rate"] = float(np.mean(self._miss)) \
            if self._miss else 0.0
        return result


# ── Full evaluation runner ────────────────────────────────────────────────────

class Evaluator:
    """
    Wraps all three evaluators for end-of-epoch evaluation.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.dataset
        e = cfg.evaluation
        self.det_05  = DetectionEvaluator(d.num_classes, d.classes)
        self.det_075 = DetectionEvaluator(d.num_classes, d.classes)
        self.trk     = TrackingEvaluator()
        self.traj    = TrajectoryEvaluator(
            hz=d.camera.fps,
            horizons_s=e.traj_horizons_s,
            miss_threshold_m=e.miss_rate_threshold_m,
        )

    def reset(self):
        self.det_05.reset()
        self.det_075.reset()
        self.trk.reset()
        self.traj.reset()

    def compute_all(self) -> dict:
        det_05  = self.det_05.compute(iou_thresh=0.5)
        det_075 = self.det_075.compute(iou_thresh=0.75)
        trk     = self.trk.compute()
        traj    = self.traj.compute()
        return {
            "detection": {
                "mAP@0.5":  det_05["mAP"],
                "mAP@0.75": det_075["mAP"],
                "per_class_AP@0.5": det_05["per_class_AP"],
            },
            "tracking":  trk,
            "trajectory": traj,
        }

    def log_and_save(self, metrics: dict, split: str,
                     epoch: int, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{split}_epoch{epoch:03d}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Metrics saved → {path}")

        # Console summary
        d = metrics["detection"]
        t = metrics["tracking"]
        tr = metrics["trajectory"]
        log.info(
            f"[{split.upper()}] "
            f"mAP@0.5={d['mAP@0.5']:.3f} | mAP@0.75={d['mAP@0.75']:.3f} | "
            f"MOTA={t['MOTA']:.3f} | MOTP={t['MOTP']:.3f} | "
            f"ADE_1s={tr.get('ADE_1.0s',0):.2f} | FDE_3s={tr.get('FDE_3.0s',0):.2f}"
        )
