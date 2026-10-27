"""
Label parsers for both annotation formats.

Camera labels  — YOLO: class_id cx cy w h  (normalised)
LiDAR labels   — JSON: position/rotation/scale per object
"""
import json
import numpy as np
from pathlib import Path

# ── Camera class map (from frequency analysis) ────────────────────────────────
# IDs 0-25 observed. Map to our 7 T-ITS classes:
# Car=0, Two_wheeler=1, Pedestrian=2, Auto_rickshaw=3, Truck=4, Bus=5, Animal=6
CAMERA_CLASS_MAP = {
    0:  0,   # Car
    1:  0,   # Car variant
    2:  1,   # Two-wheeler (most frequent non-car)
    3:  2,   # Pedestrian
    4:  3,   # Auto-rickshaw
    5:  4,   # Truck
    6:  5,   # Bus
    7:  1,   # Motorcycle
    8:  2,   # Pedestrian variant
    9:  3,   # Auto-rickshaw variant
    10: 1,   # Two-wheeler variant
    12: 4,   # Truck variant
    14: 5,   # Bus variant
    15: 6,   # Animal
    16: 6,   # Animal variant
    17: 6,
    18: 2,
    19: 2,
    20: 1,
    21: 1,
    22: 0,
    23: 3,
    24: 6,
    25: 6,
}

# Seen classes (>=100 instances) — supervised detection
# Rare classes (<100 instances) — zero-shot detection
LIDAR_CLASS_MAP = {
    # SEEN (supervised)
    "Car":            0,
    "car":            0,
    "Motorcycle":     1,
    "motorcycle":     1,
    "Trimotorcycle":  2,
    "trimotorcycle":  2,
    "Scooter":        3,
    "scooter":        3,
    # RARE (zero-shot)
    "Pedestrian":     4,
    "pedestrian":     4,
    "Unknown1":       5,
    "unknown1":       5,
    "Bus":            6,
    "bus":            6,
    "Truck":          7,
    "truck":          7,
    "Van":            8,
    "van":            8,
}

NUM_CLASSES   = 9
SEEN_CLASSES  = [0, 1, 2, 3]        # Car, Motorcycle, Trimotorcycle, Scooter
RARE_CLASSES  = [4, 5, 6, 7, 8]     # Pedestrian, Unknown1, Bus, Truck, Van
SEEN_NAMES    = ["car","motorcycle","trimotorcycle","scooter"]
RARE_NAMES    = ["pedestrian","unknown1","bus","truck","van"]
ALL_NAMES     = SEEN_NAMES + RARE_NAMES


def parse_camera_label(txt_path: str | Path,
                       max_boxes: int = 50
                       ) -> dict[str, np.ndarray]:
    """
    Returns:
        boxes  (N, 4)  float32  cx,cy,w,h  normalised
        labels (N,)    int64
    """
    boxes, labels = [], []
    try:
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                mapped = CAMERA_CLASS_MAP.get(cid, 0)
                boxes.append([cx, cy, w, h])
                labels.append(mapped)
                if len(boxes) >= max_boxes:
                    break
    except Exception:
        pass

    if not boxes:
        return {
            "boxes":  np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,),   dtype=np.int64),
        }
    return {
        "boxes":  np.array(boxes,  dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def parse_lidar_label(json_path: str | Path,
                      max_boxes: int = 50
                      ) -> dict[str, np.ndarray]:
    """
    Returns:
        boxes  (N, 7)  float32  cx,cy,cz,l,w,h,yaw
        labels (N,)    int64
    """
    boxes, labels = [], []
    try:
        with open(json_path) as f:
            objs = json.load(f)
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("obj_type", "Car")
            psr      = obj.get("psr", {})
            pos      = psr.get("position", {})
            rot      = psr.get("rotation", {})
            scale    = psr.get("scale",    {})
            cx   = float(pos.get("x", 0))
            cy   = float(pos.get("y", 0))
            cz   = float(pos.get("z", 0))
            l    = float(scale.get("x", 1))
            w    = float(scale.get("y", 1))
            h    = float(scale.get("z", 1))
            yaw  = float(rot.get("z", 0))
            cls  = LIDAR_CLASS_MAP.get(obj_type, 0)
            boxes.append([cx, cy, cz, l, w, h, yaw])
            labels.append(cls)
            if len(boxes) >= max_boxes:
                break
    except Exception:
        pass

    if not boxes:
        return {
            "boxes":  np.zeros((0, 7), dtype=np.float32),
            "labels": np.zeros((0,),   dtype=np.int64),
        }
    return {
        "boxes":  np.array(boxes,  dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def boxes_to_heatmap(boxes: np.ndarray, labels: np.ndarray,
                     num_classes: int = NUM_CLASSES,
                     num_anchors: int = 9
                     ) -> dict[str, np.ndarray]:
    """
    Convert ground-truth boxes to anchor-format targets
    expected by DetectionLoss.

    Returns:
        cls_targets  (num_anchors, num_classes)  float32  one-hot
        bbox_targets (num_anchors, 7)            float32
        pos_mask     (num_anchors,)              bool
    """
    cls_t  = np.zeros((num_anchors, num_classes), dtype=np.float32)
    bbox_t = np.zeros((num_anchors, 7),           dtype=np.float32)
    mask   = np.zeros(num_anchors,                dtype=bool)

    n = min(len(boxes), num_anchors)
    for i in range(n):
        cls_t[i, labels[i]] = 1.0
        # camera boxes are (cx,cy,w,h) — pad to 7 with zeros
        b = boxes[i]
        bbox_t[i, :len(b)] = b
        mask[i] = True

    return {
        "cls_targets":  cls_t,
        "bbox_targets": bbox_t,
        "pos_mask":     mask,
    }
