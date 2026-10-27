"""
Rebuild training shards with scene-aware augmented data (Fix 2).
Creates a fresh shard_0001.h5 with augmented samples only,
keeping shard_0000.h5 (original data) intact.
"""
import json, sys, warnings
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

warnings.filterwarnings("ignore")

AUG_ROOT     = Path("data/augmented_v2")
RARE_CLASSES = ["pedestrian", "bus", "truck", "van"]
LIDAR_CLASS_MAP = {
    "pedestrian": 4, "bus": 6, "truck": 7, "van": 8,
}
SIZE_PRIORS = {
    "pedestrian": (0.5, 0.5, 1.75),
    "bus":        (12.0, 2.5, 3.5),
    "truck":      (8.0,  2.5, 3.2),
    "van":        (5.0,  1.9, 1.8),
}


def boxes_to_targets(class_name, label,
                     num_anchors=9, num_classes=9):
    cls_t  = np.zeros((num_anchors, num_classes), dtype=np.float32)
    bbox_t = np.zeros((num_anchors, 7),           dtype=np.float32)
    mask   = np.zeros(num_anchors,                dtype=bool)
    cls_id = LIDAR_CLASS_MAP.get(class_name.lower(), 4)
    for i, obj in enumerate(label[:num_anchors]):
        psr = obj.get("psr", {})
        p   = psr.get("position", {})
        s   = psr.get("scale",    {})
        r   = psr.get("rotation", {})
        cls_t[i, cls_id] = 1.0
        bbox_t[i] = [p.get("x",0), p.get("y",0), p.get("z",0),
                     s.get("x",1), s.get("y",1), s.get("z",1),
                     r.get("z",0)]
        mask[i] = True
    return cls_t, bbox_t, mask


def run(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    H   = cfg.dataset.camera.resize_height
    W   = cfg.dataset.camera.resize_width
    MAX = cfg.dataset.lidar.max_points

    # load all augmented samples
    aug_samples = []
    for cls in RARE_CLASSES:
        cls_dir   = AUG_ROOT / cls
        pts_files = sorted(cls_dir.glob("*_pts.npy"))
        lbl_files = sorted(cls_dir.glob("*_label.json"))
        for pf, lf in zip(pts_files, lbl_files):
            aug_samples.append({
                "class": cls,
                "pts":   np.load(pf),
                "label": json.load(open(lf)),
            })
        print(f"  [{cls:<12s}] {len(pts_files):>5d} samples")

    print(f"\nTotal augmented samples: {len(aug_samples)}")

    # write to new shard file (keep original shard_0000.h5 intact)
    out_dir = Path(cfg.paths.shards) / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / "shard_0001.h5"

    print(f"Writing -> {shard_path}")
    with h5py.File(shard_path, "w") as hf:
        hf.attrs["split"]       = "train"
        hf.attrs["num_samples"] = len(aug_samples)
        hf.attrs["source"]      = "scene_aware_augmentation_v2"

        for i, s in enumerate(tqdm(aug_samples, desc="Writing")):
            key = f"aug_v2_{s['class']}_{i:04d}"
            grp = hf.require_group(key)

            # camera — zeros (LiDAR-only augmentation)
            grp.create_dataset("camera",
                data=np.zeros((1,4,3,H,W), dtype=np.float16),
                compression="lzf")

            # lidar — real scene + synthetic object
            pts = s["pts"]
            if len(pts) < MAX:
                pts = np.vstack([pts,
                    np.zeros((MAX-len(pts),4), dtype=np.float32)])
            grp.create_dataset("lidar",
                data=pts[np.newaxis, :MAX],
                compression="lzf")

            grp.create_dataset("radar",
                data=np.zeros((1,6,13),  dtype=np.float32))
            grp.create_dataset("imu_gps",
                data=np.zeros((1,9),     dtype=np.float32))
            grp.create_dataset("dt_cam",
                data=np.array([0.1],     dtype=np.float32))
            grp.create_dataset("dt_lid",
                data=np.array([0.1],     dtype=np.float32))
            grp.create_dataset("dt_rad",
                data=np.array([0.05],    dtype=np.float32))
            grp.create_dataset("dt_imu",
                data=np.array([0.02],    dtype=np.float32))

            # ground-truth labels
            cls_t, bbox_t, mask = boxes_to_targets(
                s["class"], s["label"])
            grp.create_dataset("gt_cls_targets",  data=cls_t)
            grp.create_dataset("gt_bbox_targets", data=bbox_t)
            grp.create_dataset("gt_pos_mask",     data=mask)

            grp.attrs["sample_id"]    = key
            grp.attrs["road_type"]    = "urban"
            grp.attrs["num_frames"]   = 1
            grp.attrs["is_augmented"] = True
            grp.attrs["aug_class"]    = s["class"]

    size_gb = shard_path.stat().st_size / 1e9
    print(f"\nShard written: {shard_path}  ({size_gb:.2f} GB)")

    # update dataset loader to read both shards
    print("\nBoth shards will be loaded automatically:")
    print(f"  shard_0000.h5 — original 127 real samples")
    print(f"  shard_0001.h5 — {len(aug_samples)} augmented samples")
    total = 127 + len(aug_samples)
    print(f"  Total training samples: {total}")

    # class balance report
    from collections import Counter
    dist = Counter(s["class"] for s in aug_samples)
    print("\nAugmented class distribution:")
    for cls, cnt in sorted(dist.items()):
        print(f"  {cls:<15s}: {cnt:>5d}")


if __name__ == "__main__":
    run()
