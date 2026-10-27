import json, sys, warnings
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

warnings.filterwarnings("ignore")

AUG_ROOT     = Path("data/augmented")
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


def boxes_to_targets(class_name, label, num_anchors=9, num_classes=9):
    cls_t  = np.zeros((num_anchors, num_classes), dtype=np.float32)
    bbox_t = np.zeros((num_anchors, 7),           dtype=np.float32)
    mask   = np.zeros(num_anchors,                dtype=bool)
    cls_id = LIDAR_CLASS_MAP.get(class_name.lower(), 4)
    for i, obj in enumerate(label[:num_anchors]):
        psr  = obj.get("psr", {})
        p    = psr.get("position", {})
        s    = psr.get("scale",    {})
        r    = psr.get("rotation", {})
        cls_t[i, cls_id] = 1.0
        bbox_t[i] = [p.get("x",0), p.get("y",0), p.get("z",0),
                     s.get("x",1), s.get("y",1), s.get("z",1),
                     r.get("z",0)]
        mask[i] = True
    return cls_t, bbox_t, mask


def run(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)

    # load augmented samples
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
        print(f"  [{cls:<12s}] {len(pts_files)} samples")

    print(f"\nTotal augmented: {len(aug_samples)}")

    # append to training shard
    train_shard = Path(cfg.paths.shards) / "train" / "shard_0000.h5"
    H   = cfg.dataset.camera.resize_height
    W   = cfg.dataset.camera.resize_width
    MAX = cfg.dataset.lidar.max_points

    with h5py.File(train_shard, "a") as hf:
        existing = set(hf.keys())
        added    = 0
        for i, s in enumerate(tqdm(aug_samples, desc="Writing")):
            key = f"aug_{s['class']}_{i:04d}"
            if key in existing:
                continue
            grp = hf.require_group(key)

            # sensors — zeros except lidar
            grp.create_dataset("camera",
                data=np.zeros((1,4,3,H,W), dtype=np.float16), compression="lzf")

            pts = s["pts"]
            if len(pts) < MAX:
                pts = np.vstack([pts,
                    np.zeros((MAX-len(pts),4), dtype=np.float32)])
            grp.create_dataset("lidar",
                data=pts[np.newaxis,:MAX], compression="lzf")

            grp.create_dataset("radar",   data=np.zeros((1,6,13),  dtype=np.float32))
            grp.create_dataset("imu_gps", data=np.zeros((1,9),     dtype=np.float32))
            grp.create_dataset("dt_cam",  data=np.array([0.1],     dtype=np.float32))
            grp.create_dataset("dt_lid",  data=np.array([0.1],     dtype=np.float32))
            grp.create_dataset("dt_rad",  data=np.array([0.05],    dtype=np.float32))
            grp.create_dataset("dt_imu",  data=np.array([0.02],    dtype=np.float32))

            # labels
            cls_t, bbox_t, mask = boxes_to_targets(s["class"], s["label"])
            grp.create_dataset("gt_cls_targets",  data=cls_t)
            grp.create_dataset("gt_bbox_targets", data=bbox_t)
            grp.create_dataset("gt_pos_mask",     data=mask)

            grp.attrs["sample_id"]    = key
            grp.attrs["road_type"]    = "urban"
            grp.attrs["num_frames"]   = 1
            grp.attrs["is_augmented"] = True
            grp.attrs["aug_class"]    = s["class"]
            added += 1

        total = len(hf.keys())
        print(f"\n  Added   : {added} new samples")
        print(f"  Total   : {total} samples in train shard")
        print(f"  Size    : {train_shard.stat().st_size/1e6:.1f} MB")

    print("\nClass balance after augmentation:")
    from collections import Counter
    dist = Counter(s["class"] for s in aug_samples)
    for cls, cnt in sorted(dist.items()):
        print(f"  {cls:<15s}: +{cnt} synthetic")
    print("\nDone. Run: python scripts/full_train_v2.py")


if __name__ == "__main__":
    run()
