"""
fix_dataset_labels.py — Patch dataset.py to load gt labels from HDF5
and return them as a targets dict alongside the sensor batch.

Also updates configs/config.yaml to point shards at nuscenes_mini_labeled.

Run with:
    python fix_dataset_labels.py
"""
import sys
from pathlib import Path

DATASET = Path("/home/sharon/vision_fusion/src/training/dataset.py")
CONFIG  = Path("/home/sharon/vision_fusion/configs/config.yaml")

if not DATASET.exists():
    print(f"ERROR: {DATASET} not found", file=sys.stderr)
    sys.exit(1)

src = DATASET.read_text()

# ── PATCH 1: Add label loading to __getitem__ return ─────────────────────────
OLD1 = '''\
        return {
            "camera":    cam,
            "radar":     radar,
            "lidar":     lidar,
            "imu_gps":   imu,
            "delta_t":   dt,
            "scene_id":  str(grp.attrs.get("scene_id", scene_key)),
            "road_type": str(grp.attrs.get("road_type", "unknown")),
        }'''

NEW1 = '''\
        # ── Ground-truth labels (present in nuScenes shards) ─────────────────
        targets = {}
        if "gt_cls_targets" in grp:
            import torch as _torch
            targets["detection"] = {
                "cls_targets":  _torch.from_numpy(grp["gt_cls_targets"][:]),
                "bbox_targets": _torch.from_numpy(grp["gt_bbox_targets"][:]),
                "pos_mask":     _torch.from_numpy(grp["gt_pos_mask"][:]),
            }
        # tracking / trajectory targets not yet in shards → left absent
        # losses.py already handles missing keys gracefully (returns 0)

        batch = {
            "camera":    cam,
            "radar":     radar,
            "lidar":     lidar,
            "imu_gps":   imu,
            "delta_t":   dt,
            "scene_id":  str(grp.attrs.get("scene_id", scene_key)),
            "road_type": str(grp.attrs.get("road_type", "unknown")),
        }
        return batch, targets'''

if OLD1 not in src:
    print(f"ERROR: Patch 1 anchor not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 applied: __getitem__ now returns (batch, targets)")


DATASET.write_text(src)
print(f"dataset.py updated → {DATASET}")


# ── PATCH 2: Fix config to point at labeled shards ───────────────────────────
cfg = CONFIG.read_text()
OLD2 = "  shards:         data/shards"
NEW2 = "  shards:         data/shards/nuscenes_mini_labeled"

if OLD2 not in cfg:
    # Try to find the current shards line
    import re
    m = re.search(r'  shards:\s+\S+', cfg)
    if m:
        print(f"WARNING: shards line is '{m.group()}', updating manually")
        cfg = cfg.replace(m.group(), NEW2)
    else:
        print("ERROR: could not find shards line in config.yaml", file=sys.stderr)
        sys.exit(1)
else:
    cfg = cfg.replace(OLD2, NEW2, 1)

CONFIG.write_text(cfg)
print(f"config.yaml shards → nuscenes_mini_labeled")

print("\nAll patches applied.")
print("\nNow check the trainer reads targets correctly:")
print("  grep -n 'targets\\|batch' /home/sharon/vision_fusion/src/training/trainer.py | head -20")