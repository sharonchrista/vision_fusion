"""
Joint dataset combining:
  1. TIAND original shards     (data/shards/train/)
  2. Scene-aware aug shards    (data/shards/train/shard_0001.h5)
  3. nuScenes shards           (data/shards/nuscenes_mini/)
  4. nuScenes trainval shards  (data/shards/nuscenes_trainval/) when available

Applies class-balanced sampling to prevent car domination.
"""
from __future__ import annotations
import random
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.utils.reproducibility import worker_init_fn
from src.preprocessing.parse_labels import (
    parse_camera_label, parse_lidar_label,
    boxes_to_heatmap
)

_CAM_LABEL_DIR  = Path("data/extracted/camera/DATASET")
_LID_LABEL_DIR4 = Path("data/extracted/lidar/2nd_september/Scene4/label")
_LID_LABEL_DIR5 = Path("data/extracted/lidar/2nd_september/Scene5/label")

def _build_cam_index():
    return {p.stem: p for p in _CAM_LABEL_DIR.glob("*.txt")} \
           if _CAM_LABEL_DIR.exists() else {}
def _build_lid_index():
    files = []
    for d in [_LID_LABEL_DIR4, _LID_LABEL_DIR5]:
        if d.exists():
            files += sorted(d.glob("*.json"))
    return files

_CAM_IDX = _build_cam_index()
_LID_IDX = _build_lid_index()

# Class weights for balanced sampling
# Inverse frequency weighting — rare classes get higher weight
CLASS_WEIGHTS = {
    0: 1.0,    # car          — very common
    1: 1.5,    # motorcycle   — common
    2: 3.0,    # trimotorcycle
    3: 3.0,    # scooter
    4: 5.0,    # pedestrian   — rare in TIAND
    5: 4.0,    # unknown1
    6: 8.0,    # bus          — very rare
    7: 8.0,    # truck        — very rare
    8: 10.0,   # van          — extremely rare
}
DEFAULT_WEIGHT = 2.0


class JointDataset(Dataset):
    """
    Unified dataset over all shard sources.
    Each item tagged with source and dominant class
    for balanced sampling.
    """

    def __init__(self, shard_dirs: list[str | Path],
                 cfg, augment: bool = False):
        self.cfg     = cfg
        self.augment = augment

        # discover all shards across all directories
        self._index: list[tuple[Path, str, str]] = []
        # (shard_path, sample_key, source_tag)

        for shard_dir in shard_dirs:
            shard_dir = Path(shard_dir)
            if not shard_dir.exists():
                print(f"  [SKIP] {shard_dir} — not found")
                continue
            shards = sorted(shard_dir.glob("*.h5"))
            print(f"  {shard_dir.name:<30s}: {len(shards)} shards")
            for sp in shards:
                with h5py.File(sp, "r") as hf:
                    for key in hf.keys():
                        src = str(hf[key].attrs.get(
                            "dataset", "tiand"))
                        self._index.append((sp, key, src))

        self._cache: dict[str, h5py.File] = {}
        print(f"  Total samples: {len(self._index)}")

    def __len__(self):
        return len(self._index)

    def _open(self, sp):
        k = str(sp)
        if k not in self._cache:
            self._cache[k] = h5py.File(sp, "r")
        return self._cache[k]

    def _get_tiand_labels(self, idx):
        det = {"cls_targets": None,
               "bbox_targets": None,
               "pos_mask": None}
        cam_keys = sorted(_CAM_IDX.keys())
        if cam_keys:
            key = cam_keys[idx % len(cam_keys)]
            cgt = parse_camera_label(_CAM_IDX[key])
            if len(cgt["boxes"]) > 0:
                tgt = boxes_to_heatmap(
                    cgt["boxes"], cgt["labels"],
                    num_anchors=9)
                det = {k: torch.from_numpy(v)
                       for k, v in tgt.items()}
        if _LID_IDX:
            path = _LID_IDX[idx % len(_LID_IDX)]
            lgt  = parse_lidar_label(path)
            if len(lgt["boxes"]) > 0:
                tgt = boxes_to_heatmap(
                    lgt["boxes"], lgt["labels"],
                    num_anchors=9)
                det = {k: torch.from_numpy(v)
                       for k, v in tgt.items()}
        return {"detection": det}

    def get_sample_weight(self, idx: int) -> float:
        """Return sampling weight based on rare class presence."""
        sp, key, src = self._index[idx]
        try:
            hf  = self._open(sp)
            grp = hf[key]
            if "gt_cls_targets" in grp:
                ct   = grp["gt_cls_targets"][:]   # (A, C)
                mask = grp["gt_pos_mask"][:]       # (A,)
                if mask.any():
                    cls_ids = ct[mask].argmax(-1)
                    # use highest weight among present classes
                    return max(CLASS_WEIGHTS.get(int(c), DEFAULT_WEIGHT)
                               for c in cls_ids)
        except Exception:
            pass
        return DEFAULT_WEIGHT

    def __getitem__(self, idx):
        sp, key, src = self._index[idx]
        hf  = self._open(sp)
        grp = hf[key]

        cfg = self.cfg
        H   = cfg.dataset.camera.resize_height
        W   = cfg.dataset.camera.resize_width
        NP  = cfg.dataset.lidar.max_points

        def _get(k, default):
            return grp[k][:] if k in grp else default

        cam_raw = _get("camera",
            np.zeros((1,4,3,H,W), dtype=np.float16))
        cam   = torch.from_numpy(
            cam_raw.astype(np.float32))
        lidar = torch.from_numpy(
            _get("lidar",
                 np.zeros((1,NP,4), dtype=np.float32)))
        radar = torch.from_numpy(
            _get("radar",
                 np.zeros((1,6,13), dtype=np.float32)))
        imu   = torch.from_numpy(
            _get("imu_gps",
                 np.zeros((1,9), dtype=np.float32)))
        dt_c  = _get("dt_cam",
                     np.array([0.1],  dtype=np.float32))
        dt_l  = _get("dt_lid",
                     np.array([0.1],  dtype=np.float32))
        dt_r  = _get("dt_rad",
                     np.array([0.05], dtype=np.float32))
        dt_i  = _get("dt_imu",
                     np.array([0.02], dtype=np.float32))

        if self.augment and random.random() < 0.5:
            cam = torch.flip(cam, dims=[-1])

        delta_t = {
            "cam": torch.tensor(float(dt_c[-1])),
            "lid": torch.tensor(float(dt_l[-1])),
            "rad": torch.tensor(float(dt_r[-1])),
            "imu": torch.tensor(float(dt_i[-1])),
        }

        # labels
        is_aug = bool(grp.attrs.get("is_augmented", False))
        is_nsc = (src == "nuscenes")

        if (is_aug or is_nsc) and "gt_cls_targets" in grp:
            det = {
                "cls_targets":  torch.from_numpy(
                    grp["gt_cls_targets"][:]),
                "bbox_targets": torch.from_numpy(
                    grp["gt_bbox_targets"][:]),
                "pos_mask":     torch.from_numpy(
                    grp["gt_pos_mask"][:]),
            }
            targets = {"detection": det}
        else:
            targets = self._get_tiand_labels(idx)

        return {
            "camera":    cam[-1],
            "lidar":     lidar[-1],
            "radar":     radar[-1],
            "imu_gps":   imu[-1],
            "delta_t":   delta_t,
            "targets":   targets,
            "source":    src,
            "sample_id": key,
            "road_type": str(grp.attrs.get(
                "road_type", "urban")),
        }

    def __del__(self):
        for hf in self._cache.values():
            try: hf.close()
            except: pass


def build_joint_dataloader(cfg, split: str,
                           rank: int = 0,
                           world_size: int = 1,
                           use_weighted_sampler: bool = True,
                           ) -> DataLoader:
    """
    Build dataloader combining all data sources.
    Uses WeightedRandomSampler for class balance.
    """
    # define shard directories per split
    shard_dirs = []

    if split == "train":
        # TIAND original + augmented
        shard_dirs.append(Path(cfg.paths.shards) / "train")
        # nuScenes
        for ns_dir in [
            Path("data/shards/nuscenes_mini"),
            Path("data/shards/nuscenes_trainval"),
        ]:
            if ns_dir.exists() and any(ns_dir.glob("*.h5")):
                shard_dirs.append(ns_dir)
    else:
        # val/test — TIAND only (keep evaluation on target domain)
        shard_dirs.append(Path(cfg.paths.shards) / split)

    print(f"\nBuilding [{split}] joint dataloader:")
    dataset = JointDataset(shard_dirs, cfg,
                           augment=(split == "train"))

    # weighted sampler for class balance (train only)
    sampler = None
    if split == "train" and use_weighted_sampler:
        print("  Computing sample weights for balanced sampling ...")
        weights = [dataset.get_sample_weight(i)
                   for i in range(len(dataset))]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        print(f"  Weight range: {min(weights):.1f} – "
              f"{max(weights):.1f}")

    if world_size > 1 and sampler is None:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size,
            rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
