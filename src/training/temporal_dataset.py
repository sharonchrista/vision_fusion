"""
Fix 4: Full temporal window training.

Instead of using only the last frame of each 30-frame window,
we feed the entire sequence to the LTC cells sequentially.
This properly exploits the continuous-time temporal dynamics.

Each batch item:
  camera   (T, 4, 3, H, W)  — full sequence
  lidar    (T, N, 4)
  radar    (T, 6, 13)
  imu_gps  (T, 9)
  delta_t  (T,) per modality — actual elapsed times
  targets  — from last frame (prediction target)
"""
from __future__ import annotations
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.reproducibility import worker_init_fn
from src.preprocessing.parse_labels import (
    parse_camera_label, parse_lidar_label,
    boxes_to_heatmap, NUM_CLASSES
)

_CAM_LABEL_DIR  = Path("data/extracted/camera/DATASET")
_LID_LABEL_DIR4 = Path("data/extracted/lidar/2nd_september/Scene4/label")
_LID_LABEL_DIR5 = Path("data/extracted/lidar/2nd_september/Scene5/label")

def _build_cam_label_index():
    return {p.stem: p for p in _CAM_LABEL_DIR.glob("*.txt")} \
           if _CAM_LABEL_DIR.exists() else {}

def _build_lid_label_index():
    files = []
    for d in [_LID_LABEL_DIR4, _LID_LABEL_DIR5]:
        if d.exists():
            files += sorted(d.glob("*.json"))
    return files

_CAM_LABEL_IDX = _build_cam_label_index()
_LID_LABEL_IDX = _build_lid_label_index()


class TemporalShardDataset(Dataset):
    """
    Returns full temporal windows for LTC sequential processing.
    The LTC hidden state is propagated across T timesteps per sample.
    """

    def __init__(self, shards_dir: str | Path,
                 cfg, augment: bool = False,
                 window_size: int = 30):
        self.cfg         = cfg
        self.augment     = augment
        self.window_size = window_size
        self.shard_paths = sorted(Path(shards_dir).glob("shard_*.h5"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards in {shards_dir}")

        self._index: list[tuple[Path, str]] = []
        for sp in self.shard_paths:
            with h5py.File(sp, "r") as hf:
                for key in hf.keys():
                    self._index.append((sp, key))

        self._cache: dict[str, h5py.File] = {}

    def __len__(self):
        return len(self._index)

    def _open(self, sp):
        k = str(sp)
        if k not in self._cache:
            self._cache[k] = h5py.File(sp, "r")
        return self._cache[k]

    def _get_labels(self, sample_idx):
        det_targets = {"cls_targets": None,
                       "bbox_targets": None,
                       "pos_mask": None}
        cam_keys = sorted(_CAM_LABEL_IDX.keys())
        if cam_keys:
            key  = cam_keys[sample_idx % len(cam_keys)]
            path = _CAM_LABEL_IDX[key]
            cam_gt = parse_camera_label(path)
            if len(cam_gt["boxes"]) > 0:
                tgt = boxes_to_heatmap(cam_gt["boxes"],
                                       cam_gt["labels"],
                                       num_anchors=9)
                det_targets = {k: torch.from_numpy(v)
                               for k, v in tgt.items()}
        if _LID_LABEL_IDX:
            path   = _LID_LABEL_IDX[sample_idx % len(_LID_LABEL_IDX)]
            lid_gt = parse_lidar_label(path)
            if len(lid_gt["boxes"]) > 0:
                tgt = boxes_to_heatmap(lid_gt["boxes"],
                                       lid_gt["labels"],
                                       num_anchors=9)
                det_targets = {k: torch.from_numpy(v)
                               for k, v in tgt.items()}
        return {"detection": det_targets}

    def __getitem__(self, idx):
        sp, key = self._index[idx]
        hf  = self._open(sp)
        grp = hf[key]

        cfg = self.cfg
        H   = cfg.dataset.camera.resize_height
        W   = cfg.dataset.camera.resize_width
        NP  = cfg.dataset.lidar.max_points

        def _get(k, default):
            return grp[k][:] if k in grp else default

        # ── full temporal sequences ────────────────────────────
        cam_raw = _get("camera",
                       np.zeros((1,4,3,H,W), dtype=np.float16))
        cam   = torch.from_numpy(cam_raw.astype(np.float32))   # (T,4,3,H,W)
        lidar = torch.from_numpy(
            _get("lidar",   np.zeros((1,NP,4),  dtype=np.float32)))
        radar = torch.from_numpy(
            _get("radar",   np.zeros((1,6,13),  dtype=np.float32)))
        imu   = torch.from_numpy(
            _get("imu_gps", np.zeros((1,9),     dtype=np.float32)))
        dt_c  = _get("dt_cam", np.array([0.1],  dtype=np.float32))
        dt_l  = _get("dt_lid", np.array([0.1],  dtype=np.float32))
        dt_r  = _get("dt_rad", np.array([0.05], dtype=np.float32))
        dt_i  = _get("dt_imu", np.array([0.02], dtype=np.float32))

        T = cam.shape[0]

        # augmentation
        if self.augment and random.random() < 0.5:
            cam = torch.flip(cam, dims=[-1])

        # delta_t sequences (T,) for LTC
        delta_t_seq = {
            "cam": torch.from_numpy(dt_c.astype(np.float32)),
            "lid": torch.from_numpy(dt_l.astype(np.float32)),
            "rad": torch.from_numpy(dt_r.astype(np.float32)),
            "imu": torch.from_numpy(dt_i.astype(np.float32)),
        }
        # scalar delta_t for single-step ops
        delta_t = {
            k: torch.tensor(float(v[-1]), dtype=torch.float32)
            for k, v in delta_t_seq.items()
        }

        # labels (from last frame = prediction target)
        is_aug = bool(grp.attrs.get("is_augmented", False))
        if is_aug and "gt_cls_targets" in grp:
            det_targets = {
                "cls_targets":  torch.from_numpy(
                    grp["gt_cls_targets"][:]),
                "bbox_targets": torch.from_numpy(
                    grp["gt_bbox_targets"][:]),
                "pos_mask":     torch.from_numpy(
                    grp["gt_pos_mask"][:]),
            }
            targets = {"detection": det_targets}
        else:
            sample_idx = int(key.split("_")[-1]) \
                         if "_" in key else idx
            targets    = self._get_labels(sample_idx)

        return {
            # full sequences for temporal LTC training
            "camera_seq":   cam,          # (T,4,3,H,W)
            "lidar_seq":    lidar,        # (T,N,4)
            "radar_seq":    radar,        # (T,6,13)
            "imu_seq":      imu,          # (T,9)
            "delta_t_seq":  delta_t_seq,  # dict of (T,)
            # last frame for single-step inference compatibility
            "camera":       cam[-1],      # (4,3,H,W)
            "lidar":        lidar[-1],    # (N,4)
            "radar":        radar[-1],    # (6,13)
            "imu_gps":      imu[-1],      # (9,)
            "delta_t":      delta_t,      # dict of scalar
            "targets":      targets,
            "sample_id":    key,
            "road_type":    str(grp.attrs.get("road_type","urban")),
        }

    def __del__(self):
        for hf in self._cache.values():
            try: hf.close()
            except: pass


def build_temporal_dataloader(cfg, split: str,
                              rank: int = 0,
                              world_size: int = 1) -> DataLoader:
    shards_dir = Path(cfg.paths.shards) / split
    augment    = (split == "train")
    dataset    = TemporalShardDataset(shards_dir, cfg,
                                      augment=augment)
    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size,
            rank=rank, shuffle=(split=="train"))

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
