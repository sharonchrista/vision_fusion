"""
HDF5ShardDataset — PyTorch Dataset over pre-built shards.

Each __getitem__ returns a dict:
{
  "camera":   Tensor (4, 3, H, W)       — 4 camera frames
  "radar":    Tensor (6, 13)            — 6 radar detections
  "lidar":    Tensor (N_pts, 4)         — x,y,z,intensity
  "imu_gps":  Tensor (13,)
  "delta_t":  dict[str, Tensor]         — Δt per modality
  "labels":   dict                      — boxes + classes
  "scene_id": str
  "road_type": str
}
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import load_config
from src.utils.reproducibility import worker_init_fn



# BEV coordinate normalisation constants (computed from nuscenes_mini_labeled)
_BBOX_MEAN  = [1500.0, 1197.0, 1.5,  3.0,  1.5,  1.8,  0.0]
_BBOX_SCALE = [ 100.0,  100.0, 3.0, 10.0, 10.0,  3.0,  3.14]

def _norm_bbox(t):
    """Normalise raw BEV bbox targets to ~[-1, 1] range for stable regression."""
    import torch
    mean  = torch.tensor(_BBOX_MEAN,  dtype=t.dtype)
    scale = torch.tensor(_BBOX_SCALE, dtype=t.dtype)
    return (t - mean) / scale


class HDF5ShardDataset(Dataset):

    def __init__(self, shards_dir: str | Path,
                 cfg,
                 augment: bool = False):
        self.cfg      = cfg
        self.augment  = augment
        self.shard_paths = sorted(Path(shards_dir).glob("*.h5"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shards_dir}")

        # Build flat index: [(shard_path, scene_key), ...]
        self._index: list[tuple[Path, str]] = []
        for sp in self.shard_paths:
            with h5py.File(sp, "r") as hf:
                for key in hf.keys():
                    self._index.append((sp, key))

        self._file_cache: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _open(self, shard_path: Path) -> h5py.File:
        key = str(shard_path)
        if key not in self._file_cache:
            self._file_cache[key] = h5py.File(shard_path, "r")
        return self._file_cache[key]

    # ── per-modality loaders ──────────────────────────────────────────────

    def _load_camera(self, grp: h5py.Group) -> torch.Tensor:
        """Load up to 4 camera frames; pad with zeros if fewer available."""
        cam_grp = grp.get("camera", None)
        num_cam = self.cfg.dataset.camera.num_cameras
        H = self.cfg.dataset.camera.resize_height
        W = self.cfg.dataset.camera.resize_width
        out = np.zeros((num_cam, 3, H, W), dtype=np.float32)
        if cam_grp is None:
            return torch.from_numpy(out)
        # camera is stored as a flat Dataset with shape (N, 3, H, W)
        arr = cam_grp[:]
        if arr.ndim == 5: arr = arr[0]  # (1,N,3,H,W) -> (N,3,H,W)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        n = min(arr.shape[0], num_cam)
        out[:n] = arr[:n]
        return torch.from_numpy(out)

    def _load_radar(self, grp: h5py.Group) -> torch.Tensor:
        rad_grp = grp.get("radar", None)
        n = self.cfg.dataset.radar.num_radars
        c = self.cfg.dataset.radar.num_channels
        out = np.zeros((n, c), dtype=np.float32)
        if rad_grp is None:
            return torch.from_numpy(out)
        arr = rad_grp[:]
        if arr.ndim == 3: arr = arr[0]  # (1,N,C)->(N,C)
        out[:min(n, arr.shape[0])] = arr[:n]
        return torch.from_numpy(out)

    def _load_lidar(self, grp: h5py.Group) -> torch.Tensor:
        lid_grp = grp.get("lidar", None)
        max_pts = self.cfg.dataset.lidar.max_points
        out = np.zeros((max_pts, 4), dtype=np.float32)
        if lid_grp is None:
            return torch.from_numpy(out)
        arr = lid_grp[:]
        if arr.ndim == 3: arr = arr[0]  # (1,N,4)->(N,4)
        out[:min(max_pts, arr.shape[0])] = arr[:max_pts]
        return torch.from_numpy(out)

    def _load_imu_gps(self, grp: h5py.Group) -> torch.Tensor:
        imu_grp = grp.get("imu_gps", None)
        total = (self.cfg.dataset.imu_gps.imu_channels +
                 self.cfg.dataset.imu_gps.gps_channels)
        out = np.zeros(total, dtype=np.float32)
        if imu_grp is None:
            return torch.from_numpy(out)
        arr = imu_grp[:]
        if arr.ndim == 2: arr = arr[0]  # (1,C)->(C,)
        out[:min(total, arr.shape[0])] = arr[:total]
        return torch.from_numpy(out)

    def _load_delta_t(self, grp: h5py.Group) -> dict[str, torch.Tensor]:
        dt_grp = grp.get("delta_t", None)
        out = {}
        fallback = torch.tensor([0.1], dtype=torch.float32)
        for mod in ["cam", "rad", "lid", "imu"]:
            if dt_grp is not None and mod in dt_grp:
                arr = dt_grp[mod][:]
                # Take last value (most recent Δt)
                out[mod] = torch.tensor(float(arr[-1]), dtype=torch.float32)
            else:
                out[mod] = fallback.clone()
        return out

    # ── augmentation ─────────────────────────────────────────────────────

    def _augment_camera(self, cam: torch.Tensor) -> torch.Tensor:
        """Random horizontal flip for all cameras consistently."""
        if random.random() < 0.5:
            cam = torch.flip(cam, dims=[-1])
        return cam

    def _augment_lidar(self, pts: torch.Tensor) -> torch.Tensor:
        """Random Z-axis rotation ± 15°."""
        angle = random.uniform(-0.26, 0.26)   # radians
        c, s  = np.cos(angle), np.sin(angle)
        R = torch.tensor([[c, -s, 0, 0],
                          [s,  c, 0, 0],
                          [0,  0, 1, 0],
                          [0,  0, 0, 1]], dtype=torch.float32)
        return (pts @ R.T)

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        shard_path, scene_key = self._index[idx]
        hf  = self._open(shard_path)
        grp = hf[scene_key]

        cam   = self._load_camera(grp)
        radar = self._load_radar(grp)
        lidar = self._load_lidar(grp)
        imu   = self._load_imu_gps(grp)
        dt    = self._load_delta_t(grp)

        if self.augment:
            cam   = self._augment_camera(cam)
            lidar = self._augment_lidar(lidar)

        # ── Ground-truth labels (present in nuScenes shards) ─────────────────
        targets = {}
        if "gt_cls_targets" in grp:
            import torch as _torch
            targets["detection"] = {
                "cls_targets":  _torch.from_numpy(grp["gt_cls_targets"][:]),
                "bbox_targets": _torch.from_numpy(grp["gt_bbox_targets"][:]).float(),
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
        return batch, targets

    def __del__(self):
        for hf in self._file_cache.values():
            try:
                hf.close()
            except Exception:
                pass


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloader(cfg, split: str, rank: int = 0,
                     world_size: int = 1) -> DataLoader:
    shards_dir = Path(cfg.paths.shards) / split
    augment    = (split == "train")

    dataset = HDF5ShardDataset(shards_dir, cfg, augment=augment)

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=(split == "train"))

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
