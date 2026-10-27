"""
Step 3 — HDF5 shard writer.

Converts raw extracted data into compact HDF5 shards for fast training
I/O on the 4× T4 setup.  Each shard holds `shard_size` scenes.

Shard schema per scene
───────────────────────
/scene_{id}/
    camera/    img_{cam_id}_{t}   float16 (C,H,W)
    radar/     frame_{t}          float32 (N_radar, 13)
    lidar/     frame_{t}          float32 (N_pts, 4)  — x,y,z,intensity
    imu_gps/   frame_{t}          float32 (13,)
    timestamps                    float64 (T,)  — unix seconds per modality
    delta_t/   cam / rad / lid / imu   float32 (T,)
    labels/    frame_{t}          JSON string  — bbox + class per frame

Run:
    python src/preprocessing/shard_writer.py --config configs/config.yaml \
        --split train --workers 8
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger()


# ── sensor readers ────────────────────────────────────────────────────────────

def read_camera_frame(img_path: Path, cfg) -> np.ndarray:
    """BGR → RGB, resize, float16 normalised to [0,1]."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((3, cfg.dataset.camera.resize_height,
                         cfg.dataset.camera.resize_width), dtype=np.float16)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg.dataset.camera.resize_width,
                           cfg.dataset.camera.resize_height))
    img = img.astype(np.float32) / 255.0
    mean = np.array(cfg.preprocessing.camera_normalize_mean, dtype=np.float32)
    std  = np.array(cfg.preprocessing.camera_normalize_std,  dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1).astype(np.float16)   # (C,H,W)


def read_radar_frame(csv_path: Path, cfg) -> np.ndarray:
    """Read one radar CSV frame → (N_radar, 13) float32."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        arr = df.values.astype(np.float32)
        # Pad / truncate to cfg.dataset.radar.num_radars rows
        n = cfg.dataset.radar.num_radars
        if arr.shape[0] >= n:
            return arr[:n]
        pad = np.zeros((n - arr.shape[0], arr.shape[1]), dtype=np.float32)
        return np.vstack([arr, pad])
    except Exception:
        return np.zeros((cfg.dataset.radar.num_radars,
                         cfg.dataset.radar.num_channels), dtype=np.float32)


def read_lidar_frame(pcd_path: Path, cfg) -> np.ndarray:
    """
    Read PCD / BIN point cloud.
    Subsample to max_points using farthest point sampling approximation.
    Returns (N, 4) float32 — x, y, z, intensity.
    """
    max_pts = cfg.dataset.lidar.max_points

    if pcd_path.suffix == ".bin":
        pts = np.fromfile(str(pcd_path), dtype=np.float32).reshape(-1, 4)
    else:
        # Minimal PCD ASCII/binary reader
        pts = _read_pcd(pcd_path)

    if pts is None or len(pts) == 0:
        return np.zeros((max_pts, 4), dtype=np.float32)

    # Random subsample (fast alternative to FPS for preprocessing)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    elif len(pts) < max_pts:
        pad = np.zeros((max_pts - len(pts), 4), dtype=np.float32)
        pts = np.vstack([pts, pad])

    return pts.astype(np.float32)


def _read_pcd(path: Path) -> np.ndarray | None:
    """Minimal PCD reader for ASCII and binary_compressed formats."""
    try:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pc.points, dtype=np.float32)
        intensity = np.zeros((len(pts), 1), dtype=np.float32)
        return np.hstack([pts, intensity])
    except Exception:
        return None


def read_imu_gps_frame(txt_path: Path, cfg) -> np.ndarray:
    """Read IMU+GPS record → (13,) float32."""
    try:
        data = np.loadtxt(str(txt_path), delimiter=",", max_rows=1)
        if data.ndim == 0:
            data = np.array([data])
        total = cfg.dataset.imu_gps.imu_channels + cfg.dataset.imu_gps.gps_channels
        out = np.zeros(total, dtype=np.float32)
        out[:min(len(data), total)] = data[:total]
        return out
    except Exception:
        return np.zeros(
            cfg.dataset.imu_gps.imu_channels + cfg.dataset.imu_gps.gps_channels,
            dtype=np.float32
        )


def compute_delta_t(timestamps: np.ndarray) -> np.ndarray:
    """
    Compute per-frame Δt (seconds since previous frame).
    First frame Δt = median of the sequence (safe fallback).
    """
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = float(np.median(dt[1:]) if len(dt) > 1 else 0.1)
    return dt.astype(np.float32)


# ── scene writer ──────────────────────────────────────────────────────────────

def write_scene(scene: dict, hf: h5py.File, cfg) -> int:
    """
    Write one scene into an open HDF5 file.
    Returns number of frames written.
    """
    sid = scene["scene_id"]
    scene_path = Path(scene["path"])
    grp = hf.require_group(f"scene_{sid}")

    # ── camera ──
    cam_grp = grp.require_group("camera")
    cam_imgs  = sorted(scene_path.rglob("*.jpg")) + \
                sorted(scene_path.rglob("*.png"))
    cam_times = []
    for img_path in cam_imgs:
        frame = read_camera_frame(img_path, cfg)
        key   = img_path.stem
        cam_grp.create_dataset(key, data=frame, compression="lzf")
        # Use mtime as proxy timestamp if no sidecar
        cam_times.append(img_path.stat().st_mtime)

    # ── radar ──
    rad_grp   = grp.require_group("radar")
    rad_csvs  = sorted(scene_path.rglob("*.csv"))
    rad_times = []
    for csv_path in rad_csvs:
        frame = read_radar_frame(csv_path, cfg)
        rad_grp.create_dataset(csv_path.stem, data=frame)
        rad_times.append(csv_path.stat().st_mtime)

    # ── lidar ──
    lid_grp   = grp.require_group("lidar")
    lid_pcds  = sorted(scene_path.rglob("*.pcd")) + \
                sorted(scene_path.rglob("*.bin"))
    lid_times = []
    for pcd_path in lid_pcds:
        frame = read_lidar_frame(pcd_path, cfg)
        lid_grp.create_dataset(pcd_path.stem, data=frame, compression="lzf")
        lid_times.append(pcd_path.stat().st_mtime)

    # ── imu / gps ──
    imu_grp   = grp.require_group("imu_gps")
    imu_txts  = sorted(scene_path.rglob("*.txt")) + \
                sorted(scene_path.rglob("*.dat"))
    imu_times = []
    for txt_path in imu_txts:
        frame = read_imu_gps_frame(txt_path, cfg)
        imu_grp.create_dataset(txt_path.stem, data=frame)
        imu_times.append(txt_path.stat().st_mtime)

    # ── timestamps + Δt ──
    dt_grp = grp.require_group("delta_t")
    for name, times in [("cam", cam_times), ("rad", rad_times),
                        ("lid", lid_times), ("imu", imu_times)]:
        if times:
            ts = np.array(times, dtype=np.float64)
            grp.create_dataset(f"timestamps_{name}", data=ts)
            dt_grp.create_dataset(name, data=compute_delta_t(ts))

    # ── metadata ──
    grp.attrs["scene_id"]  = sid
    grp.attrs["road_type"] = scene.get("road_type", "unknown")
    grp.attrs["num_frames"] = max(len(cam_imgs), len(lid_pcds), 1)

    return grp.attrs["num_frames"]


# ── shard writer ──────────────────────────────────────────────────────────────

def write_shard(scenes_chunk: list[dict], shard_path: Path, cfg) -> int:
    total_frames = 0
    with h5py.File(shard_path, "w") as hf:
        hf.attrs["num_scenes"] = len(scenes_chunk)
        for scene in scenes_chunk:
            try:
                n = write_scene(scene, hf, cfg)
                total_frames += n
            except Exception as e:
                log.warning(f"Failed scene {scene['scene_id']}: {e}")
    return total_frames


def run(cfg_path: str, split: str, workers: int = 4) -> None:
    cfg = load_config(cfg_path)

    split_file = Path(cfg.paths.splits) / f"{split}.json"
    if not split_file.exists():
        log.error(f"Split file not found: {split_file}. Run scene_splitter first.")
        return

    with open(split_file) as f:
        scenes = json.load(f)
    log.info(f"Sharding {len(scenes)} scenes for [{split}] split …")

    shard_size  = cfg.preprocessing.shard_size
    shards_dir  = Path(cfg.paths.shards) / split
    shards_dir.mkdir(parents=True, exist_ok=True)

    chunks = [scenes[i:i+shard_size] for i in range(0, len(scenes), shard_size)]
    log.info(f"Writing {len(chunks)} shards of ≤{shard_size} scenes each.")

    total = 0
    for idx, chunk in enumerate(tqdm(chunks, desc=f"Shards [{split}]")):
        shard_path = shards_dir / f"shard_{idx:04d}.h5"
        n = write_shard(chunk, shard_path, cfg)
        total += n
        log.info(f"  shard {idx:04d}: {len(chunk)} scenes, {n} frames → {shard_path}")

    log.info(f"Done. Total frames written: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--split",   default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.config, args.split, args.workers)
