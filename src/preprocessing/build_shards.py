"""
Build HDF5 shards from the synchronized frame index.
Each sample = one sliding window of 30 frames.
"""
import json, sys, warnings
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ── radar CSV reader ──────────────────────────────────────────────────────────
def read_radar_csv(csv_path: Path, n_channels: int = 13) -> np.ndarray:
    """
    Radar CSVs have no header row — first row IS data.
    Shape on disk: (12 rows × N_objects cols) transposed.
    We take first n_channels rows, first 6 cols (one per radar direction).
    Returns (6, 13) float32.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, header=0)
        arr = df.values.astype(np.float32)           # (rows, cols)
        # take up to 13 rows (channels) x 6 cols (objects/detections)
        arr = arr[:n_channels, :6] if arr.shape[0] >= n_channels else \
              np.pad(arr[:, :6], ((0, n_channels-arr.shape[0]), (0,0)))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.T.astype(np.float32)              # (6, 13)
    except Exception:
        return np.zeros((6, 13), dtype=np.float32)

# ── LiDAR PCD reader ──────────────────────────────────────────────────────────
def read_pcd(pcd_path: Path, max_pts: int = 16384) -> np.ndarray:
    """
    Parse PCD ASCII/binary. Returns (max_pts, 4) float32: x,y,z,intensity.
    """
    try:
        pts = []
        with open(pcd_path, "rb") as f:
            header_done = False
            data_type   = "ascii"
            fields      = []
            for line in f:
                line = line.decode("utf-8", errors="ignore").strip()
                if line.startswith("FIELDS"):
                    fields = line.split()[1:]
                if line.startswith("DATA"):
                    data_type = line.split()[1]
                    header_done = True
                    break
            if not header_done:
                return np.zeros((max_pts, 4), dtype=np.float32)
            if data_type == "ascii":
                for line in f:
                    vals = line.decode("utf-8", errors="ignore").strip().split()
                    if vals:
                        pts.append([float(v) for v in vals[:4]])
            else:
                # binary — read as float32 blocks
                raw  = np.frombuffer(f.read(), dtype=np.float32)
                ncol = max(len(fields), 4)
                if len(raw) % ncol == 0:
                    pts = raw.reshape(-1, ncol)[:, :4].tolist()

        arr = np.array(pts, dtype=np.float32) if pts else np.zeros((1,4), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # pad / subsample to max_pts
        if len(arr) >= max_pts:
            idx = np.random.choice(len(arr), max_pts, replace=False)
            arr = arr[idx]
        else:
            pad = np.zeros((max_pts - len(arr), 4), dtype=np.float32)
            arr = np.vstack([arr, pad])
        return arr.astype(np.float32)
    except Exception:
        return np.zeros((max_pts, 4), dtype=np.float32)

# ── camera reader ─────────────────────────────────────────────────────────────
def read_camera(img_path: str, H: int = 480, W: int = 640) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((3, H, W), dtype=np.float16)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return img.transpose(2, 0, 1).astype(np.float16)   # (3,H,W)

# ── write one sample ──────────────────────────────────────────────────────────
def write_sample(sample: dict, grp: h5py.Group, cfg) -> None:
    T        = len(sample["frames"])
    H        = cfg.dataset.camera.resize_height
    W        = cfg.dataset.camera.resize_width
    max_pts  = cfg.dataset.lidar.max_points

    cam_dirs = ["Front", "Front_Left", "Front_Right", "Back"]

    # allocate arrays for the window
    cam_arr  = np.zeros((T, 4, 3, H, W),   dtype=np.float16)
    rad_arr  = np.zeros((T, 6, 13),         dtype=np.float32)
    lid_arr  = np.zeros((T, max_pts, 4),    dtype=np.float32)
    imu_arr  = np.zeros((T, 9),             dtype=np.float32)
    dt_cam   = np.zeros(T,                  dtype=np.float32)
    dt_lid   = np.zeros(T,                  dtype=np.float32)
    dt_rad   = np.zeros(T,                  dtype=np.float32)
    dt_imu   = np.zeros(T,                  dtype=np.float32)

    rad_dirs = ["RADAR_Front","RADAR_Front_Left","RADAR_Front_Right",
                "RADAR_Back","RADAR_Back_Left","RADAR_Back_Right"]

    for t, frame in enumerate(sample["frames"]):
        # cameras
        for ci, direction in enumerate(cam_dirs):
            fpath = frame["cam_files"].get(direction, "")
            if fpath and Path(fpath).exists():
                cam_arr[t, ci] = read_camera(fpath, H, W)

        # radar (one csv per direction per frame)
        for ri, rd in enumerate(rad_dirs):
            fpath = frame["radar_files"].get(rd, "")
            if fpath and Path(fpath).exists():
                r = read_radar_csv(Path(fpath))  # (6,13)
                rad_arr[t, ri] = r[0] if r.shape[0] > 0 else np.zeros(13, dtype=np.float32)

        # lidar
        lpath = Path(frame["lidar_file"])
        if lpath.exists():
            lid_arr[t] = read_pcd(lpath, max_pts)

        # GNSS (9 channels: lat,lon,h, vn,ve,vu, roll,pitch,az)
        gnss = frame.get("gnss", [0.0]*9)
        imu_arr[t] = np.array(gnss[:9], dtype=np.float32)

        # delta_t
        dt_cam[t] = frame.get("delta_t_cam", 0.1)
        dt_lid[t] = frame.get("delta_t_lid", 0.1)
        dt_rad[t] = frame.get("delta_t_rad", 0.05)
        dt_imu[t] = frame.get("delta_t_imu", 0.02)

    # write to HDF5
    grp.create_dataset("camera",   data=cam_arr,  compression="lzf")
    grp.create_dataset("lidar",    data=lid_arr,  compression="lzf")
    grp.create_dataset("radar",    data=rad_arr)
    grp.create_dataset("imu_gps",  data=imu_arr)
    grp.create_dataset("dt_cam",   data=dt_cam)
    grp.create_dataset("dt_lid",   data=dt_lid)
    grp.create_dataset("dt_rad",   data=dt_rad)
    grp.create_dataset("dt_imu",   data=dt_imu)
    grp.attrs["sample_id"]  = sample["sample_id"]
    grp.attrs["road_type"]  = sample.get("road_type", "urban")
    grp.attrs["num_frames"] = T

# ── main ──────────────────────────────────────────────────────────────────────
def run(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    np.random.seed(cfg.project.seed)

    for split in ["train", "val", "test"]:
        split_file = Path(cfg.paths.splits) / f"{split}.json"
        if not split_file.exists():
            print(f"[SKIP] {split_file} not found")
            continue

        with open(split_file) as f:
            samples = json.load(f)

        out_dir = Path(cfg.paths.shards) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_size = cfg.preprocessing.shard_size  # 256 — but we have <200 samples total

        chunks = [samples[i:i+shard_size]
                  for i in range(0, len(samples), shard_size)]

        print(f"\n[{split.upper()}] {len(samples)} samples -> "
              f"{len(chunks)} shard(s)")

        for ci, chunk in enumerate(chunks):
            shard_path = out_dir / f"shard_{ci:04d}.h5"
            print(f"  Writing {shard_path.name} ({len(chunk)} samples) ...")
            with h5py.File(shard_path, "w") as hf:
                hf.attrs["split"]       = split
                hf.attrs["num_samples"] = len(chunk)
                for sample in tqdm(chunk, desc=f"  {split}/shard_{ci:04d}",
                                   unit="sample"):
                    grp = hf.require_group(sample["sample_id"])
                    write_sample(sample, grp, cfg)
            size_mb = shard_path.stat().st_size / 1e6
            print(f"  -> {shard_path.name}  {size_mb:.1f} MB")

    # final check
    print("\n=== SHARD SUMMARY ===")
    for split in ["train","val","test"]:
        shards = sorted((Path(cfg.paths.shards)/split).glob("*.h5"))
        total_mb = sum(s.stat().st_size for s in shards) / 1e6
        print(f"  [{split:5s}] {len(shards)} shard(s)  {total_mb:.1f} MB total")
        for s in shards:
            with h5py.File(s,"r") as hf:
                print(f"    {s.name}  samples={hf.attrs['num_samples']}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    run(p.parse_args().config)
