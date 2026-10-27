"""
Build a synchronized frame index from the multimodal dataset.

The dataset is one continuous ~3-min drive. We:
1. Parse the LiDAR-Camera sync CSV (1840 paired frames — master clock)
2. Match closest radar frame per camera timestamp
3. Match closest GNSS row per camera timestamp
4. Slice into windows of N frames with stride S
5. Save index as JSON -> data/splits/
"""
import json, sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

MM   = Path("data/extracted/multimodal/TIAND_MULTIMODAL_DATASET")
CAM  = MM / "CAMERA/CAMERA"
LID  = MM / "LIDAR/LIDAR"
RAD  = MM / "RADAR/RADAR"
GNSS = MM / "GNSS/GNSS/scene3-novatel-oem7-inspva.csv"
SYNC_LC  = MM / "Synchronisation_timestamps/Synchronisation_timestamps/Lidar_Camera_synchronisation.csv"
SYNC_RC  = MM / "Synchronisation_timestamps/Synchronisation_timestamps/Radar_Camera_Synchronization.csv"

WINDOW   = 30   # frames per sample (3s at ~10Hz)
STRIDE   = 10   # stride between windows
SPLITS   = [0.70, 0.15, 0.15]

def ts_to_sec(ts: str) -> float:
    """'13:19:16.491248' -> seconds since midnight"""
    h, m, rest = ts.strip().split(":")
    return int(h)*3600 + int(m)*60 + float(rest)

def run(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)

    # ── 1. LiDAR-Camera sync (master clock) ──────────────────
    df_lc = pd.read_csv(SYNC_LC)
    df_lc.columns = ["lidar_ts", "cam_ts"]
    df_lc["lidar_sec"] = df_lc["lidar_ts"].apply(ts_to_sec)
    df_lc["cam_sec"]   = df_lc["cam_ts"].apply(ts_to_sec)
    print(f"LiDAR-Camera pairs: {len(df_lc)}")

    # ── 2. Radar-Camera sync ──────────────────────────────────
    df_rc = pd.read_csv(SYNC_RC, sep="\t")
    df_rc.columns = ["cam_ts", "radar_ts"]
    df_rc["cam_sec"]   = df_rc["cam_ts"].apply(ts_to_sec)
    df_rc["radar_sec"] = df_rc["radar_ts"].apply(ts_to_sec)
    # Build lookup: cam_sec -> closest radar_ts string
    radar_lookup = dict(zip(df_rc["cam_sec"], df_rc["radar_ts"]))
    radar_secs   = np.array(df_rc["cam_sec"].values)
    print(f"Radar-Camera pairs: {len(df_rc)}")

    # ── 3. GNSS lookup ────────────────────────────────────────
    df_gps = pd.read_csv(GNSS)
    # parse time column
    df_gps["sec"] = df_gps["time"].apply(
        lambda t: ts_to_sec(t.split("/")[-1]) if "/" in str(t) else 0.0
    )
    gps_secs = df_gps["sec"].values
    gps_cols = [".latitude",".longitude",".height",
                ".north_velocity",".east_velocity",".up_velocity",
                ".roll",".pitch",".azimuth"]
    gps_arr  = df_gps[gps_cols].values.astype(np.float32)  # (N, 9)
    print(f"GNSS rows: {len(df_gps)}")

    # ── 4. Camera files per direction ─────────────────────────
    cam_dirs = {
        "Front":       sorted((CAM/"CAMERA_Front").glob("*.jpg")),
        "Front_Left":  sorted((CAM/"CAMERA_Front_Left").glob("*.jpg")),
        "Front_Right": sorted((CAM/"CAMERA_Front_Right").glob("*.jpg")),
        "Back":        sorted((CAM/"CAMERA_Back").glob("*.jpg")),
    }
    for k, v in cam_dirs.items():
        print(f"  Camera {k}: {len(v)} frames")

    # ── 5. Build per-frame index aligned to LiDAR-Camera sync ─
    frames = []
    for i, row in df_lc.iterrows():
        cam_sec   = row["cam_sec"]
        lidar_ts  = row["lidar_ts"].strip()
        cam_ts    = row["cam_ts"].strip()

        # LiDAR file (timestamp in filename uses colons)
        lidar_file = LID / f"{lidar_ts}.pcd"
        if not lidar_file.exists():
            # try dot format
            lidar_ts_dot = lidar_ts.replace(":", ".")  # shouldn't be needed but safe
            lidar_file = LID / f"{lidar_ts_dot}.pcd"

        # Camera files — match by timestamp string in filename
        cam_files = {}
        for direction, files in cam_dirs.items():
            # filenames like 13-19-16.516270.jpg  (hyphens for h:m, dot for s)
            cam_ts_hyph = cam_ts[:5].replace(":", "-") + cam_ts[5:]
            match = [f for f in files if cam_ts_hyph in f.name]
            if not match:
                # find nearest by sorted position
                idx = min(i, len(files)-1)
                match = [files[idx]]
            cam_files[direction] = str(match[0])

        # Closest radar timestamp
        diffs = np.abs(radar_secs - cam_sec)
        best_rad_idx = int(np.argmin(diffs))
        radar_ts = df_rc.iloc[best_rad_idx]["radar_ts"].strip()
        # radar files per direction (use closest filename)
        radar_files = {}
        for rad_dir in ["RADAR_Front","RADAR_Front_Left","RADAR_Front_Right",
                        "RADAR_Back","RADAR_Back_Left","RADAR_Back_Right"]:
            rdir = RAD / rad_dir
            if rdir.exists():
                csvs = sorted(rdir.glob("*.csv"))
                idx  = min(best_rad_idx, len(csvs)-1)
                radar_files[rad_dir] = str(csvs[idx])

        # Closest GNSS row
        gps_diffs = np.abs(gps_secs - cam_sec)
        gps_idx   = int(np.argmin(gps_diffs))
        gps_row   = gps_arr[gps_idx].tolist()

        frames.append({
            "frame_idx":   i,
            "cam_sec":     float(cam_sec),
            "lidar_file":  str(lidar_file),
            "cam_files":   cam_files,
            "radar_files": radar_files,
            "gnss":        gps_row,
            "delta_t_lid": float(row["lidar_sec"] - (frames[-1]["cam_sec"] if frames else row["cam_sec"])),
            "delta_t_cam": 0.1,
            "delta_t_rad": 0.05,
            "delta_t_imu": 0.02,
        })

    print(f"\nTotal synchronized frames: {len(frames)}")

    # ── 6. Sliding window -> samples ──────────────────────────
    samples = []
    for start in range(0, len(frames) - WINDOW + 1, STRIDE):
        window = frames[start:start+WINDOW]
        samples.append({
            "sample_id":  f"sample_{start:05d}",
            "start_frame": start,
            "end_frame":   start + WINDOW - 1,
            "frames":      window,
            "road_type":   "urban",   # Hyderabad
        })
    print(f"Sliding windows (size={WINDOW}, stride={STRIDE}): {len(samples)}")

    # ── 7. Train / val / test split ───────────────────────────
    n = len(samples)
    n_train = int(n * SPLITS[0])
    n_val   = int(n * SPLITS[1])
    splits  = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train+n_val],
        "test":  samples[n_train+n_val:],
    }

    out_dir = Path(cfg.paths.splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        out = out_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [{name:5s}] {len(data):4d} samples -> {out}")

    print("\nDone. Next step: python src/preprocessing/build_shards.py")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    run(p.parse_args().config)
