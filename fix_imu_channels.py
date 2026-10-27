"""
fix_imu_channels.py — Pad imu_gps to 13 channels (9 IMU + 4 GPS zeros)
so it matches IMUGPSEncoder(in_dim=13).

Run with:
    python fix_imu_channels.py
Then rebuild shards and re-split.
"""
import sys
from pathlib import Path

PREPROC = next(Path("/home/sharon/vision_fusion").rglob("preprocess_nuscenes.py"), None)
if not PREPROC:
    print("ERROR: preprocess_nuscenes.py not found", file=sys.stderr)
    sys.exit(1)

src = PREPROC.read_text()

OLD = 'imu  = np.zeros((1, 9), dtype=np.float32)'
NEW = 'imu  = np.zeros((1, 13), dtype=np.float32)  # 9 IMU + 4 GPS (lat,lon,alt,speed) zeros'

if OLD not in src:
    print("ERROR: anchor not found — already patched?", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
PREPROC.write_text(src)
print(f"Patched: {PREPROC}")
print("imu_gps now written as (1, 13) — 4 GPS channels zero-padded")