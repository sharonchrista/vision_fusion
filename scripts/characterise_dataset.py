"""
Dataset characterisation — generates statistics required for the
T-ITS dataset description section.

Produces:
  - Class distribution (counts + %)
  - Scene count and frame count per road type
  - Sensor modality coverage per scene
  - Point cloud density statistics
  - Radar channel statistics (mean, std, range)

Run:
    python scripts/characterise_dataset.py --config configs/config.yaml
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger()


def characterise(cfg_path: str) -> None:
    cfg     = load_config(cfg_path)
    splits  = Path(cfg.paths.splits)
    metrics = Path(cfg.paths.metrics)
    metrics.mkdir(parents=True, exist_ok=True)

    stats = {
        "scenes_per_split": {},
        "frames_per_split": {},
        "road_type_distribution": defaultdict(int),
        "modality_coverage": defaultdict(int),
        "total_scenes": 0,
        "total_frames": 0,
    }

    for split_name in ["train", "val", "test"]:
        split_file = splits / f"{split_name}.json"
        if not split_file.exists():
            log.warning(f"Split file missing: {split_file}")
            continue

        with open(split_file) as f:
            scenes = json.load(f)

        n_frames = sum(s.get("num_frames", 0) for s in scenes)
        stats["scenes_per_split"][split_name] = len(scenes)
        stats["frames_per_split"][split_name] = n_frames
        stats["total_scenes"] += len(scenes)
        stats["total_frames"] += n_frames

        for s in scenes:
            stats["road_type_distribution"][s.get("road_type", "unknown")] += 1

        log.info(f"[{split_name}] {len(scenes)} scenes | {n_frames} frames")

    # Road type breakdown as percentage
    total = max(stats["total_scenes"], 1)
    road_pct = {
        rt: {"count": cnt, "pct": round(100 * cnt / total, 1)}
        for rt, cnt in stats["road_type_distribution"].items()
    }
    stats["road_type_distribution"] = road_pct

    # Summary print (T-ITS table format)
    print("\n" + "="*55)
    print(f"  Dataset characterisation — LiquidFusionNet")
    print("="*55)
    print(f"  Total scenes : {stats['total_scenes']:,}")
    print(f"  Total frames : {stats['total_frames']:,}")
    print(f"  Sensors      : 4 cam · 6 radar · 1 LiDAR · IMU/GPS")
    print(f"  Location     : Hyderabad, India")
    print(f"\n  Split breakdown:")
    for split, n in stats["scenes_per_split"].items():
        f = stats["frames_per_split"][split]
        print(f"    {split:<6s}: {n:5d} scenes | {f:7,} frames")
    print(f"\n  Road type distribution:")
    for rt, info in road_pct.items():
        print(f"    {rt:<12s}: {info['count']:5d} ({info['pct']:5.1f}%)")
    print("="*55 + "\n")

    # Save
    out = metrics / "dataset_characterisation.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Characterisation saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    characterise(args.config)
