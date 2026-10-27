"""
Fix 1: nuScenes preprocessing pipeline.

Converts nuScenes format to our HDF5 shard format so
nuScenes + TIAND can be trained jointly.

nuScenes has:
  - 1000 scenes (700 train, 150 val, 150 test)
  - 23 object classes → mapped to our 9
  - 6 cameras, 1 LiDAR, 5 radars, IMU/GPS
  - 3D bounding box annotations
  - ~40,000 annotated frames

Run after downloading nuScenes:
  python src/preprocessing/nuscenes/preprocess_nuscenes.py \
    --dataroot data/nuscenes/mini \
    --version v1.0-mini \
    --output data/shards/nuscenes
"""
import argparse, json, sys
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.config import load_config

# ── nuScenes → our 9-class map ────────────────────────────────
# Our classes: car(0) motorcycle(1) trimotorcycle(2) scooter(3)
#              pedestrian(4) unknown1(5) bus(6) truck(7) van(8)
# Full nuScenes category name map (subcategory format)
NUSCENES_CLASS_MAP = {
    # ── vehicles ──────────────────────────────────────────────────────────────
    "vehicle.car":                          0,   # Car
    "vehicle.motorcycle":                   1,   # Two_wheeler
    "vehicle.bicycle":                      1,   # Two_wheeler
    "vehicle.truck":                        4,   # Truck
    "vehicle.trailer":                      4,   # Truck (trailer → Truck)
    "vehicle.construction":                 4,   # Truck (construction vehicle)
    "vehicle.bus.rigid":                    5,   # Bus
    "vehicle.bus.bendy":                    5,   # Bus
    "vehicle.emergency.ambulance":          0,   # Car (closest match)
    "vehicle.emergency.police":             0,   # Car (closest match)
    # ── pedestrians ───────────────────────────────────────────────────────────
    "human.pedestrian.adult":               2,   # Pedestrian
    "human.pedestrian.child":               2,
    "human.pedestrian.construction_worker": 2,
    "human.pedestrian.police_officer":      2,
    "human.pedestrian.personal_mobility":   2,
    # ── movable objects: no T-ITS equivalent → skip (cid=-1 handled upstream)
    # "movable_object.barrier":             -1,
    # "movable_object.trafficcone":         -1,
    # "movable_object.pushable_pullable":   -1,
    # "movable_object.debris":              -1,
    # "static_object.bicycle_rack":         -1,
}

SEEN_CLASSES = {0, 1, 2, 3}   # car, motorcycle, trimoto, scooter
RARE_CLASSES = {4, 5, 6, 7, 8}


def boxes_to_targets(boxes_3d, class_ids,
                     num_anchors=9, num_classes=7):
    """Convert 3D boxes to detection targets."""
    cls_t  = np.zeros((num_anchors, num_classes), dtype=np.float32)
    bbox_t = np.zeros((num_anchors, 7),           dtype=np.float32)
    mask   = np.zeros(num_anchors,                dtype=bool)
    for i, (box, cid) in enumerate(
            zip(boxes_3d[:num_anchors], class_ids[:num_anchors])):
        cls_t[i, cid] = 1.0
        bbox_t[i]     = box   # cx,cy,cz,l,w,h,yaw
        mask[i]       = True
    return cls_t, bbox_t, mask


def process_scene(nusc, scene, out_path: Path,
                  cfg, sample_token: str | None = None) -> int:
    """
    Process one nuScenes scene into one HDF5 group.
    Returns number of samples written.
    """
    H    = cfg.dataset.camera.resize_height
    W    = cfg.dataset.camera.resize_width
    MAX  = cfg.dataset.lidar.max_points
    import cv2

    # get all samples in scene
    samples = []
    token   = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token  = sample["next"]

    if not samples:
        return 0

    with h5py.File(out_path, "w") as hf:
        hf.attrs["scene_name"]   = scene["name"]
        hf.attrs["num_samples"]  = len(samples)
        hf.attrs["dataset"]      = "nuscenes"

        for si, sample in enumerate(tqdm(
                samples, desc=scene["name"], leave=False)):
            grp = hf.require_group(f"sample_{si:04d}")

            # ── cameras (nuScenes has 6, we use front 4) ──────
            cam_keys = ["CAM_FRONT","CAM_FRONT_LEFT",
                        "CAM_FRONT_RIGHT","CAM_BACK"]
            cam_arr  = np.zeros((1,4,3,H,W), dtype=np.float16)
            for ci, cam_key in enumerate(cam_keys):
                if cam_key not in sample["data"]:
                    continue
                cam_data = nusc.get("sample_data",
                                    sample["data"][cam_key])
                img_path = Path(nusc.dataroot) / cam_data["filename"]
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (W, H))
                        img = img.astype(np.float32) / 255.0
                        mean = np.array([0.485,0.456,0.406])
                        std  = np.array([0.229,0.224,0.225])
                        img  = (img - mean) / std
                        cam_arr[0,ci] = img.transpose(2,0,1).astype(np.float16)
            grp.create_dataset("camera", data=cam_arr, compression="lzf")

            # ── LiDAR ─────────────────────────────────────────
            lid_arr = np.zeros((1, MAX, 4), dtype=np.float32)
            if "LIDAR_TOP" in sample["data"]:
                lid_data = nusc.get("sample_data",
                                    sample["data"]["LIDAR_TOP"])
                lid_path = Path(nusc.dataroot) / lid_data["filename"]
                if lid_path.exists():
                    pts = np.fromfile(str(lid_path),
                                      dtype=np.float32).reshape(-1, 5)
                    pts = pts[:, :4]   # x,y,z,intensity
                    if len(pts) > MAX:
                        idx = np.random.choice(len(pts), MAX, replace=False)
                        pts = pts[idx]
                    elif len(pts) < MAX:
                        pad = np.zeros((MAX-len(pts),4), dtype=np.float32)
                        pts = np.vstack([pts, pad])
                    lid_arr[0] = pts
            grp.create_dataset("lidar", data=lid_arr, compression="lzf")

            # ── Radar (5 radars → stack first 6 channels) ─────
            rad_arr = np.zeros((1, 6, 13), dtype=np.float32)
            radar_keys = ["RADAR_FRONT","RADAR_FRONT_LEFT",
                          "RADAR_FRONT_RIGHT","RADAR_BACK_LEFT",
                          "RADAR_BACK_RIGHT"]
            for ri, rkey in enumerate(radar_keys[:6]):
                if rkey not in sample["data"]:
                    continue
                rad_data = nusc.get("sample_data",
                                    sample["data"][rkey])
                rad_path = Path(nusc.dataroot) / rad_data["filename"]
                if rad_path.exists():
                    try:
                        from nuscenes.utils.data_classes import RadarPointCloud
                        rpc = RadarPointCloud.from_file(str(rad_path))
                        pts = rpc.points.T   # (N, 18)
                        if len(pts) > 0:
                            # use first 13 channels
                            row = pts[0, :13].astype(np.float32)
                            rad_arr[0, ri, :len(row)] = row
                    except Exception:
                        pass
            grp.create_dataset("radar",   data=rad_arr)

            # ── IMU/GPS (ego pose) ─────────────────────────────
            ego  = nusc.get("ego_pose",
                            nusc.get("sample_data",
                                sample["data"]["LIDAR_TOP"]
                            )["ego_pose_token"])
            imu  = np.zeros((1, 13), dtype=np.float32)  # 9 IMU + 4 GPS (lat,lon,alt,speed) zeros
            # translation: x,y,z
            imu[0,:3] = ego["translation"]
            # rotation quaternion → euler approx
            q = ego["rotation"]   # w,x,y,z
            imu[0,6]  = float(np.arctan2(
                2*(q[0]*q[3]+q[1]*q[2]),
                1-2*(q[2]**2+q[3]**2)))   # yaw
            grp.create_dataset("imu_gps", data=imu)

            # ── delta_t (nuScenes 2Hz = 0.5s between samples) ─
            grp.create_dataset("dt_cam",  data=np.array([0.5], dtype=np.float32))
            grp.create_dataset("dt_lid",  data=np.array([0.5], dtype=np.float32))
            grp.create_dataset("dt_rad",  data=np.array([0.5], dtype=np.float32))
            grp.create_dataset("dt_imu",  data=np.array([0.5], dtype=np.float32))

            # ── 3D annotations ────────────────────────────────
            boxes_3d = []
            class_ids = []
            for ann_token in sample["anns"]:
                ann  = nusc.get("sample_annotation", ann_token)
                cat  = ann["category_name"]
                cid  = NUSCENES_CLASS_MAP.get(cat, -1)
                if cid < 0:
                    continue
                t    = ann["translation"]   # cx,cy,cz
                s    = ann["size"]          # w,l,h
                # yaw from quaternion
                q    = ann["rotation"]
                yaw  = float(np.arctan2(
                    2*(q[0]*q[3]+q[1]*q[2]),
                    1-2*(q[2]**2+q[3]**2)))
                boxes_3d.append([t[0],t[1],t[2],
                                  s[1],s[0],s[2],yaw])
                class_ids.append(cid)

            if boxes_3d:
                cls_t, bbox_t, mask = boxes_to_targets(
                    np.array(boxes_3d),
                    np.array(class_ids))
                grp.create_dataset("gt_cls_targets",
                                   data=cls_t)
                grp.create_dataset("gt_bbox_targets",
                                   data=bbox_t)
                grp.create_dataset("gt_pos_mask",
                                   data=mask)

            grp.attrs["sample_id"]  = f"nusc_{scene['name']}_{si:04d}"
            grp.attrs["road_type"]  = "urban"
            grp.attrs["num_frames"] = 1
            grp.attrs["dataset"]    = "nuscenes"
            grp.attrs["timestamp"]  = sample["timestamp"]

    return len(samples)


def run(dataroot: str, version: str, output: str,
        cfg_path: str = "configs/config.yaml") -> None:
    from nuscenes.nuscenes import NuScenes

    cfg    = load_config(cfg_path)
    out    = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes {version} from {dataroot} ...")
    nusc   = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scenes = nusc.scene
    print(f"Scenes: {len(scenes)}")

    # class distribution
    from collections import Counter
    class_counts = Counter()
    for sample in nusc.sample:
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            cat = ann["category_name"]
            cid = NUSCENES_CLASS_MAP.get(cat, -1)
            if cid >= 0:
                class_counts[cid] += 1
    print("\nClass distribution in nuScenes:")
    class_names = ["car","motorcycle","trimotorcycle","scooter",
                   "pedestrian","unknown1","bus","truck","van"]
    for cid, cnt in sorted(class_counts.items()):
        print(f"  {class_names[cid]:<18s}: {cnt:>7,}")

    # process each scene
    total_samples = 0
    for si, scene in enumerate(tqdm(scenes, desc="Scenes")):
        out_path = out / f"nusc_scene_{si:04d}.h5"
        if out_path.exists():
            print(f"  [SKIP] {scene['name']} — already processed")
            continue
        n = process_scene(nusc, scene, out_path, cfg)
        total_samples += n

    print(f"\nTotal samples processed: {total_samples}")
    print(f"Output: {out}")
    print(f"\nNext: update cfg.paths.shards to include nuScenes shards")
    print(f"      then run: python scripts/full_train_v3.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="data/nuscenes/mini")
    p.add_argument("--version",  default="v1.0-mini")
    p.add_argument("--output",   default="data/shards/nuscenes")
    p.add_argument("--config",   default="configs/config.yaml")
    args = p.parse_args()
    run(args.dataroot, args.version, args.output, args.config)
