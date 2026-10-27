"""
Fix 2: Scene-aware diffusion augmentation.

Instead of pasting synthetic objects into empty scenes,
we paste them into REAL TIAND LiDAR scans at valid positions.

Improvements over v1:
  - Real scene background (road, buildings, other vehicles)
  - Depth-consistent placement (object placed at valid range)
  - Occlusion handling (remove background points inside object bbox)
  - 500 samples per rare class (vs 50 before)
  - Balanced sampling during training
"""
import json, sys, gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RARE_CLASSES  = ["pedestrian", "bus", "truck", "van"]
N_PER_CLASS   = 500

SIZE_PRIORS = {
    "pedestrian": (0.5,  0.5,  1.75),
    "bus":        (12.0, 2.5,  3.5),
    "truck":      (8.0,  2.5,  3.2),
    "van":        (5.0,  1.9,  1.8),
}
POINT_COUNTS = {
    "pedestrian": 512,
    "bus":        2048,
    "truck":      2048,
    "van":        1024,
}

# Valid placement ranges per class (distance from sensor in metres)
PLACEMENT = {
    "pedestrian": (3,  30),
    "bus":        (10, 60),
    "truck":      (10, 50),
    "van":        (5,  40),
}


def load_real_pcd(pcd_path: Path, max_pts: int = 16384) -> np.ndarray:
    """Load real LiDAR scan. Returns (N, 4) x,y,z,intensity."""
    try:
        pts = []
        with open(pcd_path, "rb") as f:
            data_type = "ascii"
            n_cols    = 4
            for line in f:
                line = line.decode("utf-8", errors="ignore").strip()
                if line.startswith("FIELDS"):
                    n_cols = len(line.split()) - 1
                if line.startswith("DATA"):
                    data_type = line.split()[1]
                    break
            if data_type == "ascii":
                for line in f:
                    vals = line.decode("utf-8", errors="ignore").strip().split()
                    if vals:
                        pts.append([float(v) for v in vals[:4]])
            else:
                raw = np.frombuffer(f.read(), dtype=np.float32)
                if len(raw) % n_cols == 0:
                    pts = raw.reshape(-1, n_cols)[:, :4].tolist()

        arr = np.array(pts, dtype=np.float32) if pts else \
              np.zeros((100, 4), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 4:
            arr = np.hstack([arr,
                np.zeros((len(arr), 4-arr.shape[1]), dtype=np.float32)])
        # subsample
        if len(arr) > max_pts:
            idx = np.random.choice(len(arr), max_pts, replace=False)
            arr = arr[idx]
        return arr.astype(np.float32)
    except Exception:
        return np.zeros((100, 4), dtype=np.float32)


def generate_object_pcd(class_name: str, n_points: int,
                         seed: int = 0,
                         use_pointe: bool = False) -> np.ndarray:
    """Generate object point cloud — Point-E for pedestrian, geometric for vehicles."""
    if use_pointe and class_name == "pedestrian":
        try:
            return _pointe_generate(class_name, n_points)
        except Exception as e:
            print(f"    Point-E failed: {e} — using geometric")

    # Geometric primitive (fast, effective for vehicles)
    l, w, h = SIZE_PRIORS[class_name]
    rng = np.random.default_rng(seed)
    n_per_face = n_points // 6
    pts = []
    axes = [(l,w,h,0), (w,l,h,1), (h,l,w,2)]
    for a, b, c, ax in axes:
        for sign in [-1, 1]:
            face = np.zeros((n_per_face, 3), dtype=np.float32)
            face[:, ax]       = sign * a / 2
            face[:, (ax+1)%3] = rng.uniform(-b/2, b/2, n_per_face)
            face[:, (ax+2)%3] = rng.uniform(-c/2, c/2, n_per_face)
            pts.append(face)
    pts = np.vstack(pts)
    pts += rng.normal(0, 0.025, pts.shape).astype(np.float32)
    intensity = rng.uniform(0.1, 0.9, (len(pts), 1)).astype(np.float32)
    return np.hstack([pts, intensity])[:n_points]


def _pointe_generate(class_name: str, n_points: int) -> np.ndarray:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompts = {"pedestrian": "a person walking on a road, lidar scan"}
    base_m  = model_from_config(MODEL_CONFIGS["base40M-textvec"], device)
    base_m.eval()
    up_m    = model_from_config(MODEL_CONFIGS["upsample"], device)
    up_m.eval()
    sampler = PointCloudSampler(
        device=device,
        models=[base_m, up_m],
        diffusions=[diffusion_from_config(DIFFUSION_CONFIGS["base40M-textvec"]),
                    diffusion_from_config(DIFFUSION_CONFIGS["upsample"])],
        num_points=[1024, 4096-1024],
        aux_channels=["R","G","B"],
        guidance_scale=[3.0, 0.0],
    )
    samples = None
    for x in sampler.sample_batch_progressive(
            batch_size=1,
            model_kwargs={"texts": [prompts[class_name]]}):
        samples = x
    pc  = sampler.output_to_point_clouds(samples)[0]
    pts = np.array(pc.coords, dtype=np.float32)
    intensity = np.ones((len(pts),1), dtype=np.float32) * 0.5
    pts = np.hstack([pts, intensity])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = n_points
    return pts[:n] if len(pts) >= n else \
           np.vstack([pts, np.zeros((n-len(pts),4), dtype=np.float32)])


def place_in_scene(scene_pts: np.ndarray,
                   obj_pts:   np.ndarray,
                   class_name: str,
                   rng: np.random.Generator,
                   ) -> tuple[np.ndarray, dict]:
    """
    Place object into real scene with:
    - Valid depth range per class
    - Random lateral offset
    - Occlusion: remove background points inside object bbox
    - Random yaw rotation
    """
    l, w, h = SIZE_PRIORS[class_name]
    d_min, d_max = PLACEMENT[class_name]

    dist  = float(rng.uniform(d_min, d_max))
    lat   = float(rng.uniform(-8, 8))
    z_pos = float(rng.uniform(-1.8, -0.3))
    yaw   = float(rng.uniform(-np.pi, np.pi))

    # rotate object
    c, s  = np.cos(yaw), np.sin(yaw)
    R     = np.array([[c, -s], [s, c]])
    obj_rot = obj_pts.copy()
    xy_rot  = obj_pts[:, :2] @ R.T
    obj_rot[:, 0] = xy_rot[:, 0] + dist
    obj_rot[:, 1] = xy_rot[:, 1] + lat
    obj_rot[:, 2] = obj_pts[:, 2] + z_pos

    # remove background points inside object bounding box
    # (occlusion handling)
    cx, cy, cz = dist, lat, z_pos
    mask = ~(
        (scene_pts[:, 0] > cx - l/2) & (scene_pts[:, 0] < cx + l/2) &
        (scene_pts[:, 1] > cy - w/2) & (scene_pts[:, 1] < cy + w/2) &
        (scene_pts[:, 2] > cz - h/2) & (scene_pts[:, 2] < cz + h/2)
    )
    scene_clean = scene_pts[mask]

    # merge
    augmented = np.vstack([scene_clean, obj_rot])

    # subsample back to 16384
    max_pts = 16384
    if len(augmented) > max_pts:
        idx = np.random.choice(len(augmented), max_pts, replace=False)
        augmented = augmented[idx]
    elif len(augmented) < max_pts:
        pad = np.zeros((max_pts - len(augmented), 4), dtype=np.float32)
        augmented = np.vstack([augmented, pad])

    label = {
        "obj_type": class_name.capitalize(),
        "psr": {
            "position": {"x": dist, "y": lat,  "z": z_pos},
            "rotation": {"x": 0,    "y": 0,    "z": yaw},
            "scale":    {"x": l,    "y": w,     "z": h},
        }
    }
    return augmented.astype(np.float32), label


def run(n_per_class: int = N_PER_CLASS,
        use_pointe: bool = True) -> None:
    """
    Generate scene-aware augmented samples.
    Each sample = real LiDAR scene + synthetic rare-class object.
    """
    out_root = Path("data/augmented_v2")
    rng      = np.random.default_rng(42)

    # collect all real PCD files
    pcd_files = (
        sorted(Path("data/extracted/lidar/2nd_september/Scene4/lidar").glob("*.pcd")) +
        sorted(Path("data/extracted/lidar/2nd_september/Scene5/lidar").glob("*.pcd")) +
        sorted(Path("data/extracted/multimodal/TIAND_MULTIMODAL_DATASET/LIDAR/LIDAR").glob("*.pcd"))
    )
    print(f"Real PCD scenes available: {len(pcd_files)}")

    for class_name in RARE_CLASSES:
        out_dir = out_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # resume support
        done = len(list(out_dir.glob("*_pts.npy")))
        todo = n_per_class - done
        if todo <= 0:
            print(f"[{class_name}] already complete ({done}) — skipping")
            continue

        print(f"\n[{class_name}] generating {todo} scene-aware samples ...")

        for i in tqdm(range(done, n_per_class), desc=class_name):
            pts_path = out_dir / f"{class_name}_{i:04d}_pts.npy"
            lbl_path = out_dir / f"{class_name}_{i:04d}_label.json"
            if pts_path.exists():
                continue

            # load random real scene as background
            scene_pcd = pcd_files[i % len(pcd_files)]
            scene_pts = load_real_pcd(scene_pcd)

            # generate object
            obj_pts = generate_object_pcd(
                class_name,
                POINT_COUNTS[class_name],
                seed=i,
                use_pointe=(use_pointe and class_name == "pedestrian"),
            )

            # paste into real scene
            aug_pts, label = place_in_scene(
                scene_pts, obj_pts, class_name, rng)

            # save immediately
            np.save(pts_path, aug_pts)
            with open(lbl_path, "w") as f:
                json.dump([label], f)

        done_now = len(list(out_dir.glob("*_pts.npy")))
        print(f"  [{class_name}] {done_now} samples -> {out_dir}")

    # summary
    print("\n=== SCENE-AWARE AUGMENTATION COMPLETE ===")
    total = 0
    for cls in RARE_CLASSES:
        n = len(list((out_root/cls).glob("*_pts.npy")))
        print(f"  {cls:<15s}: {n:>5d} samples")
        total += n
    print(f"  {'Total':<15s}: {total:>5d}")
    print(f"\nNext: python src/preprocessing/rebuild_shards_v2.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_class", type=int, default=N_PER_CLASS)
    p.add_argument("--no_pointe",   action="store_true")
    args = p.parse_args()
    run(args.n_per_class, not args.no_pointe)
