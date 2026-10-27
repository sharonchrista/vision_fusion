"""
Diffusion augmentation v2 — saves each sample immediately to disk.
Robust to crashes: resumes from where it left off.
"""
import json, sys, gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RARE_CLASSES = ["pedestrian", "bus", "truck", "van"]
N_PER_CLASS  = 50

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


def _geometric_primitive(class_name, n_points, seed=0):
    """Generate one point cloud from box surface + noise."""
    l, w, h = SIZE_PRIORS[class_name]
    rng = np.random.default_rng(seed)
    n_per_face = n_points // 6
    pts = []
    dims = [(l,w,h), (w,l,h), (h,l,w)]
    for fi, (a, b, c) in enumerate(dims):
        for sign in [-1, 1]:
            face = np.zeros((n_per_face, 3), dtype=np.float32)
            face[:, fi % 3]       = sign * a / 2
            face[:, (fi+1) % 3]   = rng.uniform(-b/2, b/2, n_per_face)
            face[:, (fi+2) % 3]   = rng.uniform(-c/2, c/2, n_per_face)
            pts.append(face)
    pts = np.vstack(pts)
    pts += rng.normal(0, 0.02, pts.shape).astype(np.float32)
    intensity = rng.uniform(0.2, 0.8, (len(pts), 1)).astype(np.float32)
    pts = np.hstack([pts, intensity])
    return pts[:n_points]


def generate_pointe_sample(class_name, sample_idx, device):
    """Generate one Point-E sample. Returns (N,4) array or None on failure."""
    try:
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.models.configs import MODEL_CONFIGS, model_from_config

        prompts = {
            "pedestrian": "a person walking on a road",
            "bus":        "a large bus on a road",
            "truck":      "a heavy truck on a road",
            "van":        "a van on a road",
        }

        base_model = model_from_config(
            MODEL_CONFIGS["base40M-textvec"], device)
        base_model.eval()
        base_diff = diffusion_from_config(
            DIFFUSION_CONFIGS["base40M-textvec"])

        up_model = model_from_config(MODEL_CONFIGS["upsample"], device)
        up_model.eval()
        up_diff  = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

        sampler = PointCloudSampler(
            device=device,
            models=[base_model, up_model],
            diffusions=[base_diff, up_diff],
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
        intensity = np.ones((len(pts), 1), dtype=np.float32) * 0.5
        pts = np.hstack([pts, intensity])

        n = POINT_COUNTS[class_name]
        if len(pts) >= n:
            return pts[:n]
        return np.vstack([pts, np.zeros((n-len(pts), 4), dtype=np.float32)])

    except Exception as e:
        print(f"    Point-E failed for sample {sample_idx}: {e}")
        return None
    finally:
        # free GPU memory after each sample
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_label(class_name, seed=0):
    l, w, h = SIZE_PRIORS[class_name]
    rng = np.random.default_rng(seed)
    return [{
        "obj_type": class_name.capitalize(),
        "psr": {
            "position": {
                "x": float(rng.uniform(5, 40)),
                "y": float(rng.uniform(-10, 10)),
                "z": float(rng.uniform(-1.5, -0.5)),
            },
            "rotation": {"x": 0, "y": 0,
                         "z": float(rng.uniform(-3.14, 3.14))},
            "scale": {"x": float(l), "y": float(w), "z": float(h)},
        }
    }]


def run():
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path("data/augmented")
    print(f"Device: {device}")

    # Use Point-E only for pedestrian
    # Geometric primitive for large vehicles (just as effective, instant)
    USE_POINTE = {
        "pedestrian": True,
        "bus":        False,
        "truck":      False,
        "van":        False,
    }

    for class_name in RARE_CLASSES:
        out_dir = out_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # count already done (resume support)
        done = len(list(out_dir.glob("*_pts.npy")))
        todo = N_PER_CLASS - done
        if todo <= 0:
            print(f"[{class_name}] already complete ({done} samples) — skipping")
            continue

        print(f"\n[{class_name}] generating {todo} samples "
              f"(resuming from {done}) ...")

        for i in tqdm(range(done, N_PER_CLASS), desc=class_name):
            pts_path = out_dir / f"{class_name}_{i:04d}_pts.npy"
            lbl_path = out_dir / f"{class_name}_{i:04d}_label.json"

            if pts_path.exists():
                continue   # already done

            # generate
            if USE_POINTE[class_name]:
                pts = generate_pointe_sample(class_name, i, device)
                if pts is None:
                    print(f"  Falling back to geometric for sample {i}")
                    pts = _geometric_primitive(class_name,
                                               POINT_COUNTS[class_name], i)
            else:
                pts = _geometric_primitive(class_name,
                                           POINT_COUNTS[class_name], i)

            # save immediately
            np.save(pts_path, pts)
            with open(lbl_path, "w") as f:
                json.dump(make_label(class_name, i), f)

        done_now = len(list(out_dir.glob("*_pts.npy")))
        print(f"  [{class_name}] {done_now} samples saved -> {out_dir}")

    # summary
    print("\n=== AUGMENTATION COMPLETE ===")
    for cls in RARE_CLASSES:
        n = len(list((out_root/cls).glob("*_pts.npy")))
        print(f"  {cls:<15s}: {n:>3d} samples")
    total = len(list(out_root.rglob("*_pts.npy")))
    print(f"  Total         : {total}")
    print("\nNext: python src/preprocessing/rebuild_shards_augmented.py")


if __name__ == "__main__":
    run()
