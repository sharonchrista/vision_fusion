import argparse, zipfile, json, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

def extract_zip(zip_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=str(zip_path.name), unit="file"):
            zf.extract(m, dest)
    return len(members)

def inventory(root):
    counts = defaultdict(int)
    size   = 0
    for p in Path(root).rglob("*"):
        if p.is_file():
            counts[p.suffix.lower()] += 1
            size += p.stat().st_size
    return dict(counts), round(size/1e9, 2)

def show_tree(dest, modality):
    print(f"\n--- {modality.upper()} tree ({dest}) ---")
    items = sorted(Path(dest).iterdir()) if Path(dest).exists() else []
    for item in items[:20]:
        tag = "DIR " if item.is_dir() else "FILE"
        print(f"  {tag}  {item.name}")
        if item.is_dir():
            for sub in sorted(item.iterdir())[:6]:
                stag = "DIR " if sub.is_dir() else "FILE"
                print(f"         {stag}  {sub.name}")

def run(cfg_path):
    cfg  = load_config(cfg_path)
    raw  = Path(cfg.paths.raw_data)
    extr = Path(cfg.paths.extracted)

    zips = {
        "camera":     cfg.dataset.zips.camera,
        "radar":      cfg.dataset.zips.radar,
        "lidar":      cfg.dataset.zips.lidar,
        "multimodal": cfg.dataset.zips.multimodal,
    }

    for modality, zip_name in zips.items():
        dest = extr / modality
        # try exact name first, then glob
        zip_path = raw / zip_name
        if not zip_path.exists():
            matches = list(raw.glob(f"*{modality.upper()}*")) + \
                      list(raw.glob(f"*{modality}*")) + \
                      list(raw.glob(f"*TIAND*{modality[:3].upper()}*"))
            zip_path = matches[0] if matches else None

        if zip_path is None or not zip_path.exists():
            print(f"[SKIP] {modality}: not found in {raw}")
            continue

        print(f"\n[{modality.upper()}] {zip_path.name}  "
              f"({zip_path.stat().st_size/1e9:.2f} GB)")
        n = extract_zip(zip_path, dest)
        exts, gb = inventory(dest)
        print(f"  files: {n}  |  size on disk: {gb} GB")
        print(f"  types: {exts}")
        show_tree(dest, modality)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    run(args.config)
