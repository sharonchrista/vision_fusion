"""
fix_class_map.py — Fix NUSCENES_CLASS_MAP in preprocess_nuscenes.py so all
class IDs fall within the T-ITS 7-class range (0-6).

T-ITS schema:  Car=0, Two_wheeler=1, Pedestrian=2, Auto_rickshaw=3,
               Truck=4, Bus=5, Animal=6

Run with:
    python fix_class_map.py
Then rebuild shards:
    python /home/sharon/vision_fusion/src/preprocessing/nuscenes/preprocess_nuscenes.py \
        --nuscenes_root /home/sharon/vision_fusion/data/nuscenes/v1.0-mini \
        --output_dir   /home/sharon/vision_fusion/data/shards/nuscenes_mini_labeled \
        --split        mini \
        --num_workers  4
Then re-split into train/val/test:
    python fix_shard_split.py   (or run the split logic below)
"""
import sys
import re
from pathlib import Path

# ── Locate the file ──────────────────────────────────────────────────────────
candidates = list(Path("/home/sharon/vision_fusion").rglob("preprocess_nuscenes.py"))
if not candidates:
    print("ERROR: preprocess_nuscenes.py not found under /home/sharon/vision_fusion",
          file=sys.stderr)
    sys.exit(1)
PREPROC = candidates[0]
print(f"Patching: {PREPROC}")

src = PREPROC.read_text()

# ── New map (valid IDs 0-6 only) ─────────────────────────────────────────────
NEW_MAP = '''\
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
}'''

# ── Find the existing NUSCENES_CLASS_MAP block ────────────────────────────────
# Match from "NUSCENES_CLASS_MAP = {" to the closing "}"
pattern = re.compile(
    r'NUSCENES_CLASS_MAP\s*=\s*\{[^}]*\}',
    re.DOTALL
)
m = pattern.search(src)
if not m:
    print("ERROR: NUSCENES_CLASS_MAP block not found", file=sys.stderr)
    sys.exit(1)

src = src[:m.start()] + NEW_MAP + src[m.end():]
PREPROC.write_text(src)
print("NUSCENES_CLASS_MAP patched — all IDs now in range 0-6")

# ── Quick sanity: no IDs > 6 ─────────────────────────────────────────────────
bad = [line.strip() for line in NEW_MAP.splitlines()
       if re.search(r':\s*([7-9]|[1-9]\d)', line) and not line.strip().startswith('#')]
if bad:
    print(f"WARNING: possible out-of-range IDs still present:\n  " + "\n  ".join(bad))
else:
    print("Sanity check passed: no IDs > 6 in new map")

print(f"\nPatched file: {PREPROC}")
print("\nNext step — rebuild mini labeled shards (takes ~2 min on 4 workers):")
print("  python", PREPROC,
      "\\\n    --nuscenes_root /home/sharon/vision_fusion/data/nuscenes/v1.0-mini",
      "\\\n    --output_dir   /home/sharon/vision_fusion/data/shards/nuscenes_mini_labeled",
      "\\\n    --split        mini",
      "\\\n    --num_workers  4")
print("\nThen run the dry-run:")
print("  PYTHONPATH=/home/sharon/vision_fusion \\")
print("    python /home/sharon/vision_fusion/scripts/run_ablations_v2.py --dry-run 2>&1 | tail -40")