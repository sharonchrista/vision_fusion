"""
fix_zsd_classes.py — Update zero_shot.py to T-ITS 7-class schema.

T-ITS classes: Car=0, Two_wheeler=1, Pedestrian=2,
               Auto_rickshaw=3, Truck=4, Bus=5, Animal=6

Run with:
    python fix_zsd_classes.py
"""
import sys
from pathlib import Path

ZSD = Path("/home/sharon/vision_fusion/src/models/zero_shot.py")
if not ZSD.exists():
    print(f"ERROR: {ZSD} not found", file=sys.stderr)
    sys.exit(1)

NEW_CONTENT = '''"""
Zero-Shot Detection module for LiquidFusionNet.

T-ITS 7-class schema:
  Car=0, Two_wheeler=1, Pedestrian=2, Auto_rickshaw=3,
  Truck=4, Bus=5, Animal=6

Semantic attributes per class (9 dims):
  0: avg_length_m        — physical length
  1: avg_width_m         — physical width
  2: avg_height_m        — physical height
  3: max_speed_kmh       — typical max speed
  4: radar_rcs_dbsm      — radar cross-section
  5: is_motorised        — binary
  6: is_person_or_animal — binary
  7: is_large_vehicle    — binary
  8: indian_road_freq    — frequency on Indian roads (0-1)

Seen classes (supervised):  Car, Two_wheeler, Pedestrian (0,1,2)
Rare classes (ZSD):         Auto_rickshaw, Truck, Bus, Animal (3,4,5,6)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── T-ITS 7-class attribute matrix ───────────────────────────────────────────
# Rows: Car, Two_wheeler, Pedestrian, Auto_rickshaw, Truck, Bus, Animal
# Cols: length, width, height, max_speed, radar_rcs,
#       is_motorised, is_person_or_animal, is_large_vehicle, indian_freq
ATTRIBUTE_MATRIX = np.array([
    # len    wid    hgt   spd    rcs   mot  per  lrg  freq
    [4.50,  1.80,  1.50, 120,  10.0,   1,   0,   0,  0.95],  # 0: Car
    [2.20,  0.80,  1.20,  90,   3.0,   1,   0,   0,  0.85],  # 1: Two_wheeler
    [0.50,  0.50,  1.75,   6,   0.5,   0,   1,   0,  0.60],  # 2: Pedestrian
    [2.50,  1.30,  1.70,  50,   5.0,   1,   0,   0,  0.75],  # 3: Auto_rickshaw
    [8.00,  2.50,  3.20,  90,  18.0,   1,   0,   1,  0.35],  # 4: Truck
    [12.0,  2.50,  3.50,  80,  20.0,   1,   0,   1,  0.40],  # 5: Bus
    [1.00,  0.40,  1.00,  20,   0.2,   0,   1,   0,  0.20],  # 6: Animal
], dtype=np.float32)

NUM_CLASSES  = 7
SEEN_IDS     = [0, 1, 2]          # Car, Two_wheeler, Pedestrian
RARE_IDS     = [3, 4, 5, 6]       # Auto_rickshaw, Truck, Bus, Animal
CLASS_NAMES  = ["Car", "Two_wheeler", "Pedestrian",
                "Auto_rickshaw", "Truck", "Bus", "Animal"]


def normalise_attributes(mat: np.ndarray) -> np.ndarray:
    """Normalise each attribute column to [0,1]."""
    mn = mat.min(0, keepdims=True)
    mx = mat.max(0, keepdims=True)
    return (mat - mn) / np.clip(mx - mn, 1e-6, None)


class AttributeEmbedding(nn.Module):
    """
    Projects normalised semantic attributes into a shared
    embedding space where visual features can be compared.

    Input:  (num_classes, attr_dim)  — fixed attribute matrix
    Output: (num_classes, embed_dim) — learned class prototypes
    """
    def __init__(self, attr_dim: int = 9, embed_dim: int = 128):
        super().__init__()
        attrs = normalise_attributes(ATTRIBUTE_MATRIX)
        self.register_buffer("attributes",
                             torch.from_numpy(attrs))   # (C, attr_dim)
        self.proj = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self) -> torch.Tensor:
        """Returns (C, embed_dim) class prototype embeddings."""
        return F.normalize(self.proj(self.attributes), dim=-1)


class ZeroShotDetectionHead(nn.Module):
    """
    Generalised Zero-Shot Detection (GZSD) head for T-ITS 7 classes.

    Seen classes  (0-2): standard classification (cross-entropy)
    Rare classes  (3-6): cosine similarity to attribute prototypes

    Input:  visual feature (B, feat_dim) from motor neurons
    Output:
        seen_logits  (B, num_seen)    — supervised logits
        zsd_logits   (B, num_classes) — full class scores via attributes
        feat_embed   (B, embed_dim)   — visual embedding for loss
    """
    def __init__(self, feat_dim: int = 3,
                 embed_dim: int = 128,
                 num_seen: int = 3,
                 num_classes: int = 7,
                 temp: float = 0.07):
        super().__init__()
        self.num_seen    = num_seen
        self.num_classes = num_classes
        self.temp        = temp

        # Visual feature → embedding space
        self.visual_proj = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        # Semantic attribute embeddings (class prototypes)
        self.attr_embed = AttributeEmbedding(attr_dim=9, embed_dim=embed_dim)
        # Standard head for seen classes only
        self.seen_head = nn.Linear(embed_dim, num_seen)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Visual embedding
        vis_emb = F.normalize(self.visual_proj(x), dim=-1)  # (B, E)

        # Class prototype embeddings from attributes
        proto = self.attr_embed()                            # (C, E)

        # Cosine similarity to ALL classes (seen + rare)
        zsd_logits = vis_emb @ proto.T / self.temp           # (B, C)

        # Standard supervised logits for seen classes
        seen_logits = self.seen_head(vis_emb)                # (B, num_seen)

        return {
            "seen_logits": seen_logits,
            "zsd_logits":  zsd_logits,
            "vis_emb":     vis_emb,
            "proto":       proto,
        }


class ZeroShotLoss(nn.Module):
    """
    Combined loss for GZSD:
      L = L_seen (cross-entropy on seen classes)
        + lambda_zsd * L_zsd (attribute-space alignment)
        + lambda_reg * L_reg (seen/unseen margin, hubness reduction)
    """
    def __init__(self, lambda_zsd: float = 0.5,
                 lambda_reg: float = 0.1, temp: float = 0.07):
        super().__init__()
        self.lambda_zsd = lambda_zsd
        self.lambda_reg = lambda_reg
        self.temp       = temp

    def forward(self, pred: dict,
                gt_labels: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        seen_logits = pred["seen_logits"]   # (B, num_seen)
        zsd_logits  = pred["zsd_logits"]    # (B, C)
        vis_emb     = pred["vis_emb"]       # (B, E)
        proto       = pred["proto"]         # (C, E)

        B = seen_logits.shape[0]
        breakdown = {}

        l_seen = seen_logits.mean() * 0
        l_zsd  = vis_emb.mean() * 0

        if gt_labels is not None and gt_labels.numel() > 0:
            gl = gt_labels.view(B, -1)
            gl_per_sample = gl[:, 0].clamp(0, NUM_CLASSES - 1)

            seen_mask = gl_per_sample < len(SEEN_IDS)
            if seen_mask.any():
                l_seen = F.cross_entropy(
                    seen_logits[seen_mask],
                    gl_per_sample[seen_mask],
                )

            gt_proto = proto[gl_per_sample]
            l_zsd    = 1 - (vis_emb * gt_proto).sum(-1).mean()

        breakdown["loss_seen"] = l_seen.item()
        breakdown["loss_zsd"]  = l_zsd.item()

        # Hubness reduction: push seen and rare prototypes apart
        seen_proto = proto[SEEN_IDS]
        rare_proto = proto[RARE_IDS]
        cross_sim  = (seen_proto @ rare_proto.T).mean()
        l_reg      = F.relu(cross_sim - 0.3)
        breakdown["loss_zsd_reg"] = l_reg.item()

        total = l_seen + self.lambda_zsd * l_zsd + self.lambda_reg * l_reg
        breakdown["loss_zsd_total"] = total.item()
        return total, breakdown
'''

ZSD.write_text(NEW_CONTENT)
print(f"Updated {ZSD}")
print("T-ITS 7-class schema: Car=0, Two_wheeler=1, Pedestrian=2, "
      "Auto_rickshaw=3, Truck=4, Bus=5, Animal=6")
print("Seen: 0-2 (Car, Two_wheeler, Pedestrian)")
print("Rare: 3-6 (Auto_rickshaw, Truck, Bus, Animal)")