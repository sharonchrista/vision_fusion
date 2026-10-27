"""
Task heads for LiquidFusionNet.

DetectionHead      — 3-D bounding box regression + classification
TrackingHead       — per-object state embedding for association
TrajectoryHead     — multi-modal waypoint forecasting
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims: list[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1])]
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(),
                       nn.Dropout(dropout)]
    return nn.Sequential(*layers)


# ── Detection head ────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Input : (B, motor_dim)  — motor neuron activations
    Output:
        cls_logits  (B, num_anchors, num_classes)
        bbox_deltas (B, num_anchors, 7)   — dx,dy,dz,dw,dl,dh,dθ
    """
    def __init__(self, in_dim: int = 3, head_dim: int = 256,
                 num_classes: int = 7, num_anchors: int = 9):
        super().__init__()
        self.num_anchors = num_anchors
        self.shared = _mlp([in_dim, head_dim, head_dim])
        self.cls    = nn.Linear(head_dim, num_anchors * num_classes)
        self.bbox   = nn.Linear(head_dim, num_anchors * 7)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.shared(x)
        B = x.shape[0]
        cls  = self.cls(feat).view(B, self.num_anchors, self.num_classes)
        bbox = self.bbox(feat).view(B, self.num_anchors, 7)
        return {"cls_logits": cls, "bbox_deltas": bbox}


# ── Tracking head ─────────────────────────────────────────────────────────────

class TrackingHead(nn.Module):
    """
    Produces a re-ID embedding for each detected object.
    Used with contrastive association loss.

    Input : (B, motor_dim)
    Output: (B, embed_dim)  — L2-normalised embedding
    """
    def __init__(self, in_dim: int = 3, head_dim: int = 128,
                 embed_dim: int = 128):
        super().__init__()
        self.net = _mlp([in_dim, head_dim, embed_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return nn.functional.normalize(emb, p=2, dim=-1)


# ── Trajectory head ───────────────────────────────────────────────────────────

class TrajectoryHead(nn.Module):
    """
    Multi-modal trajectory forecasting.
    Predicts K hypotheses over T future timesteps.

    Input : (B, motor_dim)
    Output:
        waypoints (B, K, T, 2)   — (x, y) offsets per mode
        mode_logits (B, K)        — mixture weights
    """
    def __init__(self, in_dim: int = 3, head_dim: int = 256,
                 forecast_horizon: int = 30, num_modes: int = 6):
        super().__init__()
        self.K = num_modes
        self.T = forecast_horizon
        self.shared = _mlp([in_dim, head_dim, head_dim])
        self.wp_head    = nn.Linear(head_dim, num_modes * forecast_horizon * 2)
        self.mode_head  = nn.Linear(head_dim, num_modes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        feat = self.shared(x)
        wp   = self.wp_head(feat).view(B, self.K, self.T, 2)
        mode = self.mode_head(feat)
        return {"waypoints": wp, "mode_logits": mode}
