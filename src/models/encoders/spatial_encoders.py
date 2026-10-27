"""
Spatial encoders for each sensor modality.

CameraEncoder   — ResNet-50 backbone → 256-d feature vector
RadarEncoder    — MLP over 13-channel radar detections
LiDAREncoder    — PointNet++ style set abstraction
IMUGPSEncoder   — Simple MLP for pose + dynamics
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# ── Camera ────────────────────────────────────────────────────────────────────

class CameraEncoder(nn.Module):
    """
    ResNet-50 backbone, ImageNet pretrained.
    Input:  (B, num_cameras, 3, H, W)
    Output: (B, feat_dim)  — max-pooled across cameras
    """
    def __init__(self, feat_dim: int = 256, pretrained: bool = True,
                 num_cameras: int = 4):
        super().__init__()
        self.num_cameras = num_cameras

        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        # Replace final FC with projection to feat_dim
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_feats, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)           # (B*N, feat_dim)
        feats = feats.view(B, N, -1)
        return feats.max(dim=1).values     # (B, feat_dim) — max pool over cams


# ── Radar ─────────────────────────────────────────────────────────────────────

class RadarEncoder(nn.Module):
    """
    MLP encoder for raw radar detections.
    Input:  (B, num_radars, num_channels)
    Output: (B, feat_dim)
    """
    def __init__(self, in_channels: int = 13, num_radars: int = 6,
                 hidden_dims: list[int] = None, feat_dim: int = 128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        layers = []
        prev = in_channels
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        self.per_radar_mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(prev, feat_dim)
        self.num_radars = num_radars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_radars, in_channels)
        B, N, C = x.shape
        x = x.view(B * N, C)
        x = self.per_radar_mlp(x)          # (B*N, hidden)
        x = x.view(B, N, -1).max(dim=1).values  # (B, hidden)
        return self.proj(x)                # (B, feat_dim)


# ── LiDAR ─────────────────────────────────────────────────────────────────────

class PointNetSetAbstraction(nn.Module):
    """
    Simplified PointNet++ set abstraction layer.
    Input:  (B, N, 3+extra)
    Output: (B, out_dim)
    """
    def __init__(self, in_dim: int, mlp_dims: list[int]):
        super().__init__()
        layers = []
        prev = in_dim
        for d in mlp_dims:
            layers += [nn.Conv1d(prev, d, 1), nn.BatchNorm1d(d), nn.GELU()]
            prev = d
        self.mlp = nn.Sequential(*layers)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, C)
        x = pts.permute(0, 2, 1)       # (B, C, N)
        x = self.mlp(x)                # (B, D, N)
        return x.max(dim=-1).values    # (B, D)


class LiDAREncoder(nn.Module):
    """
    Two-stage PointNet++ style encoder.
    Input:  (B, N_pts, 4)  — x, y, z, intensity
    Output: (B, feat_dim)
    """
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(in_dim=4,   mlp_dims=[32, 64])
        self.sa2 = PointNetSetAbstraction(in_dim=64,  mlp_dims=[128, 256])
        self.proj = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, 4)
        x = self.sa1(pts)              # not a proper SA (no grouping) —
                                       # valid simplification for T4 budget
        x = x.unsqueeze(1)             # (B, 1, 64)
        x = self.sa2(x)                # (B, 256)
        return self.proj(x)


# ── IMU / GPS ─────────────────────────────────────────────────────────────────

class IMUGPSEncoder(nn.Module):
    """
    Simple MLP for 13-channel IMU + GPS vector.
    Input:  (B, 13)
    Output: (B, feat_dim)
    """
    def __init__(self, in_dim: int = 13,
                 hidden_dims: list[int] = None, feat_dim: int = 64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, feat_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
