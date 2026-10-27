"""
Fix 3 + Fix 4: LiquidFusionNet with pretrained LiDAR backbone
and full temporal window processing.

Key changes from v2:
  1. PretrainedLiDAREncoder replaces scratch PointNet++
  2. Forward accepts full sequences (T frames)
  3. LTC hidden state propagated across T timesteps
  4. Returns predictions at each timestep (for temporal loss)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from src.models.encoders.spatial_encoders import (
    CameraEncoder, RadarEncoder, IMUGPSEncoder
)
from src.models.encoders.lidar_pretrained import PretrainedLiDAREncoder
from src.models.fusion.ltc_ncp import LTCNCPFusion, DeltaTInjector
from src.models.heads.task_heads import (
    DetectionHead, TrackingHead, TrajectoryHead
)
from src.models.zero_shot import ZeroShotDetectionHead


class TemporalLiquidFusionNet(nn.Module):
    """
    LiquidFusionNet v3 — temporal + pretrained backbone.

    Forward accepts either:
      - Single frame: camera (B,4,3,H,W)   → standard inference
      - Sequence:     camera (B,T,4,3,H,W) → temporal training
    """

    def __init__(self, cfg, use_pretrained_lidar: bool = True):
        super().__init__()
        m = cfg.model
        d = cfg.dataset

        # ── Fix 3: pretrained LiDAR encoder ──────────────────
        if use_pretrained_lidar:
            self.lid_enc = PretrainedLiDAREncoder(
                feat_dim=m.lidar_feat_dim,
                pretrained=True,
            )
            print("Using pretrained LiDAR encoder (Point-MAE)")
        else:
            from src.models.encoders.spatial_encoders import LiDAREncoder
            self.lid_enc = LiDAREncoder(feat_dim=m.lidar_feat_dim)
            print("Using scratch LiDAR encoder")

        self.cam_enc = CameraEncoder(
            feat_dim=m.camera_feat_dim,
            pretrained=m.camera_pretrained,
            num_cameras=d.camera.num_cameras,
        )
        self.rad_enc = RadarEncoder(
            in_channels=d.radar.num_channels,
            num_radars=d.radar.num_radars,
            hidden_dims=m.radar_hidden_dims,
            feat_dim=m.radar_feat_dim,
        )
        self.imu_enc = IMUGPSEncoder(
            in_dim=9,
            hidden_dims=m.imu_hidden_dims,
            feat_dim=m.imu_feat_dim,
        )

        self.dt_inject = DeltaTInjector()
        ltc_in = m.fused_dim + 4

        # ── LTC-NCP temporal core ─────────────────────────────
        self.fusion = LTCNCPFusion(
            fused_dim=ltc_in,
            inter_neurons=m.ltc.inter_neurons,
            command_neurons=m.ltc.command_neurons,
            motor_neurons=m.ltc.motor_neurons,
            sparsity_level=m.ltc.sparsity_level,
            mode=m.ltc.mode,
        )

        motor = m.ltc.motor_neurons

        # ── Task heads ────────────────────────────────────────
        self.det_head  = DetectionHead(
            in_dim=motor,
            head_dim=m.detection_head_dim,
            num_classes=d.num_classes,
        )
        self.trk_head  = TrackingHead(
            in_dim=motor, head_dim=m.tracking_head_dim)
        self.traj_head = TrajectoryHead(
            in_dim=motor,
            head_dim=m.forecast_head_dim,
            forecast_horizon=m.forecast_horizon,
        )
        self.zsd_head  = ZeroShotDetectionHead(
            feat_dim=motor, embed_dim=128,
            num_seen=4, num_classes=9,
        )

    def _encode_frame(self, cam, lid, rad, imu, dt):
        """Encode a single frame across all modalities."""
        # Guard against NaN inputs from corrupted shards
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=-1.0)
        lid = torch.nan_to_num(lid, nan=0.0, posinf=100.0, neginf=-100.0)
        rad = torch.nan_to_num(rad, nan=0.0, posinf=100.0, neginf=-100.0)
        imu = torch.nan_to_num(imu, nan=0.0, posinf=10.0,  neginf=-10.0)
        f_cam = self.cam_enc(cam)
        f_lid = self.lid_enc(lid)
        f_rad = self.rad_enc(rad)
        f_imu = self.imu_enc(imu)
        fused    = torch.cat([f_cam, f_lid, f_rad, f_imu], dim=-1)
        fused_dt = self.dt_inject(fused, dt)
        return fused_dt

    def forward_sequence(self, batch: dict,
                         hx: torch.Tensor | None = None
                         ) -> tuple[dict, torch.Tensor]:
        """
        Fix 4: Process full temporal window sequentially.
        Propagates LTC hidden state across T timesteps.

        Returns predictions at the LAST timestep.
        """
        cam_seq = batch["camera_seq"]    # (B,T,4,3,H,W)
        lid_seq = batch["lidar_seq"]     # (B,T,N,4)
        rad_seq = batch["radar_seq"]     # (B,T,6,13)
        imu_seq = batch["imu_seq"]       # (B,T,9)
        dt_seq  = batch["delta_t_seq"]   # dict of (B,T)

        B, T = cam_seq.shape[:2]

        # encode each timestep
        encoded_seq = []
        for t in range(T):
            dt_t = {k: v[:, t] if v.dim() > 1 else v
                    for k, v in dt_seq.items()}
            cam_t = cam_seq[:, t]   # (B,4,3,H,W)
            lid_t = lid_seq[:, t]   # (B,N,4)
            rad_t = rad_seq[:, t]   # (B,6,13)
            imu_t = imu_seq[:, t]   # (B,9)
            enc_t = self._encode_frame(cam_t, lid_t,
                                       rad_t, imu_t, dt_t)
            encoded_seq.append(enc_t)

        # stack to sequence (B, T, fused_dim+4)
        seq = torch.stack(encoded_seq, dim=1)

        # LTC processes entire sequence, hidden state propagates
        motor_out, hx_new = self.fusion(seq, hx=hx)

        return {
            "detection":  self.det_head(motor_out),
            "tracking":   self.trk_head(motor_out),
            "trajectory": self.traj_head(motor_out),
            "zero_shot":  self.zsd_head(motor_out),
        }, hx_new

    def forward(self, batch: dict,
                hx: torch.Tensor | None = None,
                use_temporal: bool = False,
                ) -> tuple[dict, torch.Tensor]:
        """
        Unified forward — temporal or single-frame.
        """
        if use_temporal and "camera_seq" in batch:
            return self.forward_sequence(batch, hx)

        # single-frame path (backward compatible)
        dt_t = {k: v if v.dim() == 1 else v[:, -1]
                for k, v in batch.get(
                    "delta_t", {}).items()}
        enc = self._encode_frame(
            batch["camera"],
            batch["lidar"],
            batch["radar"],
            batch["imu_gps"],
            dt_t,
        )
        motor_out, hx_new = self.fusion(enc, hx=hx)
        return {
            "detection":  self.det_head(motor_out),
            "tracking":   self.trk_head(motor_out),
            "trajectory": self.traj_head(motor_out),
            "zero_shot":  self.zsd_head(motor_out),
        }, hx_new

    def freeze_encoders(self):
        for enc in [self.cam_enc, self.rad_enc,
                    self.imu_enc]:
            for p in enc.parameters():
                p.requires_grad = False
        # freeze only the transformer blocks in LiDAR encoder
        # keep projection head trainable for domain adaptation
        for p in self.lid_enc.encoder.blocks.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for enc in [self.cam_enc, self.rad_enc,
                    self.lid_enc, self.imu_enc]:
            for p in enc.parameters():
                p.requires_grad = True

    def count_parameters(self):
        def count(m):
            return sum(p.numel() for p in m.parameters()
                       if p.requires_grad)
        return {
            "camera_encoder":     count(self.cam_enc),
            "lidar_encoder":      count(self.lid_enc),
            "radar_encoder":      count(self.rad_enc),
            "imu_encoder":        count(self.imu_enc),
            "ltc_ncp_fusion":     count(self.fusion),
            "detection_head":     count(self.det_head),
            "tracking_head":      count(self.trk_head),
            "trajectory_head":    count(self.traj_head),
            "zsd_head":           count(self.zsd_head),
            "total":              count(self),
        }
