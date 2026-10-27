"""
LiquidFusionNet — full model assembly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders.spatial_encoders import (
    CameraEncoder, RadarEncoder, LiDAREncoder, IMUGPSEncoder
)
from src.models.fusion.ltc_ncp import LTCNCPFusion, DeltaTInjector
from src.models.heads.task_heads import DetectionHead, TrackingHead, TrajectoryHead


class LiquidFusionNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        d = cfg.dataset

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
        self.lid_enc = LiDAREncoder(feat_dim=m.lidar_feat_dim)
        self.imu_enc = IMUGPSEncoder(
            in_dim=d.imu_gps.imu_channels + d.imu_gps.gps_channels,
            hidden_dims=m.imu_hidden_dims,
            feat_dim=m.imu_feat_dim,
        )

        self.cfg      = cfg                               # needed by forward for ablation flags

        self.dt_inject = DeltaTInjector()
        ltc_in = m.fused_dim + 4   # 704 + 4 delta_t scalars

        _use_ltc         = getattr(cfg, "ablation_use_ltc",         True)
        _use_transformer = getattr(cfg, "ablation_use_transformer", False)

        self._fusion_is_gru         = False
        self._fusion_is_transformer = False

        if _use_transformer:
            # Transformer baseline — d_model = ltc_in, 4 heads, 2 layers
            # projects to motor_neurons width after encoding
            _d_model = ltc_in
            _nhead   = 4
            _enc_layer = torch.nn.TransformerEncoderLayer(
                d_model=_d_model, nhead=_nhead,
                dim_feedforward=_d_model * 2,
                dropout=0.1, batch_first=True,
            )
            self.fusion = torch.nn.TransformerEncoder(_enc_layer, num_layers=2)
            self._fusion_proj = torch.nn.Linear(_d_model, m.ltc.motor_neurons)
            self._fusion_is_transformer = True
        elif _use_ltc:
            self.fusion = LTCNCPFusion(
                fused_dim=ltc_in,
                inter_neurons=m.ltc.inter_neurons,
                command_neurons=m.ltc.command_neurons,
                motor_neurons=m.ltc.motor_neurons,
                sparsity_level=m.ltc.sparsity_level,
                mode=m.ltc.mode,
            )
        else:
            # GRU baseline — comparable parameter budget to LTC-NCP
            _gru_hidden = m.ltc.motor_neurons   # match output width
            self.fusion = torch.nn.GRU(
                input_size=ltc_in,
                hidden_size=_gru_hidden,
                num_layers=2,
                batch_first=True,
            )
            self._fusion_proj = torch.nn.Linear(_gru_hidden, m.ltc.motor_neurons)
            self._fusion_is_gru = True

        motor = m.ltc.motor_neurons
        self.det_head  = DetectionHead(
            in_dim=motor, head_dim=m.detection_head_dim,
            num_classes=d.num_classes,
        )
        self.trk_head  = TrackingHead(
            in_dim=motor, head_dim=m.tracking_head_dim,
        )
        self.traj_head = TrajectoryHead(
            in_dim=motor, head_dim=m.forecast_head_dim,
            forecast_horizon=m.forecast_horizon,
        )

    def forward(self, batch: dict,
                hx: torch.Tensor | None = None
                ) -> tuple[dict, torch.Tensor]:
        f_cam  = self.cam_enc(batch["camera"])
        f_rad  = self.rad_enc(batch["radar"])
        f_lid  = self.lid_enc(batch["lidar"])
        f_imu  = self.imu_enc(batch["imu_gps"])

        # ── Ablation: zero out disabled modalities ───────────────────────────
        _active = getattr(self.cfg, "ablation_modalities",
                          ["camera", "lidar", "radar", "imu", "imu_gps"])
        if "radar" not in _active:
            f_rad = torch.zeros_like(f_rad)
        if "lidar" not in _active:
            f_lid = torch.zeros_like(f_lid)
        if "imu" not in _active and "imu_gps" not in _active:
            f_imu = torch.zeros_like(f_imu)
        # camera is always active

        fused = torch.cat([f_cam, f_rad, f_lid, f_imu], dim=-1)

        # ── Ablation: optionally skip Δt injection ───────────────────────────
        _inject_dt = getattr(self.cfg, "ablation_inject_dt", True)
        if _inject_dt:
            fused_dt = self.dt_inject(fused, batch["delta_t"])
        else:
            # pad 4 zeros so downstream tensor width stays consistent
            fused_dt = torch.cat(
                [fused, torch.zeros(fused.shape[0], 4, device=fused.device)],
                dim=-1,
            )

        # ── Ablation: LTC-NCP vs GRU vs Transformer fusion ──────────────────
        if self._fusion_is_transformer:
            _x = fused_dt.unsqueeze(1)           # (B, 1, d_model)
            _out = self.fusion(_x)               # (B, 1, d_model)
            motor_out = self._fusion_proj(_out.squeeze(1))
            hx_new = None
        elif not self._fusion_is_gru:
            motor_out, hx_new = self.fusion(fused_dt, hx=hx)
        else:
            _x = fused_dt.unsqueeze(1)           # (B, 1, features)
            _out, hx_new = self.fusion(_x, hx)   # hx: (num_layers, B, hidden)
            motor_out = self._fusion_proj(_out.squeeze(1))

        return {
            "detection":  self.det_head(motor_out),
            "tracking":   self.trk_head(motor_out),
            "trajectory": self.traj_head(motor_out),
        }, hx_new

    def freeze_encoders(self) -> None:
        for enc in [self.cam_enc, self.rad_enc, self.lid_enc, self.imu_enc]:
            for p in enc.parameters():
                p.requires_grad = False

    def unfreeze_encoders(self) -> None:
        for enc in [self.cam_enc, self.rad_enc, self.lid_enc, self.imu_enc]:
            for p in enc.parameters():
                p.requires_grad = True

    def count_parameters(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "camera_encoder":  count(self.cam_enc),
            "radar_encoder":   count(self.rad_enc),
            "lidar_encoder":   count(self.lid_enc),
            "imu_encoder":     count(self.imu_enc),
            "ltc_ncp_fusion":  count(self.fusion),
            "detection_head":  count(self.det_head),
            "tracking_head":   count(self.trk_head),
            "trajectory_head": count(self.traj_head),
            "total":           count(self),
        }
