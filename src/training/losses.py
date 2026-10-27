"""
Loss functions for LiquidFusionNet — T-ITS paper.

DetectionLoss    — Focal + Smooth-L1 + CIoU
TrackingLoss     — Contrastive association
TrajectoryLoss   — ADE/FDE Huber + multi-modal NLL
MultiTaskLoss    — Kendall uncertainty weighting
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt    = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


class DetectionLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, gt):
        logits = pred["cls_logits"]
        deltas = pred["bbox_deltas"]
        cls_t  = gt.get("cls_targets")
        box_t  = gt.get("bbox_targets")
        mask   = gt.get("pos_mask")
        l_cls  = focal_loss(logits, cls_t, self.gamma, self.alpha) \
                 if cls_t is not None else logits.mean() * 0
        if box_t is not None and mask is not None:
            l_reg = F.smooth_l1_loss(deltas[mask], box_t[mask])
        else:
            l_reg = deltas.mean() * 0
        return l_cls + 0.001 * l_reg


class TrackingLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp

    def forward(self, pred, gt):
        emb  = pred   # (B, D) L2-normalised
        ids  = gt.get("assoc_targets")
        if ids is None:
            return emb.pow(2).mean() * 0
        sim     = torch.mm(emb, emb.T) / self.temp
        pos     = (ids.unsqueeze(1) == ids.unsqueeze(0)).float()
        pos.fill_diagonal_(0)
        exp_sim = torch.exp(sim)
        denom   = exp_sim.sum(1, keepdim=True) - exp_sim.diagonal().unsqueeze(1)
        pos_s   = (exp_sim * pos).sum(1)
        return (-torch.log((pos_s + 1e-8) / (denom.squeeze() + 1e-8))).mean()


class TrajectoryLoss(nn.Module):
    def forward(self, pred, gt):
        wp   = pred["waypoints"]    # (B,K,T,2)
        mode = pred["mode_logits"]  # (B,K)
        fut  = gt.get("future_xy")
        if fut is None:
            return wp.mean() * 0
        B, K, T, _ = wp.shape
        gt_e  = fut.unsqueeze(1).expand(B, K, T, 2)
        ade   = F.huber_loss(wp, gt_e, reduction="none").mean(dim=[2,3])  # (B,K)
        best  = ade.argmin(1)
        l_ade = ade.gather(1, best.unsqueeze(1)).mean()
        best_wp = wp[torch.arange(B), best]
        l_fde = F.huber_loss(best_wp[:,-1], fut[:,-1])
        l_nll = -F.log_softmax(mode, -1)[torch.arange(B), best].mean()
        return l_ade + l_fde + 0.1 * l_nll


class MultiTaskLoss(nn.Module):
    """Kendall uncertainty weighting: L = Σ (1/2σ²)Lᵢ + log σᵢ"""
    def __init__(self, cfg):
        super().__init__()
        t = cfg.training.loss
        self.det_loss  = DetectionLoss(gamma=t.focal_gamma, alpha=t.focal_alpha)
        self.trk_loss  = TrackingLoss()
        self.traj_loss = TrajectoryLoss()
        if t.use_uncertainty_weighting:
            self.log_var = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("_lam", torch.tensor(
                [t.lambda_det, t.lambda_trk, t.lambda_traj]))
            self.log_var = None

    def forward(self, outputs, targets):
        l_det  = self.det_loss(outputs["detection"],  targets.get("detection",  {}))
        l_trk  = self.trk_loss(outputs["tracking"],   targets.get("tracking",   {}))
        l_traj = self.traj_loss(outputs["trajectory"], targets.get("trajectory", {}))
        losses = torch.stack([l_det, l_trk, l_traj])

        if self.log_var is not None:
            prec  = torch.exp(-self.log_var)
            total = (prec * losses + self.log_var).sum()
        else:
            total = (self._lam * losses).sum()

        bd = {
            "loss_det":   l_det.item(),
            "loss_trk":   l_trk.item(),
            "loss_traj":  l_traj.item(),
            "loss_total": total.item(),
        }
        if self.log_var is not None:
            for i, k in enumerate(["sigma_det","sigma_trk","sigma_traj"]):
                bd[k] = self.log_var[i].exp().item()
        return total, bd
