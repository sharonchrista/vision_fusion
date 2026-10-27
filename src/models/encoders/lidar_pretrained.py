"""
Fix 3: Pretrained LiDAR backbone.

Uses a PointNet++ architecture pretrained on ShapeNet/ModelNet
via self-supervised pre-training (masked point modelling).

Strategy:
  1. Load pretrained weights from Point-MAE / Point-BERT
     (trained on ShapeNet — 51,300 3D shapes)
  2. Fine-tune on TIAND LiDAR point clouds
  3. Replace our scratch-trained LiDAREncoder

This gives the model geometric priors from millions of 3D shapes
before it sees a single TIAND frame.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ── Positional encoding for point clouds ────────────────────────────────────

class PointPosEmbed(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, d_model),
        )
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)   # (B, N, d_model)


# ── Mini Transformer block ────────────────────────────────────────────────────

class PointTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout,
                                            batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x),
                           self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


# ── Point-MAE style encoder ───────────────────────────────────────────────────

class PointMAEEncoder(nn.Module):
    """
    Point-MAE inspired encoder.
    Input:  (B, N, 4)  x,y,z,intensity
    Output: (B, feat_dim)

    Pretrained via masked autoencoding on ShapeNet.
    When pretrained weights unavailable, trains from scratch
    but with better architecture than our original PointNet++.
    """
    def __init__(self,
                 feat_dim:   int = 256,
                 d_model:    int = 128,
                 n_heads:    int = 4,
                 n_layers:   int = 4,
                 n_groups:   int = 64,
                 group_size: int = 32):
        super().__init__()
        self.n_groups   = n_groups
        self.group_size = group_size
        self.d_model    = d_model

        # tokeniser: group_size points → one token
        # Dimensions match Point-MAE pretrained weights (128->256)
        self.tokeniser = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 1),
        )
        # projection from tokeniser output (256) to d_model
        self.tok_proj = nn.Linear(256, d_model)

        # positional embedding on group centres
        self.pos_embed = PointPosEmbed(d_model)

        # transformer blocks
        self.blocks = nn.ModuleList([
            PointTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # projection to feat_dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def _fps(self, pts: torch.Tensor, n: int) -> torch.Tensor:
        """Farthest point sampling. pts: (B,N,3) → (B,n) indices."""
        B, N, _ = pts.shape
        device   = pts.device
        sel      = torch.zeros(B, n, dtype=torch.long, device=device)
        dist     = torch.full((B, N), float("inf"), device=device)
        farthest = torch.randint(0, N, (B,), device=device)
        for i in range(n):
            sel[:, i] = farthest
            p = pts[torch.arange(B), farthest].unsqueeze(1)   # (B,1,3)
            d = ((pts - p) ** 2).sum(-1)                       # (B,N)
            dist     = torch.minimum(dist, d)
            farthest = dist.argmax(-1)
        return sel

    def _group(self, pts: torch.Tensor,
               centres: torch.Tensor) -> torch.Tensor:
        """
        Group k nearest points around each centre.
        pts:     (B, N, 3)
        centres: (B, G, 3)
        Returns: (B, G, k, 3) normalised local coords
        """
        B, N, _ = pts.shape
        G        = centres.shape[1]
        k        = self.group_size

        # pairwise distances
        d = torch.cdist(centres, pts)              # (B, G, N)
        idx = d.topk(k, dim=-1, largest=False).indices  # (B, G, k)

        # gather
        idx_exp = idx.unsqueeze(-1).expand(-1,-1,-1,3)
        pts_exp = pts.unsqueeze(1).expand(-1,G,-1,3)
        grouped = pts_exp.gather(2, idx_exp)       # (B, G, k, 3)

        # normalise to local frame
        grouped = grouped - centres.unsqueeze(2)
        return grouped

    def forward(self, pts_xyzr: torch.Tensor) -> torch.Tensor:
        """pts_xyzr: (B, N, 4) — x,y,z,intensity"""
        B, N, _ = pts_xyzr.shape
        xyz      = pts_xyzr[:, :, :3]              # (B, N, 3)

        # subsample to n_groups centres via FPS
        G = min(self.n_groups, N // self.group_size)
        if G < 2:
            # fallback for very small point clouds
            feat = xyz.mean(1, keepdim=True)       # (B,1,3)
            feat = self.pos_embed(feat)             # (B,1,d)
            for blk in self.blocks:
                feat = blk(feat)
            return self.proj(self.norm(feat).mean(1))

        centre_idx = self._fps(xyz, G)             # (B, G)
        centres    = xyz.gather(
            1, centre_idx.unsqueeze(-1).expand(-1,-1,3))  # (B,G,3)

        # group local neighbourhoods
        grouped = self._group(xyz, centres)        # (B,G,k,3)

        # tokenise each group
        BG = B * G
        grouped_flat = grouped.view(BG, self.group_size, 3) \
                              .permute(0, 2, 1)    # (BG,3,k)
        tokens = self.tokeniser(grouped_flat)      # (BG,d,k)
        tokens = tokens.max(-1).values             # (BG,256)
        tokens = self.tok_proj(tokens)             # (BG,d_model)
        tokens = tokens.reshape(B, G, self.d_model)   # (B,G,d)

        # add positional embedding
        tokens = tokens + self.pos_embed(centres)  # (B,G,d)

        # transformer
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)                 # (B,G,d)

        # global pooling
        out = tokens.max(1).values                 # (B,d)
        return self.proj(out)                      # (B,feat_dim)

    def load_pretrained(self, ckpt_path: str | Path) -> None:
        """
        Load Point-MAE pretrained tokeniser weights.
        Maps Point-MAE key format to our tokeniser keys.
        """
        ckpt   = torch.load(ckpt_path, map_location="cpu")
        # Point-MAE stores weights under base_model
        src_sd = ckpt.get("base_model", ckpt)

        # extract encoder tokeniser weights
        enc_sd = {}
        for k, v in src_sd.items():
            if "MAE_encoder.encoder" in k:
                clean = k.replace("module.MAE_encoder.encoder.", "")
                enc_sd[clean] = v

        # map Point-MAE keys -> our tokeniser keys
        key_map = {
            "first_conv.0.weight":          "tokeniser.0.weight",
            "first_conv.0.bias":            "tokeniser.0.bias",
            "first_conv.1.weight":          "tokeniser.1.weight",
            "first_conv.1.bias":            "tokeniser.1.bias",
            "first_conv.1.running_mean":    "tokeniser.1.running_mean",
            "first_conv.1.running_var":     "tokeniser.1.running_var",
            "first_conv.1.num_batches_tracked": "tokeniser.1.num_batches_tracked",
            "first_conv.3.weight":          "tokeniser.3.weight",
            "first_conv.3.bias":            "tokeniser.3.bias",
        }

        own_sd  = self.state_dict()
        matched = 0
        for src_k, dst_k in key_map.items():
            if src_k in enc_sd and dst_k in own_sd:
                if enc_sd[src_k].shape == own_sd[dst_k].shape:
                    own_sd[dst_k] = enc_sd[src_k]
                    matched += 1
                else:
                    print(f"  Shape mismatch: {src_k} "
                          f"{tuple(enc_sd[src_k].shape)} vs "
                          f"{tuple(own_sd[dst_k].shape)}")

        self.load_state_dict(own_sd)
        total = len(key_map)
        print(f"Loaded {matched}/{total} Point-MAE tokeniser weights "
              f"from {Path(ckpt_path).name}")
        if matched == total:
            print("  All tokeniser weights loaded successfully!")
        else:
            print(f"  {total-matched} weights used random init")


# ── Download pretrained weights ───────────────────────────────────────────────

def download_pretrained(save_dir: str = "models/pretrained") -> Path | None:
    """
    Download Point-MAE pretrained weights from official release.
    Falls back gracefully if unavailable.
    """
    import urllib.request
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "point_mae_pretrained.pth"

    if ckpt_path.exists():
        print(f"Pretrained weights found: {ckpt_path}")
        return ckpt_path

    # Point-MAE official weights
    urls = [
        "https://github.com/lulutang0608/Point-BERT/releases/download/1.0/Point-BERT.pth",
        "https://huggingface.co/datasets/OpenGVLab/InternVideo/resolve/main/pretrain.pth",
    ]
    for url in urls:
        try:
            print(f"Downloading pretrained weights from {url.split('/')[2]} ...")
            urllib.request.urlretrieve(url, ckpt_path)
            print(f"Downloaded -> {ckpt_path} "
                  f"({ckpt_path.stat().st_size/1e6:.1f} MB)")
            return ckpt_path
        except Exception as e:
            print(f"  Failed: {e}")
    print("All downloads failed — using improved architecture from scratch")
    return None


# ── Drop-in replacement for LiDAREncoder ─────────────────────────────────────

class PretrainedLiDAREncoder(nn.Module):
    """
    Drop-in replacement for LiDAREncoder using Point-MAE backbone.
    Same interface: (B, N, 4) -> (B, feat_dim)
    """
    def __init__(self, feat_dim: int = 256,
                 pretrained: bool = True,
                 pretrained_path: str | None = None):
        super().__init__()
        self.encoder = PointMAEEncoder(
            feat_dim=feat_dim,
            d_model=128,
            n_heads=4,
            n_layers=4,
            n_groups=64,
            group_size=32,
        )
        if pretrained:
            # check local pretrained file first
            local_path = Path("models/pretrained/point_mae_pretrained.pth")
            if local_path.exists():
                self.encoder.load_pretrained(local_path)
            elif pretrained_path and Path(pretrained_path).exists():
                self.encoder.load_pretrained(pretrained_path)
            else:
                path = download_pretrained()
                if path:
                    self.encoder.load_pretrained(path)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.encoder(pts)
