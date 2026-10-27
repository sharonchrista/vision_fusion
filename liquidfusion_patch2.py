"""
liquidfusion_patch2.py  —  Wire ablation flags into liquidfusionnet.py
(corrected anchors based on actual file content)

Applies 4 patches:
  1. Store cfg as self.cfg in __init__
  2. Add GRU fallback instantiation in __init__ (when ablation_use_ltc=False)
  3. Add modality zeroing + Δt toggle in forward
  4. Swap fusion call to use GRU when ablation_use_ltc=False

Run with:
    python liquidfusion_patch2.py
"""

import sys
from pathlib import Path

MODEL = Path("/home/sharon/vision_fusion/src/models/liquidfusionnet.py")

if not MODEL.exists():
    print(f"ERROR: {MODEL} not found", file=sys.stderr)
    sys.exit(1)

src = MODEL.read_text()

# ── PATCH 1: store cfg as self.cfg ───────────────────────────────────────────
OLD1 = '        self.dt_inject = DeltaTInjector()'
NEW1 = '''\
        self.cfg      = cfg                               # needed by forward for ablation flags

        self.dt_inject = DeltaTInjector()'''

if OLD1 not in src:
    print(f"ERROR: Patch 1 anchor not found:\n{OLD1}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 applied: self.cfg = cfg")


# ── PATCH 2: GRU fallback replacing self.fusion = LTCNCPFusion(...) ──────────
OLD2 = '''\
        self.fusion = LTCNCPFusion(
            fused_dim=ltc_in,
            inter_neurons=m.ltc.inter_neurons,
            command_neurons=m.ltc.command_neurons,
            motor_neurons=m.ltc.motor_neurons,
            sparsity_level=m.ltc.sparsity_level,
            mode=m.ltc.mode,
        )'''

NEW2 = '''\
        _use_ltc = getattr(cfg, "ablation_use_ltc", True)
        if _use_ltc:
            self.fusion = LTCNCPFusion(
                fused_dim=ltc_in,
                inter_neurons=m.ltc.inter_neurons,
                command_neurons=m.ltc.command_neurons,
                motor_neurons=m.ltc.motor_neurons,
                sparsity_level=m.ltc.sparsity_level,
                mode=m.ltc.mode,
            )
            self._fusion_is_gru = False
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
            self._fusion_is_gru = True'''

if OLD2 not in src:
    print(f"ERROR: Patch 2 anchor not found:\n{OLD2}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD2, NEW2, 1)
print("Patch 2 applied: GRU fallback in __init__")


# ── PATCH 3: modality masking + Δt toggle in forward ─────────────────────────
OLD3 = '''\
        fused    = torch.cat([f_cam, f_rad, f_lid, f_imu], dim=-1)
        fused_dt = self.dt_inject(fused, batch["delta_t"])'''

NEW3 = '''\
        # ── Ablation: zero out disabled modalities ───────────────────────────
        _active = getattr(self.cfg, "ablation_modalities",
                          ["camera", "lidar", "radar", "imu"])
        if "radar" not in _active:
            f_rad = torch.zeros_like(f_rad)
        if "lidar" not in _active:
            f_lid = torch.zeros_like(f_lid)
        if "imu" not in _active:
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
            )'''

if OLD3 not in src:
    print(f"ERROR: Patch 3 anchor not found:\n{OLD3}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD3, NEW3, 1)
print("Patch 3 applied: modality masking + Δt toggle in forward")


# ── PATCH 4: dispatch LTC vs GRU in forward ──────────────────────────────────
OLD4 = '        motor_out, hx_new = self.fusion(fused_dt, hx=hx)'

NEW4 = '''\
        # ── Ablation: LTC-NCP vs GRU fusion ─────────────────────────────────
        if not self._fusion_is_gru:
            motor_out, hx_new = self.fusion(fused_dt, hx=hx)
        else:
            _x = fused_dt.unsqueeze(1)           # (B, 1, features)
            _out, hx_new = self.fusion(_x, hx)   # hx: (num_layers, B, hidden)
            motor_out = self._fusion_proj(_out.squeeze(1))'''

if OLD4 not in src:
    print(f"ERROR: Patch 4 anchor not found:\n{OLD4}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD4, NEW4, 1)
print("Patch 4 applied: GRU/LTC dispatch in forward")


# ── Write patched file ────────────────────────────────────────────────────────
MODEL.write_text(src)
print(f"\nAll 4 patches applied → {MODEL}")
print("\nVerify with:")
print("  grep -n 'ablation_modalities\\|ablation_use_ltc\\|ablation_inject_dt\\|_fusion_is_gru' \\")
print(f"    {MODEL}")