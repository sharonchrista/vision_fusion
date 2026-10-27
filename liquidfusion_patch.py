"""
liquidfusion_patch.py  —  Wire ablation flags into liquidfusionnet.py

Applies 4 patches:
  1. Store cfg as self.cfg in __init__
  2. Add GRU fallback instantiation in __init__ (when ablation_use_ltc=False)
  3. Add modality zeroing + Δt toggle in forward
  4. Swap fusion call to use GRU when ablation_use_ltc=False

Run with:
    python liquidfusion_patch.py
"""

import sys
from pathlib import Path

MODEL = Path("/home/sharon/vision_fusion/src/models/liquidfusionnet.py")

if not MODEL.exists():
    print(f"ERROR: {MODEL} not found", file=sys.stderr)
    sys.exit(1)

src = MODEL.read_text()

# ── PATCH 1: store cfg as self.cfg right after super().__init__() ─────────────
# We look for the __init__ signature line and the first self.xxx assignment
# to insert self.cfg = cfg before it.
OLD1 = '        self.cam_enc = CameraEncoder('
NEW1 = '''\
        self.cfg      = cfg                               # needed by forward for ablation flags

        self.cam_enc = CameraEncoder('''

if OLD1 not in src:
    print(f"ERROR: Patch 1 anchor not found:\n{OLD1}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 applied: self.cfg = cfg")


# ── PATCH 2: Add GRU fallback after LTCNCPFusion instantiation ───────────────
# Target the line that sets self.fusion = LTCNCPFusion(...)
# We'll wrap it so that when ablation_use_ltc=False a GRU is used instead.
OLD2 = '''\
        self.fusion = LTCNCPFusion(fused_dim=ltc_in,'''

NEW2 = '''\
        _use_ltc = getattr(cfg, "ablation_use_ltc", True)
        if _use_ltc:
            self.fusion = LTCNCPFusion(fused_dim=ltc_in,'''

if OLD2 not in src:
    print(f"ERROR: Patch 2 anchor not found:\n{OLD2}", file=sys.stderr)
    sys.exit(1)

# We need to find the full LTCNCPFusion(...) call — it may span multiple lines.
# Find the block that starts with OLD2 and ends with the closing ')' + newline.
import re
# Find position of OLD2
pos = src.index(OLD2)
# From pos, find the matching closing paren of the LTCNCPFusion call
depth = 0
end = pos + len("        self.fusion = LTCNCPFusion(")
for i in range(end, len(src)):
    if src[i] == '(':
        depth += 1
    elif src[i] == ')':
        if depth == 0:
            end = i + 1
            break
        depth -= 1

ltc_call_block = src[pos:end]  # full "        self.fusion = LTCNCPFusion(...)"

# Build the hidden_size from the ltc_in variable.  GRU equivalent:
#   hidden_size ~ same as LTCNCPFusion's hidden units (typically fused_dim)
# We'll derive it from ltc_in at runtime.
replacement = '''\
        _use_ltc = getattr(cfg, "ablation_use_ltc", True)
        if _use_ltc:
            ''' + ltc_call_block.lstrip() + '''
        else:
            # GRU baseline — comparable parameter budget
            _gru_hidden = ltc_in          # same width as LTC input
            self.fusion = torch.nn.GRU(
                input_size=ltc_in,
                hidden_size=_gru_hidden,
                num_layers=2,
                batch_first=True,
            )
            self._fusion_proj = torch.nn.Linear(_gru_hidden, ltc_in)
            self._fusion_is_gru = True'''

src = src[:pos] + replacement + src[end:]
print("Patch 2 applied: GRU fallback in __init__")


# ── PATCH 3: Wire modality masking + Δt toggle in forward ────────────────────
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
        if "imu"   not in _active:
            f_imu = torch.zeros_like(f_imu)
        # camera is always kept (every variant includes camera)

        fused = torch.cat([f_cam, f_rad, f_lid, f_imu], dim=-1)

        # ── Ablation: optionally skip Δt injection ───────────────────────────
        _inject_dt = getattr(self.cfg, "ablation_inject_dt", True)
        if _inject_dt:
            fused_dt = self.dt_inject(fused, batch["delta_t"])
        else:
            # pad to same width so downstream dims stay consistent
            fused_dt = torch.cat(
                [fused, torch.zeros(fused.shape[0], 4, device=fused.device)],
                dim=-1,
            )'''

if OLD3 not in src:
    print(f"ERROR: Patch 3 anchor not found:\n{OLD3}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD3, NEW3, 1)
print("Patch 3 applied: modality masking + Δt toggle in forward")


# ── PATCH 4: Swap fusion call for GRU path ───────────────────────────────────
OLD4 = '''\
        motor_out, hx_new = self.fusion(fused_dt, hx=hx)'''

NEW4 = '''\
        # ── Ablation: LTC-NCP vs GRU fusion ─────────────────────────────────
        if getattr(self.cfg, "ablation_use_ltc", True):
            motor_out, hx_new = self.fusion(fused_dt, hx=hx)
        else:
            # GRU path: (batch, 1, features) → (batch, 1, hidden) → proj
            _x = fused_dt.unsqueeze(1)          # add seq dim
            _out, hx_new = self.fusion(_x, hx)  # hx: (num_layers, B, H)
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
print("  grep -n 'ablation_modalities\\|ablation_use_ltc\\|ablation_inject_dt\\|_fusion_is_gru\\|zeros_like'")
print(f"    {MODEL}")