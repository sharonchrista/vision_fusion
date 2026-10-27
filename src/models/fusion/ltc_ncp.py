"""
LTC-NCP temporal fusion core.

Uses Closed-form Continuous-time (CfC) cells from the `ncps` library —
same dynamics as LTC but solved analytically, ~10× faster on T4 GPUs
vs ODE-solver variants.

Architecture:
    Fused sensor vector (B, fused_dim + 4 Δt values)
        ↓
    CfC RNN  (inter_neurons)
        ↓
    NCP wiring  (command_neurons → motor_neurons)
        ↓
    Shared latent  (B, motor_neurons)

The motor neuron outputs fan out to the three task heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LTCNCPFusion(nn.Module):
    """
    Continuous-time multimodal fusion block.

    Parameters
    ----------
    fused_dim : int
        Dimensionality of the concatenated spatial encoder outputs.
    inter_neurons : int
        Number of LTC inter-neurons (hidden units).
    command_neurons : int
        Number of NCP command neurons.
    motor_neurons : int
        Number of output neurons (one per task head).
    sparsity_level : float
        Fraction of inter→command connections that are active.
    mode : str
        CfC mode: 'default' | 'pure' | 'no_gate'
    """

    def __init__(self, fused_dim: int = 708,
                 inter_neurons: int = 64,
                 command_neurons: int = 32,
                 motor_neurons: int = 3,
                 sparsity_level: float = 0.5,
                 mode: str = "default"):
        super().__init__()
        self.motor_neurons  = motor_neurons
        self.inter_neurons  = inter_neurons
        self.command_neurons = command_neurons

        try:
            from ncps.torch import CfC
            from ncps.wirings import AutoNCP
            wiring = AutoNCP(
                units=inter_neurons + command_neurons,
                output_size=motor_neurons,
                sparsity_level=sparsity_level,
            )
            self.rnn = CfC(fused_dim, wiring, mode=mode, batch_first=True)
            self._use_ncps = True
        except ImportError:
            # Graceful fallback to GRU for environments without ncps installed
            import warnings
            warnings.warn(
                "ncps not installed — falling back to GRU. "
                "Install with: pip install ncps==0.0.7",
                stacklevel=2,
            )
            self.rnn = nn.GRU(
                input_size=fused_dim,
                hidden_size=inter_neurons,
                batch_first=True,
            )
            self.motor_proj = nn.Linear(inter_neurons, motor_neurons)
            self._use_ncps = False

    def forward(self, fused: torch.Tensor,
                delta_t: torch.Tensor | None = None,
                hx: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused : (B, T, fused_dim)  or  (B, fused_dim) for single-step
        delta_t : (B, T) elapsed seconds per timestep (required for LTC)
        hx : previous hidden state

        Returns
        -------
        output : (B, motor_neurons)  — last timestep motor activations
        hx_new : new hidden state
        """
        if fused.dim() == 2:
            fused = fused.unsqueeze(1)          # (B, 1, fused_dim)

        if self._use_ncps:
            if delta_t is not None:
                if delta_t.dim() == 1:
                    delta_t = delta_t.unsqueeze(1)   # (B, 1)
                out, hx_new = self.rnn(fused, hx, timespans=delta_t)
            else:
                out, hx_new = self.rnn(fused, hx)
        else:
            out, hx_new = self.rnn(fused, hx)
            out = self.motor_proj(out)

        # out: (B, T, motor_neurons) — take last step
        return out[:, -1, :], hx_new


class DeltaTInjector(nn.Module):
    """
    Concatenates per-sensor Δt values onto the fused feature vector.

    delta_t dict keys: cam, rad, lid, imu → each (B,) scalar
    Appended as 4 extra dimensions to fused vector.
    """
    def forward(self, fused: torch.Tensor,
                delta_t: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [fused]
        for key in ["cam", "rad", "lid", "imu"]:
            dt = delta_t.get(key, torch.zeros(fused.shape[0], 1,
                                              device=fused.device))
            if dt.dim() == 0:
                dt = dt.unsqueeze(0).expand(fused.shape[0], 1)
            elif dt.dim() == 1:
                dt = dt.unsqueeze(1)
            parts.append(dt)
        return torch.cat(parts, dim=-1)    # (B, fused_dim + 4)
