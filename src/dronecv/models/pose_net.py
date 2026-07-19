"""Absolute pose regression (APR) with learned heteroscedastic uncertainty.

Predicts ENU position, heading (sin/cos), and per-sample log-variances for
both (Kendall-style aleatoric uncertainty): the loss automatically
down-weights views the network finds ambiguous, and at inference the
predicted sigma feeds the EKF measurement covariance (after the calibration
scale fitted by the evaluator — networks are overconfident out of the box).

The filter attitude is yaw-only (kinematic drone, fixed camera tilt), so the
orientation target is heading sin/cos rather than a full 3D rotation; the
uncertainty treatment is what matters for fusion.
"""

from __future__ import annotations

import torch
from torch import nn

from dronecv.models.backbone import TinyConvNet


class PoseNet(nn.Module):
    def __init__(self, width: int = 32, pos_scale_m: float = 500.0):
        super().__init__()
        self.backbone = TinyConvNet(width)
        self.pos_scale_m = pos_scale_m
        d = self.backbone.out_dim
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(inplace=True))
        self.pos_head = nn.Linear(d, 3)  # ENU position / pos_scale_m
        self.heading_head = nn.Linear(d, 2)  # (sin, cos) of true-north heading
        self.logvar_pos = nn.Linear(d, 1)  # log sigma^2 in scaled units
        self.logvar_heading = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.head(self.backbone(x))
        return {
            "pos_scaled": self.pos_head(f),
            "heading_vec": self.heading_head(f),
            "logvar_pos": self.logvar_pos(f).squeeze(-1).clamp(-8, 6),
            "logvar_heading": self.logvar_heading(f).squeeze(-1).clamp(-8, 6),
        }

    def loss(
        self, out: dict[str, torch.Tensor], pos_enu: torch.Tensor, heading_sincos: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = pos_enu / self.pos_scale_m
        pos_err = (out["pos_scaled"] - target).pow(2).sum(-1)
        lv_p = out["logvar_pos"]
        loss_pos = (pos_err * torch.exp(-lv_p) + lv_p).mean()

        head_err = (out["heading_vec"] - heading_sincos).pow(2).sum(-1)
        lv_h = out["logvar_heading"]
        loss_head = (head_err * torch.exp(-lv_h) + lv_h).mean()

        loss = loss_pos + 0.5 * loss_head
        stats = {
            "loss": float(loss.detach()),
            "pos_rmse_m": float(pos_err.detach().mean().sqrt() * self.pos_scale_m),
        }
        return loss, stats

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Physical units: pos_enu (m), heading_deg, sigma_pos_m, sigma_heading_deg."""
        out = self(x)
        pos = out["pos_scaled"] * self.pos_scale_m
        v = out["heading_vec"]
        heading = torch.rad2deg(torch.atan2(v[..., 0], v[..., 1])) % 360.0
        # Per-axis sigma in meters: sqrt of (scaled variance / 3 axes).
        sigma_pos = torch.exp(0.5 * out["logvar_pos"]) * self.pos_scale_m / (3.0**0.5)
        sigma_heading = torch.rad2deg(torch.exp(0.5 * out["logvar_heading"])).clamp(1.0, 90.0)
        return {
            "pos_enu": pos,
            "heading_deg": heading,
            "sigma_pos_m": sigma_pos,
            "sigma_heading_deg": sigma_heading,
        }
