"""Local KL anchor helpers for DGPO (truth‑p_T–weighted velocity MSE vs frozen reference).

This module was restored as a minimal implementation so ``dgpo_trainer.py`` imports resolve
after optional files were removed from the working tree. Extend here if you need full behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

_log = logging.getLogger(__name__)


@dataclass
class LocalKlAnchorTrainConfig:
    enabled: bool = False
    min_count: int = 50


def _dgpo_block(dg: Any, key: str) -> Any | None:
    if dg is None:
        return None
    if isinstance(dg, dict):
        return dg.get(key)
    return getattr(dg, key, None)


def resolve_local_kl_anchor_train_config(dg: Any) -> LocalKlAnchorTrainConfig:
    block = _dgpo_block(dg, "local_kl_anchor")
    if block is None:
        return LocalKlAnchorTrainConfig()
    en = bool(block.get("enabled", False)) if isinstance(block, dict) else bool(
        getattr(block, "enabled", False)
    )
    mc = block.get("min_count", 50) if isinstance(block, dict) else getattr(block, "min_count", 50)
    try:
        mc_i = int(mc)
    except (TypeError, ValueError):
        mc_i = 50
    return LocalKlAnchorTrainConfig(enabled=en, min_count=max(1, mc_i))


def truth_pt_slot_kb_gev(
    batch_rep: dict[str, Any],
    *,
    cartesian: bool,
    num_slots: int,
) -> Tensor:
    """Truth neutrino p_T [GeV] per row, shape ``(B_kb, num_slots)``."""
    if cartesian and isinstance(batch_rep.get("x_invisible_cartesian"), Tensor):
        xyz = batch_rep["x_invisible_cartesian"]
        pt = torch.sqrt(xyz[..., 0].pow(2) + xyz[..., 1].pow(2) + 1e-12)
    elif isinstance(batch_rep.get("x_invisible"), Tensor):
        t = batch_rep["x_invisible"]
        pt = torch.expm1(t[..., 0].clamp(-10.0, 10.0))
    else:
        B = int(batch_rep["x"].shape[0])
        dev, dt = batch_rep["x"].device, batch_rep["x"].dtype
        return torch.zeros((B, num_slots), device=dev, dtype=dt)
    n = min(int(pt.shape[1]), int(num_slots))
    out = pt[:, :n]
    if n < num_slots:
        pad = num_slots - n
        out = torch.cat(
            [out, torch.zeros(out.shape[0], pad, device=out.device, dtype=out.dtype)],
            dim=1,
        )
    return out


def compute_local_kl_anchor_loss(
    model_v: Tensor,
    ref_v: Tensor,
    noise_mask_rep: Tensor,
    truth_pt_kb: Tensor,
    *,
    cfg: LocalKlAnchorTrainConfig,
    p0_ref: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked mean squared velocity error vs detached reference (optional Gaussian weighting)."""
    del truth_pt_kb, p0_ref  # reserved for full Gaussian-by-truth-p_T weighting
    if not cfg.enabled:
        z = model_v.new_zeros(())
        return z, {}
    d = (model_v - ref_v.detach()).pow(2)
    if noise_mask_rep.dim() == 3:
        m = noise_mask_rep.squeeze(-1).to(dtype=d.dtype)
    else:
        m = noise_mask_rep.to(dtype=d.dtype)
    while m.dim() < d.dim():
        m = m.unsqueeze(-1)
    num = (d * m).sum()
    den = m.sum().clamp(min=1e-8)
    loss = num / den
    diag: dict[str, Tensor] = {
        "local_kl_anchor/loss": loss.detach(),
        "local_kl_anchor/active": torch.ones((), device=loss.device, dtype=torch.float64),
    }
    return loss, diag


def fit_reference_pt_residual_profile_p0(
    tt: np.ndarray,
    dd: np.ndarray,
    *,
    min_nonempty_bins: int = 2,
    min_total_slots: int = 1,
) -> tuple[float, float, float, float]:
    """Linear fit ``dd ≈ slope * tt + intercept``; return ``(p0, slope, intercept, n_slots)``."""
    del min_nonempty_bins
    tt = np.asarray(tt, dtype=np.float64).reshape(-1)
    dd = np.asarray(dd, dtype=np.float64).reshape(-1)
    m = np.isfinite(tt) & np.isfinite(dd)
    tt = tt[m]
    dd = dd[m]
    ns = float(tt.size)
    if tt.size < max(1, int(min_total_slots)):
        return float("nan"), float("nan"), float("nan"), ns
    if tt.size == 1:
        return float("nan"), float("nan"), float("nan"), ns
    slope, intercept = np.polyfit(tt, dd, 1)
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        p0 = float("nan")
    else:
        p0 = float(-intercept / slope)
    return p0, float(slope), float(intercept), ns
