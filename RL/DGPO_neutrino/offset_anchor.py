"""Optional residual offset anchor for DGPO neutrino training.

Anchors pooled mean kinematic residuals inside a finite truth ``p_T`` band:

- ``pt_mean`` (default): ``mean(pred p_T − truth p_T)``.
- ``xyz_mean``: ``mean(pred (p_x,p_y,p_z) − truth (p_x,p_y,p_z))`` per component after
  Cartesian conversion when needed.
- ``pt_eta_phi_mean``: ``mean(pred (p_T,eta,phi) - truth (p_T,eta,phi))`` per
  component, with wrapped ``phi`` residual and per-component scaling.

Computed from differentiable single-step diffusion predictions (``model_v -> x_0_hat``).

This module is intentionally self-contained so the anchor can be enabled/disabled via config
without spreading calibration logic across the trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from RL.DGPO_neutrino.dgpo_utils import (
    _dgpo_cfg_get,
    piecewise_linear_schedule_at,
    predict_x0_normalized_from_velocity_diffusion,
)
from RL.DGPO_neutrino.rewards import (
    cartesian_to_log_pt_eta_phi,
    get_event_valid_mask,
    invisible_kinematics_to_cartesian,
)


def _finite_pt_scale_gev(normalization_dict: Mapping[str, Any] | dict[str, Any] | None) -> float:
    """Linear ``p_T`` scale in GeV (same precedence as dgpo_trainer._pt_scale_from_normalization)."""
    if normalization_dict is None:
        raise ValueError(
            "offset_anchor scale=auto requires a normalization.pt dict; set scale to a positive "
            "float in YAML otherwise."
        )
    if "invisible_pt_std" in normalization_dict:
        std_t = normalization_dict["invisible_pt_std"]["Source"]
        std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
        if not std_list:
            raise ValueError("invisible_pt_std['Source'] must be non-empty.")
        scale = float(std_list[0])
        if scale > 0.0:
            return scale
        raise ValueError(f"invisible_pt_std['Source'][0] must be positive, got {scale}")
    if "invisible_cartesian_std" not in normalization_dict:
        raise ValueError(
            "offset_anchor scale=auto requires 'invisible_pt_std' or 'invisible_cartesian_std' "
            "in normalization.pt. Set dgpo.offset_anchor.scale to an explicit positive float."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std_list) < 2:
        raise ValueError("invisible_cartesian_std['Source'] must contain at least [px, py].")
    scale = math.sqrt(0.5 * (float(std_list[0]) ** 2 + float(std_list[1]) ** 2))
    if scale <= 0.0:
        raise ValueError(
            "transverse pT scale from invisible_cartesian_std must be positive, got "
            f"{scale}"
        )
    return scale


def _finite_xyz_scale_gev(
    normalization_dict: Mapping[str, Any] | dict[str, Any] | None,
) -> tuple[float, float, float]:
    """XYZ linear scales ``(σ_px, σ_py, σ_pz)`` in GeV from ``normalization.pt``."""
    if normalization_dict is None:
        raise ValueError(
            "offset_anchor mode=xyz_mean with scale=auto requires normalization.pt dict; "
            "set dgpo.offset_anchor.scale or scale_xyz explicitly otherwise."
        )
    if "invisible_cartesian_std" not in normalization_dict:
        raise ValueError(
            "offset_anchor mode=xyz_mean with scale=auto requires "
            "'invisible_cartesian_std' in normalization.pt (Source entries [σ_px, σ_py, σ_pz]). "
            "Set dgpo.offset_anchor.scale / scale_xyz to explicit positive values otherwise."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std_list) < 3:
        raise ValueError(
            "invisible_cartesian_std['Source'] must contain at least [px, py, pz] for xyz_mean "
            "scale=auto."
        )
    scales = tuple(float(std_list[i]) for i in range(3))
    for i, s in enumerate(scales):
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError(
                f"invisible_cartesian_std Source[{i}] must be a positive finite float for xyz "
                f"scales, got {s}"
            )
    return scales


def _finite_pt_eta_phi_scale(
    normalization_dict: Mapping[str, Any] | dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Scales for ``(pT, eta, phi)``: linear-pT scale plus training ``invisible_std`` angular scales."""
    pt_scale = _finite_pt_scale_gev(normalization_dict)
    if normalization_dict is None or "invisible_std" not in normalization_dict:
        raise ValueError(
            "offset_anchor mode=pt_eta_phi_mean with scale_pt_eta_phi=auto requires "
            "normalization.pt['invisible_std']['Source'] for eta/phi scales."
        )
    std_t = normalization_dict["invisible_std"]["Source"]
    std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std_list) < 3:
        raise ValueError(
            "invisible_std['Source'] must contain at least [log_pt, eta, phi] for "
            "pt_eta_phi_mean scale_pt_eta_phi=auto."
        )
    eta_scale = float(std_list[1])
    phi_scale = float(std_list[2])
    for name, val in (("eta", eta_scale), ("phi", phi_scale)):
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(
                f"invisible_std['Source'] {name} scale must be positive and finite, got {val}"
            )
    return (float(pt_scale), eta_scale, phi_scale)


@dataclass(frozen=True)
class OffsetAnchorDualControlConfig:
    """Optional PI dual term on pooled pT residual (``pt_mean`` / ``pt_eta_phi_mean``).

    ``leak`` and ``deadband`` implement anti-windup: ``dual_pt`` leaks toward zero when EMA drift
    is inside the deadband, instead of accumulating one-sided corrections forever.
    """

    enabled: bool
    component: str
    target: float
    ema_decay: float
    dual_lr: float
    dual_clip: float
    warmup_steps: int
    leak: float
    deadband: float


@dataclass(frozen=True)
class OffsetAnchorTrainConfig:
    """Resolved DGPO ``offset_anchor`` block for training-time loss."""

    enabled: bool
    mode: str
    pt_min: float
    pt_max: float
    scale: float
    scale_xyz: tuple[float, float, float]
    scale_pt_eta_phi: tuple[float, float, float]
    lambda_coef: float
    mu_ref: float
    mu_ref_xyz: tuple[float, float, float]
    mu_ref_pt_eta_phi: tuple[float, float, float]
    loss_type: str
    min_count: int
    apply_to: str
    dual_control: OffsetAnchorDualControlConfig


def resolve_offset_anchor_dual_control(
    block: Mapping[str, Any] | Any | None,
    *,
    mode: str,
    mu_ref_scalar: float,
    mu_ref_pt_eta_phi: tuple[float, float, float],
) -> OffsetAnchorDualControlConfig:
    """Resolve ``dgpo.offset_anchor.dual_control`` (plug-in PI / augmented Lagrangian on pT)."""
    dc = _dgpo_cfg_get(block, "dual_control", None) if block is not None else None
    default = OffsetAnchorDualControlConfig(
        enabled=False,
        component="pt",
        target=float(mu_ref_scalar if mode != "pt_eta_phi_mean" else mu_ref_pt_eta_phi[0]),
        ema_decay=0.95,
        dual_lr=0.02,
        dual_clip=5.0,
        warmup_steps=0,
        leak=1.0,
        deadband=0.0,
    )
    if dc is None or not bool(_dgpo_cfg_get(dc, "enabled", False)):
        return default

    component = str(_dgpo_cfg_get(dc, "component", "pt") or "pt").strip().lower()
    if component != "pt":
        raise ValueError(
            "dgpo.offset_anchor.dual_control.component currently only supports 'pt', "
            f"got {component!r}"
        )
    if mode not in ("pt_mean", "pt_eta_phi_mean"):
        raise ValueError(
            "dgpo.offset_anchor.dual_control requires mode=pt_mean or pt_eta_phi_mean, "
            f"got {mode!r}"
        )

    target_raw = _dgpo_cfg_get(dc, "target", None)
    if target_raw is None:
        tgt = (
            float(mu_ref_pt_eta_phi[0]) if mode == "pt_eta_phi_mean" else float(mu_ref_scalar)
        )
    else:
        tgt = float(target_raw)

    if not math.isfinite(tgt):
        raise ValueError(f"dgpo.offset_anchor.dual_control.target must be finite, got {tgt}")

    expected_ref = (
        float(mu_ref_pt_eta_phi[0]) if mode == "pt_eta_phi_mean" else float(mu_ref_scalar)
    )
    if abs(float(tgt) - float(expected_ref)) > 1e-6:
        raise ValueError(
            "dgpo.offset_anchor.dual_control.target must match the anchor pT reference "
            f"(mu_ref_pt_eta_phi[0]={expected_ref} for pt_eta_phi_mean, mu_ref={expected_ref} for "
            f"pt_mean); got target={tgt}. Align YAML or omit target to default to anchor ref."
        )

    ema_decay = float(_dgpo_cfg_get(dc, "ema_decay", 0.95))
    if not math.isfinite(ema_decay) or not (0.0 <= ema_decay < 1.0):
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.ema_decay must be finite in [0, 1), got {ema_decay}"
        )

    dual_lr = float(_dgpo_cfg_get(dc, "dual_lr", 0.02))
    if not math.isfinite(dual_lr) or dual_lr < 0.0:
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.dual_lr must be finite and nonnegative, got {dual_lr}"
        )

    dual_clip = float(_dgpo_cfg_get(dc, "dual_clip", 5.0))
    if not math.isfinite(dual_clip) or dual_clip <= 0.0:
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.dual_clip must be positive and finite, got {dual_clip}"
        )

    warmup_steps = int(_dgpo_cfg_get(dc, "warmup_steps", 20))
    if warmup_steps < 0:
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.warmup_steps must be >= 0, got {warmup_steps}"
        )

    leak = float(_dgpo_cfg_get(dc, "leak", 1.0))
    if not math.isfinite(leak) or leak <= 0.0 or leak > 1.0:
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.leak must be finite in (0, 1], got {leak}"
        )

    deadband_v = float(_dgpo_cfg_get(dc, "deadband", 0.0))
    if not math.isfinite(deadband_v) or deadband_v < 0.0:
        raise ValueError(
            f"dgpo.offset_anchor.dual_control.deadband must be finite and nonnegative, got {deadband_v}"
        )

    return OffsetAnchorDualControlConfig(
        enabled=True,
        component=component,
        target=tgt,
        ema_decay=ema_decay,
        dual_lr=dual_lr,
        dual_clip=dual_clip,
        warmup_steps=warmup_steps,
        leak=leak,
        deadband=deadband_v,
    )


def offset_anchor_dual_state_step(
    *,
    cfg: OffsetAnchorDualControlConfig,
    z_pt_batch: float,
    dual_pt: float,
    z_ema_pt: float,
    global_step: int,
) -> tuple[float, float, dict[str, float]]:
    """One detached dual controller update after an optimizer step.

    Args:
        cfg: Resolved dual control config.
        z_pt_batch: DDP-mean of detached ``z_pt`` for this step.
        dual_pt: Current dual multiplier.
        z_ema_pt: Current EMA of ``z_pt``.
        global_step: Training step (0-based); dual updates only after ``warmup_steps``.

    Returns:
        ``(new_dual_pt, new_z_ema_pt, detached_metrics)``.
    """
    if not cfg.enabled:
        return dual_pt, z_ema_pt, {}

    if int(global_step) < int(cfg.warmup_steps):
        return dual_pt, z_ema_pt, {"offset_anchor/dual_active": 0.0}

    zb = float(z_pt_batch)
    if not math.isfinite(zb):
        return dual_pt, z_ema_pt, {"offset_anchor/dual_active": 0.0}

    ema = float(cfg.ema_decay)
    z_new = ema * float(z_ema_pt) + (1.0 - ema) * zb
    leak_f = float(cfg.leak)
    db = float(cfg.deadband)
    dp_prev = float(dual_pt)
    integrating = abs(float(z_new)) >= db

    if integrating:
        du_raw = leak_f * dp_prev + float(cfg.dual_lr) * float(z_new)
    else:
        du_raw = leak_f * dp_prev

    clip_v = float(cfg.dual_clip)
    du_clipped = min(clip_v, max(-clip_v, du_raw))

    out = {
        "offset_anchor/dual_active": 1.0,
        "offset_anchor/z_pt_batch_mean": zb,
        "offset_anchor/z_ema_pt_after": float(z_new),
        "offset_anchor/dual_integrating": 1.0 if integrating else 0.0,
        "offset_anchor/dual_leak": float(leak_f),
        "offset_anchor/dual_deadband": float(db),
        "offset_anchor/dual_update_pt": float(du_clipped - dp_prev),
        "offset_anchor/dual_pt_after": float(du_clipped),
    }
    return du_clipped, z_new, out


def resolve_offset_anchor_train_config(
    dgpo_cfg: Mapping[str, Any] | Any | None,
    normalization_dict: Mapping[str, Any] | dict[str, Any] | None,
    *,
    stored_mu_ref: float | None = None,
    stored_mu_ref_xyz: tuple[float, float, float] | None = None,
) -> OffsetAnchorTrainConfig:
    """Read YAML ``dgpo.offset_anchor`` and return a frozen config for training.

    Args:
        dgpo_cfg: ``global_config.dgpo`` mapping or DotDict.
        normalization_dict: ``normalization.pt`` contents when ``scale: auto``.
        stored_mu_ref: Frozen baseline from checkpoint; overrides YAML ``mu_ref`` when not None
            (``pt_mean``).
        stored_mu_ref_xyz: Frozen XYZ baseline triple from checkpoint (``xyz_mean``); overrides YAML
            ``mu_ref_xyz`` when not None.

    Raises:
        ValueError: Misconfiguration.
    """
    block = _dgpo_cfg_get(dgpo_cfg, "offset_anchor", None)
    if block is None or not bool(_dgpo_cfg_get(block, "enabled", False)):
        _mr = float(_dgpo_cfg_get(block, "mu_ref", 0.0)) if block is not None else 0.0
        return OffsetAnchorTrainConfig(
            enabled=False,
            mode="pt_mean",
            pt_min=20.0,
            pt_max=80.0,
            scale=1.0,
            scale_xyz=(1.0, 1.0, 1.0),
            scale_pt_eta_phi=(1.0, 1.0, 1.0),
            lambda_coef=0.0,
            mu_ref=_mr,
            mu_ref_xyz=(0.0, 0.0, 0.0),
            mu_ref_pt_eta_phi=(0.0, 0.0, 0.0),
            loss_type="huber",
            min_count=50,
            apply_to=str(_dgpo_cfg_get(block, "apply_to", "all_candidates") or "all_candidates"),
            dual_control=OffsetAnchorDualControlConfig(
                enabled=False,
                component="pt",
                target=_mr,
                ema_decay=0.95,
                dual_lr=0.02,
                dual_clip=5.0,
                warmup_steps=0,
                leak=1.0,
                deadband=0.0,
            ),
        )

    mode_raw = str(_dgpo_cfg_get(block, "mode", "pt_mean") or "pt_mean").strip().lower()
    if mode_raw == "ptetaphi_mean":
        mode_raw = "pt_eta_phi_mean"
    if mode_raw not in ("pt_mean", "xyz_mean", "pt_eta_phi_mean"):
        raise ValueError(
            "dgpo.offset_anchor.mode must be pt_mean, xyz_mean, or pt_eta_phi_mean, "
            f"got {mode_raw!r}"
        )

    pt_min = float(_dgpo_cfg_get(block, "pt_min", 20.0))
    pt_max = float(_dgpo_cfg_get(block, "pt_max", 80.0))
    if not (math.isfinite(pt_min) and math.isfinite(pt_max) and pt_max > pt_min):
        raise ValueError(
            f"dgpo.offset_anchor requires finite pt_min < pt_max, got {pt_min} {pt_max}"
        )

    scale_raw = _dgpo_cfg_get(block, "scale", "auto")
    scale_xyz_explicit = _dgpo_cfg_get(block, "scale_xyz", None)
    scale_pt_eta_phi_explicit = _dgpo_cfg_get(block, "scale_pt_eta_phi", None)

    def _triple_from_seq(
        seq: Any,
        *,
        label: str,
        auto_values: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float]:
        lst = list(seq) if hasattr(seq, "__iter__") and not isinstance(seq, (str, bytes)) else None
        if lst is None or len(lst) != 3:
            raise ValueError(
                f"dgpo.offset_anchor.{label} must be a length-3 sequence [sx, sy, sz], "
                f"got {seq!r}"
            )
        vals: list[float] = []
        for i, x in enumerate(lst):
            if isinstance(x, str) and x.strip().lower() == "auto":
                if auto_values is None:
                    raise ValueError(
                        f"dgpo.offset_anchor.{label}[{i}]=auto requires an auto scale source"
                    )
                vals.append(float(auto_values[i]))
            else:
                vals.append(float(x))
        out = tuple(vals)
        for i, s in enumerate(out):
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError(
                    f"dgpo.offset_anchor.{label}[{i}] must be a positive finite float, got {s}"
                )
        return out

    scale_val: float
    scale_xyz_val: tuple[float, float, float]
    scale_pt_eta_phi_val: tuple[float, float, float]

    if mode_raw == "xyz_mean":
        if scale_pt_eta_phi_explicit is not None:
            raise ValueError(
                "dgpo.offset_anchor.scale_pt_eta_phi is only used when mode=pt_eta_phi_mean "
                f"(got mode={mode_raw!r})."
            )
        if scale_xyz_explicit is not None:
            scale_xyz_val = _triple_from_seq(scale_xyz_explicit, label="scale_xyz")
        elif isinstance(scale_raw, str) and str(scale_raw).strip().lower() == "auto":
            scale_xyz_val = _finite_xyz_scale_gev(normalization_dict)
        else:
            s = float(scale_raw)
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError(
                    "dgpo.offset_anchor.scale must be 'auto', a positive float, or set scale_xyz for "
                    f"xyz_mean mode, got {scale_raw!r}"
                )
            scale_xyz_val = (s, s, s)
        # Informative combined scale for logging / legacy fields (RMS of component scales).
        scale_val = float(math.sqrt(sum(x * x for x in scale_xyz_val) / 3.0))
        scale_pt_eta_phi_val = (scale_val, 1.0, 1.0)
    elif mode_raw == "pt_eta_phi_mean":
        if scale_xyz_explicit is not None:
            raise ValueError(
                "dgpo.offset_anchor.scale_xyz is only used when mode=xyz_mean "
                f"(got mode={mode_raw!r})."
            )
        if isinstance(scale_raw, str) and str(scale_raw).strip().lower() == "auto":
            scale_val = float(_finite_pt_scale_gev(normalization_dict))
        else:
            scale_val = float(scale_raw)
            if not math.isfinite(scale_val) or scale_val <= 0.0:
                raise ValueError(
                    "dgpo.offset_anchor.scale must be 'auto' or a positive finite float, "
                    f"got {scale_raw!r}"
                )
        if scale_pt_eta_phi_explicit is None:
            scale_pt_eta_phi_val = (scale_val, 1.0, 1.0)
        elif (
            isinstance(scale_pt_eta_phi_explicit, str)
            and str(scale_pt_eta_phi_explicit).strip().lower() == "auto"
        ):
            scale_pt_eta_phi_val = _finite_pt_eta_phi_scale(normalization_dict)
        else:
            auto_pt_eta_phi = _finite_pt_eta_phi_scale(normalization_dict)
            scale_pt_eta_phi_val = _triple_from_seq(
                scale_pt_eta_phi_explicit,
                label="scale_pt_eta_phi",
                auto_values=auto_pt_eta_phi,
            )
        scale_xyz_val = (scale_val, scale_val, scale_val)
    else:
        if scale_xyz_explicit is not None or scale_pt_eta_phi_explicit is not None:
            raise ValueError(
                "dgpo.offset_anchor.scale_xyz / scale_pt_eta_phi are only used in vector modes "
                f"(got mode={mode_raw!r})."
            )
        if isinstance(scale_raw, str) and str(scale_raw).strip().lower() == "auto":
            scale_val = float(_finite_pt_scale_gev(normalization_dict))
        else:
            scale_val = float(scale_raw)
            if not math.isfinite(scale_val) or scale_val <= 0.0:
                raise ValueError(
                    "dgpo.offset_anchor.scale must be 'auto' or a positive finite float, "
                    f"got {scale_raw!r}"
                )
        scale_xyz_val = (scale_val, scale_val, scale_val)
        scale_pt_eta_phi_val = (scale_val, 1.0, 1.0)

    lam = float(_dgpo_cfg_get(block, "lambda", 0.0) or 0.0)
    if lam < 0.0 or not math.isfinite(lam):
        raise ValueError(f"dgpo.offset_anchor.lambda must be finite and nonnegative, got {lam}")

    if stored_mu_ref is not None:
        mu_ref = float(stored_mu_ref)
    else:
        mu_ref = float(_dgpo_cfg_get(block, "mu_ref", 0.0))
    if not math.isfinite(mu_ref):
        raise ValueError(f"dgpo.offset_anchor.mu_ref must be finite, got {mu_ref}")

    mu_ref_xyz_raw = _dgpo_cfg_get(block, "mu_ref_xyz", None)
    if mu_ref_xyz_raw is not None:
        lst = (
            list(mu_ref_xyz_raw)
            if hasattr(mu_ref_xyz_raw, "__iter__") and not isinstance(mu_ref_xyz_raw, (str, bytes))
            else None
        )
        if lst is None or len(lst) != 3:
            raise ValueError(
                f"dgpo.offset_anchor.mu_ref_xyz must be a length-3 sequence, got {mu_ref_xyz_raw!r}"
            )
        mu_ref_xyz = tuple(float(x) for x in lst)
        for i, v in enumerate(mu_ref_xyz):
            if not math.isfinite(v):
                raise ValueError(f"dgpo.offset_anchor.mu_ref_xyz[{i}] must be finite, got {v}")
    else:
        mu_ref_xyz = (0.0, 0.0, 0.0)

    if stored_mu_ref_xyz is not None:
        if len(stored_mu_ref_xyz) != 3:
            raise ValueError("stored_mu_ref_xyz must be a length-3 tuple")
        mu_ref_xyz = tuple(float(x) for x in stored_mu_ref_xyz)
        for i, v in enumerate(mu_ref_xyz):
            if not math.isfinite(v):
                raise ValueError(f"stored_mu_ref_xyz[{i}] must be finite, got {v}")

    mu_ref_pt_eta_phi_raw = _dgpo_cfg_get(block, "mu_ref_pt_eta_phi", None)
    if mu_ref_pt_eta_phi_raw is not None:
        lst = (
            list(mu_ref_pt_eta_phi_raw)
            if hasattr(mu_ref_pt_eta_phi_raw, "__iter__")
            and not isinstance(mu_ref_pt_eta_phi_raw, (str, bytes))
            else None
        )
        if lst is None or len(lst) != 3:
            raise ValueError(
                "dgpo.offset_anchor.mu_ref_pt_eta_phi must be a length-3 sequence, "
                f"got {mu_ref_pt_eta_phi_raw!r}"
            )
        mu_ref_pt_eta_phi = tuple(float(x) for x in lst)
        for i, v in enumerate(mu_ref_pt_eta_phi):
            if not math.isfinite(v):
                raise ValueError(
                    f"dgpo.offset_anchor.mu_ref_pt_eta_phi[{i}] must be finite, got {v}"
                )
    else:
        mu_ref_pt_eta_phi = (mu_ref, 0.0, 0.0)

    loss_type = str(_dgpo_cfg_get(block, "loss_type", "huber") or "huber").strip().lower()
    if loss_type not in ("huber", "mse"):
        raise ValueError(f"dgpo.offset_anchor.loss_type must be huber or mse, got {loss_type!r}")

    min_count = int(_dgpo_cfg_get(block, "min_count", 50))
    if min_count < 1:
        raise ValueError(f"dgpo.offset_anchor.min_count must be >= 1, got {min_count}")

    apply_to = str(_dgpo_cfg_get(block, "apply_to", "all_candidates") or "all_candidates").strip()
    allowed_apply = {"all_candidates", "best_candidate", "weighted_candidates"}
    if apply_to not in allowed_apply:
        raise ValueError(
            f"dgpo.offset_anchor.apply_to must be one of {sorted(allowed_apply)}, got {apply_to!r}"
        )
    if apply_to == "weighted_candidates":
        raise NotImplementedError(
            "dgpo.offset_anchor.apply_to=weighted_candidates is reserved; use all_candidates "
            "or best_candidate until soft candidate weights are wired into the trainer."
        )

    dual_cfg = resolve_offset_anchor_dual_control(
        block,
        mode=mode_raw,
        mu_ref_scalar=mu_ref,
        mu_ref_pt_eta_phi=mu_ref_pt_eta_phi,
    )

    return OffsetAnchorTrainConfig(
        enabled=True,
        mode=mode_raw,
        pt_min=pt_min,
        pt_max=pt_max,
        scale=scale_val,
        scale_xyz=scale_xyz_val,
        scale_pt_eta_phi=scale_pt_eta_phi_val,
        lambda_coef=lam,
        mu_ref=mu_ref,
        mu_ref_xyz=mu_ref_xyz,
        mu_ref_pt_eta_phi=mu_ref_pt_eta_phi,
        loss_type=loss_type,
        min_count=min_count,
        apply_to=apply_to,
        dual_control=dual_cfg,
    )


def resolve_offset_anchor_lambda_coef(
    dgpo_cfg: Mapping[str, Any] | Any | None,
    *,
    fallback_lambda: float,
    global_step: int,
    epoch: int,
) -> float:
    """Return effective ``offset_anchor`` weight, optionally from a piecewise schedule."""
    fallback = float(fallback_lambda)
    if not math.isfinite(fallback) or fallback < 0.0:
        raise ValueError(f"offset_anchor fallback lambda must be finite and nonnegative, got {fallback}")
    block = _dgpo_cfg_get(dgpo_cfg, "offset_anchor", None)
    if block is None:
        return fallback
    sched = _dgpo_cfg_get(block, "lambda_schedule", None)
    if sched is None:
        sched = _dgpo_cfg_get(block, "weight_schedule", None)
    if sched is None or not bool(_dgpo_cfg_get(sched, "enabled", False)):
        return fallback
    mode = str(_dgpo_cfg_get(sched, "mode", "piecewise_linear") or "piecewise_linear").strip().lower()
    if mode != "piecewise_linear":
        raise ValueError(
            f"dgpo.offset_anchor.lambda_schedule.mode must be piecewise_linear, got {mode!r}"
        )
    axis = str(_dgpo_cfg_get(sched, "axis", "epoch") or "epoch").strip().lower()
    if axis in ("epoch",):
        coord = float(int(epoch))
    elif axis in ("step", "global_step", "globalstep"):
        coord = float(int(global_step))
    else:
        raise ValueError(
            "dgpo.offset_anchor.lambda_schedule.axis must be epoch or global_step, "
            f"got {axis!r}"
        )
    raw_points = _dgpo_cfg_get(sched, "points", None)
    if raw_points is None:
        raise ValueError("dgpo.offset_anchor.lambda_schedule.enabled requires points")
    parsed: list[tuple[float, float]] = []
    for i, p in enumerate(list(raw_points)):
        at = _dgpo_cfg_get(p, "at", None)
        val = _dgpo_cfg_get(p, "value", None)
        if at is None or val is None:
            raise ValueError(
                f"dgpo.offset_anchor.lambda_schedule.points[{i}] must have at and value"
            )
        fa = float(at)
        fv = float(val)
        if fa < 0.0 or not math.isfinite(fa) or not math.isfinite(fv) or fv < 0.0:
            raise ValueError(
                "dgpo.offset_anchor.lambda_schedule points require finite at>=0 and value>=0, "
                f"got at={at} value={val}"
            )
        parsed.append((fa, fv))
    parsed.sort(key=lambda z: z[0])
    return piecewise_linear_schedule_at(parsed, coord)


def resolve_offset_anchor_adaptive_lambda_multiplier(
    dgpo_cfg: Mapping[str, Any] | Any | None,
    cfg: OffsetAnchorTrainConfig,
    diag: Mapping[str, Tensor],
) -> tuple[float, dict[str, float]]:
    """Detached adaptive multiplier for offset-anchor weight based on current residual drift."""
    block = _dgpo_cfg_get(dgpo_cfg, "offset_anchor", None)
    adaptive = _dgpo_cfg_get(block, "adaptive_lambda", None) if block is not None else None
    if adaptive is None or not bool(_dgpo_cfg_get(adaptive, "enabled", False)):
        return 1.0, {}

    component = str(_dgpo_cfg_get(adaptive, "component", "pt") or "pt").strip().lower()
    direction = str(_dgpo_cfg_get(adaptive, "direction", "below_ref") or "below_ref").strip().lower()
    gain = float(_dgpo_cfg_get(adaptive, "gain", 1.0))
    deadband = float(_dgpo_cfg_get(adaptive, "deadband", 0.0))
    min_multiplier = float(_dgpo_cfg_get(adaptive, "min_multiplier", 1.0))
    max_multiplier = float(_dgpo_cfg_get(adaptive, "max_multiplier", 5.0))
    base_multiplier = float(_dgpo_cfg_get(adaptive, "base_multiplier", 1.0))
    if gain < 0.0 or deadband < 0.0:
        raise ValueError("dgpo.offset_anchor.adaptive_lambda gain/deadband must be nonnegative")
    if min_multiplier < 0.0 or max_multiplier < min_multiplier:
        raise ValueError("adaptive_lambda requires 0 <= min_multiplier <= max_multiplier")

    key_by_component = {
        "pt": ("offset_anchor/delta_mu_pt", "offset_anchor/delta_mu"),
        "eta": ("offset_anchor/delta_mu_eta",),
        "phi": ("offset_anchor/delta_mu_phi",),
        "px": ("offset_anchor/delta_mu_px",),
        "py": ("offset_anchor/delta_mu_py",),
        "pz": ("offset_anchor/delta_mu_pz",),
    }
    if component not in key_by_component:
        raise ValueError(
            "dgpo.offset_anchor.adaptive_lambda.component must be one of "
            f"{sorted(key_by_component)}, got {component!r}"
        )

    delta_t: Tensor | None = None
    chosen_key = ""
    for key in key_by_component[component]:
        maybe = diag.get(key)
        if isinstance(maybe, Tensor):
            delta_t = maybe.detach().reshape(())
            chosen_key = key
            break
    if delta_t is None:
        return 1.0, {
            "offset_anchor/adaptive_lambda_multiplier": 1.0,
            "offset_anchor/adaptive_lambda_trigger": 0.0,
        }
    delta = float(delta_t.cpu().item())
    if not math.isfinite(delta):
        return 1.0, {
            "offset_anchor/adaptive_lambda_multiplier": 1.0,
            "offset_anchor/adaptive_lambda_trigger": 0.0,
        }

    scale_by_component = {
        "pt": cfg.scale_pt_eta_phi[0] if cfg.mode == "pt_eta_phi_mean" else cfg.scale,
        "eta": cfg.scale_pt_eta_phi[1],
        "phi": cfg.scale_pt_eta_phi[2],
        "px": cfg.scale_xyz[0],
        "py": cfg.scale_xyz[1],
        "pz": cfg.scale_xyz[2],
    }
    scale = max(float(scale_by_component[component]), 1e-12)
    z = delta / scale
    if direction == "below_ref":
        trigger = max(0.0, -z - deadband)
    elif direction == "above_ref":
        trigger = max(0.0, z - deadband)
    elif direction == "both":
        trigger = max(0.0, abs(z) - deadband)
    else:
        raise ValueError(
            "dgpo.offset_anchor.adaptive_lambda.direction must be below_ref, above_ref, or both, "
            f"got {direction!r}"
        )
    multiplier = base_multiplier + gain * trigger
    multiplier = min(max_multiplier, max(min_multiplier, multiplier))
    return multiplier, {
        "offset_anchor/adaptive_lambda_multiplier": float(multiplier),
        "offset_anchor/adaptive_lambda_trigger": float(trigger),
        "offset_anchor/adaptive_lambda_z": float(z),
        "offset_anchor/adaptive_lambda_component_pt": 1.0 if component == "pt" else 0.0,
        "offset_anchor/adaptive_lambda_key_found": 1.0 if chosen_key else 0.0,
    }


def _pred_truth_pt_slots_kb(
    nu_phys_kb: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    *,
    cartesian: bool,
) -> tuple[Tensor, Tensor]:
    """``pred_pt``, ``truth_pt`` each ``(KB, 2)`` for the first two neutrino slots."""
    KB = nu_phys_kb.shape[0]
    if cartesian:
        pred_pt = torch.sqrt(
            nu_phys_kb[..., :2, 0].pow(2) + nu_phys_kb[..., :2, 1].pow(2) + 1e-12
        )
        if not isinstance(batch_kb.get("x_invisible_cartesian"), Tensor):
            raise KeyError("offset_anchor cartesian=True requires batch['x_invisible_cartesian']")
        truth = batch_kb["x_invisible_cartesian"].to(device=pred_pt.device, dtype=pred_pt.dtype)[
            :, :2, :
        ]
        truth_pt = torch.sqrt(truth[..., 0].pow(2) + truth[..., 1].pow(2) + 1e-12)
    else:
        pred_pt = torch.expm1(nu_phys_kb[..., :2, 0].clamp(-10.0, 10.0))
        if not isinstance(batch_kb.get("x_invisible"), Tensor):
            raise KeyError("offset_anchor cartesian=False requires batch['x_invisible']")
        truth = batch_kb["x_invisible"].to(device=pred_pt.device, dtype=pred_pt.dtype)[:, :2, :]
        truth_pt = torch.expm1(truth[..., 0].clamp(-10.0, 10.0))

    if tuple(pred_pt.shape) != (KB, 2) or tuple(truth_pt.shape) != (KB, 2):
        raise ValueError(
            f"expected pred_pt and truth_pt shape (KB, 2), got {tuple(pred_pt.shape)} vs "
            f"{tuple(truth_pt.shape)}"
        )
    return pred_pt, truth_pt


def _pred_truth_xyz_slots_kb(
    nu_phys_kb: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    *,
    cartesian: bool,
) -> tuple[Tensor, Tensor]:
    """``pred_xyz``, ``truth_xyz`` each ``(KB, 2, 3)`` for the first two neutrino slots."""
    KB = nu_phys_kb.shape[0]
    if cartesian:
        pred_xyz = nu_phys_kb[:, :2, :3].contiguous()
        if not isinstance(batch_kb.get("x_invisible_cartesian"), Tensor):
            raise KeyError("offset_anchor cartesian=True requires batch['x_invisible_cartesian']")
        truth_raw = batch_kb["x_invisible_cartesian"].to(
            device=pred_xyz.device, dtype=pred_xyz.dtype
        )[:, :2, :]
        truth_xyz = truth_raw[:, :, :3].contiguous()
    else:
        pred_xyz = invisible_kinematics_to_cartesian(nu_phys_kb[:, :2, :])
        if not isinstance(batch_kb.get("x_invisible"), Tensor):
            raise KeyError("offset_anchor cartesian=False requires batch['x_invisible']")
        truth = batch_kb["x_invisible"].to(device=pred_xyz.device, dtype=pred_xyz.dtype)[:, :2, :]
        truth_xyz = invisible_kinematics_to_cartesian(truth)

    if tuple(pred_xyz.shape) != (KB, 2, 3) or tuple(truth_xyz.shape) != (KB, 2, 3):
        raise ValueError(
            f"expected pred_xyz and truth_xyz shape (KB, 2, 3), got {tuple(pred_xyz.shape)} vs "
            f"{tuple(truth_xyz.shape)}"
        )
    return pred_xyz, truth_xyz


def _wrap_phi_delta(delta_phi: Tensor) -> Tensor:
    """Wrap angular residual to ``[-pi, pi)``."""
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _pred_truth_pt_eta_phi_slots_kb(
    nu_phys_kb: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    *,
    cartesian: bool,
) -> tuple[Tensor, Tensor]:
    """``pred`` and ``truth`` each ``(KB, 2, 3)`` ordered ``[pT, eta, phi]``."""
    KB = nu_phys_kb.shape[0]
    if cartesian:
        plp, pred_eta, pred_phi = cartesian_to_log_pt_eta_phi(
            nu_phys_kb[:, :2, 0],
            nu_phys_kb[:, :2, 1],
            nu_phys_kb[:, :2, 2],
        )
        if not isinstance(batch_kb.get("x_invisible_cartesian"), Tensor):
            raise KeyError("offset_anchor cartesian=True requires batch['x_invisible_cartesian']")
        truth = batch_kb["x_invisible_cartesian"].to(device=plp.device, dtype=plp.dtype)[:, :2, :]
        tlp, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
            truth[..., 0],
            truth[..., 1],
            truth[..., 2],
        )
    else:
        plp = nu_phys_kb[:, :2, 0]
        pred_eta = nu_phys_kb[:, :2, 1]
        pred_phi = nu_phys_kb[:, :2, 2]
        if not isinstance(batch_kb.get("x_invisible"), Tensor):
            raise KeyError("offset_anchor cartesian=False requires batch['x_invisible']")
        truth = batch_kb["x_invisible"].to(device=plp.device, dtype=plp.dtype)[:, :2, :]
        tlp = truth[..., 0]
        truth_eta = truth[..., 1]
        truth_phi = truth[..., 2]

    pred_pt = torch.expm1(plp.clamp(-10.0, 10.0))
    truth_pt = torch.expm1(tlp.clamp(-10.0, 10.0))
    pred = torch.stack((pred_pt, pred_eta, pred_phi), dim=-1)
    truth_pep = torch.stack((truth_pt, truth_eta, truth_phi), dim=-1)
    if tuple(pred.shape) != (KB, 2, 3) or tuple(truth_pep.shape) != (KB, 2, 3):
        raise ValueError(
            f"expected pt/eta/phi shape (KB, 2, 3), got {tuple(pred.shape)} vs "
            f"{tuple(truth_pep.shape)}"
        )
    return pred, truth_pep


def _offset_anchor_dense_mask(
    *,
    pred_pt: Tensor,
    truth_pt: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    K: int,
    cfg: OffsetAnchorTrainConfig,
    candidate_weights_kb: Tensor | None,
) -> Tensor:
    """Truth ``p_T`` band + slot/event/candidate mask shared by scalar and XYZ anchor paths."""
    KB = pred_pt.shape[0]
    B_expected = KB // max(int(K), 1)

    if isinstance(batch_kb.get("x_invisible_mask"), Tensor):
        sm = batch_kb["x_invisible_mask"].to(device=pred_pt.device)
        if sm.dim() == 3 and sm.shape[-1] == 1:
            sm = sm.squeeze(-1)
        slot_ok = sm[:, :2] > 0
    else:
        slot_ok = torch.ones_like(pred_pt, dtype=torch.bool)

    event_ok = torch.ones(B_expected, device=pred_pt.device, dtype=torch.bool)
    if batch_kb.get("x") is not None:
        vb = get_event_valid_mask(batch_kb, B_expected, pred_pt.device, pred_pt.dtype) > 0
        event_ok = vb.bool()

    event_ok_kb = event_ok.unsqueeze(0).expand(int(K), -1).reshape(KB)

    if cfg.apply_to == "best_candidate":
        if candidate_weights_kb is None:
            raise ValueError(
                "dgpo.offset_anchor.apply_to=best_candidate requires candidate_weights_kb"
            )
        if tuple(candidate_weights_kb.shape) != (int(K), B_expected):
            raise ValueError(
                "candidate_weights_kb must have shape "
                f"(K, B)=({int(K)}, {B_expected}), got {tuple(candidate_weights_kb.shape)}"
            )
        candidate_ok = (candidate_weights_kb.to(device=pred_pt.device).detach() > 0).reshape(KB)
    else:
        candidate_ok = torch.ones(KB, device=pred_pt.device, dtype=torch.bool)

    dense = (
        (truth_pt > float(cfg.pt_min))
        & (truth_pt < float(cfg.pt_max))
        & torch.isfinite(truth_pt)
        & torch.isfinite(pred_pt)
        & slot_ok
        & event_ok_kb.unsqueeze(-1)
        & candidate_ok.unsqueeze(-1)
    )
    return dense


def compute_pt_offset_anchor(
    *,
    model_v: Tensor,
    x_t: Tensor,
    t_rep: Tensor,
    noise_mask_rep: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    core_model: torch.nn.Module,
    cartesian: bool,
    K: int,
    cfg: OffsetAnchorTrainConfig,
    candidate_weights_kb: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Differentiable offset anchor; see ``mode`` on :class:`OffsetAnchorTrainConfig`.

    Args:
        model_v: Current velocity ``(K*B, N_nu, F)``.
        x_t: Noised neutrinos ``(K*B, N_nu, F)``.
        t_rep: Diffusion times ``(K*B,)``.
        noise_mask_rep: ``(K*B, N_nu, 1)``.
        batch_kb: Candidate-tiled conditioning batch ``repeat_batch_for_candidates``.
        core_model: Unwrapped EveNet model (normalizer attrs).
        cartesian: ``TruthGeneration.cartesian`` kinematic convention.
        K: Candidate count per event.
        cfg: Resolved training config from :func:`resolve_offset_anchor_train_config`.
        candidate_weights_kb: Optional detached ``(K, B)`` selector. Required for
            ``apply_to='best_candidate'`` where it is a one-hot best-candidate mask.

    Returns:
        Scalar ``loss`` differentiable w.r.t. ``model_v`` when enabled and count suffices, and
        a diagnostic dict keyed for logging (detach outside if needed).

    Distributed:
        Computes local-batch pooled mean only (no cross-rank autograd-safe reduction yet).
    """
    zero = torch.zeros((), device=model_v.device, dtype=model_v.dtype)
    if not cfg.enabled or cfg.lambda_coef <= 0.0:
        return zero + model_v.sum() * 0.0, {}

    x0_hat, _, _ = predict_x0_normalized_from_velocity_diffusion(x_t, model_v, t_rep)
    pad = int(getattr(core_model, "invisible_padding", 0))
    xm = noise_mask_rep
    remove_padding = pad > 0
    invisible_normalizer = getattr(core_model, "invisible_normalizer", None)
    if invisible_normalizer is None:
        raise AttributeError(
            "EveNet core model must expose ``invisible_normalizer`` for offset_anchor denormalization"
        )

    denorm_mask = xm if xm.dim() == 3 else xm.unsqueeze(-1)
    nu_phys = invisible_normalizer.denormalize(
        x0_hat,
        mask=denorm_mask,
        remove_padding=remove_padding,
    )

    KB = nu_phys.shape[0]
    B_expected = KB // max(int(K), 1)
    if KB != int(K) * B_expected:
        raise ValueError(
            f"offset_anchor expected K*B={(int(K), B_expected)}, got flattened rows KB={KB}"
        )

    pred_pt, truth_pt = _pred_truth_pt_slots_kb(nu_phys, batch_kb, cartesian=cartesian)

    diag: dict[str, Tensor] = {
        "offset_anchor/active": torch.tensor(1.0, device=model_v.device, dtype=model_v.dtype),
    }

    dense = _offset_anchor_dense_mask(
        pred_pt=pred_pt,
        truth_pt=truth_pt,
        batch_kb=batch_kb,
        K=int(K),
        cfg=cfg,
        candidate_weights_kb=candidate_weights_kb,
    )

    if cfg.mode == "pt_eta_phi_mean":
        pred_pep, truth_pep = _pred_truth_pt_eta_phi_slots_kb(
            nu_phys,
            batch_kb,
            cartesian=cartesian,
        )
        delta_v = pred_pep - truth_pep
        delta_v = torch.stack(
            (
                delta_v[..., 0],
                delta_v[..., 1],
                _wrap_phi_delta(delta_v[..., 2]),
            ),
            dim=-1,
        )
        finite_mask = dense & torch.isfinite(delta_v).all(dim=-1)
        cnt = finite_mask.sum()
        diag["offset_anchor/mode_xyz"] = torch.tensor(
            0.0, device=model_v.device, dtype=torch.float64
        )
        diag["offset_anchor/mode_pt_eta_phi"] = torch.tensor(
            1.0, device=model_v.device, dtype=torch.float64
        )
        diag["offset_anchor/mask_count"] = cnt.detach().to(dtype=torch.float64)

        names = ("pt", "eta", "phi")
        nan64 = delta_v.new_tensor(float("nan"), dtype=torch.float64)
        if int(cnt.item()) < int(cfg.min_count):
            for j, nm in enumerate(names):
                diag[f"offset_anchor/mu_theta_{nm}"] = nan64
                diag[f"offset_anchor/mu_ref_{nm}"] = nan64.new_tensor(
                    float(cfg.mu_ref_pt_eta_phi[j])
                )
                diag[f"offset_anchor/delta_mu_{nm}"] = nan64
            diag.update(
                {
                    "offset_anchor/raw_loss": zero.detach(),
                    "offset_anchor/raw_loss_pt_eta_phi": zero.detach(),
                    "offset_anchor/skipped_small_mask": nan64.new_tensor(1.0),
                }
            )
            return zero + model_v.sum() * 0.0, diag

        cnt_f = cnt.to(delta_v.dtype)
        mu_theta_pep = delta_v.masked_fill(~finite_mask.unsqueeze(-1), 0.0).sum(dim=(0, 1)) / cnt_f
        spt, seta, sphi = cfg.scale_pt_eta_phi
        denom = mu_theta_pep.new_tensor(
            [max(spt, 1e-12), max(seta, 1e-12), max(sphi, 1e-12)]
        )
        mu_ref_t = mu_theta_pep.new_tensor(
            [
                float(cfg.mu_ref_pt_eta_phi[0]),
                float(cfg.mu_ref_pt_eta_phi[1]),
                float(cfg.mu_ref_pt_eta_phi[2]),
            ]
        )
        z_scaled = (mu_theta_pep - mu_ref_t) / denom
        diag["offset_anchor/z_pt"] = z_scaled[0]
        if cfg.loss_type == "huber":
            loss_raw = F.huber_loss(
                z_scaled,
                torch.zeros_like(z_scaled),
                delta=1.0,
                reduction="mean",
            )
        else:
            loss_raw = z_scaled.square().mean()

        for j, nm in enumerate(names):
            diag[f"offset_anchor/mu_theta_{nm}"] = mu_theta_pep[j].detach().to(
                dtype=torch.float64
            )
            diag[f"offset_anchor/mu_ref_{nm}"] = torch.tensor(
                float(cfg.mu_ref_pt_eta_phi[j]), device=model_v.device, dtype=torch.float64
            )
            diag[f"offset_anchor/delta_mu_{nm}"] = (
                mu_theta_pep[j].detach() - float(cfg.mu_ref_pt_eta_phi[j])
            ).to(dtype=torch.float64)
        diag.update(
            {
                "offset_anchor/raw_loss": loss_raw.detach(),
                "offset_anchor/raw_loss_pt_eta_phi": loss_raw.detach(),
                "offset_anchor/skipped_small_mask": torch.zeros(
                    (), device=model_v.device, dtype=torch.float64
                ),
            }
        )
        return loss_raw, diag

    if cfg.mode == "xyz_mean":
        pred_xyz, truth_xyz = _pred_truth_xyz_slots_kb(nu_phys, batch_kb, cartesian=cartesian)
        delta_v = pred_xyz - truth_xyz
        finite_mask = dense & torch.isfinite(delta_v).all(dim=-1)
        cnt = finite_mask.sum()
        diag["offset_anchor/mode_xyz"] = torch.tensor(
            1.0, device=model_v.device, dtype=torch.float64
        )
        diag["offset_anchor/mode_pt_eta_phi"] = torch.tensor(
            0.0, device=model_v.device, dtype=torch.float64
        )
        diag["offset_anchor/mask_count"] = cnt.detach().to(dtype=torch.float64)

        names = ("px", "py", "pz")
        nan64 = delta_v.new_tensor(float("nan"), dtype=torch.float64)

        if int(cnt.item()) < int(cfg.min_count):
            for j, nm in enumerate(names):
                diag[f"offset_anchor/mu_theta_{nm}"] = nan64
                diag[f"offset_anchor/mu_ref_{nm}"] = nan64.new_tensor(float(cfg.mu_ref_xyz[j]))
                diag[f"offset_anchor/delta_mu_{nm}"] = nan64
            diag.update(
                {
                    "offset_anchor/raw_loss": zero.detach(),
                    "offset_anchor/skipped_small_mask": nan64.new_tensor(1.0),
                }
            )
            return zero + model_v.sum() * 0.0, diag

        cnt_f = cnt.to(delta_v.dtype)
        mu_theta_xyz = delta_v.masked_fill(~finite_mask.unsqueeze(-1), 0.0).sum(dim=(0, 1)) / cnt_f
        sx, sy, sz = cfg.scale_xyz
        denom = mu_theta_xyz.new_tensor([max(sx, 1e-12), max(sy, 1e-12), max(sz, 1e-12)])
        mu_ref_t = mu_theta_xyz.new_tensor(
            [float(cfg.mu_ref_xyz[0]), float(cfg.mu_ref_xyz[1]), float(cfg.mu_ref_xyz[2])]
        )
        z_scaled = (mu_theta_xyz - mu_ref_t) / denom
        if cfg.loss_type == "huber":
            loss_raw = F.huber_loss(
                z_scaled,
                torch.zeros_like(z_scaled),
                delta=1.0,
                reduction="mean",
            )
        else:
            loss_raw = z_scaled.square().mean()

        for j, nm in enumerate(names):
            diag[f"offset_anchor/mu_theta_{nm}"] = mu_theta_xyz[j].detach().to(dtype=torch.float64)
            diag[f"offset_anchor/mu_ref_{nm}"] = torch.tensor(
                float(cfg.mu_ref_xyz[j]), device=model_v.device, dtype=torch.float64
            )
            diag[f"offset_anchor/delta_mu_{nm}"] = (
                mu_theta_xyz[j].detach() - float(cfg.mu_ref_xyz[j])
            ).to(dtype=torch.float64)
        diag.update(
            {
                "offset_anchor/raw_loss": loss_raw.detach(),
                "offset_anchor/skipped_small_mask": torch.zeros(
                    (), device=model_v.device, dtype=torch.float64
                ),
            }
        )
        return loss_raw, diag

    # --- pt_mean (legacy scalar) ---
    diag["offset_anchor/mode_xyz"] = torch.tensor(
        0.0, device=model_v.device, dtype=torch.float64
    )
    diag["offset_anchor/mode_pt_eta_phi"] = torch.tensor(
        0.0, device=model_v.device, dtype=torch.float64
    )
    delta = pred_pt - truth_pt
    finite_mask = dense & torch.isfinite(delta)
    cnt = finite_mask.sum()
    diag["offset_anchor/mask_count"] = cnt.detach().to(dtype=torch.float64)

    if int(cnt.item()) < int(cfg.min_count):
        nan64 = delta.new_tensor(float("nan"), dtype=torch.float64)
        diag.update(
            {
                "offset_anchor/mu_theta": nan64,
                "offset_anchor/mu_ref": nan64.new_tensor(float(cfg.mu_ref)),
                "offset_anchor/delta_mu": nan64,
                "offset_anchor/raw_loss": zero.detach(),
                "offset_anchor/skipped_small_mask": nan64.new_tensor(1.0),
            }
        )
        return zero + model_v.sum() * 0.0, diag

    mu_theta = delta.masked_fill(~finite_mask, 0.0).sum() / cnt.to(delta.dtype)

    denom = mu_theta.new_tensor(max(float(cfg.scale), 1e-12))
    z_scaled = (mu_theta - mu_theta.new_tensor(float(cfg.mu_ref))) / denom
    diag["offset_anchor/z_pt"] = z_scaled

    if cfg.loss_type == "huber":
        loss_raw = F.huber_loss(z_scaled, torch.zeros_like(z_scaled), delta=1.0, reduction="mean")
    else:
        loss_raw = z_scaled.square()

    diag.update(
        {
            "offset_anchor/mu_theta": mu_theta.detach().to(dtype=torch.float64),
            "offset_anchor/mu_ref": torch.tensor(
                float(cfg.mu_ref), device=model_v.device, dtype=torch.float64
            ),
            "offset_anchor/delta_mu": (mu_theta.detach() - float(cfg.mu_ref)).to(
                dtype=torch.float64
            ),
            "offset_anchor/raw_loss": loss_raw.detach(),
            "offset_anchor/z_scaled": z_scaled.detach().to(dtype=torch.float64),
            "offset_anchor/skipped_small_mask": torch.zeros(
                (), device=model_v.device, dtype=torch.float64
            ),
        }
    )
    return loss_raw, diag


def dense_pt_residual_mean_numpy(
    truth_pt_flat: np.ndarray,
    delta_pt_flat: np.ndarray,
    *,
    pt_min: float,
    pt_max: float,
) -> tuple[float, int]:
    """Global mean residual over entries with ``pt_min < truth_pt < pt_max``."""
    t = np.asarray(truth_pt_flat, dtype=np.float64).reshape(-1)
    d = np.asarray(delta_pt_flat, dtype=np.float64).reshape(-1)
    if t.size != d.size:
        raise ValueError(f"truth_pt and delta_pt length mismatch ({t.size} vs {d.size})")
    m = np.isfinite(t) & np.isfinite(d) & (t > float(pt_min)) & (t < float(pt_max))
    n = int(np.sum(m))
    if n <= 0:
        return float("nan"), 0
    return float(np.mean(d[m])), n


def dense_xyz_residual_means_numpy(
    truth_pt_flat: np.ndarray,
    delta_px_flat: np.ndarray,
    delta_py_flat: np.ndarray,
    delta_pz_flat: np.ndarray,
    *,
    pt_min: float,
    pt_max: float,
) -> tuple[tuple[float, float, float], int]:
    """Global mean ``(Δpx, Δpy, Δpz)`` over entries with ``pt_min < truth_pt < pt_max``."""
    t = np.asarray(truth_pt_flat, dtype=np.float64).reshape(-1)
    dx = np.asarray(delta_px_flat, dtype=np.float64).reshape(-1)
    dy = np.asarray(delta_py_flat, dtype=np.float64).reshape(-1)
    dz = np.asarray(delta_pz_flat, dtype=np.float64).reshape(-1)
    if not (t.size == dx.size == dy.size == dz.size):
        raise ValueError("truth_pt and delta_* length mismatch for dense xyz baseline")
    m = (
        np.isfinite(t)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & np.isfinite(dz)
        & (t > float(pt_min))
        & (t < float(pt_max))
    )
    n = int(np.sum(m))
    if n <= 0:
        return (float("nan"), float("nan"), float("nan")), 0
    return (
        float(np.mean(dx[m])),
        float(np.mean(dy[m])),
        float(np.mean(dz[m])),
    ), n
