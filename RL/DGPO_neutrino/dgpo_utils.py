"""Pure utilities for DGPO neutrino RL: advantages, batch tiling, and DGPO loss."""

from __future__ import annotations

from typing import Any, Mapping

import math
import torch
from torch import Tensor

from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma

MW_REF_GEV = 80.379


def _dgpo_cfg_get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    """Read YAML/DotDict key from either a dict or DotDict."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def piecewise_linear_schedule_at(
    sorted_points: list[tuple[float, float]], x: float
) -> float:
    """Evaluate a piecewise-linear schedule; hold endpoints outside ``[first.at, last.at]``.

    ``sorted_points`` must be sorted by ``at`` ascending, each ``(at, value)``.
    """
    if not sorted_points:
        raise ValueError("sorted_points must be non-empty")
    if not math.isfinite(x):
        raise ValueError(f"schedule coordinate must be finite, got {x}")
    if x <= sorted_points[0][0]:
        return float(sorted_points[0][1])
    if x >= sorted_points[-1][0]:
        return float(sorted_points[-1][1])
    for i in range(len(sorted_points) - 1):
        a0, v0 = sorted_points[i]
        a1, v1 = sorted_points[i + 1]
        if a0 <= x <= a1:
            span = a1 - a0
            if abs(span) < 1e-15:
                return float(v1)
            t = (x - a0) / span
            return float(v0 + t * (v1 - v0))
    return float(sorted_points[-1][1])


def resolve_beta_kl_schedule(
    dgpo_cfg: Mapping[str, Any] | Any,
    *,
    global_step: int,
    epoch: int,
) -> float:
    """YAML-driven piecewise-linear ``beta_kl`` compatible with resumed ``global_step`` / epoch.

    When ``dgpo.beta_kl_schedule`` is absent or ``enabled: false``, returns
    ``float(dgpo.beta_kl)`` (fallback).

    Supported:

    - ``axis: global_step`` uses the training loop ``global_step`` (resume-safe).
    - ``axis: epoch`` uses the outer epoch index (resume-safe).

    Raises:
        ValueError: On invalid modes, malformed points, or non-monotone/non-finite anchors.
    """
    fallback = float(_dgpo_cfg_get(dgpo_cfg, "beta_kl", 0.0) or 0.0)
    if not math.isfinite(fallback):
        raise ValueError(f"dgpo.beta_kl must be finite, got {fallback}")

    sched = _dgpo_cfg_get(dgpo_cfg, "beta_kl_schedule", None)
    if sched is None:
        return fallback
    if not bool(_dgpo_cfg_get(sched, "enabled", False)):
        return fallback

    mode = str(_dgpo_cfg_get(sched, "mode", "piecewise_linear")).strip().lower()
    if mode != "piecewise_linear":
        raise ValueError(
            f"beta_kl_schedule.mode must be 'piecewise_linear', got {mode!r}"
        )

    axis_raw = str(_dgpo_cfg_get(sched, "axis", "global_step")).strip().lower()
    if axis_raw in ("step", "global_step", "globalstep"):
        coord = float(int(global_step))
    elif axis_raw in ("epoch",):
        coord = float(int(epoch))
    else:
        raise ValueError(
            f"beta_kl_schedule.axis must be 'global_step' or 'epoch', got {axis_raw!r}"
        )

    raw_points = _dgpo_cfg_get(sched, "points", None)
    if raw_points is None:
        raise ValueError("beta_kl_schedule.enabled requires non-null points list")
    parsed: list[tuple[float, float]] = []
    for i, p in enumerate(list(raw_points)):
        at = _dgpo_cfg_get(p, "at", None)
        val = _dgpo_cfg_get(p, "value", None)
        if at is None or val is None:
            raise ValueError(f"beta_kl_schedule.points[{i}] must have 'at' and 'value'")
        fa = float(at)
        fv = float(val)
        if fa < 0.0:
            raise ValueError(f"beta_kl_schedule point at must be >= 0, got {fa}")
        if not math.isfinite(fa) or not math.isfinite(fv):
            raise ValueError(f"non-finite point at index {i}: at={at} value={val}")
        parsed.append((fa, fv))

    parsed.sort(key=lambda z: z[0])
    return piecewise_linear_schedule_at(parsed, coord)


def compute_per_event_advantage(
    rewards: Tensor,
    eps: float = 1e-6,
    *,
    positive_only: bool = False,
    mode: str = "zscore",
    temperature: float = 1.0,
    candidate_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-event advantages over candidates (dim 0).

    Args:
        rewards: Shape ``(K, B)`` — K candidates, B events.
        eps: Added to std for numerical stability (``mode == "zscore"`` only).
        positive_only: If True, clamp advantages to ``[0, inf)`` after the transform.
        mode: ``"zscore"`` (default), ``"centered"``, or ``"softmax_centered"``.
        temperature: Must be finite and > 0 when ``mode == "softmax_centered"``.
        candidate_mask: Optional ``(K, B)`` boolean/float mask. When provided,
            per-event statistics are computed only over masked candidates, and
            unmasked candidates receive zero advantage.

    ``centered`` uses ``advantages = rewards - mean_K(rewards)`` without dividing by the
    group standard deviation.

    ``softmax_centered`` uses ``advantages = K * softmax(rewards / temperature, dim=0) - 1``,
    so ``sum_k advantages[k, b] = 0`` per event and higher reward yields higher advantage.

    Returns:
        ``(advantages, weights)`` each ``(K, B)``; ``weights = |advantages|`` (after optional clamp).
    """
    mode_str = str(mode).strip().lower()
    if candidate_mask is not None:
        if candidate_mask.shape != rewards.shape:
            raise ValueError(
                f"candidate_mask shape {tuple(candidate_mask.shape)} must match rewards "
                f"shape {tuple(rewards.shape)}"
            )
        mask = candidate_mask.to(device=rewards.device, dtype=torch.bool)
        mf = mask.to(dtype=rewards.dtype)
        count = mf.sum(dim=0).clamp(min=1.0)
    else:
        mask = torch.ones_like(rewards, dtype=torch.bool)
        mf = torch.ones_like(rewards)
        count = rewards.new_full((rewards.shape[1],), float(rewards.shape[0]))

    if mode_str == "zscore":
        mu = (rewards * mf).sum(dim=0) / count
        var = ((rewards - mu.unsqueeze(0)).pow(2) * mf).sum(dim=0) / count
        std = torch.sqrt(var) + eps
        advantages = (rewards - mu.unsqueeze(0)) / std.unsqueeze(0)
    elif mode_str in ("centered", "mean_centered"):
        mu = (rewards * mf).sum(dim=0) / count
        advantages = rewards - mu.unsqueeze(0)
    elif mode_str in ("softmax_centered", "softmax"):
        t = float(temperature)
        if not math.isfinite(t) or t <= 0.0:
            raise ValueError(
                f"advantage temperature must be finite and positive for softmax_centered, got {temperature!r}"
            )
        logits = rewards / t
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask, logits, logits.new_full((), neg_inf))
        logits_max = masked_logits.max(dim=0, keepdim=True).values
        has_any = mask.any(dim=0, keepdim=True)
        logits_max = torch.where(has_any, logits_max, torch.zeros_like(logits_max))
        exp_logits = torch.exp(masked_logits - logits_max) * mf
        denom = exp_logits.sum(dim=0, keepdim=True).clamp(min=eps)
        prob = exp_logits / denom
        advantages = prob * count.unsqueeze(0) - 1.0
    else:
        raise ValueError(
            "advantage mode must be 'zscore', 'centered', or 'softmax_centered', "
            f"got {mode!r}"
        )
    advantages = torch.where(mask, advantages, torch.zeros_like(advantages))
    if positive_only:
        advantages = torch.clamp(advantages, min=0.0)
    weights = advantages.abs()
    return advantages, weights

def repeat_batch_for_candidates(batch: dict[str, Any], K: int) -> dict[str, Any]:
    """Tile each tensor value K times along batch (dim 0) for flattened K*B forwards.

    For tensor ``v`` of shape ``(B, ...)``, the result has shape ``(K*B, ...)`` with
    layout ``[cand0_evt0..evtB-1, cand1_evt0.., ...]``.

    Non-tensor values are shallow-copied into the output dict unchanged.

    Args:
        batch: Mapping of string keys to tensors or other objects.
        K: Number of candidate repetitions per event.

    Returns:
        New dict with the same keys; tensor values expanded as above.
    """
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, Tensor):
            v = val
            out[key] = (
                v.unsqueeze(0)
                .expand(K, *v.shape)
                .reshape(K * v.shape[0], *v.shape[1:])
                .contiguous()
            )
        else:
            out[key] = val
    return out


def build_dgpo_loss(
    L_cur_2d: Tensor,
    L_ref_2d: Tensor,
    advantages: Tensor,
    beta_dgpo: float,
    K: int,
    *,
    kl_per_row: Tensor | None = None,
    kl_weights: Tensor | None = None,
    beta_kl: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Velocity-space DGPO loss with detached per-event gate and optional KL anchor.

    Detached gate: ``Delta = stopgrad(L_cur - L_ref)``, ``M_e`` from
    ``(beta_dgpo / K) * sum_i A_{i,e} Delta_{i,e}``, ``w_e = stopgrad(sigmoid(M_e))``.
    Main term: ``mean_{e,i}( w_e * A_{i,e} * L_cur )`` — gradients only through ``L_cur_2d``.

    Optional anchor: ``loss_kl = mean(kl_weights * kl_per_row)`` with
    ``loss_total = loss_main + beta_kl * loss_kl``.
    ``kl_per_row`` should use a detached reference velocity (e.g. MSE vs ``ref_v.detach()``).

    Args:
        L_cur_2d: Per-(candidate, event) current velocity MSE, shape ``(K, B)`` (trainable).
        L_ref_2d: Same for frozen reference, shape ``(K, B)`` (no grad).
        advantages: Shape ``(K, B)``; ``advantages.shape[0]`` must equal ``K``.
        beta_dgpo: Scales the detached group statistic ``M_e``.
        K: Number of candidates per event.
        kl_per_row: Per flattened row KL surrogate, shape ``(K * B,)`` or ``(K, B)``; optional.
        kl_weights: Optional detached KL multipliers, shape ``(B,)`` for per-event
            weights or ``(K, B)`` / ``(K * B,)`` for per-candidate weights.
            Higher values anchor those rows more strongly to the reference.
        beta_kl: Weight on ``loss_kl``; if zero, KL term is skipped.

    Returns:
        Scalar ``loss_total`` and diagnostics:
        ``loss_total``, ``loss_main``, ``loss_kl``, ``L_cur_mean``, ``L_ref_mean``,
        ``delta_abs_mean``, ``w_e_mean``, ``w_e_std``, ``w_e_min``, ``w_e_max`` (detached).
    """
    if int(advantages.shape[0]) != int(K):
        raise ValueError(
            f"advantages.shape[0]={advantages.shape[0]} must equal K={K}"
        )

    # Delta, M_e, w_e: no gradient into L_cur / L_ref / gate path
    Delta = L_cur_2d.detach() - L_ref_2d.detach()  # (K, B)
    M_e = (float(beta_dgpo) / float(K)) * (advantages * Delta).sum(dim=0)  # (B,)
    w_e = torch.sigmoid(M_e).detach()  # (B,)
    # Batch statistics for W&B ``parameter/w_e_*`` (per-event gate in [0, 1]).
    w_e_mean = w_e.mean()
    w_e_std = w_e.std(unbiased=False) if w_e.numel() > 1 else torch.zeros((), device=w_e.device, dtype=w_e.dtype)

    loss_main = (w_e.unsqueeze(0) * advantages * L_cur_2d).mean(dim=0).mean()

    loss_kl = torch.zeros((), device=L_cur_2d.device, dtype=L_cur_2d.dtype)
    kl_weights_norm: Tensor | None = None
    kl_event_weights_norm: Tensor | None = None
    if kl_per_row is not None and float(beta_kl) > 0.0:
        kr = kl_per_row
        if kr.dim() == 1:
            kr2 = kr.reshape(K, -1)
        else:
            kr2 = kr
        if kl_weights is not None:
            kw = kl_weights
            if kw.dim() == 1 and int(kw.numel()) == int(kr2.shape[1]):
                kl_event_weights_norm = kw.to(device=kr2.device, dtype=kr2.dtype).detach()
                kl_event_weights_norm = torch.where(
                    torch.isfinite(kl_event_weights_norm) & (kl_event_weights_norm > 0.0),
                    kl_event_weights_norm,
                    torch.ones_like(kl_event_weights_norm),
                )
            else:
                if kw.dim() == 1:
                    kw = kw.reshape(K, -1)
                if tuple(kw.shape) != tuple(kr2.shape):
                    raise ValueError(
                        f"kl_weights shape {tuple(kw.shape)} must be (B,) or match "
                        f"kl_per_row {tuple(kr2.shape)}"
                    )
                kl_weights_norm = kw.to(device=kr2.device, dtype=kr2.dtype).detach()
                kl_weights_norm = torch.where(
                    torch.isfinite(kl_weights_norm) & (kl_weights_norm > 0.0),
                    kl_weights_norm,
                    torch.ones_like(kl_weights_norm),
                )
                kr2 = kr2 * kl_weights_norm
        per_event_kl = kr2.mean(dim=0)
        if kl_event_weights_norm is not None:
            per_event_kl = per_event_kl * kl_event_weights_norm
        loss_kl = per_event_kl.mean()

    loss_total = loss_main + float(beta_kl) * loss_kl

    diag: dict[str, Tensor] = {
        "loss_total": loss_total.detach(),
        "loss_main": loss_main.detach(),
        "loss_kl": loss_kl.detach(),
        "L_cur_mean": L_cur_2d.detach().mean(),
        "L_ref_mean": L_ref_2d.detach().mean(),
        "delta_abs_mean": (L_cur_2d - L_ref_2d).detach().abs().mean(),
        "w_e_mean": w_e_mean.detach(),
        "w_e_std": w_e_std.detach(),
        "w_e_min": w_e.min().detach(),
        "w_e_max": w_e.max().detach(),
    }
    if kl_weights_norm is not None:
        diag["kl_weight_mean"] = kl_weights_norm.detach().mean()
        diag["kl_weight_min"] = kl_weights_norm.detach().min()
        diag["kl_weight_max"] = kl_weights_norm.detach().max()
    if kl_event_weights_norm is not None:
        diag["kl_weight_mean"] = kl_event_weights_norm.detach().mean()
        diag["kl_weight_min"] = kl_event_weights_norm.detach().min()
        diag["kl_weight_max"] = kl_event_weights_norm.detach().max()
    return loss_total, diag


def predict_x0_normalized_from_velocity_diffusion(
    x_t: Tensor,
    v_pred: Tensor,
    t_rep: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct normalized clean neutrinos :math:`x_0` from velocity ``v``.

    Uses the same inversion as ``DDIMSampler.sample`` (velocity branch):
    ``eps_hat = alpha * v + sigma * x_t``, ``x_0 = (x_t - sigma * eps_hat) / alpha``.
    Scheduler ``alpha_t``, ``sigma_t`` match ``policy_evaluation_step`` /
    ``get_logsnr_alpha_sigma(time)``.
    """
    _, alpha, sigma = get_logsnr_alpha_sigma(t_rep, shape=(t_rep.shape[0], 1, 1))
    eps_hat = v_pred * alpha + x_t * sigma
    x0 = (x_t - sigma * eps_hat) / alpha.clamp(min=1e-8)
    return x0, alpha.view(-1), sigma.view(-1)


def _invariant_mass_gev_torch(
    e: Tensor,
    px: Tensor,
    py: Tensor,
    pz: Tensor,
) -> Tensor:
    """Scalar invariant mass sqrt(E^2 - |p|^2) in GeV (non-negative clamp).

    ``torch.clamp(min=0)`` does not handle ``NaN``; sanitize first so callers (e.g. the
    W-mass anchor in :func:`tt2l_soft_w_mass_weighted_anchor`) never propagate NaN/inf
    into the loss when neutrino kinematics blow up early in training.
    """
    m2 = e * e - (px * px + py * py + pz * pz)
    m2 = torch.nan_to_num(m2, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt(torch.clamp(m2, min=0.0))


def _lepton_four_momentum_timestep_row(pc_row: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sequential invisible/point-cloud row: ``logE``, ``logPt``, ``eta``, ``phi`` … → (E, px, py, pz)."""
    log_e = pc_row[..., 0]
    log_pt = pc_row[..., 1]
    eta = pc_row[..., 2]
    phi = pc_row[..., 3]
    energy = torch.expm1(log_e)
    pt = torch.expm1(log_pt)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return energy, px, py, pz


def _neutrino_four_massless_log_pt_eta_phi(nu_row: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Neutrino row stores ``log1p(pT), eta, phi`` … → massless four-momentum.

    ``log_pt`` and ``eta`` are clamped before ``expm1`` / ``cosh`` / ``sinh`` so the
    W-mass anchor stays finite when the diffusion model emits out-of-distribution
    kinematics during DGPO rollouts (``expm1(50)`` overflows fp32; ``cosh(50)``
    saturates well past the ttbar regime).
    """
    log_pt = nu_row[..., 0].clamp(-10.0, 10.0)
    eta = nu_row[..., 1].clamp(-8.0, 8.0)
    phi = nu_row[..., 2]
    pt = torch.expm1(log_pt)
    e = pt * torch.cosh(eta)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return e, px, py, pz


def _neutrino_four_massless_cartesian(pxyz: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Neutrino row stores ``px, py, pz``; treat as massless: ``E = |p|``."""
    px_, py_, pz_ = pxyz[..., 0], pxyz[..., 1], pxyz[..., 2]
    e = torch.sqrt(px_ * px_ + py_ * py_ + pz_ * pz_ + 1e-24)
    return e, px_, py_, pz_


def _add_four(
    a: tuple[Tensor, Tensor, Tensor, Tensor],
    b: tuple[Tensor, Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def get_truth_assignment_indices(batch: dict[str, Any]) -> Tensor:
    """Return truth assignment targets from either Ray-unflattened or legacy batch keys."""
    if "assignments-indices" in batch:
        return batch["assignments-indices"]
    if "assignment_indices" in batch:
        return batch["assignment_indices"]
    raise KeyError(
        "Missing truth assignment targets. Expected batch['assignments-indices'] "
        "(Ray/unflattened parquet key) or batch['assignment_indices'] (legacy key)."
    )


def tt2l_w_mass_per_row_and_means(
    nu_phys_kb: Tensor,
    x_kb: Tensor,
    x_mask_kb: Tensor,
    assignment_indices_kb: Tensor,
    *,
    cartesian: bool,
    mw_ref: float = MW_REF_GEV,
) -> tuple[Tensor, Tensor, Tensor]:
    """TT2L ``W -> l nu``: per-flat-row mass residuals and reconstructed masses.

    Uses lepton indices from assignment ``[..., top, 1]`` and neutrino slots ``[:, 0]`` /
    ``[:, 1]``. Invalid rows contribute zero loss contribution (caller averages).

    Returns:
        ``L_w_sq_per_row`` shape ``(R,)``: ``(m_W+ - mw_ref)^2 + (m_W- - mw_ref)^2``.
        ``m_plus``, ``m_minus`` invariant masses in GeV (NaN where invalid).
    """
    rows = nu_phys_kb.shape[0]
    ai = assignment_indices_kb.long()
    idx = torch.arange(rows, device=x_kb.device)
    lp0 = ai[:, 0, 1]
    lp1 = ai[:, 1, 1]
    valid_assign = (lp0 >= 0) & (lp1 >= 0)
    xm = x_mask_kb.to(dtype=nu_phys_kb.dtype)
    if xm.dim() == 2:
        valid_slot = (
            xm[idx, lp0.clamp(min=0)] > 0.5
        ) & (xm[idx, lp1.clamp(min=0)] > 0.5)
        valid_assign = valid_assign & valid_slot
    valid_assign = valid_assign & torch.isfinite(nu_phys_kb).reshape(rows, -1).all(dim=1)

    lep0_row = x_kb[idx, lp0.clamp(min=0)]
    lep1_row = x_kb[idx, lp1.clamp(min=0)]
    l4_0 = _lepton_four_momentum_timestep_row(lep0_row)
    l4_1 = _lepton_four_momentum_timestep_row(lep1_row)

    if cartesian:
        n4_0 = _neutrino_four_massless_cartesian(nu_phys_kb[:, 0])
        n4_1 = _neutrino_four_massless_cartesian(nu_phys_kb[:, 1])
    else:
        n4_0 = _neutrino_four_massless_log_pt_eta_phi(nu_phys_kb[:, 0])
        n4_1 = _neutrino_four_massless_log_pt_eta_phi(nu_phys_kb[:, 1])

    wplus = _add_four(l4_0, n4_0)
    wminus = _add_four(l4_1, n4_1)
    mp = _invariant_mass_gev_torch(*wplus)
    mm = _invariant_mass_gev_torch(*wminus)
    lw = (mp - mw_ref).pow(2) + (mm - mw_ref).pow(2)
    lw = torch.where(valid_assign, lw, torch.zeros_like(lw))
    mp_out = torch.where(valid_assign, mp, mp.new_tensor(float("nan")))
    mm_out = torch.where(valid_assign, mm, mm.new_tensor(float("nan")))
    return lw, mp_out.detach(), mm_out.detach()


def histogram_mode_bin_index(counts_1d: Tensor) -> tuple[int, Tensor]:
    """Argmax bin index with deterministic tie-break toward smaller bin index.

    Args:
        counts_1d: nonnegative histogram counts ``(num_bins,)``.

    Returns:
        ``(mode_idx, counts_1d_detached)``.
    """
    if counts_1d.dim() != 1:
        raise ValueError(f"counts_1d must be 1-D, got shape {tuple(counts_1d.shape)}")
    c = counts_1d.detach().to(dtype=torch.float64)
    if c.numel() == 0:
        raise ValueError("counts_1d must be non-empty")
    idx = torch.arange(c.numel(), device=c.device, dtype=torch.float64)
    pair = torch.stack((c, -idx), dim=0)
    best = pair.max(dim=1).indices.max().item()
    return int(best), c


def truth_pt_slots_b2_from_batch(
    batch: Mapping[str, Any] | dict[str, Any],
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Truth neutrino pT for slots 0–1: ``pt_truth`` ``(B, 2)``, ``slot_valid`` ``(B, 2)`` bool."""
    if cartesian and isinstance(batch.get("x_invisible_cartesian"), Tensor):
        truth = batch["x_invisible_cartesian"].to(device=device, dtype=dtype)
        pt = torch.sqrt(truth[..., 0].pow(2) + truth[..., 1].pow(2) + 1e-12)
    elif isinstance(batch.get("x_invisible"), Tensor):
        truth = batch["x_invisible"].to(device=device, dtype=dtype)
        pt = torch.expm1(truth[..., 0].clamp(-10.0, 10.0))
    else:
        raise KeyError(
            "truth_pt_slots_b2_from_batch requires batch['x_invisible'] "
            "or batch['x_invisible_cartesian']"
        )

    pt = pt[:, :2]
    if isinstance(batch.get("x_invisible_mask"), Tensor):
        slot_mask = batch["x_invisible_mask"].to(device=device)
        if slot_mask.dim() == 3 and slot_mask.shape[-1] == 1:
            slot_mask = slot_mask.squeeze(-1)
        slot_valid = slot_mask[:, :2] > 0
    else:
        slot_valid = torch.ones_like(pt, dtype=torch.bool)
    return pt, slot_valid


def directional_pt_gate_eligibility_kb(
    candidates_phys: Tensor,
    batch: Mapping[str, Any] | dict[str, Any],
    valid_b: Tensor,
    *,
    cartesian: bool,
    num_bins: int,
    pt_min: float,
    pt_max: float,
    slot_aggregation: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float]]:
    """Batch-level directional pT gate: zero advantages unless candidates push pT away from the mode.

    Histogram is built from truth pT (slots 0–1) over valid entries in the batch. For each truth
    bin ``j``, candidates must satisfy ``pred_pt <= truth_pt`` when truth pT is in a bin
    below the mode bin, and ``pred_pt >= truth_pt`` when it is above the mode bin.
    Truth pT inside the mode bin is intentionally not eligible.

    Args:
        candidates_phys: ``(K, B, N_nu, F_phys)`` DDIM outputs in invisible physical space.
        batch: Truth neutrinos + masks (same convention as rollout rewards).
        valid_b: ``(B,)`` event validity mask (>0 valid).
        cartesian: ``TruthGeneration.cartesian`` flag for truth/candidate pT extraction.
        num_bins: Histogram bins between ``pt_min`` and ``pt_max``.
        pt_min/pt_max: Closed histogram domain; values are clamped before binning.
        slot_aggregation: ``any`` (either ν slot can authorize the row) or ``all``
            (both slots must authorize).
        device/dtype: Target tensors for masks.

    Returns:
        ``eligible_kb`` ``(K, B)`` float {0,1} and scalar diagnostics for W&B.
    """
    if num_bins < 1:
        raise ValueError(f"directional_pt_gate.num_bins must be >= 1, got {num_bins}")
    if pt_max <= pt_min:
        raise ValueError(
            f"directional_pt_gate requires pt_min < pt_max, got {pt_min} {pt_max}"
        )
    agg = str(slot_aggregation or "any").strip().lower()
    if agg not in ("any", "all"):
        raise ValueError(
            "directional_pt_gate.slot_aggregation must be 'any' or 'all', "
            f"got {slot_aggregation!r}"
        )

    K, B = int(candidates_phys.shape[0]), int(candidates_phys.shape[1])
    vb = valid_b.reshape(-1).to(device=device) > 0

    pt_truth, slot_valid = truth_pt_slots_b2_from_batch(
        batch, cartesian=cartesian, device=device, dtype=dtype
    )

    edges = torch.linspace(
        float(pt_min),
        float(pt_max),
        steps=num_bins + 1,
        device=device,
        dtype=torch.float64,
    )
    bin_w = float(pt_max - pt_min) / float(num_bins)
    centers = edges[:-1] + 0.5 * bin_w

    pt_clamped = pt_truth.clamp(min=float(pt_min), max=float(pt_max))
    idx_f = (pt_clamped - float(pt_min)) / bin_w
    idx = idx_f.floor().to(dtype=torch.long).clamp(min=0, max=num_bins - 1)

    contrib = vb.unsqueeze(-1) & slot_valid
    counts = torch.zeros(num_bins, device=device, dtype=torch.float64)
    for j in range(num_bins):
        m = contrib & (idx == j)
        counts[j] = m.sum()

    mode_j, counts_det = histogram_mode_bin_index(counts)
    mode_center = float(centers[mode_j].detach().cpu())

    eligible_slots = torch.zeros((K, B, 2), device=device, dtype=dtype)
    truth_below_mode = torch.zeros((B, 2), device=device, dtype=torch.bool)
    truth_above_mode = torch.zeros((B, 2), device=device, dtype=torch.bool)

    for s in range(2):
        slot_ok = vb & slot_valid[:, s]
        if cartesian:
            px = candidates_phys[..., s, 0]
            py = candidates_phys[..., s, 1]
            pred_pt = torch.sqrt(px.pow(2) + py.pow(2) + 1e-12)
        else:
            pred_pt = torch.expm1(candidates_phys[..., s, 0].clamp(-10.0, 10.0))

        tt = pt_truth[:, s]
        below = idx[:, s] < mode_j
        above = idx[:, s] > mode_j
        truth_below_mode[:, s] = below
        truth_above_mode[:, s] = above

        elig = torch.zeros((K, B), device=device, dtype=torch.bool)
        elig = elig | (below.unsqueeze(0) & (pred_pt <= tt.unsqueeze(0)))
        elig = elig | (above.unsqueeze(0) & (pred_pt >= tt.unsqueeze(0)))

        mask_kb = slot_ok.unsqueeze(0).expand(K, -1)
        eligible_slots[..., s] = torch.where(
            elig & mask_kb,
            torch.ones_like(eligible_slots[..., s]),
            torch.zeros_like(eligible_slots[..., s]),
        )

    if agg == "any":
        eligible_kb = (
            (eligible_slots[..., 0] > 0.5) | (eligible_slots[..., 1] > 0.5)
        ).to(dtype=dtype)
    else:
        eligible_kb = (
            (eligible_slots[..., 0] > 0.5) & (eligible_slots[..., 1] > 0.5)
        ).to(dtype=dtype)

    denom_kb = float(K * max(int(vb.sum().item()), 1))
    frac_kb = float(eligible_kb.sum().detach().cpu()) / denom_kb

    event_any = (eligible_kb.sum(dim=0) > 0.5) & vb
    frac_events = (
        float(event_any.sum().detach().cpu()) / float(max(int(vb.sum().item()), 1))
    )

    metrics: dict[str, float] = {
        "diagnostics/directional_pt_gate/enabled": 1.0,
        "diagnostics/directional_pt_gate/num_bins": float(num_bins),
        "diagnostics/directional_pt_gate/pt_min": float(pt_min),
        "diagnostics/directional_pt_gate/pt_max": float(pt_max),
        "diagnostics/directional_pt_gate/mode_bin_index": float(mode_j),
        "diagnostics/directional_pt_gate/mode_bin_center": mode_center,
        "diagnostics/directional_pt_gate/mode_bin_low_edge": float(
            edges[mode_j].detach().cpu()
        ),
        "diagnostics/directional_pt_gate/mode_bin_high_edge": float(
            edges[mode_j + 1].detach().cpu()
        ),
        "diagnostics/directional_pt_gate/mode_bin_truth_count": float(
            counts_det[mode_j].detach().cpu()
        ),
        "diagnostics/directional_pt_gate/eligible_fraction_kb": frac_kb,
        "diagnostics/directional_pt_gate/events_with_any_eligible": frac_events,
        "diagnostics/directional_pt_gate/slot_agg_any": 1.0 if agg == "any" else 0.0,
    }

    # Fraction of valid batch slots that lie strictly on one side of the mode center (informative gate).
    side_mask = vb.unsqueeze(-1) & slot_valid
    denom_slots = float(max(int(side_mask.sum().item()), 1))
    below_frac = (
        float((side_mask & truth_below_mode).sum().item()) / denom_slots
    )
    above_frac = (
        float((side_mask & truth_above_mode).sum().item()) / denom_slots
    )
    metrics["diagnostics/directional_pt_gate/truth_below_mode_frac_slots"] = below_frac
    metrics["diagnostics/directional_pt_gate/truth_above_mode_frac_slots"] = above_frac

    return eligible_kb, metrics


def bracketing_truth_pt_gate_eligibility_kb(
    candidates_phys: Tensor,
    batch: Mapping[str, Any] | dict[str, Any],
    valid_b: Tensor,
    *,
    cartesian: bool,
    slot_aggregation: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float]]:
    """Event-level pT support gate: keep only events whose candidates bracket truth pT.

    An event passes when the rollout contains both a candidate above truth pT and a
    candidate below truth pT. With ``slot_aggregation="any"``, one valid neutrino
    slot bracketing truth is enough; with ``"all"``, every valid slot must bracket.
    Passing events expose all K candidates to advantage calculation; failing events
    receive an all-zero candidate mask.
    """
    agg = str(slot_aggregation or "any").strip().lower()
    if agg not in ("any", "all"):
        raise ValueError(
            "bracketing_truth_pt_gate.slot_aggregation must be 'any' or 'all', "
            f"got {slot_aggregation!r}"
        )

    K, B = int(candidates_phys.shape[0]), int(candidates_phys.shape[1])
    vb = valid_b.reshape(-1).to(device=device) > 0
    pt_truth, slot_valid = truth_pt_slots_b2_from_batch(
        batch, cartesian=cartesian, device=device, dtype=dtype
    )

    slot_brackets = torch.zeros((B, 2), device=device, dtype=torch.bool)
    for s in range(2):
        if cartesian:
            px = candidates_phys[..., s, 0]
            py = candidates_phys[..., s, 1]
            pred_pt = torch.sqrt(px.pow(2) + py.pow(2) + 1e-12)
        else:
            pred_pt = torch.expm1(candidates_phys[..., s, 0].clamp(-10.0, 10.0))

        tt = pt_truth[:, s].unsqueeze(0)
        has_above = (pred_pt > tt).any(dim=0)
        has_below = (pred_pt < tt).any(dim=0)
        slot_brackets[:, s] = has_above & has_below & slot_valid[:, s]

    valid_slot_count = slot_valid.sum(dim=1)
    if agg == "any":
        event_pass = slot_brackets.any(dim=1)
    else:
        event_pass = (valid_slot_count > 0) & (
            slot_brackets.sum(dim=1) == valid_slot_count
        )
    event_pass = event_pass & vb
    eligible_kb = event_pass.unsqueeze(0).expand(K, B).to(dtype=dtype)

    n_valid_events = max(int(vb.sum().item()), 1)
    n_valid_slots = max(int((slot_valid & vb.unsqueeze(-1)).sum().item()), 1)
    events_frac = float(event_pass.sum().detach().cpu()) / float(n_valid_events)
    slot_frac = (
        float((slot_brackets & vb.unsqueeze(-1)).sum().detach().cpu())
        / float(n_valid_slots)
    )
    metrics: dict[str, float] = {
        "diagnostics/directional_pt_gate/enabled": 1.0,
        "diagnostics/directional_pt_gate/mode_bracket_truth": 1.0,
        "diagnostics/directional_pt_gate/eligible_fraction_kb": events_frac,
        "diagnostics/directional_pt_gate/events_with_any_eligible": events_frac,
        "diagnostics/directional_pt_gate/events_bracketing_truth": events_frac,
        "diagnostics/directional_pt_gate/slots_bracketing_truth": slot_frac,
        "diagnostics/directional_pt_gate/slot_agg_any": 1.0 if agg == "any" else 0.0,
    }
    return eligible_kb, metrics


def tt2l_soft_w_mass_weighted_anchor(
    *,
    model_v: Tensor,
    x_t: Tensor,
    t_rep: Tensor,
    batch_rep_kb: dict[str, Any],
    noise_mask_kb: Tensor,
    invisible_normalizer: Any,
    invisible_padding: int,
    cartesian: bool,
) -> dict[str, Tensor]:
    """Full soft anchor from velocity: ``x_0_pred`` … denorm … ``L_w`` weighted by ``alpha_t^2``.

    Args:
        model_v: Normalized neutrino velocity ``(KB, N_nu, F)``.
        x_t: Noised normalized state ``(KB, N_nu, F)``.
        t_rep: Timesteps ``(KB,)``.
        batch_rep_kb: Flattened batch including truth assignment indices.
        invisible_normalizer: ``EveNetModel.invisible_normalizer``.
        invisible_padding: Trailing padded features before denormalizing.
        cartesian: From ``TruthGeneration.cartesian``.
    """
    x0_pred, alpha_b, _sigma_b = predict_x0_normalized_from_velocity_diffusion(x_t, model_v, t_rep)
    pad = int(invisible_padding)
    xm = noise_mask_kb
    # ``model_v`` / ``x0_pred`` already use ``invisible_input_dim`` (the unpadded
    # neutrino width). ``remove_padding=True`` selects the matching normalizer
    # statistics; slicing here would drop real neutrino features when padding > 0.
    remove_padding = pad > 0
    nu_phys = invisible_normalizer.denormalize(
        x0_pred,
        mask=xm,
        remove_padding=remove_padding,
    )
    assign_kb = get_truth_assignment_indices(batch_rep_kb)
    vis_kb = batch_rep_kb["x"]
    vis_mask_kb = batch_rep_kb["x_mask"]

    lw_sq_per_row, m_plus, m_minus = tt2l_w_mass_per_row_and_means(
        nu_phys,
        vis_kb,
        vis_mask_kb,
        assign_kb,
        cartesian=cartesian,
    )
    # Defensive: zero out any residual non-finite rows so a single bad event cannot
    # turn the whole anchor (and the backbone gradients) into NaN.
    lw_sq_per_row = torch.nan_to_num(lw_sq_per_row, nan=0.0, posinf=0.0, neginf=0.0)
    ws = (alpha_b * alpha_b).clamp(max=1.0)
    lw_weighted_scalar = (ws * lw_sq_per_row).mean()
    lw_raw_mean = lw_sq_per_row.detach().mean()

    fin_p = torch.isfinite(m_plus)
    fin_m = torch.isfinite(m_minus)
    mp_mean_t = m_plus.new_tensor(float("nan"))
    mm_mean_t = m_minus.new_tensor(float("nan"))
    if bool(fin_p.any().item()):
        mp_mean_t = m_plus[fin_p].mean().detach()
    if bool(fin_m.any().item()):
        mm_mean_t = m_minus[fin_m].mean().detach()
    alpha_sq_mean_t = ws.detach().mean()

    return {
        "loss_w_anchor": lw_weighted_scalar,
        "loss_w_raw_mean": lw_raw_mean.detach(),
        "mw_plus_mean_batch": mp_mean_t.detach(),
        "mw_minus_mean_batch": mm_mean_t.detach(),
        "alpha_squared_mean_batch": alpha_sq_mean_t.detach(),
    }
