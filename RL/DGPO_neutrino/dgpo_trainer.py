"""
Standalone DGPO fine-tuning loop for neutrino diffusion (Step 5 of the neutrino RL plan).

Uses the same ``EveNetModel`` backbone and Parquet → Ray → ``iter_torch_batches`` path as
``evenet/train.py``, but replaces Lightning with a plain PyTorch optimizer step and the DGPO
objective from ``dgpo_utils.py``.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = Path(__file__).resolve()
# Layout (post-restructure): ``DGPO-NeurIPS-Addon/`` is the repo root and ``evenet`` /
# ``preprocessing`` live as a nested companion checkout at
# ``DGPO-NeurIPS-Addon/EveNet-Full/`` (cloned via ``git clone --recurse-submodules``;
# see top-level README "Install" section). ``parents[2]`` here = ``DGPO-NeurIPS-Addon/``
# (resolves ``RL.*``, ``shared.*``, ``event_selection.*``); the EveNet root resolves
# ``evenet.*`` and ``preprocessing.*``. We no longer add ``parents[3]`` to ``sys.path``.
_REPO_ROOT = str(_HERE.parents[2])
_EVENET_ROOT = str(_HERE.parents[2] / "EveNet-Full")
for _p in (_REPO_ROOT, _EVENET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from evenet.control.global_config import global_config
from shared.shared import make_process_fn, prepare_datasets
from evenet.utilities.diffusion_sampler import DDIMSampler, get_logsnr_alpha_sigma

from RL.DGPO_neutrino.dgpo_utils import (
    _dgpo_cfg_get,
    bracketing_truth_pt_gate_eligibility_kb,
    build_dgpo_loss,
    compute_per_event_advantage,
    directional_pt_gate_eligibility_kb,
    repeat_batch_for_candidates,
    resolve_beta_kl_schedule,
)
from RL.DGPO_neutrino.model_utils import (
    apply_component_freezes,
    freeze_reference_model,
    load_evenet_model_for_dgpo,
    make_ema,
    make_ema_rollout,
    make_reference_model,
    parse_dgpo_resume_from_checkpoint,
    save_lightning_compatible_checkpoint,
)
from RL.DGPO_neutrino.offset_anchor import (
    compute_pt_offset_anchor,
    offset_anchor_dual_state_step,
    resolve_offset_anchor_adaptive_lambda_multiplier,
    resolve_offset_anchor_lambda_coef,
    resolve_offset_anchor_train_config,
)
from RL.DGPO_neutrino.local_kl_anchor import (
    compute_local_kl_anchor_loss,
    fit_reference_pt_residual_profile_p0,
    resolve_local_kl_anchor_train_config,
    truth_pt_slot_kb_gev,
)
from RL.DGPO_neutrino.reference_reward_kl import (
    ReferenceRewardKlStore,
    event_key_pair_columns_from_batch,
    multiply_event_and_row_kl_weights,
    resolve_reference_reward_kl_train_config,
    run_reference_reward_kl_training_baseline,
)
from RL.DGPO_neutrino.rewards import (
    ComponentNormalizedTruthDistanceReward,
    LogPtTruthReward,
    PtTruthReward,
    RelativeTruthReward,
    RewardAggregator,
    TruthDistanceReward,
    WMassProjectionReward,
    WMassTruthNormalizedReward,
    cartesian_to_log_pt_eta_phi,
    compute_truth_l2_distances_kb,
    get_event_valid_mask,
    log_pt_eta_phi_to_cartesian,
)

_log = logging.getLogger(__name__)

GRAD_CLIP_NORM = 1.0


def _dgpo_offset_anchor_checkpoint_fields(
    dgpo_cfg: Any,
    norm: dict[str, Any] | None,
    *,
    effective_mu_ref: float | None,
    effective_mu_ref_xyz: tuple[float, float, float] | None,
) -> tuple[float | None, Tensor | None]:
    """Scalar and/or vector ``mu_ref`` payload for :func:`save_lightning_compatible_checkpoint`."""
    resolved = resolve_offset_anchor_train_config(
        dgpo_cfg,
        norm,
        stored_mu_ref=effective_mu_ref,
        stored_mu_ref_xyz=effective_mu_ref_xyz,
    )
    if not resolved.enabled:
        return None, None
    if resolved.mode == "xyz_mean":
        t = torch.tensor(resolved.mu_ref_xyz, dtype=torch.float32)
        if t.numel() != 3 or not torch.isfinite(t).all():
            return None, None
        return None, t
    mr = float(resolved.mu_ref)
    return (mr if math.isfinite(mr) else None), None

def _dgpo_rollout_ema_decay(global_step: int) -> float:
    """Effective decay for rollout EMA update: ``min(max, ramp * step)`` (Flow GRPO ``ema_ref`` style)."""
    dg = global_config.dgpo
    decay_max = float(dg.get("ema_rollout_decay_max", 0.3))
    decay_ramp = float(dg.get("ema_rollout_decay_ramp", 0.001))
    return min(decay_max, decay_ramp * float(global_step))


def _resolve_beta_kl_from_config(dg: Any, *, global_step: int, epoch: int) -> float:
    """Resolve ``dgpo.beta_kl`` with optional YAML schedule (resume-safe when ``axis: global_step``).

    Delegates to :func:`~RL.DGPO_neutrino.dgpo_utils.resolve_beta_kl_schedule`; see YAML
    ``dgpo.beta_kl_schedule`` in ``RL/DGPO_neutrino/config.yaml``.
    """
    return resolve_beta_kl_schedule(dg, global_step=int(global_step), epoch=int(epoch))


def _dgpo_loss_sum_k_mean_b(
    L_cur_2d: Tensor,
    L_ref_2d: Tensor,
    advantages: Tensor,
    beta_dgpo: float,
    K: int,
    *,
    kl_per_row: Tensor | None,
    kl_weights: Tensor | None = None,
    beta_kl: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Same gate and sample-mean reduction as Flow-GRPO, with optional clipped ``L_cur``.

    Used with PPO-style clipped ``L_cur``.
    """
    if int(advantages.shape[0]) != int(K):
        raise ValueError(
            f"advantages.shape[0]={advantages.shape[0]} must equal K={K}"
        )
    Delta = L_cur_2d.detach() - L_ref_2d.detach()
    M_e = (float(beta_dgpo) / float(K)) * (advantages * Delta).sum(dim=0)
    w_e = torch.sigmoid(M_e).detach()
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


class _DGPODDPForward(nn.Module):
    """Routes DDP ``forward`` to ``EveNetModel.predict_diffusion_vector`` (neutrino mode)."""

    def __init__(self, eve_net: nn.Module) -> None:
        super().__init__()
        self.eve_net = eve_net

    def forward(
        self,
        noise_x: Tensor,
        cond_x: dict[str, Any],
        time: Tensor,
        noise_mask: Tensor,
    ) -> Tensor:
        return self.eve_net.predict_diffusion_vector(
            noise_x=noise_x,
            cond_x=cond_x,
            time=time,
            mode="neutrino",
            noise_mask=noise_mask,
        )


def _unwrap_core_evenet(model: nn.Module) -> nn.Module:
    """Unwrap ``DDP(_DGPODDPForward(eve))`` to the underlying ``EveNetModel``."""
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "eve_net") and isinstance(getattr(m, "eve_net"), nn.Module):
        return m.eve_net
    return m


def _next_batch_synced(
    iterator: Any,
    *,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, Any] | None, bool]:
    """Pull the next batch from a per-rank Ray DataIterator with cross-rank termination sync.

    Each rank fetches its own batch from its own shard. To keep DDP collectives in lock-step,
    we all-reduce a "has-more" flag with ``MIN``: the loop terminates as soon as **any** rank
    runs out of data. This may drop a few batches from longer shards but prevents NCCL hangs.

    Returns ``(batch, all_have)``.  ``batch`` is ``None`` when the local shard is exhausted.
    """
    try:
        batch = next(iterator)
        local_has = 1
    except StopIteration:
        batch = None
        local_has = 0

    if world_size > 1:
        flag = torch.tensor([local_has], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_have = bool(flag.item() > 0)
    else:
        all_have = local_has > 0
    return batch, all_have


def _truth_generation_cartesian() -> bool:
    tg = global_config.options.Training.Components.TruthGeneration
    return bool(getattr(tg, "cartesian", False))


@torch.no_grad()
def _kin_hist_candidate_indices_per_event(
    rewards_kb: Tensor,
    candidates_kb: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
) -> Tensor:
    """Per-event candidate index ``(B,)`` for ``train_dist/*`` and ``val_neutrino/*`` histograms.

    When ``reward_config.rule_based`` is enabled with ``mode: \"truth_distance\"``, use the
    candidate that minimizes the same masked truth L2 as :class:`TruthDistanceReward` (via
    :func:`compute_truth_l2_distances_kb`). That matches the usual \"best of K\" neutrino
    kinematics view.

    For ``component_normalized_truth_distance`` or when rule-based reward is off, use the
    scalar ``rewards_kb`` argmax (combined reward).
    """
    rc = global_config.reward_config
    rb = getattr(rc, "rule_based", None)
    use_truth_l2_argmin = False
    if rb is not None and bool(getattr(rb, "enabled", False)):
        mode = str(getattr(rb, "mode", "truth_distance"))
        use_truth_l2_argmin = mode == "truth_distance"
    if use_truth_l2_argmin:
        d = compute_truth_l2_distances_kb(
            candidates_kb, batch, cartesian=cartesian, mask=None
        )
        return torch.nanargmin(d, dim=0)
    return rewards_kb.argmax(dim=0)


@torch.no_grad()
def compute_reward_mean_gap(rewards_kb: Tensor, valid_b: Tensor) -> float:
    """Mean over valid events of (mean reward above median − mean reward below median) along ``K``."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0 or rewards_kb.shape[0] < 2:
        return float("nan")
    r = rewards_kb[:, vb]
    med = r.median(dim=0).values.unsqueeze(0)
    good_m = r > med
    bad_m = r < med
    good_den = good_m.sum(dim=0).clamp(min=1).to(r.dtype)
    bad_den = bad_m.sum(dim=0).clamp(min=1).to(r.dtype)
    good_mean = (r * good_m.to(r.dtype)).sum(dim=0) / good_den
    bad_mean = (r * bad_m.to(r.dtype)).sum(dim=0) / bad_den
    return float((good_mean - bad_mean).mean().cpu())


@torch.no_grad()
def compute_reward_advantage_pos_neg_gap(
    rewards_kb: Tensor,
    advantages_kb: Tensor,
    valid_b: Tensor,
) -> float:
    """Mean reward where advantage > 0 minus mean reward where advantage < 0 (valid events only)."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return float("nan")
    r = rewards_kb[:, vb]
    a = advantages_kb[:, vb]
    pos = a > 0
    neg = a < 0
    if not pos.any() or not neg.any():
        return float("nan")
    pos_m = r[pos].mean()
    neg_m = r[neg].mean()
    return float((pos_m - neg_m).cpu())


@torch.no_grad()
def _reward_based_kl_weights_from_config(
    rewards_kb: Tensor,
    valid_b: Tensor,
) -> tuple[Tensor | None, dict[str, float]]:
    """Per-event KL multipliers from mean reward.

    Each event gets one multiplier from the mean reward over its K candidates. Better
    events get stronger reference anchoring; worse events get weaker KL.
    """
    cfg = _dgpo_cfg_get(global_config.dgpo, "reward_kl_weight", None)
    if cfg is None or not bool(_dgpo_cfg_get(cfg, "enabled", False)):
        return None, {}

    min_weight = float(_dgpo_cfg_get(cfg, "min_weight", 0.2))
    max_weight = float(_dgpo_cfg_get(cfg, "max_weight", 1.0))
    eps = float(_dgpo_cfg_get(cfg, "eps", 1e-8))
    if min_weight <= 0.0 or max_weight <= 0.0 or max_weight < min_weight:
        raise ValueError(
            "dgpo.reward_kl_weight requires 0 < min_weight <= max_weight, got "
            f"min_weight={min_weight} max_weight={max_weight}"
        )
    if eps <= 0.0:
        raise ValueError(f"dgpo.reward_kl_weight.eps must be positive, got {eps}")

    event_reward = rewards_kb.detach().mean(dim=0)  # (B,)
    vb = valid_b.reshape(-1).to(device=event_reward.device) > 0
    if bool(vb.any().item()):
        valid_event_reward = event_reward[vb]
    else:
        valid_event_reward = event_reward
    r_min = valid_event_reward.min()
    r_max = valid_event_reward.max()
    span = r_max - r_min
    score = torch.ones_like(event_reward)
    if float(span.detach().cpu()) > eps:
        score = (event_reward - r_min) / span.clamp(min=eps)
    weights = min_weight + (max_weight - min_weight) * score
    weights = torch.where(vb, weights, torch.ones_like(weights))

    valid_weights = weights[vb]
    if valid_weights.numel() == 0:
        valid_weights = weights.reshape(-1)
    metrics = {
        "dgpo/reward_kl_weight/enabled": 1.0,
        "dgpo/reward_kl_weight/min_config": min_weight,
        "dgpo/reward_kl_weight/max_config": max_weight,
        "dgpo/reward_kl_weight/event_reward_mean": float(
            valid_event_reward.mean().detach().cpu()
        ),
        "dgpo/reward_kl_weight/mean": float(valid_weights.mean().detach().cpu()),
        "dgpo/reward_kl_weight/min": float(valid_weights.min().detach().cpu()),
        "dgpo/reward_kl_weight/max": float(valid_weights.max().detach().cpu()),
    }
    return weights, metrics


@torch.no_grad()
def _directional_pt_gate_from_config(
    candidates_phys: Tensor,
    batch: dict[str, Any],
    valid_b: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor | None, dict[str, float]]:
    """Optional batch-mode directional pT gate mask ``(K,B)`` for DGPO advantages."""
    cfg = _dgpo_cfg_get(global_config.dgpo, "directional_pt_gate", None)
    if cfg is None or not bool(_dgpo_cfg_get(cfg, "enabled", False)):
        return None, {"diagnostics/directional_pt_gate/enabled": 0.0}

    agg = str(_dgpo_cfg_get(cfg, "slot_aggregation", "any"))
    mode = str(_dgpo_cfg_get(cfg, "mode", "mode_bin_directional")).strip().lower()

    if mode in ("bracket_truth", "bracketing_truth", "truth_bracket"):
        return bracketing_truth_pt_gate_eligibility_kb(
            candidates_phys,
            batch,
            valid_b,
            cartesian=_truth_generation_cartesian(),
            slot_aggregation=agg,
            device=device,
            dtype=dtype,
        )

    if mode not in ("mode_bin_directional", "directional", "mode_bin"):
        raise ValueError(
            "dgpo.directional_pt_gate.mode must be 'bracket_truth' or "
            f"'mode_bin_directional', got {mode!r}"
        )

    num_bins = int(_dgpo_cfg_get(cfg, "num_bins", 20))
    pt_min = float(_dgpo_cfg_get(cfg, "pt_min", 0.0))
    pt_max = float(_dgpo_cfg_get(cfg, "pt_max", 300.0))

    eligible_kb, metrics = directional_pt_gate_eligibility_kb(
        candidates_phys,
        batch,
        valid_b,
        cartesian=_truth_generation_cartesian(),
        num_bins=num_bins,
        pt_min=pt_min,
        pt_max=pt_max,
        slot_aggregation=agg,
        device=device,
        dtype=dtype,
    )
    return eligible_kb, metrics


def _grad_norm_pre_clip_and_clip_active(
    model: nn.Module,
    max_norm: float,
) -> tuple[float, float]:
    """Return (total L2 grad norm before clipping, 1.0 if norm exceeded ``max_norm`` else 0.0).

    ``torch.nn.utils.clip_grad_norm_`` returns the norm **before** scaling; clipping applies when
    that norm exceeds ``max_norm``.
    """
    gn = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))
    )
    active = 1.0 if gn > float(max_norm) + 1e-12 else 0.0
    return gn, active


def _dgpo_assert_train_step_invariants(
    L_ref: Tensor,
    advantages: Tensor,
    rewards: Tensor,
) -> None:
    """Cheap per-step guards for DGPO training."""
    assert not L_ref.requires_grad, "[DGPO CHECK] L_ref must not require grad."
    assert not advantages.requires_grad, "[DGPO CHECK] advantages must not require grad."
    assert torch.isfinite(rewards).all(), "[DGPO CHECK] rewards must be finite."


_REWARD_DIST_OVERLAY_BINS = 40
_REL_PT_DIST_BINS = 50


def _reward_dist_overlaid_figure(
    best: np.ndarray,
    worst: np.ndarray,
    med: np.ndarray,
) -> Any:
    """Three overlapped 1D histograms (density), EveNet validation style, as ``wandb.Image``."""
    import wandb

    stacked = np.concatenate([best, worst, med])
    lo = float(np.min(stacked))
    hi = float(np.max(stacked))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    elif hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    else:
        span = hi - lo
        pad = max(1e-6 * span, 1e-9)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REWARD_DIST_OVERLAY_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    labels = ("best (max per event)", "worst (min per event)", "median along K")
    for arr, c, lab in (
        (best, colors[0], labels[0]),
        (worst, colors[1], labels[1]),
        (med, colors[2], labels[2]),
    ):
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.42,
            label=lab,
            color=c,
            histtype="stepfilled",
        )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title("Per-event reward (best / worst / median among K)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _finite_1d_numpy(arr: np.ndarray) -> np.ndarray:
    """Return finite 1D float values for histogram plotting."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _rel_pt_distribution_figure(
    all_rel_pt: np.ndarray,
    best_rel_pt: np.ndarray,
) -> Any:
    """Overlaid density plot for ``pT_pred / pT_truth - 1`` diagnostics."""
    import wandb

    all_rel_pt = _finite_1d_numpy(all_rel_pt)
    best_rel_pt = _finite_1d_numpy(best_rel_pt)
    stacked = np.concatenate([all_rel_pt, best_rel_pt])
    if stacked.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(stacked, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arr, color, label in (
        (all_rel_pt, "#1f77b4", "all K candidates"),
        (best_rel_pt, "#d62728", "reward-best candidate"),
    ):
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.65,
            label=f"{label}: mean={arr.mean():+.3f}, mean abs={np.abs(arr).mean():.3f}",
            color=color,
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title("Relative pT residual distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _single_rel_pt_distribution_figure(
    rel_pt: np.ndarray,
    *,
    title: str,
    label: str,
) -> Any:
    """Single-series density plot for reference/rollout relative-pT bias diagnostics."""
    import wandb

    rel_pt = _finite_1d_numpy(rel_pt)
    if rel_pt.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(rel_pt, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(rel_pt)) if rel_pt.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if rel_pt.size > 0:
        ax.hist(
            rel_pt,
            bins=bins,
            density=True,
            alpha=0.75,
            label=f"{label}: mean={rel_pt.mean():+.3f}, mean abs={np.abs(rel_pt).mean():.3f}",
            color="#9467bd",
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _pt_delta_vs_truth_pt_figure(
    truth_pt: np.ndarray,
    delta_pt: np.ndarray,
    *,
    title: str,
) -> Any:
    """Profile plot of mean ``pT_pred - pT_truth`` in truth-pT bins."""
    import wandb

    truth_pt = np.asarray(truth_pt, dtype=np.float64).reshape(-1)
    delta_pt = np.asarray(delta_pt, dtype=np.float64).reshape(-1)
    if truth_pt.shape != delta_pt.shape:
        n = min(truth_pt.size, delta_pt.size)
        truth_pt = truth_pt[:n]
        delta_pt = delta_pt[:n]
    keep = np.isfinite(truth_pt) & np.isfinite(delta_pt) & (truth_pt >= 0.0)
    truth_pt = truth_pt[keep]
    delta_pt = delta_pt[keep]

    bin_edges = np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.full(_VAL_KIN_NUM_BINS, np.nan, dtype=np.float64)
    errors = np.full(_VAL_KIN_NUM_BINS, np.nan, dtype=np.float64)
    counts = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.int64)
    if truth_pt.size > 0:
        bin_idx = np.digitize(truth_pt, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < _VAL_KIN_NUM_BINS)
        for i in range(_VAL_KIN_NUM_BINS):
            vals = delta_pt[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    has_points = np.isfinite(means)
    if np.any(has_points):
        ax.errorbar(
            centers[has_points],
            means[has_points],
            yerr=errors[has_points],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=r"mean $(p_T^{pred} - p_T^{truth})$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    ax_count.bar(
        centers,
        counts,
        width=float(bin_edges[1] - bin_edges[0]) * 0.85,
        alpha=0.12,
        color="gray",
        label="entries",
    )
    ax_count.set_ylabel("Entries")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _binned_delta_profile(
    truth_value: np.ndarray,
    delta_value: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return truth-value bin centers, mean residual, standard error, and counts."""
    truth_value = np.asarray(truth_value, dtype=np.float64).reshape(-1)
    delta_value = np.asarray(delta_value, dtype=np.float64).reshape(-1)
    if truth_value.shape != delta_value.shape:
        n = min(truth_value.size, delta_value.size)
        truth_value = truth_value[:n]
        delta_value = delta_value[:n]
    keep = np.isfinite(truth_value) & np.isfinite(delta_value)
    truth_value = truth_value[keep]
    delta_value = delta_value[keep]

    if bin_edges is None:
        bin_edges = np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    num_bins = len(bin_edges) - 1
    means = np.full(num_bins, np.nan, dtype=np.float64)
    errors = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    if truth_value.size > 0:
        bin_idx = np.digitize(truth_value, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        for i in range(num_bins):
            vals = delta_value[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return centers, means, errors, counts


def _pt_delta_selection_profiles_figure(
    truth_pt_all: np.ndarray,
    delta_pt_all: np.ndarray,
    truth_pt_best: np.ndarray,
    delta_pt_best: np.ndarray,
    truth_pt_oracle: np.ndarray | None = None,
    delta_pt_oracle: np.ndarray | None = None,
    *,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional pT-oracle pT delta."""
    import wandb

    centers, mean_all, err_all, _ = _binned_delta_profile(truth_pt_all, delta_pt_all)
    _, mean_best, err_best, counts = _binned_delta_profile(truth_pt_best, delta_pt_best)
    mean_oracle = err_oracle = None
    if truth_pt_oracle is not None and delta_pt_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_pt_oracle, delta_pt_oracle
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    has_all = np.isfinite(mean_all)
    has_best = np.isfinite(mean_best)
    if np.any(has_all):
        ax.errorbar(
            centers[has_all],
            mean_all[has_all],
            yerr=err_all[has_all],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="all rollout candidates",
        )
    if np.any(has_best):
        ax.errorbar(
            centers[has_best],
            mean_best[has_best],
            yerr=err_best[has_best],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="reward-best candidates",
        )
    if mean_oracle is not None and err_oracle is not None:
        has_oracle = np.isfinite(mean_oracle)
        if np.any(has_oracle):
            ax.errorbar(
                centers[has_oracle],
                mean_oracle[has_oracle],
                yerr=err_oracle[has_oracle],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                label="pT-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    has_gap = np.isfinite(best_gap)
    if np.any(has_gap):
        ax_gap.plot(
            centers[has_gap],
            best_gap[has_gap],
            "o-",
            linewidth=1.8,
            markersize=4,
            color="#d62728",
            label="reward-best - all",
        )
    if oracle_gap is not None:
        has_oracle_gap = np.isfinite(oracle_gap)
        if np.any(has_oracle_gap):
            ax_gap.plot(
                centers[has_oracle_gap],
                oracle_gap[has_oracle_gap],
                "^-",
                linewidth=1.8,
                markersize=4,
                color="#2ca02c",
                label="pT-oracle - all",
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"$\Delta p_T$ gap [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _profile_bin_edges(profile_name: str, truth_arrays: list[np.ndarray]) -> np.ndarray:
    """Bin edges for residual profile plots keyed by the profiled truth variable."""
    if profile_name == "pt":
        return np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
    if profile_name == "eta":
        return np.linspace(-4.0, 4.0, _VAL_KIN_NUM_BINS + 1)
    if profile_name == "phi":
        return np.linspace(-3.2, 3.2, _VAL_KIN_NUM_BINS + 1)

    finite_parts = [
        np.asarray(arr, dtype=np.float64).reshape(-1)
        for arr in truth_arrays
        if isinstance(arr, np.ndarray) and arr.size > 0
    ]
    if not finite_parts:
        lo, hi = -100.0, 100.0
    else:
        values = np.concatenate(finite_parts, axis=0)
        values = values[np.isfinite(values)]
        if values.size == 0:
            lo, hi = -100.0, 100.0
        else:
            lo, hi = [float(x) for x in np.nanpercentile(values, [0.5, 99.5])]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                center = float(np.nanmean(values)) if values.size > 0 else 0.0
                lo, hi = center - 100.0, center + 100.0
            hi = max(abs(lo), abs(hi), 1.0)
            lo = -hi
    pad = max(0.05 * (hi - lo), 1e-3)
    return np.linspace(lo - pad, hi + pad, _VAL_KIN_NUM_BINS + 1)


def _profile_axis_labels(profile_name: str) -> tuple[str, str, str]:
    """Return x-label, y-label, and display name for a residual profile variable."""
    display = {
        "pt": "pT",
        "eta": "eta",
        "phi": "phi",
        "px": "px",
        "py": "py",
        "pz": "pz",
    }.get(profile_name, profile_name)
    if profile_name in {"pt", "px", "py", "pz"}:
        return f"Truth {display} [GeV]", f"Mean delta {display} [GeV]", display
    if profile_name == "phi":
        return "Truth phi [rad]", "Mean wrapped delta phi [rad]", display
    return f"Truth {display}", f"Mean delta {display}", display


def _delta_selection_profiles_figure(
    truth_all: np.ndarray,
    delta_all: np.ndarray,
    truth_best: np.ndarray,
    delta_best: np.ndarray,
    truth_oracle: np.ndarray | None = None,
    delta_oracle: np.ndarray | None = None,
    *,
    profile_name: str,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional variable-oracle residuals."""
    import wandb

    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(
        profile_name,
        [truth_all, truth_best] + ([] if truth_oracle is None else [truth_oracle]),
    )
    centers, mean_all, err_all, _ = _binned_delta_profile(
        truth_all, delta_all, bin_edges=bin_edges
    )
    _, mean_best, err_best, counts = _binned_delta_profile(
        truth_best, delta_best, bin_edges=bin_edges
    )
    mean_oracle = err_oracle = None
    if truth_oracle is not None and delta_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_oracle, delta_oracle, bin_edges=bin_edges
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for means, errs, fmt, label, color in (
        (mean_all, err_all, "o-", "all rollout candidates", "#1f77b4"),
        (mean_best, err_best, "s-", "reward-best candidates", "#d62728"),
    ):
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )
    if mean_oracle is not None and err_oracle is not None:
        keep = np.isfinite(mean_oracle)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                mean_oracle[keep],
                yerr=err_oracle[keep],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color="#2ca02c",
                label=f"{display}-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    gap_series = (
        (best_gap, "o-", "reward-best - all", "#d62728"),
        (oracle_gap, "^-", f"{display}-oracle - all", "#2ca02c"),
    )
    for gap, fmt, label, color in gap_series:
        if gap is None:
            continue
        keep = np.isfinite(gap)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gap[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(x_label)
    ax_gap.set_ylabel("Selection gap")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


_DIAG_PROFILE_NAMES = ("pt", "eta", "phi", "px", "py", "pz")


def _diag_profile_raw_key(profile_name: str, suffix: str) -> str:
    """Internal metric key for raw arrays used by accumulated W&B profile images."""
    return f"_diag_{profile_name}_profile_{suffix}"


def _diag_profile_log_key(profile_name: str, *, accumulated: bool = False) -> str:
    """Public W&B image key for a binned residual profile."""
    suffix = "_accumulated" if accumulated else ""
    return (
        f"diagnostics/reward_hacking/profile/"
        f"{profile_name}_delta_vs_truth_{profile_name}{suffix}"
    )


def _diag_profile_title(profile_name: str, *, accumulated_batches: int | None = None) -> str:
    """Human-readable title for a binned residual profile."""
    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    title = f"Reward selection {display} bias vs truth {display}"
    if accumulated_batches is not None:
        title += f" ({accumulated_batches} train batches)"
    return title


def _align_truth_tensor_to_delta(truth: Tensor, delta: Tensor) -> Tensor:
    """Expand cached truth tensors from ``(1, B, S)`` to the candidate shape when needed."""
    if truth.shape == delta.shape:
        return truth
    if truth.dim() == delta.dim() and truth.shape[0] == 1 and truth.shape[1:] == delta.shape[1:]:
        return truth.expand_as(delta)
    return truth


def _finite_profile_numpy(truth: Tensor, delta: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired truth and residual arrays for plotting."""
    mask = torch.isfinite(truth) & torch.isfinite(delta)
    return (
        truth[mask].detach().float().cpu().numpy(),
        delta[mask].detach().float().cpu().numpy(),
    )


def _pt_delta_prefix_vs_full_figure(
    truth_pt_reward_prefix: np.ndarray,
    delta_pt_reward_prefix: np.ndarray,
    truth_pt_reward_full: np.ndarray,
    delta_pt_reward_full: np.ndarray,
    truth_pt_oracle_prefix: np.ndarray,
    delta_pt_oracle_prefix: np.ndarray,
    truth_pt_oracle_full: np.ndarray,
    delta_pt_oracle_full: np.ndarray,
    *,
    prefix_k: int,
    full_k: int,
    title: str,
) -> Any:
    """Compare first-prefix-K vs full-K selection for reward-best and pT-oracle."""
    import wandb

    centers, reward_prefix, reward_prefix_err, counts = _binned_delta_profile(
        truth_pt_reward_prefix, delta_pt_reward_prefix
    )
    _, reward_full, reward_full_err, _ = _binned_delta_profile(
        truth_pt_reward_full, delta_pt_reward_full
    )
    _, oracle_prefix, oracle_prefix_err, _ = _binned_delta_profile(
        truth_pt_oracle_prefix, delta_pt_oracle_prefix
    )
    _, oracle_full, oracle_full_err, _ = _binned_delta_profile(
        truth_pt_oracle_full, delta_pt_oracle_full
    )

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    series = (
        (reward_prefix, reward_prefix_err, "o-", f"reward-best first {prefix_k}", "#ff7f0e"),
        (reward_full, reward_full_err, "s-", f"reward-best full {full_k}", "#d62728"),
        (oracle_prefix, oracle_prefix_err, "^-", f"pT-oracle first {prefix_k}", "#2ca02c"),
        (oracle_full, oracle_full_err, "v-", f"pT-oracle full {full_k}", "#1f77b4"),
    )
    for means, errs, fmt, label, color in series:
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    reward_gain = reward_full - reward_prefix
    oracle_gain = oracle_full - oracle_prefix
    for gain, fmt, label, color in (
        (reward_gain, "o-", f"reward full {full_k} - first {prefix_k}", "#d62728"),
        (oracle_gain, "^-", f"oracle full {full_k} - first {prefix_k}", "#2ca02c"),
    ):
        keep = np.isfinite(gain)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gain[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"Full - prefix [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def _build_reference_bias_metrics(
    candidates: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
    log_distribution: bool = False,
) -> dict[str, Any]:
    """Diagnostics for the rollout/reference policy's raw kinematic bias vs truth."""
    out: dict[str, Any] = {}
    K, B, N_nu, F = candidates.shape
    S = min(2, N_nu)
    if S == 0:
        return out

    if "x_invisible_mask" in batch:
        mask = batch["x_invisible_mask"].to(device=candidates.device, dtype=candidates.dtype)
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() != 2:
            return out
        slot_valid = mask[:, :S] > 0
    else:
        slot_valid = torch.ones(B, S, device=candidates.device, dtype=torch.bool)

    event_valid = get_event_valid_mask(batch, B, candidates.device, candidates.dtype) > 0
    valid_kbs = (slot_valid & event_valid.unsqueeze(-1)).unsqueeze(0).expand(K, B, S)

    if cartesian:
        truth = batch.get("x_invisible_cartesian")
        if not isinstance(truth, Tensor) or truth.dim() != 3 or truth.shape[0] != B:
            return out
        truth_xyz = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_xyz = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
            truth_xyz[..., 0],
            truth_xyz[..., 1],
            truth_xyz[..., 2],
        )
        cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
            cand_xyz[..., 0],
            cand_xyz[..., 1],
            cand_xyz[..., 2],
        )
    else:
        truth = batch.get("x_invisible")
        if (
            not isinstance(truth, Tensor)
            or truth.dim() != 3
            or truth.shape[0] != B
            or F < 3
        ):
            return out
        truth_kin = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_kin = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = truth_kin.unbind(dim=-1)
        cand_log_pt, cand_eta, cand_phi = cand_kin.unbind(dim=-1)

    truth_pt = torch.expm1(truth_log_pt.clamp(-10.0, 10.0)).unsqueeze(0)
    cand_pt = torch.expm1(cand_log_pt.clamp(-10.0, 10.0))
    residuals = {
        "pt": cand_pt - truth_pt,
        "rel_pt": (cand_pt - truth_pt) / truth_pt.clamp(min=1e-6),
        "eta": cand_eta - truth_eta.unsqueeze(0),
        "phi": torch.atan2(
            torch.sin(cand_phi - truth_phi.unsqueeze(0)),
            torch.cos(cand_phi - truth_phi.unsqueeze(0)),
        ),
    }

    finite_residuals: dict[str, Tensor] = {}
    for name, tensor in residuals.items():
        values = tensor[valid_kbs]
        values = values[torch.isfinite(values)]
        finite_residuals[name] = values
        if name == "rel_pt":
            mean_key = f"diagnostics/reference_bias/all/{name}/mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/abs_mean"
        else:
            mean_key = f"diagnostics/reference_bias/all/{name}/delta_mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/delta_abs_mean"
        if values.numel() > 0:
            out[mean_key] = float(values.mean().detach().cpu())
            out[abs_mean_key] = float(values.abs().mean().detach().cpu())
        else:
            out[mean_key] = float("nan")
            out[abs_mean_key] = float("nan")

    if log_distribution:
        rel_pt = finite_residuals.get("rel_pt")
        if rel_pt is not None:
            try:
                import wandb  # noqa: F401

                out["diagnostics/reference_bias/dist/rel_pt"] = (
                    _single_rel_pt_distribution_figure(
                        rel_pt.detach().float().cpu().numpy(),
                        title="Reference / frozen-rollout relative pT bias",
                        label="rollout candidates",
                    )
                )
            except Exception:
                pass
        delta_pt = residuals["pt"][valid_kbs]
        truth_pt_rep = truth_pt.expand(K, B, S)[valid_kbs]
        profile_mask = torch.isfinite(delta_pt) & torch.isfinite(truth_pt_rep)
        delta_pt = delta_pt[profile_mask]
        truth_pt_rep = truth_pt_rep[profile_mask]
        try:
            import wandb  # noqa: F401

            out["diagnostics/reference_bias/profile/pt_delta_vs_truth_pt"] = (
                _pt_delta_vs_truth_pt_figure(
                    truth_pt_rep.detach().float().cpu().numpy(),
                    delta_pt.detach().float().cpu().numpy(),
                    title="Reference / frozen-rollout pT bias vs truth pT",
                )
            )
        except Exception:
            pass
    return out


@torch.no_grad()
def build_reward_distribution_histograms(
    rewards: Tensor,
    valid_b: Tensor,
) -> dict[str, Any]:
    """Panel ``reward/dist``: overlapped 1D histograms for best / worst / median as ``wandb.Image``.

    Logs a **single** media key each time so the W&B Images panel shows one series with a
    **step slider** (same pattern as ``wandb.Image`` validation plots in ``evenet/``).
    """
    try:
        import wandb  # noqa: F401 — require package; figure built in _reward_dist_overlaid_figure
    except ImportError:
        return {}
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return {}
    rv = rewards[:, vb]
    best = rv.max(dim=0).values.detach().float().cpu().numpy()
    worst = rv.min(dim=0).values.detach().float().cpu().numpy()
    med = rv.median(dim=0).values.detach().float().cpu().numpy()
    return {"reward/dist/overlap": _reward_dist_overlaid_figure(best, worst, med)}


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a Ray/Lightning-style batch dict to ``device``."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _save_trainable_weights(model: torch.nn.Module) -> dict[str, Tensor]:
    """Buffer for EMA rollout swap (only parameters that participate in EMA shadow)."""
    core = _unwrap_core_evenet(model)
    return {n: p.data.clone() for n, p in core.named_parameters() if p.requires_grad}


def _restore_trainable_weights(model: torch.nn.Module, buf: dict[str, Tensor]) -> None:
    core = _unwrap_core_evenet(model)
    for n, p in core.named_parameters():
        if n in buf:
            p.data.copy_(buf[n])


def _scales_from_normalization(normalization_dict: dict[str, Any] | None) -> dict[str, float]:
    """Build the 6 per-component scales from ``normalization.pt['invisible_cartesian_std']``.

    Reads shape-``(3,)`` std in order ``[px, py, pz]`` (computed sample-wise at
    preprocessing over both ``nu1`` and ``nu2`` slots) and applies the same px/py/pz
    std to both neutrinos. Raises if the key is absent — re-run preprocessing first.
    """
    if normalization_dict is None or "invisible_cartesian_std" not in normalization_dict:
        raise ValueError(
            "component_normalized_truth_distance requires 'invisible_cartesian_std' in "
            "normalization.pt (shape (3,) [px, py, pz]). Re-run preprocessing so the "
            "Cartesian std is saved."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std) < 3:
        raise ValueError(
            f"invisible_cartesian_std must have 3 entries [px, py, pz], got {len(std)}"
        )
    return {
        "nu1_px": float(std[0]), "nu1_py": float(std[1]), "nu1_pz": float(std[2]),
        "nu2_px": float(std[0]), "nu2_py": float(std[1]), "nu2_pz": float(std[2]),
    }


def _log_pt_scale_from_normalization(normalization_dict: dict[str, Any] | None) -> float:
    """Resolve ``reward_config.log_pt_truth.scale: auto`` / ``w_projection.scale: auto``.

    Uses ``normalization_dict['invisible_std']['Source'][0]`` — the preprocessing
    std over stored ``log1p(pT)`` (same layout as invisible features).
    """
    if normalization_dict is None or "invisible_std" not in normalization_dict:
        raise ValueError(
            "log-pT–scale rewards with scale: auto require 'invisible_std' in "
            "normalization.pt (log1p(pT) std at index 0). Re-run preprocessing or set "
            "reward_config.log_pt_truth.scale and/or reward_config.w_projection.scale "
            "to an explicit positive float."
        )
    std_t = normalization_dict["invisible_std"]["Source"]
    std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if not std_list:
        raise ValueError("invisible_std['Source'] must be non-empty.")
    scale = float(std_list[0])
    if scale <= 0.0:
        raise ValueError(f"log_pT scale from invisible_std[0] must be positive, got {scale}")
    return scale


def _pt_scale_from_normalization(normalization_dict: dict[str, Any] | None) -> float:
    """Resolve ``reward_config.pt_truth.scale: auto`` in linear pT [GeV].

    Prefer an explicit linear-pT statistic if future preprocessing writes one.
    Otherwise use the transverse Cartesian RMS from ``invisible_cartesian_std``
    as a stable GeV-scale proxy.
    """
    if normalization_dict is None:
        raise ValueError(
            "pt_truth reward with scale: auto requires a normalization.pt. "
            "Set reward_config.pt_truth.scale to an explicit GeV value instead."
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
            "pt_truth reward with scale: auto requires 'invisible_pt_std' or "
            "'invisible_cartesian_std' in normalization.pt. Set an explicit GeV scale "
            "if neither key is available."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std_list = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std_list) < 2:
        raise ValueError("invisible_cartesian_std['Source'] must contain at least [px, py].")
    scale = math.sqrt(0.5 * (float(std_list[0]) ** 2 + float(std_list[1]) ** 2))
    if scale <= 0.0:
        raise ValueError(f"transverse pT scale from invisible_cartesian_std must be positive, got {scale}")
    return scale


def _build_rule_based_reward(
    rule_cfg: Any,
    *,
    cartesian: bool,
    normalization_dict: dict[str, Any] | None = None,
) -> Any:
    """Pick the rule-based reward variant to use based on ``rule_cfg.mode``.

    Modes:
        - ``"truth_distance"`` (default): :class:`TruthDistanceReward` — original behavior.
        - ``"component_normalized_truth_distance"``: :class:`ComponentNormalizedTruthDistanceReward`
          with scales loaded from ``normalization_dict['invisible_cartesian_std']``.
        - ``"relative-reward"`` / ``"relative_reward"``: :class:`RelativeTruthReward`
          using relative Cartesian residuals without normalization.pt component scales.
    """
    mode = str(getattr(rule_cfg, "mode", "truth_distance"))
    mode_key = mode.strip().lower().replace("_", "-")
    if mode_key == "truth-distance":
        return TruthDistanceReward(cartesian=cartesian)
    if mode_key == "component-normalized-truth-distance":
        cn = getattr(rule_cfg, "component_normalized", None)
        eps = float(getattr(cn, "eps", 1e-8)) if cn is not None else 1e-8
        scales = _scales_from_normalization(normalization_dict)
        _log.info(
            "[DGPO/reward] component_normalized scales from normalization.pt "
            "invisible_cartesian_std [px, py, pz]: %s",
            {k: round(v, 4) for k, v in scales.items()},
        )
        return ComponentNormalizedTruthDistanceReward(scales, cartesian=cartesian, eps=eps)
    if mode_key == "relative-reward":
        rr = getattr(rule_cfg, "relative_reward", None)
        eps = float(getattr(rr, "eps", 1e-6)) if rr is not None else 1e-6
        clip = float(getattr(rr, "clip", 5.0)) if rr is not None else 5.0
        _log.info(
            "[DGPO/reward] relative_reward using clipped relative Cartesian residuals "
            "eps=%.3g clip=%.3g cartesian=%s (no normalization.pt component scales)",
            eps,
            clip,
            cartesian,
        )
        return RelativeTruthReward(cartesian=cartesian, eps=eps, clip=clip)
    raise ValueError(
        f"unknown reward_config.rule_based.mode={mode!r}; "
        "expected 'truth_distance', 'component_normalized_truth_distance', or 'relative-reward'"
    )


def build_reward_aggregator(
    model: torch.nn.Module,
    device: torch.device,
    normalization_dict: dict[str, Any] | None = None,
) -> RewardAggregator:
    """Construct weighted rewards from ``reward_config`` in the loaded global config.

    ``normalization_dict`` is forwarded to the rule-based reward builder so the
    component-normalized variant can auto-load px/py/pz scales from
    ``invisible_cartesian_std`` (saved by preprocessing). Pass ``None`` to require
    explicit YAML scales. The optional ``log_pt_truth`` additive reward uses
    ``normalization_dict['invisible_std']['Source'][0]`` when YAML sets
    ``reward_config.log_pt_truth.scale`` or ``reward_config.w_projection.scale``
    to ``auto``. The optional ``pt_truth`` additive reward uses a linear-pT GeV scale from
    ``invisible_pt_std`` when present, otherwise a transverse Cartesian proxy from
    ``invisible_cartesian_std``.
    """
    rc = global_config.reward_config
    agg = RewardAggregator()
    if bool(rc.rule_based.enabled):
        agg.add(
            _build_rule_based_reward(
                rc.rule_based,
                cartesian=_truth_generation_cartesian(),
                normalization_dict=normalization_dict,
            ),
            float(rc.rule_based.weight),
        )
    log_pt_truth_cfg = getattr(rc, "log_pt_truth", None)
    if log_pt_truth_cfg is not None and bool(getattr(log_pt_truth_cfg, "enabled", False)):
        lpt_eps = float(getattr(log_pt_truth_cfg, "eps", 1e-8))
        scale_raw = getattr(log_pt_truth_cfg, "scale", "auto")
        scale_str = str(scale_raw).strip().lower()
        if scale_str == "auto":
            scale_val = _log_pt_scale_from_normalization(normalization_dict)
        else:
            scale_val = float(scale_raw)
            if scale_val <= 0.0:
                raise ValueError(f"log_pt_truth.scale must be positive or 'auto', got {scale_raw!r}")
        _log.info(
            "[DGPO/reward] log_pt_truth resolved scale=%.6g eps=%.3g cartesian=%s",
            scale_val,
            lpt_eps,
            _truth_generation_cartesian(),
        )
        agg.add(
            LogPtTruthReward(
                cartesian=_truth_generation_cartesian(),
                scale=scale_val,
                eps=lpt_eps,
            ),
            float(getattr(log_pt_truth_cfg, "weight", 0.05)),
        )
    pt_truth_cfg = getattr(rc, "pt_truth", None)
    if pt_truth_cfg is not None and bool(getattr(pt_truth_cfg, "enabled", False)):
        pt_eps = float(getattr(pt_truth_cfg, "eps", 1e-8))
        scale_raw = getattr(pt_truth_cfg, "scale", "auto")
        scale_str = str(scale_raw).strip().lower()
        if scale_str == "auto":
            scale_val = _pt_scale_from_normalization(normalization_dict)
        else:
            scale_val = float(scale_raw)
            if scale_val <= 0.0:
                raise ValueError(f"pt_truth.scale must be positive or 'auto', got {scale_raw!r}")
        _log.info(
            "[DGPO/reward] pt_truth resolved linear-pT scale=%.6g GeV eps=%.3g cartesian=%s",
            scale_val,
            pt_eps,
            _truth_generation_cartesian(),
        )
        agg.add(
            PtTruthReward(
                cartesian=_truth_generation_cartesian(),
                scale=scale_val,
                eps=pt_eps,
            ),
            float(getattr(pt_truth_cfg, "weight", 0.05)),
        )
    w_mass_cfg = getattr(rc, "w_mass", None)
    if w_mass_cfg is not None and bool(getattr(w_mass_cfg, "enabled", False)):
        wm_eps = float(getattr(w_mass_cfg, "eps", 1e-8))
        wm_w = float(getattr(w_mass_cfg, "weight", 0.2))
        _log.info(
            "[DGPO/reward] w_mass truth-mean/std normalized eps=%.3g cartesian=%s weight=%s",
            wm_eps,
            _truth_generation_cartesian(),
            wm_w,
        )
        agg.add(
            WMassTruthNormalizedReward(
                cartesian=_truth_generation_cartesian(),
                eps=wm_eps,
            ),
            wm_w,
        )
    w_proj_cfg = getattr(rc, "w_projection", None)
    if w_proj_cfg is not None and bool(getattr(w_proj_cfg, "enabled", False)):
        wp_eps = float(getattr(w_proj_cfg, "eps", 1e-8))
        wp_min_pt = float(getattr(w_proj_cfg, "min_pt", 0.1))
        wp_max_pt = float(getattr(w_proj_cfg, "max_pt", 1000.0))
        wp_w = float(getattr(w_proj_cfg, "weight", 0.1))
        scale_raw = getattr(w_proj_cfg, "scale", "auto")
        scale_str = str(scale_raw).strip().lower()
        if scale_str == "auto":
            wp_scale = _log_pt_scale_from_normalization(normalization_dict)
        else:
            wp_scale = float(scale_raw)
            if wp_scale <= 0.0:
                raise ValueError(f"w_projection.scale must be positive or 'auto', got {scale_raw!r}")
        _log.info(
            "[DGPO/reward] w_projection PDG projection penalty scale=%.6g eps=%.3g "
            "cartesian=%s min_pt=%s max_pt=%s weight=%s",
            wp_scale,
            wp_eps,
            _truth_generation_cartesian(),
            wp_min_pt,
            wp_max_pt,
            wp_w,
        )
        agg.add(
            WMassProjectionReward(
                cartesian=_truth_generation_cartesian(),
                scale=wp_scale,
                eps=wp_eps,
                min_pt=wp_min_pt,
                max_pt=wp_max_pt,
            ),
            wp_w,
        )
    if not agg.sources:
        raise RuntimeError(
            "No reward sources enabled; enable rule_based, log_pt_truth, pt_truth, w_mass, "
            "and/or w_projection in reward_config."
        )
    return agg


def _physics_informed_flag() -> bool:
    tg = global_config.options.Training.Components.TruthGeneration
    return bool(getattr(tg, "physics_informed", False))


@torch.no_grad()
def generate_neutrino_candidates(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sampler: DDIMSampler,
    *,
    K: int,
    num_ddim_steps: int,
    device: torch.device,
    tqdm_k_chains: bool = False,
    use_tqdm_ddim: bool = False,
    chain_progress_desc: str = "DGPO DDIM chains",
) -> Tensor:
    """DDIM rollouts in physical invisible space, shape ``(K, B, N_nu, F)``.

    ``data_shape`` for the DDIM prior must match what ``predict_diffusion_vector``
    receives as ``noise_x``:  ``(B, N_nu, invisible_input_dim)``.  When
    ``TruthGeneration.cartesian: true`` that is 3 (px, py, pz), not the full
    7-D ``x_invisible`` from the parquet.
    """
    if "x_invisible" not in batch:
        raise KeyError("batch missing x_invisible for DDIM data_shape.")
    B, N_nu = batch["x_invisible"].shape[:2]
    inv_dim = int(getattr(model, "invisible_input_dim", batch["x_invisible"].shape[-1]))
    data_shape = (B, N_nu, inv_dim)
    candidates: list[Tensor] = []
    noise_mask = batch["x_invisible_mask"].unsqueeze(-1)
    pred_partial = partial(
        model.predict_diffusion_vector,
        mode="neutrino",
        cond_x=batch,
        noise_mask=noise_mask,
    )
    if _physics_informed_flag():
        raise NotImplementedError(
            "options.Training.Components.TruthGeneration.physics_informed=true is not "
            "supported with the public EveNet DDIMSampler (no predict_x0 / x0-prediction "
            "branch). Set physics_informed: false in your YAML or supply an extended sampler."
        )
    k_iter: Any = range(K)
    if tqdm_k_chains:
        try:
            from tqdm.auto import tqdm

            k_iter = tqdm(
                range(K),
                desc=chain_progress_desc,
                leave=False,
                unit="chain",
            )
        except ImportError:
            k_iter = range(K)

    inner_name = f"{chain_progress_desc} steps"
    for _ in k_iter:
        gen = sampler.sample(
            data_shape=data_shape,
            pred_fn=pred_partial,
            num_steps=num_ddim_steps,
            normalize_fn=model.invisible_normalizer,
            remove_padding=True,
            noise_mask=noise_mask,
            use_tqdm=use_tqdm_ddim,
            process_name=inner_name,
        )
        candidates.append(gen)
    return torch.stack(candidates, dim=0)


def _normalize_candidates_for_policy(
    model: torch.nn.Module,
    c_phys: Tensor,
    inv_mask: Tensor,
) -> Tensor:
    """Map denormalized DDIM output to the normalized space for ``predict_diffusion_vector``.

    Matches training / DDIM: raw invisible is padded to ``sequential_input_dim``, then
    ``invisible_normalizer`` runs on that width. ``predict_diffusion_vector`` (neutrino) then
    applies ``F.pad(..., invisible_padding)`` itself, so ``noise_x`` must be only the first
    ``invisible_input_dim`` channels (same width as ``DDIMSampler`` uses from ``x_invisible``),
    not the full padded-normalized tensor — otherwise features become ``sequential + padding``
    and ``torch.cat`` with jets fails (e.g. 7 vs 11).
    """
    # c_phys: (R, N_nu, F_phys)
    pad = int(getattr(model, "invisible_padding", 0))
    m = inv_mask.unsqueeze(-1).to(dtype=c_phys.dtype)
    x = c_phys
    if pad > 0:
        x = F.pad(x, (0, pad))
    full_norm = model.invisible_normalizer(x=x, mask=m)
    inv_in = int(getattr(model, "invisible_input_dim", full_norm.shape[-1]))
    return full_norm[..., :inv_in]


def per_row_velocity_mse(
    pred_v: Tensor,
    target_v: Tensor,
    noise_mask_bn11: Tensor,
    invisible_padding: int,
) -> Tensor:
    """Masked mean squared error per row (one scalar per event×candidate row)."""
    m = noise_mask_bn11.expand_as(pred_v).to(dtype=pred_v.dtype)
    if invisible_padding > 0:
        m = m.clone()
        m[:, :, -invisible_padding:] = 0.0
    sq = (pred_v - target_v).pow(2) * m
    den = m.sum(dim=(1, 2)).clamp(min=1e-8)
    return sq.sum(dim=(1, 2)) / den


def _diag_scalar_float(diag: dict[str, Tensor], key: str) -> float:
    """Single scalar tensor from diagnostics, or NaN if absent / non-finite."""
    t = diag.get(key)
    if t is None:
        return float("nan")
    v = float(t.detach().float().cpu())
    return v if math.isfinite(v) else float("nan")


def _parameter_panel_from_diag(diag_last: dict[str, Tensor]) -> dict[str, float]:
    """Map loss diagnostics to W&B ``parameter/*`` keys (extend with more tensors here later)."""
    out: dict[str, float] = {}
    mapping = (
        ("w_e_mean", "parameter/w_e/mean"),
        ("w_e_std", "parameter/w_e/std"),
        ("w_e_min", "parameter/w_e/min"),
        ("w_e_max", "parameter/w_e/max"),
        ("kl_weight_mean", "parameter/kl_weight/mean"),
        ("kl_weight_min", "parameter/kl_weight/min"),
        ("kl_weight_max", "parameter/kl_weight/max"),
    )
    for src, dst in mapping:
        t = diag_last.get(src)
        if t is None:
            continue
        v = float(t.detach().cpu())
        if math.isfinite(v):
            out[dst] = v
    return out


def _mean_diag_dict(diags: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Elementwise mean of detached loss diagnostics across training sub-steps (accumulate mode).

    Keys are unioned across sub-steps so optional panels (offset anchor / future terms) survive
    when some substeps omit them—the missing slice is averaged as NaN (then typically filtered by
    the logger).
    """
    if not diags:
        return {}
    all_keys: set[str] = set()
    for d in diags:
        all_keys.update(d.keys())
    out: dict[str, Tensor] = {}
    for k in sorted(all_keys):  # stable ordering helps debugging parity across ranks.
        first: Tensor | None = None
        for d in diags:
            t = d.get(k)
            if t is None:
                continue
            t = t.detach()
            first = t
            break
        if first is None:
            continue
        filled: list[Tensor] = []
        for d in diags:
            t = d.get(k)
            if t is None:
                filled.append(torch.tensor(float("nan"), device=first.device, dtype=first.dtype))
            else:
                filled.append(t.detach())
        out[k] = torch.stack(filled, dim=0).mean(dim=0)
    return out


def _finite_mean_float(values: Tensor) -> float:
    """Mean over finite tensor entries, or NaN if none are finite."""
    flat = values.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().detach().cpu())


@torch.no_grad()
def _build_reward_source_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None,
) -> dict[str, float]:
    """Generic per-source reward decomposition for spotting competing reward terms."""
    out: dict[str, float] = {}
    if not reward_breakdown:
        return out

    weight_by_name: dict[str, float] = {}
    for src, weight in reward_agg.sources:
        weight_by_name[src.name] = weight_by_name.get(src.name, 0.0) + float(weight)

    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        for name in reward_breakdown:
            for suffix in (
                "mean",
                "weighted_mean",
                "selected_by_total_mean",
                "selected_by_total_weighted_mean",
                "source_best_of_k",
                "source_last_place",
                "selection_gap",
            ):
                out[f"reward/sources/{name}/{suffix}"] = float("nan")
        return out

    total_v = rewards[:, vb]
    total_best_k = total_v.argmax(dim=0)
    cols = torch.arange(int(total_best_k.numel()), device=rewards.device, dtype=torch.long)

    for name, tensor in reward_breakdown.items():
        if tensor.dim() != 2:
            continue
        rv = tensor[:, vb]
        weight = float(weight_by_name.get(name, 1.0))
        selected = rv[total_best_k, cols]
        mean = _finite_mean_float(rv)
        selected_mean = _finite_mean_float(selected)
        out[f"reward/sources/{name}/mean"] = mean
        out[f"reward/sources/{name}/weighted_mean"] = weight * mean
        out[f"reward/sources/{name}/selected_by_total_mean"] = selected_mean
        out[f"reward/sources/{name}/selected_by_total_weighted_mean"] = weight * selected_mean
        out[f"reward/sources/{name}/source_best_of_k"] = _finite_mean_float(
            rv.max(dim=0).values
        )
        out[f"reward/sources/{name}/source_last_place"] = _finite_mean_float(
            rv.min(dim=0).values
        )
        out[f"reward/sources/{name}/selection_gap"] = selected_mean - mean
    return out


@torch.no_grad()
def _build_reward_extra_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None = None,
    *,
    log_distribution: bool = False,
) -> dict[str, Any]:
    """Light extras for W&B: ``reward/raw/{mean,std}``, optional ``reward/w_mass/*`` /
    ``physics/w_mass/*``, optional ``reward/w_projection/*`` / ``physics/w_projection/*``,
    and per-component contributions.

    Per-component contributions are emitted only when the active rule-based reward is
    :class:`ComponentNormalizedTruthDistanceReward` or :class:`RelativeTruthReward`.
    They are reported under the independent ``components/`` panel as **negative** squared errors
    (i.e. per-component reward contributions, ``-err_c``) so the sign convention
    matches ``reward/raw/*`` (``<= 0``, larger = better, increasing = improving).
    By construction ``sum_c components/{c}/mean == reward/raw/mean`` for the
    component-normalized reward.

    Reward-hacking checks keep compact scalar breakdowns for the competing reward
    components: axis reward means, raw residual means, raw absolute residual means,
    and the relative-pT distribution panel.
    """
    out: dict[str, Any] = {}
    vb = valid_b.reshape(-1) > 0

    def _log_mean_abs(prefix: str, values: Tensor) -> Tensor:
        finite = values.reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() > 0:
            out[f"{prefix}/delta_mean"] = float(finite.mean().detach().cpu())
            out[f"{prefix}/delta_abs_mean"] = float(finite.abs().mean().detach().cpu())
        else:
            out[f"{prefix}/delta_mean"] = float("nan")
            out[f"{prefix}/delta_abs_mean"] = float("nan")
        return finite

    if vb.sum() > 0:
        r = rewards[:, vb]
        out["reward/raw/mean"] = float(r.mean().detach().cpu())
        out["reward/raw/std"] = float(r.std(unbiased=False).detach().cpu()) if r.numel() > 1 else 0.0
    else:
        out["reward/raw/mean"] = float("nan")
        out["reward/raw/std"] = float("nan")

    out.update(_build_reward_source_metrics(rewards, valid_b, reward_agg, reward_breakdown))

    for src, _w in reward_agg.sources:
        if isinstance(src, (ComponentNormalizedTruthDistanceReward, RelativeTruthReward)):
            comps = src.last_component_errors()
            if comps is None:
                continue
            for cname, ctensor in comps.items():
                if vb.sum() == 0:
                    out[f"components/{cname}/mean"] = float("nan")
                    continue
                cv = ctensor[:, vb]
                # Negate so the sign matches ``reward/raw/*`` (per-component reward, not error).
                out[f"components/{cname}/mean"] = float((-cv).mean().detach().cpu())
            if vb.sum() > 0:
                rewards_v = rewards[:, vb]
                best_k = rewards_v.argmax(dim=0)
                bv = int(best_k.numel())
                cols = torch.arange(bv, device=rewards.device, dtype=torch.long)
                axis_pairs = {
                    "px": ("nu1_px", "nu2_px"),
                    "py": ("nu1_py", "nu2_py"),
                    "pz": ("nu1_pz", "nu2_pz"),
                }
                for axis, (a, b) in axis_pairs.items():
                    axis_reward = -(comps[a] + comps[b])[:, vb]
                    out[f"diagnostics/reward_hacking/all/{axis}/reward_mean"] = float(
                        axis_reward.mean().detach().cpu()
                    )
                    out[f"diagnostics/reward_hacking/best/{axis}/reward_mean"] = float(
                        axis_reward[best_k, cols].mean().detach().cpu()
                    )
                profile_tensors: dict[str, tuple[Tensor, Tensor]] = {}
                deltas = src.last_component_deltas()
                truths = src.last_component_truths()
                if deltas is not None:
                    for axis, (a, b) in axis_pairs.items():
                        all_delta = torch.stack((deltas[a], deltas[b]), dim=-1)[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{axis}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{axis}",
                            best_delta,
                        )
                    if truths is not None:
                        for axis, (a, b) in axis_pairs.items():
                            truth_axis = torch.stack((truths[a], truths[b]), dim=-1)
                            delta_axis = torch.stack((deltas[a], deltas[b]), dim=-1)
                            profile_tensors[axis] = (truth_axis, delta_axis)
                kin_deltas = src.last_kinematic_deltas()
                if kin_deltas is not None:
                    rel_pt = kin_deltas.get("rel_pt")
                    if rel_pt is None:
                        break
                    for name in ("pt", "eta", "phi"):
                        tensor = kin_deltas.get(name)
                        if tensor is None:
                            continue
                        all_delta = tensor[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{name}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{name}",
                            best_delta,
                        )
                        truth_tensor = kin_deltas.get(f"truth_{name}")
                        if truth_tensor is not None:
                            profile_tensors[name] = (truth_tensor, tensor)
                    all_rel_pt = rel_pt[:, vb].reshape(-1)
                    all_rel_pt = all_rel_pt[torch.isfinite(all_rel_pt)]
                    t_sel = rel_pt[:, vb, :]
                    best_rel_pt = t_sel[best_k, cols, :].reshape(-1)
                    best_rel_pt = best_rel_pt[torch.isfinite(best_rel_pt)]
                    if all_rel_pt.numel() > 0:
                        out["diagnostics/reward_hacking/all/rel_pt/mean"] = float(
                            all_rel_pt.mean().detach().cpu()
                        )
                        out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float(
                            all_rel_pt.abs().mean().detach().cpu()
                        )
                    else:
                        out["diagnostics/reward_hacking/all/rel_pt/mean"] = float("nan")
                        out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float("nan")
                    if best_rel_pt.numel() > 0:
                        out["diagnostics/reward_hacking/best/rel_pt/mean"] = float(
                            best_rel_pt.mean().detach().cpu()
                        )
                        out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float(
                            best_rel_pt.abs().mean().detach().cpu()
                        )
                    else:
                        out["diagnostics/reward_hacking/best/rel_pt/mean"] = float("nan")
                        out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float("nan")
                    if log_distribution:
                        try:
                            import wandb  # noqa: F401

                            out["diagnostics/reward_hacking/dist/rel_pt"] = (
                                _rel_pt_distribution_figure(
                                    all_rel_pt.detach().float().cpu().numpy(),
                                    best_rel_pt.detach().float().cpu().numpy(),
                                )
                            )
                            pt_delta = kin_deltas.get("pt")
                            truth_pt = kin_deltas.get("truth_pt")
                            if pt_delta is not None and truth_pt is not None:
                                pt_all = pt_delta[:, vb]
                                truth_all = truth_pt[:, vb, :]
                                if truth_all.shape[0] == 1 and pt_all.shape[0] > 1:
                                    truth_all = truth_all.expand_as(pt_all)
                                best_pt_delta = pt_all[best_k, cols, :]
                                best_truth_pt = truth_all[best_k, cols, :]
                                pt_oracle_k = pt_all.abs().sum(dim=-1).argmin(dim=0)
                                oracle_pt_delta = pt_all[pt_oracle_k, cols, :]
                                oracle_truth_pt = truth_all[pt_oracle_k, cols, :]
                                _log_mean_abs(
                                    "diagnostics/reward_hacking/pt_oracle/pt",
                                    oracle_pt_delta,
                                )
                                prefix_k = min(10, int(pt_all.shape[0]))
                                pt_prefix = pt_all[:prefix_k]
                                truth_prefix = truth_all[:prefix_k]
                                rewards_prefix = rewards_v[:prefix_k]
                                prefix_reward_k = rewards_prefix.argmax(dim=0)
                                prefix_cols = torch.arange(
                                    int(prefix_reward_k.numel()),
                                    device=rewards.device,
                                    dtype=torch.long,
                                )
                                prefix_reward_delta = pt_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_reward_truth = truth_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_oracle_k = pt_prefix.abs().sum(dim=-1).argmin(dim=0)
                                prefix_oracle_delta = pt_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_oracle_truth = truth_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_reward_mask = (
                                    torch.isfinite(prefix_reward_truth)
                                    & torch.isfinite(prefix_reward_delta)
                                )
                                prefix_oracle_mask = (
                                    torch.isfinite(prefix_oracle_truth)
                                    & torch.isfinite(prefix_oracle_delta)
                                )
                                all_mask = torch.isfinite(truth_all) & torch.isfinite(pt_all)
                                best_mask = torch.isfinite(best_truth_pt) & torch.isfinite(best_pt_delta)
                                oracle_mask = (
                                    torch.isfinite(oracle_truth_pt)
                                    & torch.isfinite(oracle_pt_delta)
                                )
                                out["_diag_pt_profile_truth_all"] = (
                                    truth_all[all_mask].detach().float().cpu().numpy()
                                )
                                out["_diag_pt_profile_delta_all"] = (
                                    pt_all[all_mask].detach().float().cpu().numpy()
                                )
                                out["_diag_pt_profile_truth_best"] = (
                                    best_truth_pt[best_mask].detach().float().cpu().numpy()
                                )
                                out["_diag_pt_profile_delta_best"] = (
                                    best_pt_delta[best_mask].detach().float().cpu().numpy()
                                )
                                out["_diag_pt_profile_truth_oracle"] = (
                                    oracle_truth_pt[oracle_mask].detach().float().cpu().numpy()
                                )
                                out["_diag_pt_profile_delta_oracle"] = (
                                    oracle_pt_delta[oracle_mask].detach().float().cpu().numpy()
                                )
                                if int(pt_all.shape[0]) > prefix_k:
                                    out[
                                        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK"
                                    ] = _pt_delta_prefix_vs_full_figure(
                                        prefix_reward_truth[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_reward_delta[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_truth_pt[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_pt_delta[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_truth[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_delta[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_truth_pt[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_pt_delta[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_k=prefix_k,
                                        full_k=int(pt_all.shape[0]),
                                        title=(
                                            "pT response: first 10 candidates vs full "
                                            f"{int(pt_all.shape[0])}"
                                        ),
                                    )
                                out[
                                    "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt"
                                ] = _pt_delta_selection_profiles_figure(
                                    truth_all.detach().float().cpu().numpy(),
                                    pt_all.detach().float().cpu().numpy(),
                                    best_truth_pt.detach().float().cpu().numpy(),
                                    best_pt_delta.detach().float().cpu().numpy(),
                                    oracle_truth_pt.detach().float().cpu().numpy(),
                                    oracle_pt_delta.detach().float().cpu().numpy(),
                                    title="Reward selection pT bias vs truth pT",
                                )
                            for profile_name in ("eta", "phi", "px", "py", "pz"):
                                tensors = profile_tensors.get(profile_name)
                                if tensors is None:
                                    continue
                                truth_tensor, delta_tensor = tensors
                                truth_tensor = _align_truth_tensor_to_delta(
                                    truth_tensor, delta_tensor
                                )
                                profile_delta_all = delta_tensor[:, vb]
                                profile_truth_all = truth_tensor[:, vb]
                                profile_best_delta = profile_delta_all[best_k, cols, :]
                                profile_best_truth = profile_truth_all[best_k, cols, :]
                                profile_oracle_k = profile_delta_all.abs().sum(dim=-1).argmin(dim=0)
                                profile_oracle_delta = profile_delta_all[
                                    profile_oracle_k, cols, :
                                ]
                                profile_oracle_truth = profile_truth_all[
                                    profile_oracle_k, cols, :
                                ]
                                _log_mean_abs(
                                    f"diagnostics/reward_hacking/{profile_name}_oracle/{profile_name}",
                                    profile_oracle_delta,
                                )
                                truth_all_np, delta_all_np = _finite_profile_numpy(
                                    profile_truth_all, profile_delta_all
                                )
                                truth_best_np, delta_best_np = _finite_profile_numpy(
                                    profile_best_truth, profile_best_delta
                                )
                                truth_oracle_np, delta_oracle_np = _finite_profile_numpy(
                                    profile_oracle_truth, profile_oracle_delta
                                )
                                raw_items = {
                                    "truth_all": truth_all_np,
                                    "delta_all": delta_all_np,
                                    "truth_best": truth_best_np,
                                    "delta_best": delta_best_np,
                                    "truth_oracle": truth_oracle_np,
                                    "delta_oracle": delta_oracle_np,
                                }
                                for suffix, arr in raw_items.items():
                                    out[_diag_profile_raw_key(profile_name, suffix)] = arr
                                out[_diag_profile_log_key(profile_name)] = (
                                    _delta_selection_profiles_figure(
                                        truth_all_np,
                                        delta_all_np,
                                        truth_best_np,
                                        delta_best_np,
                                        truth_oracle_np,
                                        delta_oracle_np,
                                        profile_name=profile_name,
                                        title=_diag_profile_title(profile_name),
                                    )
                                )
                        except Exception:
                            pass
            else:
                nan_f = float("nan")
                for scope in ("all", "best"):
                    for axis in ("px", "py", "pz"):
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/reward_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_abs_mean"] = nan_f
                    for name in ("pt", "eta", "phi"):
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_abs_mean"] = nan_f
                out["diagnostics/reward_hacking/all/rel_pt/mean"] = nan_f
                out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = nan_f
                out["diagnostics/reward_hacking/best/rel_pt/mean"] = nan_f
                out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = nan_f
            break

    for src, w_mass_weight in reward_agg.sources:
        if not isinstance(src, WMassTruthNormalizedReward):
            continue
        rk, mp_kb, mm_kb = src.last_reward_tensors()
        tm, ts = src.last_truth_normalization()
        if rk is None or mp_kb is None or mm_kb is None:
            continue
        if tm is not None:
            out["physics/w_mass/truth_mean"] = float(tm)
        if ts is not None:
            out["physics/w_mass/truth_std"] = float(ts)
        if vb.sum() > 0:
            rv = rk[:, vb]
            out["reward/w_mass/mean"] = float(rv.mean().detach().cpu())
            out["reward/w_mass/std"] = (
                float(rv.std(unbiased=False).detach().cpu()) if rv.numel() > 1 else 0.0
            )
            out["reward/w_mass/weighted_mean"] = float(w_mass_weight * rv.mean().detach().cpu())
            out["reward/w_mass/best_of_k"] = float(rv.max(dim=0).values.mean().detach().cpu())
            out["reward/w_mass/last_place"] = float(rv.min(dim=0).values.mean().detach().cpu())
        else:
            nan = float("nan")
            out["reward/w_mass/mean"] = nan
            out["reward/w_mass/std"] = nan
            out["reward/w_mass/weighted_mean"] = nan
            out["reward/w_mass/best_of_k"] = nan
            out["reward/w_mass/last_place"] = nan

        mp_flat = mp_kb[:, vb].reshape(-1) if vb.sum() > 0 else mp_kb.reshape(-1)
        mm_flat = mm_kb[:, vb].reshape(-1) if vb.sum() > 0 else mm_kb.reshape(-1)
        mp_fin = mp_flat[torch.isfinite(mp_flat)]
        mm_fin = mm_flat[torch.isfinite(mm_flat)]
        if mp_fin.numel() > 0:
            out["physics/w_mass/plus_mean"] = float(mp_fin.mean().detach().cpu())
            out["physics/w_mass/plus_std"] = (
                float(mp_fin.std(unbiased=False).detach().cpu())
                if mp_fin.numel() > 1
                else 0.0
            )
        else:
            out["physics/w_mass/plus_mean"] = float("nan")
            out["physics/w_mass/plus_std"] = float("nan")
        if mm_fin.numel() > 0:
            out["physics/w_mass/minus_mean"] = float(mm_fin.mean().detach().cpu())
            out["physics/w_mass/minus_std"] = (
                float(mm_fin.std(unbiased=False).detach().cpu())
                if mm_fin.numel() > 1
                else 0.0
            )
        else:
            out["physics/w_mass/minus_mean"] = float("nan")
            out["physics/w_mass/minus_std"] = float("nan")
        break

    nan_f = float("nan")
    for src, wp_weight in reward_agg.sources:
        if not isinstance(src, WMassProjectionReward):
            continue
        rk, dkb2, vkb2 = src.last_projection_tensors()
        if rk is None or dkb2 is None or vkb2 is None:
            continue
        if vb.sum() > 0:
            rv = rk[:, vb]
            out["reward/w_projection/mean"] = float(rv.mean().detach().cpu())
            out["reward/w_projection/std"] = (
                float(rv.std(unbiased=False).detach().cpu()) if rv.numel() > 1 else 0.0
            )
            out["reward/w_projection/weighted_mean"] = float(wp_weight * rv.mean().detach().cpu())
            out["reward/w_projection/best_of_k"] = float(rv.max(dim=0).values.mean().detach().cpu())
            out["reward/w_projection/last_place"] = float(rv.min(dim=0).values.mean().detach().cpu())
            both_v = (vkb2[:, vb, 0] & vkb2[:, vb, 1]).detach()
            frac = both_v.float().mean()
            out["physics/w_projection/valid_fraction"] = float(frac.cpu())
            dk_v = dkb2[:, vb].detach()
            if bool(both_v.any().item()):
                d0 = dk_v[..., 0][both_v].abs().float()
                d1 = dk_v[..., 1][both_v].abs().float()
                cat = torch.cat((d0.reshape(-1), d1.reshape(-1)), dim=0)
                out["physics/w_projection/logpt_delta_abs_mean"] = float(cat.mean().cpu())
                out["physics/w_projection/logpt_delta_abs_p90"] = float(
                    torch.quantile(cat, 0.9).detach().cpu()
                )
            else:
                out["physics/w_projection/logpt_delta_abs_mean"] = nan_f
                out["physics/w_projection/logpt_delta_abs_p90"] = nan_f
        else:
            out["reward/w_projection/mean"] = nan_f
            out["reward/w_projection/std"] = nan_f
            out["reward/w_projection/weighted_mean"] = nan_f
            out["reward/w_projection/best_of_k"] = nan_f
            out["reward/w_projection/last_place"] = nan_f
            out["physics/w_projection/valid_fraction"] = nan_f
            out["physics/w_projection/logpt_delta_abs_mean"] = nan_f
            out["physics/w_projection/logpt_delta_abs_p90"] = nan_f
        break

    return out


@torch.no_grad()
def _build_train_metrics(
    diag_last: dict[str, Tensor],
    rewards: Tensor,
    valid_b: Tensor,
    advantages: Tensor | None = None,
) -> dict[str, float]:
    """Training panels ``reward/monitor/*``, ``train/loss/*``, and ``parameter/*`` for Weights & Biases."""
    param = _parameter_panel_from_diag(diag_last)
    vb = valid_b.reshape(-1) > 0
    adv_gap = (
        compute_reward_advantage_pos_neg_gap(rewards, advantages, valid_b)
        if advantages is not None
        else float("nan")
    )
    if vb.sum() == 0:
        nan = float("nan")
        base: dict[str, float] = {
            "train/loss/total": float(diag_last["loss_total"].cpu()),
            "train/loss/velocity": _diag_scalar_float(
                diag_last, "loss_velocity_training"
            ),
            "train/loss/offset_anchor": _diag_scalar_float(
                diag_last, "loss_offset_anchor"
            ),
            "train/loss/local_kl_anchor": _diag_scalar_float(
                diag_last, "loss_local_kl_anchor"
            ),
            "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
            "train/loss/kl": float(diag_last["loss_kl"].cpu()),
            "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
            "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
            "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
            "reward/monitor/best_of_k": nan,
            "reward/monitor/median": nan,
            "reward/monitor/mean_gap": nan,
            "reward/monitor/last_place": nan,
            "reward/monitor/p10": nan,
            "reward/monitor/p30": nan,
            "reward/monitor/p70": nan,
            "reward/monitor/p90": nan,
            "reward/monitor/advantage_pos_neg_gap": adv_gap,
        }
        if not math.isfinite(base["train/loss/velocity"]):
            base["train/loss/velocity"] = float(diag_last["loss_total"].cpu())
        if not math.isfinite(base["train/loss/offset_anchor"]):
            base["train/loss/offset_anchor"] = 0.0
        if not math.isfinite(base["train/loss/local_kl_anchor"]):
            base["train/loss/local_kl_anchor"] = 0.0
        base.update(param)
        if "loss_offset_anchor_quadratic_term" in diag_last:
            base["train/loss/offset_anchor_quadratic"] = _diag_scalar_float(
                diag_last, "loss_offset_anchor_quadratic_term"
            )
        for dk, dv in diag_last.items():
            if isinstance(dk, str) and dk.startswith("offset_anchor/"):
                base[dk] = float(dv.detach().float().cpu())
            if isinstance(dk, str) and dk.startswith("local_kl_anchor/"):
                base[dk] = float(dv.detach().float().cpu())
        return base

    r = rewards[:, vb]
    K = r.shape[0]
    best_of_k = float(r.max(dim=0).values.mean().cpu())
    median_e = float(r.median(dim=0).values.mean().cpu())
    last_place = float(r.min(dim=0).values.mean().cpu())
    p10 = float(r.quantile(0.1, dim=0).mean().cpu()) if K >= 2 else last_place
    p30 = float(r.quantile(0.3, dim=0).mean().cpu()) if K >= 2 else last_place
    p70 = float(r.quantile(0.7, dim=0).mean().cpu()) if K >= 2 else last_place
    p90 = float(r.quantile(0.9, dim=0).mean().cpu()) if K >= 2 else last_place
    mean_gap = compute_reward_mean_gap(rewards, valid_b)

    lv = _diag_scalar_float(diag_last, "loss_velocity_training")
    if not math.isfinite(lv):
        lv = float(diag_last["loss_total"].cpu())
    out: dict[str, float] = {
        "train/loss/total": float(diag_last["loss_total"].cpu()),
        "train/loss/velocity": lv,
        "train/loss/offset_anchor": _diag_scalar_float(
            diag_last, "loss_offset_anchor"
        ),
        "train/loss/local_kl_anchor": _diag_scalar_float(
            diag_last, "loss_local_kl_anchor"
        ),
        "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
        "train/loss/kl": float(diag_last["loss_kl"].cpu()),
        "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
        "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
        "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
        "reward/monitor/best_of_k": best_of_k,
        "reward/monitor/median": median_e,
        "reward/monitor/mean_gap": mean_gap,
        "reward/monitor/last_place": last_place,
        "reward/monitor/p10": p10,
        "reward/monitor/p30": p30,
        "reward/monitor/p70": p70,
        "reward/monitor/p90": p90,
        "reward/monitor/advantage_pos_neg_gap": adv_gap,
    }
    loa = out["train/loss/offset_anchor"]
    if not math.isfinite(loa):
        out["train/loss/offset_anchor"] = 0.0
    lkl = out["train/loss/local_kl_anchor"]
    if not math.isfinite(lkl):
        out["train/loss/local_kl_anchor"] = 0.0
    out.update(param)
    if "loss_offset_anchor_quadratic_term" in diag_last:
        out["train/loss/offset_anchor_quadratic"] = _diag_scalar_float(
            diag_last, "loss_offset_anchor_quadratic_term"
        )
    for dk, dv in diag_last.items():
        if isinstance(dk, str) and dk.startswith("offset_anchor/"):
            out[dk] = float(dv.detach().float().cpu())
        if isinstance(dk, str) and dk.startswith("local_kl_anchor/"):
            out[dk] = float(dv.detach().float().cpu())
    return out


def policy_evaluation_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Tensor,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    t: Tensor | None = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    """Shared ``eps`` per event (when ``shared_noise``).

    Returns:
        ``L_cur``, ``L_ref`` each ``(K, B)``, ``t`` ``(B,)``, ``model_v``, ``ref_v`` each
        ``(K*B, N_nu, F)``, ``noise_mask_rep`` ``(K*B, N_nu, 1)`` for KL anchor MSE,
        ``x_t``, ``target_v``, ``t_rep``, ``batch_rep`` for optional rollout-EMA PPO ratio.
    """
    B = batch["x"].shape[0]
    inv_mask = batch["x_invisible_mask"]
    c_flat = candidates_phys.reshape(K * B, *candidates_phys.shape[2:])
    # Row order matches ``(K, B, ...)`` reshape: all events for k=0, then k=1, ...
    inv_kb = inv_mask.unsqueeze(0).expand(K, -1, -1).reshape(K * B, *inv_mask.shape[1:])
    eve = _unwrap_core_evenet(model)
    c_norm = _normalize_candidates_for_policy(eve, c_flat, inv_kb)

    if t is None:
        t = torch.rand(B, device=device, dtype=torch.float32) * (t_max - t_min) + t_min
    _, alpha, sigma = get_logsnr_alpha_sigma(t, shape=(B, 1, 1))
    alpha_rep = alpha.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, 1, 1).to(dtype)
    sigma_rep = sigma.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, 1, 1).to(dtype)

    N_nu, F_eff = c_norm.shape[1], c_norm.shape[2]
    if shared_noise:
        eps = torch.randn(B, N_nu, F_eff, device=device, dtype=dtype)
        eps_rep = eps.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, N_nu, F_eff)
    else:
        eps_rep = torch.randn(K * B, N_nu, F_eff, device=device, dtype=dtype)

    x_t = alpha_rep * c_norm + sigma_rep * eps_rep
    target_v = alpha_rep * eps_rep - sigma_rep * c_norm

    t_rep = t.unsqueeze(0).expand(K, -1).reshape(K * B)
    batch_rep = repeat_batch_for_candidates(batch, K)
    noise_mask_rep = batch_rep["x_invisible_mask"].unsqueeze(-1)

    if isinstance(model, DDP):
        model_v = model(x_t, batch_rep, t_rep, noise_mask_rep)
    else:
        model_v = model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
    # ``predict_diffusion_vector`` already strips invisible_padding from its output,
    # so ``model_v`` and ``target_v`` share the same width (``invisible_input_dim``).
    L_cur = per_row_velocity_mse(model_v, target_v, noise_mask_rep, invisible_padding=0)

    with torch.no_grad():
        ref_v = ref_model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
        L_ref = per_row_velocity_mse(ref_v, target_v, noise_mask_rep, invisible_padding=0)

    return (
        L_cur.reshape(K, B),
        L_ref.reshape(K, B),
        t,
        model_v,
        ref_v,
        noise_mask_rep,
        x_t,
        target_v,
        t_rep,
        batch_rep,
    )


class _DgpoOptimizerWithSchedule:
    """Bundle ``(AdamW, LambdaLR)`` for DGPO: ``scheduler_step()`` once per batch.

    When ``dgpo.accumulate_train_timesteps`` is true, the train loop performs a single
    ``optimizer.step()`` per batch; when false, it may call ``step()`` per training sub-step
    (legacy behavior).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.step(*args, **kwargs)

    def scheduler_step(self) -> None:
        self.scheduler.step()

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Any) -> None:
        if isinstance(state, dict) and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            return
        try:
            self.optimizer.load_state_dict(state)
        except (ValueError, RuntimeError):
            raise
        _log.warning(
            "[DGPO] Loaded legacy optimizer state dict without scheduler keys; "
            "LambdaLR keeps its current step counter (may be out of sync with resume step).",
        )


def train_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_rollout: Any | None,
    ema_save: Any | None,
    batch: dict[str, Any],
    optimizer: Any,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    beta: float,
    beta_kl: float,
    advantage_positive_only: bool,
    advantage_mode: str = "zscore",
    advantage_temperature: float = 1.0,
    K: int,
    num_ddim_steps: int,
    shared_noise: bool,
    use_ema_for_rollout: bool,
    update_ema_rollout: bool,
    global_step: int,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    log_reward_dist: bool = False,
    log_diagnostic_dist: bool = False,
    num_inner_epochs: int = 1,
    num_train_timesteps: int = 1,
    adv_clip_max: float | None = None,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    ppo_clip_range: float | None = None,
    policy_eval_t_min: float = 0.0,
    policy_eval_t_max: float = 1.0,
    accumulate_train_timesteps: bool = False,
    normalization_dict: dict[str, Any] | None = None,
    dgpo_offset_anchor_stored_mu_ref: float | None = None,
    dgpo_offset_anchor_stored_mu_ref_xyz: tuple[float, float, float] | None = None,
    offset_anchor_dual_state: dict[str, float] | None = None,
    dgpo_local_kl_anchor_stored_p0_ref: float | None = None,
    reference_reward_kl_store: ReferenceRewardKlStore | None = None,
) -> dict[str, Any]:
    """Rollout once, then either one optimizer update per batch (accumulate) or one per sub-step (legacy)."""
    model.train()
    ref_model.eval()
    freeze_reference_model(ref_model)

    core = _unwrap_core_evenet(model)
    B = int(batch["x"].shape[0])

    if ppo_clip_range is not None and ema_rollout is None:
        _log.warning(
            "[DGPO] ppo_clip_range is set but rollout EMA is disabled; "
            "PPO ratio clipping is skipped (no rollout snapshot to compare).",
        )

    use_ppo_clip = ppo_clip_range is not None and ema_rollout is not None

    # --- Phase 1: generation (optional rollout EMA weights) ---
    buf: dict[str, Tensor] = {}
    if use_ema_for_rollout:
        if ema_rollout is None:
            _log.warning("[DGPO] use_ema_for_rollout=True but EMA disabled; using trainable weights.")
        else:
            buf = _save_trainable_weights(model)
            ema_rollout.copy_to(core)
    try:
        with torch.no_grad():
            candidates_phys = generate_neutrino_candidates(
                core,
                batch,
                sampler,
                K=K,
                num_ddim_steps=num_ddim_steps,
                device=device,
            )
    finally:
        if use_ema_for_rollout and buf and ema_rollout is not None:
            _restore_trainable_weights(model, buf)

    rewards, reward_breakdown = reward_agg.compute(candidates_phys, batch)
    valid_b = get_event_valid_mask(batch, B, device, dtype)
    gate_kb, gate_metrics = _directional_pt_gate_from_config(
        candidates_phys,
        batch,
        valid_b,
        device=device,
        dtype=dtype,
    )
    advantages, _ = compute_per_event_advantage(
        rewards,
        positive_only=advantage_positive_only,
        mode=advantage_mode,
        temperature=advantage_temperature,
        candidate_mask=gate_kb,
    )

    if adv_clip_max is not None:
        advantages = torch.clamp(advantages, -float(adv_clip_max), float(adv_clip_max))

    if gate_kb is not None:
        advantages = advantages * gate_kb

    kl_weights, kl_weight_metrics = _reward_based_kl_weights_from_config(rewards, valid_b)
    kl_weights_eff = kl_weights

    rrkl_cfg = resolve_reference_reward_kl_train_config(global_config.dgpo)
    rrkl_substitutes_pt_local_kl = (
        rrkl_cfg.enabled
        and reference_reward_kl_store is not None
        and len(reference_reward_kl_store) > 0
    )
    rrkl_miss_frac = float("nan")
    rrkl_evt_ref_wb = None
    rrkl_ref_reward_batch = None
    if rrkl_substitutes_pt_local_kl:
        assert reference_reward_kl_store is not None
        ek0, ek1, _ek_hint = event_key_pair_columns_from_batch(
            batch, rrkl_cfg, device=device,
        )
        rrkl_evt_ref_wb, rrkl_miss_frac = reference_reward_kl_store.lookup_weights(
            ek0, ek1, valid_b,
        )
        rrkl_ref_reward_batch = reference_reward_kl_store.lookup_reference_rewards(
            ek0, ek1,
        )
        kl_weights_eff = multiply_event_and_row_kl_weights(rrkl_evt_ref_wb, kl_weights)

    lka_cfg = resolve_local_kl_anchor_train_config(global_config.dgpo)
    compute_pt_local_anchor = bool(lka_cfg.enabled) and not rrkl_substitutes_pt_local_kl

    event_weights = None
    event_weight_metrics: dict[str, float] = {}

    beta_dgpo = float(beta)
    offset_cfg = resolve_offset_anchor_train_config(
        global_config.dgpo,
        normalization_dict,
        stored_mu_ref=dgpo_offset_anchor_stored_mu_ref,
        stored_mu_ref_xyz=dgpo_offset_anchor_stored_mu_ref_xyz,
    )
    offset_anchor_candidate_weights: Tensor | None = None
    if offset_cfg.enabled and offset_cfg.apply_to == "best_candidate":
        best_k = rewards.detach().argmax(dim=0)
        cols = torch.arange(B, device=device, dtype=torch.long)
        offset_anchor_candidate_weights = torch.zeros_like(rewards, dtype=torch.bool)
        offset_anchor_candidate_weights[best_k, cols] = True
    n_inner = max(1, int(num_inner_epochs))
    n_t = max(1, int(num_train_timesteps))
    acc_steps = n_inner * n_t

    dual_z_local = [0.0, 0]  # local sum(z_pt), local count
    optimizer_ran = False

    def _append_dual_snapshot_if_needed(diag: dict[str, Tensor]) -> None:
        dc_snap = offset_cfg.dual_control
        if (
            not dc_snap.enabled
            or not offset_cfg.enabled
            or offset_anchor_dual_state is None
            or int(global_step) < int(dc_snap.warmup_steps)
        ):
            return
        zp = diag.get("offset_anchor/z_pt")
        if isinstance(zp, Tensor) and int(zp.numel()) == 1:
            fv = float(zp.detach().reshape(-1)[0].cpu())
            if math.isfinite(fv):
                dual_z_local[0] += fv
                dual_z_local[1] += int(1)

    def _dgpo_substep() -> tuple[Tensor, dict[str, Tensor]]:
        (
            L_cur,
            L_ref,
            _,
            model_v,
            ref_v,
            noise_mask_rep,
            x_t,
            target_v,
            t_rep,
            batch_rep,
        ) = policy_evaluation_step(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            t=None,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
        )
        _dgpo_assert_train_step_invariants(L_ref, advantages, rewards)
        kl_per_row = per_row_velocity_mse(
            model_v,
            ref_v.detach(),
            noise_mask_rep,
            invisible_padding=0,
        )

        if use_ppo_clip:
            clip_range = float(ppo_clip_range)  # validated by use_ppo_clip
            buf_clip = _save_trainable_weights(model)
            ema_rollout.copy_to(core)
            with torch.no_grad():
                if isinstance(model, DDP):
                    ema_v = model(x_t, batch_rep, t_rep, noise_mask_rep)
                else:
                    ema_v = core.predict_diffusion_vector(
                        noise_x=x_t,
                        cond_x=batch_rep,
                        time=t_rep,
                        mode="neutrino",
                        noise_mask=noise_mask_rep,
                    )
                L_ema = per_row_velocity_mse(
                    ema_v, target_v, noise_mask_rep, invisible_padding=0
                ).reshape(K, B)
            _restore_trainable_weights(model, buf_clip)

            ratio = torch.exp(-L_cur + L_ema.detach())
            should_clip = torch.where(
                advantages > 0,
                ratio > 1.0 + clip_range,
                ratio < 1.0 - clip_range,
            )
            L_cur_clipped = torch.where(should_clip, L_cur.detach(), L_cur)
            loss_vel, dlast = _dgpo_loss_sum_k_mean_b(
                L_cur_clipped,
                L_ref,
                advantages,
                beta_dgpo,
                K,
                kl_per_row=kl_per_row,
                kl_weights=kl_weights_eff,
                beta_kl=float(beta_kl),
            )
        else:
            loss_vel, dlast = build_dgpo_loss(
                L_cur,
                L_ref,
                advantages,
                beta_dgpo,
                K,
                kl_per_row=kl_per_row,
                kl_weights=kl_weights_eff,
                beta_kl=float(beta_kl),
            )

        lam_base = (
            resolve_offset_anchor_lambda_coef(
                global_config.dgpo,
                fallback_lambda=float(offset_cfg.lambda_coef),
                global_step=int(global_step),
                epoch=int(epoch),
            )
            if offset_cfg.enabled
            else 0.0
        )
        loss_off_scalar, off_diag = compute_pt_offset_anchor(
            model_v=model_v,
            x_t=x_t,
            t_rep=t_rep,
            noise_mask_rep=noise_mask_rep,
            batch_kb=batch_rep,
            core_model=core,
            cartesian=_truth_generation_cartesian(),
            K=K,
            cfg=offset_cfg,
            candidate_weights_kb=offset_anchor_candidate_weights,
        )
        adaptive_mult = 1.0
        adaptive_diag: dict[str, float] = {}
        if off_diag and offset_cfg.enabled:
            adaptive_mult, adaptive_diag = resolve_offset_anchor_adaptive_lambda_multiplier(
                global_config.dgpo,
                offset_cfg,
                off_diag,
            )
        lam_o = float(lam_base) * float(adaptive_mult)
        if lam_o > 0.0 and offset_cfg.enabled:
            loss_anchor_term = lam_o * loss_off_scalar
        else:
            loss_anchor_term = loss_vel * 0.0

        dc = offset_cfg.dual_control
        loss_dual_term = loss_vel.new_zeros(())
        if (
            dc.enabled
            and offset_cfg.enabled
            and offset_cfg.mode in ("pt_mean", "pt_eta_phi_mean")
            and offset_anchor_dual_state is not None
            and off_diag
            and int(global_step) >= int(dc.warmup_steps)
        ):
            zp_maybe = off_diag.get("offset_anchor/z_pt")
            if isinstance(zp_maybe, Tensor) and int(zp_maybe.numel()) == 1:
                zp_v = zp_maybe.reshape(())
                dp = float(offset_anchor_dual_state.get("dual_pt", 0.0))
                if math.isfinite(dp):
                    dp_t = loss_vel.new_tensor(dp)
                    loss_dual_term = dp_t * zp_v

        nu_slots_lka = int(model_v.shape[1])
        cart_lka = _truth_generation_cartesian()
        loss_lka_scalar: Tensor
        lka_diag: dict[str, Tensor]
        if compute_pt_local_anchor:
            truth_pt_kb = truth_pt_slot_kb_gev(
                batch_rep, cartesian=cart_lka, num_slots=nu_slots_lka,
            )
            loss_lka_scalar, lka_diag = compute_local_kl_anchor_loss(
                model_v,
                ref_v,
                noise_mask_rep,
                truth_pt_kb,
                cfg=lka_cfg,
                p0_ref=dgpo_local_kl_anchor_stored_p0_ref,
            )
        else:
            loss_lka_scalar = loss_vel.new_zeros(())
            lka_diag = {}

        loss_backward = loss_vel + loss_anchor_term + loss_dual_term + loss_lka_scalar

        if off_diag:
            dlast.update(off_diag)
            for ak, av in adaptive_diag.items():
                dlast[ak] = torch.tensor(float(av), device=device, dtype=torch.float64)
            dlast["offset_anchor/lambda_base"] = torch.tensor(
                float(lam_base), device=device, dtype=torch.float64
            )
            dlast["offset_anchor/lambda_effective"] = torch.tensor(
                float(lam_o), device=device, dtype=torch.float64
            )
            if offset_anchor_dual_state is not None and offset_cfg.dual_control.enabled:
                dlast["offset_anchor/dual_pt"] = torch.tensor(
                    float(offset_anchor_dual_state.get("dual_pt", 0.0)),
                    device=device,
                    dtype=torch.float64,
                )
                z0 = offset_anchor_dual_state.get("z_ema_pt")
                zi = float(z0) if z0 is not None and math.isfinite(float(z0)) else 0.0
                dlast["offset_anchor/z_ema_pt"] = torch.tensor(zi, device=device, dtype=torch.float64)
                dlast["offset_anchor/dual_active"] = torch.tensor(
                    1.0
                    if int(global_step) >= int(offset_cfg.dual_control.warmup_steps)
                    else 0.0,
                    device=device,
                    dtype=torch.float64,
                )
                if dc.enabled:
                    dlast["offset_anchor/dual_loss_pt"] = loss_dual_term.detach().to(dtype=torch.float64)
                else:
                    dlast["offset_anchor/dual_loss_pt"] = torch.zeros(
                        (), device=device, dtype=torch.float64
                    )
        if lka_diag:
            dlast.update(lka_diag)

        dlast["loss_velocity_training"] = loss_vel.detach()
        dlast["loss_offset_anchor_raw"] = loss_off_scalar.detach()
        anchor_plus_dual = loss_anchor_term + loss_dual_term
        dlast["loss_offset_anchor"] = anchor_plus_dual.detach()
        if offset_cfg.dual_control.enabled:
            dlast["loss_offset_anchor_quadratic_term"] = loss_anchor_term.detach()
        dlast["loss_local_kl_anchor"] = loss_lka_scalar.detach()
        dlast["loss_total"] = loss_backward.detach()
        return loss_backward, dlast

    grad_norm_pre_clip_max = 0.0
    grad_clip_active_any = False
    diag_last: dict[str, Tensor] = {}
    skipped_substeps = 0
    if accumulate_train_timesteps:
        diags: list[dict[str, Tensor]] = []
        optimizer.zero_grad(set_to_none=True)
        sub = 0
        for _inner in range(n_inner):
            for _t_step in range(n_t):
                sub += 1
                is_last = sub == acc_steps
                loss, dlast = _dgpo_substep()
                diags.append(dlast)
                if not torch.isfinite(loss):
                    skipped_substeps += 1
                    _log.warning(
                        "[DGPO] non-finite substep loss (%s); skipping backward "
                        "(step=%s sub=%s/%s).",
                        float(loss.detach().float().cpu()),
                        global_step, sub, acc_steps,
                    )
                    continue
                ctx = model.no_sync() if isinstance(model, DDP) and not is_last else nullcontext()
                with ctx:
                    (loss / float(acc_steps)).backward()
                _append_dual_snapshot_if_needed(dlast)
        gn, clip_on = _grad_norm_pre_clip_and_clip_active(
            model, float(grad_clip_norm)
        )
        grad_norm_pre_clip_max = gn
        grad_clip_active_any = clip_on > 0.5
        if skipped_substeps == acc_steps or not math.isfinite(gn):
            _log.warning(
                "[DGPO] all substeps non-finite or grad-norm non-finite (%s); "
                "skipping optimizer.step at global_step=%s.",
                gn, global_step,
            )
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            optimizer_ran = True
        diag_last = _mean_diag_dict(diags)
    else:
        for _inner in range(n_inner):
            for _t_step in range(n_t):
                loss, diag_last = _dgpo_substep()
                if not torch.isfinite(loss):
                    skipped_substeps += 1
                    _log.warning(
                        "[DGPO] non-finite substep loss (%s); skipping optimizer "
                        "step at global_step=%s.",
                        float(loss.detach().float().cpu()), global_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn, clip_on = _grad_norm_pre_clip_and_clip_active(
                    model, float(grad_clip_norm)
                )
                if not math.isfinite(gn):
                    _log.warning(
                        "[DGPO] non-finite grad-norm (%s); skipping optimizer.step "
                        "at global_step=%s.",
                        gn, global_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                grad_norm_pre_clip_max = max(grad_norm_pre_clip_max, gn)
                grad_clip_active_any = grad_clip_active_any or (clip_on > 0.5)
                optimizer.step()
                optimizer_ran = True
                _append_dual_snapshot_if_needed(diag_last)

    if optimizer_ran:
        if (
            offset_cfg.dual_control.enabled
            and offset_anchor_dual_state is not None
            and dual_z_local[1] > 0
        ):
            dc_pi = offset_cfg.dual_control
            zp_buf = torch.tensor(
                [float(dual_z_local[0]), float(dual_z_local[1])],
                device=device,
                dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(zp_buf, op=dist.ReduceOp.SUM)
            z_sum = float(zp_buf[0].detach().cpu())
            n_sum = float(zp_buf[1].detach().cpu())
            zp_mean = float(z_sum / max(n_sum, 1.0))

            dp0 = float(offset_anchor_dual_state.get("dual_pt", 0.0))
            zem0 = float(offset_anchor_dual_state.get("z_ema_pt", 0.0))
            dp_new, zem_new, dc_metrics = offset_anchor_dual_state_step(
                cfg=dc_pi,
                z_pt_batch=zp_mean,
                dual_pt=dp0,
                z_ema_pt=zem0,
                global_step=int(global_step),
            )
            offset_anchor_dual_state["dual_pt"] = float(dp_new)
            offset_anchor_dual_state["z_ema_pt"] = float(zem_new)
            diag_last["offset_anchor/dual_pt_next"] = torch.tensor(
                dp_new, device=device, dtype=torch.float64
            )
            diag_last["offset_anchor/z_ema_pt_next"] = torch.tensor(
                zem_new, device=device, dtype=torch.float64
            )
            for mk, mv in dc_metrics.items():
                diag_last[mk] = torch.tensor(
                    float(mv), device=device, dtype=torch.float64
                )

    if optimizer_ran and update_ema_rollout and ema_rollout is not None:
        ema_rollout.update(core, decay_=_dgpo_rollout_ema_decay(global_step))
    if ema_save is not None:
        ema_cfg = global_config.options.Training.get("EMA", None) or {}
        ema_every_n = max(1, int(ema_cfg.get("update_every_n_steps", 1)))
        if global_step % ema_every_n == 0:
            ema_save.update(core)

    out: dict[str, Any] = _build_train_metrics(
        diag_last,
        rewards,
        valid_b,
        advantages=advantages,
    )
    out.update(event_weight_metrics)
    out.update(kl_weight_metrics)
    out.update(gate_metrics)

    if log_reward_dist:
        out.update(build_reward_distribution_histograms(rewards, valid_b))
    out.update(
        _build_reward_extra_metrics(
            rewards,
            valid_b,
            reward_agg,
            reward_breakdown,
            log_distribution=log_diagnostic_dist,
        )
    )
    out.update(
        _build_reference_bias_metrics(
            candidates_phys,
            batch,
            cartesian=_truth_generation_cartesian(),
            log_distribution=log_diagnostic_dist,
        )
    )
    out["train/grad/global_norm_pre_clip"] = float(grad_norm_pre_clip_max)
    out["train/grad/clip_active"] = 1.0 if grad_clip_active_any else 0.0
    out["dgpo/beta_kl_current"] = float(beta_kl)
    out["dgpo/rollout_ema_update_enabled"] = 1.0 if update_ema_rollout else 0.0

    if rrkl_cfg.enabled:
        vb_rr = valid_b.reshape(-1) > 0
        out["reference_reward_kl/active"] = 1.0 if rrkl_substitutes_pt_local_kl else 0.0
        out["reference_reward_kl/missing_key_fraction"] = float(rrkl_miss_frac)
        if rrkl_substitutes_pt_local_kl:
            mr = rrkl_ref_reward_batch.detach().reshape(-1)
            vb_line = vb_rr.reshape(-1)
            finite_r = vb_line & torch.isfinite(mr)
            if bool(finite_r.any().item()):
                sel = mr[finite_r]
                out["reference_reward_kl/ref_reward_mean"] = float(sel.mean().detach().cpu())
                out["reference_reward_kl/ref_reward_min"] = float(sel.min().detach().cpu())
                out["reference_reward_kl/ref_reward_max"] = float(sel.max().detach().cpu())
            else:
                nan_f = float("nan")
                out["reference_reward_kl/ref_reward_mean"] = nan_f
                out["reference_reward_kl/ref_reward_min"] = nan_f
                out["reference_reward_kl/ref_reward_max"] = nan_f
            if rrkl_evt_ref_wb is not None:
                ww = rrkl_evt_ref_wb.detach().reshape(-1)
                selw = ww[vb_line]
                if selw.numel() > 0:
                    out["reference_reward_kl/weight_mean"] = float(selw.mean().detach().cpu())
                    out["reference_reward_kl/weight_min"] = float(selw.min().detach().cpu())
                    out["reference_reward_kl/weight_max"] = float(selw.max().detach().cpu())
                else:
                    nf = float("nan")
                    out["reference_reward_kl/weight_mean"] = nf
                    out["reference_reward_kl/weight_min"] = nf
                    out["reference_reward_kl/weight_max"] = nf

    # Per-batch kinematic histogram counts for epoch-level training-distribution plots.
    cartesian = _truth_generation_cartesian()
    k_sel = _kin_hist_candidate_indices_per_event(
        rewards, candidates_phys, batch, cartesian=cartesian
    )
    ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
        candidates_phys, batch, k_sel, cartesian=cartesian, device=device
    )
    k1_sel = torch.zeros(B, device=device, dtype=torch.long)
    k1_pt, k1_eta, k1_phi, k1_tpt, k1_teta, k1_tphi = _val_pred_truth_kin_flat(
        candidates_phys, batch, k1_sel, cartesian=cartesian, device=device
    )
    _td_pt_edges = np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
    _td_eta_edges = np.linspace(-4.0, 4.0, _VAL_KIN_NUM_BINS + 1)
    _td_phi_edges = np.linspace(-3.2, 3.2, _VAL_KIN_NUM_BINS + 1)
    out["_kin_h_pt_p"] = np.histogram(ppt, bins=_td_pt_edges)[0].astype(np.float64)
    out["_kin_h_pt_t"] = np.histogram(tpt, bins=_td_pt_edges)[0].astype(np.float64)
    out["_kin_h_e_p"] = np.histogram(peta, bins=_td_eta_edges)[0].astype(np.float64)
    out["_kin_h_e_t"] = np.histogram(teta, bins=_td_eta_edges)[0].astype(np.float64)
    out["_kin_h_p_p"] = np.histogram(pphi, bins=_td_phi_edges)[0].astype(np.float64)
    out["_kin_h_p_t"] = np.histogram(tphi, bins=_td_phi_edges)[0].astype(np.float64)
    out["_kin_h_pt_k1_p"] = np.histogram(k1_pt, bins=_td_pt_edges)[0].astype(np.float64)
    out["_kin_h_pt_k1_t"] = np.histogram(k1_tpt, bins=_td_pt_edges)[0].astype(np.float64)
    out["_kin_h_e_k1_p"] = np.histogram(k1_eta, bins=_td_eta_edges)[0].astype(np.float64)
    out["_kin_h_e_k1_t"] = np.histogram(k1_teta, bins=_td_eta_edges)[0].astype(np.float64)
    out["_kin_h_p_k1_p"] = np.histogram(k1_phi, bins=_td_phi_edges)[0].astype(np.float64)
    out["_kin_h_p_k1_t"] = np.histogram(k1_tphi, bins=_td_phi_edges)[0].astype(np.float64)

    optimizer.scheduler_step()
    return out


def build_optimizer(
    model: torch.nn.Module,
    *,
    steps_per_epoch: int,
    warmup_steps: int,
    is_rank0: bool = True,
) -> _DgpoOptimizerWithSchedule:
    """AdamW with EveNet-style ``optimizer_group`` LR/WD (no world-size scaling) + linear warmup then constant LR.

    ``warmup_steps`` counts **batches** (one ``scheduler_step`` per call to :func:`train_step`, after
    which ``train_step`` may have performed one or several ``optimizer.step()`` calls depending on
    ``dgpo.accumulate_train_timesteps``).

    Parameters
    ----------
    model:
        ``EveNetModel`` or ``DDP(_DGPODDPForward(EveNetModel))``; parameters are taken from the
        unwrapped core for grouping (same tensor objects as ``model.parameters()``).
    steps_per_epoch:
        Logged on rank 0 for traceability (matches the worker's batch count per epoch).
    warmup_steps:
        Batches for linear LR ramp ``min(1, epoch / warmup_steps)`` on groups with ``warm_up: true``.
    is_rank0:
        When True, log one line per optimizer group on construction.
    """
    core = _unwrap_core_evenet(model)
    train_opt = global_config.options.Training
    components = train_opt.Components
    default_lr = float(train_opt.learning_rate)
    default_wd = float(train_opt.weight_decay)

    group_meta: dict[str, dict[str, Any]] = {}
    group_modules: dict[str, list[str]] = defaultdict(list)

    for comp_key, cfg in components.items():
        if cfg is None:
            continue
        group = cfg.get("optimizer_group", None)
        if not group:
            continue
        module_attr = getattr(core, comp_key, None)
        if module_attr is None:
            continue
        gname = str(group)
        group_modules[gname].append(str(comp_key))
        if gname not in group_meta:
            lr = float(cfg.get("learning_rate", default_lr))
            wd = float(cfg.get("weight_decay", default_wd))
            warm_up = bool(cfg.get("warm_up", True))
            opt_type = str(cfg.get("optimizer_type", "AdamW"))
            group_meta[gname] = {
                "lr": lr,
                "weight_decay": wd,
                "warm_up": warm_up,
                "optimizer_type": opt_type,
            }

    bad = [
        (g, m["optimizer_type"])
        for g, m in group_meta.items()
        if str(m["optimizer_type"]).lower() != "adamw"
    ]
    if bad:
        raise ValueError(
            "DGPO build_optimizer only supports AdamW parameter groups; got: "
            + ", ".join(f"{g}={t!r}" for g, t in bad)
        )

    ws = max(1, int(warmup_steps))
    param_groups: list[dict[str, Any]] = []
    lr_lambdas: list[Any] = []
    nonempty_group_order: list[str] = []

    for gname, meta in group_meta.items():
        seen_ids: set[int] = set()
        params: list[nn.Parameter] = []
        for comp_key in group_modules[gname]:
            mod = getattr(core, comp_key, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    params.append(p)
        if not params:
            continue
        nonempty_group_order.append(gname)
        param_groups.append(
            {
                "params": params,
                "lr": float(meta["lr"]),
                "weight_decay": float(meta["weight_decay"]),
            }
        )
        if meta["warm_up"]:
            lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))
        else:
            lr_lambdas.append(lambda _epoch: 1.0)

    if not param_groups:
        raise ValueError(
            "[DGPO] build_optimizer: no trainable parameters matched Components with "
            "optimizer_group (check include/freeze settings)."
        )

    # Fallback group for any trainable parameter not covered by an optimizer_group
    # (preserves the original permissive behavior of optimizing everything trainable).
    assigned = {id(p) for pg in param_groups for p in pg["params"]}
    leftover = [
        p for p in core.parameters() if p.requires_grad and id(p) not in assigned
    ]
    if leftover:
        if is_rank0:
            n_leftover = sum(p.numel() for p in leftover)
            _log.warning(
                "[DGPO] %s trainable parameters (%s elements) are not covered by any "
                "Components.<X>.optimizer_group; adding to a fallback group with "
                "lr=%s wd=%s warm_up=true.",
                len(leftover),
                n_leftover,
                default_lr,
                default_wd,
            )
        nonempty_group_order.append("__fallback__")
        param_groups.append(
            {
                "params": leftover,
                "lr": default_lr,
                "weight_decay": default_wd,
            }
        )
        lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    if is_rank0:
        _log.info(
            "[DGPO] Optimizer: AdamW groups=%s steps/epoch≈%s warmup_batches=%s (linear→constant).",
            len(param_groups),
            int(steps_per_epoch),
            ws,
        )
        for i, gname in enumerate(nonempty_group_order):
            pg = optimizer.param_groups[i]
            npar = sum(p.numel() for p in pg["params"])
            if gname in group_meta:
                mods = ", ".join(group_modules[gname])
                warm = group_meta[gname]["warm_up"]
            else:
                mods = "<fallback>"
                warm = True
            _log.info(
                "[DGPO]   group %r modules=[%s] params=%s lr=%s wd=%s warm_up=%s",
                gname,
                mods,
                npar,
                pg["lr"],
                pg["weight_decay"],
                warm,
            )

    return _DgpoOptimizerWithSchedule(optimizer, scheduler)


def _dgpo_wandb_metric_definition_map() -> dict[str, str]:
    """Explicit definitions for W&B Config → dgpo_metric_definitions (visible in the UI)."""
    return {
        "epoch": "Training epoch index (x-axis for most plots).",
        # --- reward/dist (overlaid figure, every log_reward_dist_every steps) ---
        "reward/dist/overlap": "Matplotlib figure: three overlapped 1D density histograms (best / worst / median per valid event). wandb.Image — use the media step slider to compare across training steps.",
        # --- reward/monitor (scalars, every step) ---
        "reward/monitor/best_of_k": "Mean reward of the argmax (best) candidate per valid event.",
        "reward/monitor/median": "Mean over events of the median reward along K.",
        "reward/monitor/mean_gap": "Mean over events: (mean reward strictly above per-event median) − (mean reward strictly below median).",
        "reward/monitor/last_place": "Mean reward of the worst (min) candidate per valid event.",
        "reward/monitor/p10": "Mean over events of the 10th percentile of rewards along K.",
        "reward/monitor/p30": "Mean over events of the 30th percentile of rewards along K.",
        "reward/monitor/p70": "Mean over events of the 70th percentile of rewards along K.",
        "reward/monitor/p90": "Mean over events of the 90th percentile of rewards along K.",
        "reward/monitor/advantage_pos_neg_gap": "Mean reward where advantage > 0 minus mean where advantage < 0 (valid slots).",
        "reward/sources/*/mean": "Raw mean reward for each additive reward source over all valid rollout candidates.",
        "reward/sources/*/weighted_mean": "Configured reward weight times reward/sources/*/mean; these add up to reward/raw/mean.",
        "reward/sources/*/selected_by_total_mean": "Per-source raw reward after selecting the candidate with highest combined total reward per valid event.",
        "reward/sources/*/selected_by_total_weighted_mean": "Configured reward weight times selected_by_total_mean.",
        "reward/sources/*/source_best_of_k": "For each source alone, mean over events of max_k source_reward[k,event].",
        "reward/sources/*/source_last_place": "For each source alone, mean over events of min_k source_reward[k,event].",
        "reward/sources/*/selection_gap": "selected_by_total_mean - mean. Large shifts show how combined reward selection biases that source.",
        # --- dgpo (train scalars) ---
        "dgpo/beta_kl_current": "Effective KL weight used in this training step: DGPO scalar = loss_main + beta_kl_current * loss_kl (before optional offset-anchor and local KL anchor terms). With beta_kl_schedule enabled, this is piecewise-linear in global_step or epoch (resume continues from checkpoint global_step).",
        "dgpo/rollout_ema_update_enabled": "1 when the fast rollout EMA is updated after the batch; 0 when dgpo.update_ema_rollout=false freezes rollouts at the initial loaded policy for bias diagnostics.",
        # --- train/loss ---
        "train/loss/total": "Scalar passed to backward(): DGPO main + β_kl weighted velocity KL; plus optional λ·offset-anchor; optional local KL anchor unless replaced by dgpo.reference_reward_kl frozen weights.",
        "train/loss/dgpo": "DGPO main term (detached gate × advantage × L_cur). Lower is better.",
        "train/loss/kl": "Velocity KL anchor aggregate (||vθ−v_ref||²), optionally per-event weighted by dgpo.reward_kl_weight **and/or** frozen dgpo.reference_reward_kl LUT from epoch −1 ref rollouts.",
        "train/loss/L_cur": "Mean velocity MSE for the trainable policy (DDIM target). Lower is better.",
        "train/loss/L_ref": "Mean velocity MSE for the frozen reference policy. Lower is better.",
        "train/loss/delta": "mean(|L_cur - L_ref|): average absolute gap between current and reference velocity MSE. Shows how far the policy has moved from reference.",
        "train/loss/velocity": "Detached DGPO(+KL) part of the objective—the same differentiable scalar as backward when additional anchors are disabled. With offset and/or local KL anchor on, backward total adds train/loss/offset_anchor + train/loss/local_kl_anchor.",
        "train/loss/offset_anchor": "Detached effective anchor backward group: λ_eff * Huber/MSE on pooled residuals **plus optional dual_pt * z_pt** when dgpo.offset_anchor.dual_control.enabled; see train/loss/offset_anchor_quadratic for λ-only slice.",
        "train/loss/local_kl_anchor": "Detached extra fixed-reference velocity MSE weighted by truth-p_T Gaussian around frozen p0_ref (see dgpo.local_kl_anchor). Zero when disabled or p0_ref non-finite.",
        # --- offset_anchor (dense truth-pT residual anchor + val baseline) ---
        "offset_anchor/baseline_mu_ref": "Dense-band pooled mean Δp_T (pred−truth, GeV) from validation at epoch −1 when baseline_mu_ref_from_epoch_minus_one_val is True; used for pt_mean if ckpt omits dgpo_offset_anchor_mu_ref (not used as primary target when mode=xyz_mean).",
        "offset_anchor/baseline_mu_ref_px": "Dense-band pooled mean Δpx [GeV] for xyz_mean when ckpt omits dgpo_offset_anchor_mu_ref_xyz.",
        "offset_anchor/baseline_mu_ref_py": "Dense-band pooled mean Δpy [GeV] (same mask as baseline_mu_ref).",
        "offset_anchor/baseline_mu_ref_pz": "Dense-band pooled mean Δpz [GeV] (same mask).",
        "offset_anchor/baseline_mask_count_total": "Σ mask entries summed across ranks (both ν slots, open interval pt_min<truth_pt<pt_max).",
        "offset_anchor/lambda_base": "Offset-anchor weight after optional lambda_schedule, before adaptive multiplier.",
        "offset_anchor/lambda_effective": "Effective offset-anchor weight after optional lambda_schedule and adaptive_lambda multiplier.",
        "offset_anchor/adaptive_lambda_multiplier": "Detached adaptive multiplier applied to lambda_base.",
        "offset_anchor/adaptive_lambda_trigger": "Nonnegative scaled residual drift that triggered the adaptive multiplier.",
        "offset_anchor/adaptive_lambda_z": "Selected residual divided by its scale before adaptive direction/deadband.",
        "offset_anchor/mode_xyz": "1 when dgpo.offset_anchor.mode=xyz_mean; 0 otherwise.",
        "offset_anchor/mode_pt_eta_phi": "1 when dgpo.offset_anchor.mode=pt_eta_phi_mean; 0 otherwise.",
        "offset_anchor/mask_count": "Per training sub-step: count of entries in the pooled residual mean.",
        "offset_anchor/mu_theta": "Detached pooled mean Δp_T (GeV); pt_mean mode only.",
        "offset_anchor/mu_ref": "Detached scalar target μ_ref for pt_mean (ckpt dgpo_offset_anchor_mu_ref > epoch−1 baseline > YAML).",
        "offset_anchor/delta_mu": "Detached μ_θ − μ_ref (pt_mean).",
        "offset_anchor/z_scaled": "Detached (μ_θ − μ_ref) / scale (pt_mean only).",
        "offset_anchor/mu_theta_px": "Detached pooled mean Δpx [GeV] (xyz_mean).",
        "offset_anchor/mu_theta_py": "Detached pooled mean Δpy [GeV] (xyz_mean).",
        "offset_anchor/mu_theta_pz": "Detached pooled mean Δpz [GeV] (xyz_mean).",
        "offset_anchor/mu_ref_px": "Detached target Δpx component (ckpt vector > YAML mu_ref_xyz).",
        "offset_anchor/mu_ref_py": "Detached target Δpy component.",
        "offset_anchor/mu_ref_pz": "Detached target Δpz component.",
        "offset_anchor/delta_mu_px": "Detached μ_θ_px − μ_ref_px.",
        "offset_anchor/delta_mu_py": "Detached μ_θ_py − μ_ref_py.",
        "offset_anchor/delta_mu_pz": "Detached μ_θ_pz − μ_ref_pz.",
        "offset_anchor/mu_theta_pt": "Detached pooled mean ΔpT [GeV] for pt_eta_phi_mean.",
        "offset_anchor/mu_theta_eta": "Detached pooled mean Δη for pt_eta_phi_mean.",
        "offset_anchor/mu_theta_phi": "Detached pooled mean wrapped Δφ [rad] for pt_eta_phi_mean.",
        "offset_anchor/mu_ref_pt": "Detached target ΔpT component for pt_eta_phi_mean.",
        "offset_anchor/mu_ref_eta": "Detached target Δη component for pt_eta_phi_mean.",
        "offset_anchor/mu_ref_phi": "Detached target Δφ component for pt_eta_phi_mean.",
        "offset_anchor/delta_mu_pt": "Detached μ_θ_pt − μ_ref_pt.",
        "offset_anchor/delta_mu_eta": "Detached μ_θ_eta − μ_ref_eta.",
        "offset_anchor/delta_mu_phi": "Detached μ_θ_phi − μ_ref_phi.",
        "offset_anchor/active": "1 when the anchor diagnostics block is emitting values for this sub-step.",
        "offset_anchor/raw_loss": "Detached mean Huber/MSE on scaled residuals (scalar for pt_mean; averaged over axes for vector modes).",
        "offset_anchor/raw_loss_pt_eta_phi": "Same as raw_loss for pt_eta_phi_mean; logged separately for mode-specific plots.",
        "offset_anchor/skipped_small_mask": "1 if mask_count < dgpo.offset_anchor.min_count for this sub-step (backward anchor term is omitted).",
        "offset_anchor/z_pt": "Differentiable scaled pT statistic for dual_control: equals z_scaled pt_mean diagnostics, or pt component z_scaled[0] when mode=pt_eta_phi_mean.",
        "offset_anchor/dual_pt": "Dual multiplier λ_dual on z_pt immediately before optional PI step at batch end.",
        "offset_anchor/z_ema_pt": "Detached EMA of global batch-mean z_pt feeding the dual increment.",
        "offset_anchor/dual_active": "1 when dual_control is enabled and global_step≥warmup with finite detached batch mean; else 0 during warmup/off.",
        "offset_anchor/dual_loss_pt": "Detached differentiable dual term dual_pt * z_pt contributing to train/loss/offset_anchor alongside λ * raw_anchor_loss.",
        "offset_anchor/dual_leak": "YAML dual_control leak in (0,1]; scales dual_pt before optional dual_lr*z_ema_after integration.",
        "offset_anchor/dual_deadband": "YAML dual_control deadband on |EMA(z_pt)| after batch mean; skips integration when smaller (anti-windup leak-only decay).",
        "offset_anchor/dual_integrating": "1 when |z_ema_after|≥deadband on this detached PI step, else PI step is leak-only on dual_pt.",
        "offset_anchor/dual_pt_next": "Dual multiplier after clipped PI update (post successful optimizer.step).",
        "offset_anchor/z_ema_pt_next": "EMA after PI update.",
        "offset_anchor/z_pt_batch_mean": "all_reduce SUM/COUNT global mean of detached z_pt used for PI (same statistic as summed in train_step when dual aggregates).",
        "offset_anchor/z_ema_pt_after": "Detached EMA value after absorbing z_pt_batch_mean (controller diagnostic).",
        "offset_anchor/dual_update_pt": "Clipped dual increment applied this PI step.",
        "offset_anchor/dual_pt_after": "Dual multiplier after clipping (equals dual_pt_next).",
        "train/loss/offset_anchor_quadratic": "Detached λ_eff * raw anchor loss without dual_pt * z_pt (logged only when dual_control.enabled adds a separate quadratic term metric).",
        # --- local_kl_anchor (fixed-region reference velocity anchor) ---
        "local_kl_anchor/p0_used": "Truth p_T [GeV] center used for Gaussian weights on this sub-step (equals frozen p0_ref when active).",
        "local_kl_anchor/loss": "Detached mean weighted ||v_θ − v_ref||² with slot Gaussian weights by truth pT (same mask as KL).",
        "local_kl_anchor/weight_mean": "Mean Gaussian+baseline slot weight among valid-masked neutrino slots in the minibatch.",
        "local_kl_anchor/weight_max": "Maximum slot weight among valid-masked slots.",
        "local_kl_anchor/weight_min": "Minimum slot weight among valid-masked slots.",
        "local_kl_anchor/truth_pt_mean": "Mean truth p_T [GeV] over valid-masked slots in the minibatch.",
        "local_kl_anchor/active": "1 when the local KL anchor contributes a differentiable term this sub-step.",
        "local_kl_anchor/p0_ref_fit": "Fitted truth p_T [GeV] where frozen-reference mean Δp_T vs truth-p_T profile crosses zero (epoch −1 ref DDIM rollout, globally gathered). Used to freeze dgpo.local_kl_anchor p0_ref when YAML p0_ref and checkpoint omit it.",
        "local_kl_anchor/fit_slope": "Slope [GeV/GeV] of binned linear fit mean(Δp_T^ref) vs truth p_T used for p0_ref_fit.",
        "local_kl_anchor/fit_intercept": "Intercept [GeV] of that linear fit.",
        "local_kl_anchor/ref_profile_slot_count": "Number of neutrino slot samples entering the epoch −1 ref-profile fit (rank-0 concatenation of all shards).",
        "reference_reward_kl/active": "1 when dgpo.reference_reward_kl LUT is nonempty and substitutes the pT-slot local KL anchor for this training run.",
        "reference_reward_kl/missing_key_fraction": "Fraction of minibatch rows with no LUT hit when LUT substitution is active (missing rows use weight 1.0).",
        "reference_reward_kl/ref_reward_mean": "Batch mean frozen r_ref (epoch−1 ComponentNormalizedTruthDistanceReward, mean over baseline K chains).",
        "reference_reward_kl/ref_reward_min": "Batch min frozen r_ref among valid-masked lookups.",
        "reference_reward_kl/ref_reward_max": "Batch max frozen r_ref.",
        "reference_reward_kl/weight_mean": "Batch mean LUT weight; gaussian mode uses base_weight + weight_scale * exp(-(abs(r_ref)/sigma)^2).",
        "reference_reward_kl/weight_min": "Batch min LUT weight.",
        "reference_reward_kl/weight_max": "Batch max LUT weight.",
        "reference_reward_kl/lut_fill": "(epoch−1 diagnostics) LUT unique key count.",
        "reference_reward_kl/ref_reward_mean_lut": "Global LUT mean frozen r_ref (rank‑0 histogram over entire merged map).",
        "reference_reward_kl/ref_reward_min_lut": "Global LUT min r_ref.",
        "reference_reward_kl/ref_reward_max_lut": "Global LUT max r_ref.",
        "reference_reward_kl/weight_mean_lut": "Global LUT mean weight.",
        "reference_reward_kl/weight_min_lut": "Global LUT min weight.",
        "reference_reward_kl/weight_max_lut": "Global LUT max weight.",
        "reward/w_mass/mean": "Mean raw W-mass reward r_W (truth-mean/std normalized) before applying reward_config.w_mass.weight.",
        "reward/w_mass/std": "Population std of r_W over valid (candidate × event) pairs in the batch.",
        "reward/w_mass/weighted_mean": "reward_config.w_mass.weight × reward/w_mass/mean.",
        "reward/w_mass/best_of_k": "Mean over valid events of max_k r_W.",
        "reward/w_mass/last_place": "Mean over valid events of min_k r_W.",
        "physics/w_mass/plus_mean": "Mean reconstructed m_W+ from candidates (GeV), valid finite entries only.",
        "physics/w_mass/plus_std": "Std of reconstructed m_W+ over valid finite candidate-event pairs.",
        "physics/w_mass/minus_mean": "Mean reconstructed m_W− (GeV).",
        "physics/w_mass/minus_std": "Std of reconstructed m_W−.",
        "physics/w_mass/truth_mean": "Batch truth-W mean used as normalization target (GeV).",
        "physics/w_mass/truth_std": "Batch truth-W std used as normalization scale (GeV).",
        "reward/w_projection/mean": "Mean raw W–pT projection reward (PDG m_W shell) before reward_config.w_projection.weight.",
        "reward/w_projection/std": "Population std of the projection reward over valid (candidate × event) pairs.",
        "reward/w_projection/weighted_mean": "reward_config.w_projection.weight × reward/w_projection/mean.",
        "reward/w_projection/best_of_k": "Mean over valid events of max_k projection reward.",
        "reward/w_projection/last_place": "Mean over valid events of min_k projection reward.",
        "physics/w_projection/valid_fraction": "Fraction of (candidate × event) pairs with valid W-mass projection on both ν slots.",
        "physics/w_projection/logpt_delta_abs_mean": "Mean |Δ log1p(pT)| over both ν slots where both projections are valid.",
        "physics/w_projection/logpt_delta_abs_p90": "90th percentile of |Δ log1p(pT)| (pooled over valid slots).",
        "diagnostics/reward_hacking/all/px/reward_mean": "Mean px reward contribution (ν1+ν2, negative normalized squared error) over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/py/reward_mean": "Mean py reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/pz/reward_mean": "Mean pz reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/best/px/reward_mean": "Mean px reward contribution after selecting the combined-reward argmax candidate per valid event.",
        "diagnostics/reward_hacking/best/py/reward_mean": "Mean py reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/reward_mean": "Mean pz reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/all/px/delta_mean": "Signed mean px residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_mean": "Signed mean py residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_mean": "Signed mean pz residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/px/delta_abs_mean": "Mean absolute px residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_abs_mean": "Mean absolute py residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_abs_mean": "Mean absolute pz residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/px/delta_mean": "Signed mean px residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_mean": "Signed mean py residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_mean": "Signed mean pz residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/px/delta_abs_mean": "Mean absolute px residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_abs_mean": "Mean absolute py residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_abs_mean": "Mean absolute pz residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pt/delta_abs_mean": "Mean absolute pT residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/pt/delta_mean": "Signed mean pT residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pt/delta_abs_mean": "Mean absolute pT residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/eta/delta_mean": "Signed mean η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/eta/delta_abs_mean": "Mean absolute η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/eta/delta_mean": "Signed mean η residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/eta/delta_abs_mean": "Mean absolute η residual on reward-best candidates.",
        "diagnostics/reward_hacking/all/phi/delta_mean": "Signed mean wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/phi/delta_mean": "Signed mean wrapped φ residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/phi/delta_abs_mean": "Mean absolute wrapped φ residual on reward-best candidates.",
        "diagnostics/reward_hacking/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all valid rollout candidates and both ν slots. Negative values indicate pT shrink.",
        "diagnostics/reward_hacking/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/rel_pt/mean": "Mean pT_pred / pT_truth - 1 after selecting the combined-reward argmax candidate per valid event. Compare to all/rel_pt/mean to spot reward-driven pT shrink.",
        "diagnostics/reward_hacking/best/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) on reward-best candidates.",
        "diagnostics/reward_hacking/dist/rel_pt": "Matplotlib density overlay of pT_pred / pT_truth - 1 for all rollout candidates vs reward-best candidates (wandb.Image).",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_mean": "Signed mean pT residual pred−truth in GeV after selecting, per event, the candidate with smallest |ΔpT_nu1| + |ΔpT_nu2|. This is a truth oracle for candidate-support diagnosis, not a deployable selector.",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV for the pT-oracle-best candidate.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt": "Profile plot by truth-pT bin: top panel compares mean delta pT = pT_pred - pT_truth for all rollout candidates, reward-best candidates, and pT-oracle-best candidates; bottom panel shows reward-best minus all and pT-oracle minus all. Gray bars show truth event-slot counts per bin, not K-times all-candidate counts. If pT-oracle fixes high-pT bins, support exists and ranking/reward is the bottleneck; if pT-oracle remains low, generator support is insufficient.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta": "Profile plot by truth-eta bin, with the same all/reward-best/eta-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi": "Profile plot by truth-phi bin using wrapped phi residuals, with the same all/reward-best/phi-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px": "Profile plot by truth-px bin, comparing mean px residual for all rollout, reward-best, and px-oracle candidates.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py": "Profile plot by truth-py bin, comparing mean py residual for all rollout, reward-best, and py-oracle candidates.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz": "Profile plot by truth-pz bin, comparing mean pz residual for all rollout, reward-best, and pz-oracle candidates.",
        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK": "Profile plot by truth-pT bin comparing first-10-candidate selection against full-K selection on the same rollout pool. Curves show reward-best first 10, reward-best full K, pT-oracle first 10, and pT-oracle full K; bottom panel shows full-K minus first-10 gains.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt_accumulated": "Same as diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt, but concatenates raw diagnostic samples over dgpo.diagnostic_profile_accumulate_steps train batches before plotting. Use this for more stable high-pT tail statistics.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta_accumulated": "Accumulated truth-bin eta residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi_accumulated": "Accumulated truth-bin phi residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px_accumulated": "Accumulated truth-bin px residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py_accumulated": "Accumulated truth-bin py residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz_accumulated": "Accumulated truth-bin pz residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reference_bias/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all rollout candidates and the first two ν slots. With dgpo.update_ema_rollout=false, this reflects the initial/reference policy bias.",
        "diagnostics/reference_bias/all/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_mean": "Signed mean η residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_abs_mean": "Mean absolute η residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_mean": "Signed mean wrapped φ residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all rollout candidates and the first two ν slots. Negative values indicate pT shrink. Freeze rollout updates to isolate initial/reference bias.",
        "diagnostics/reference_bias/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/dist/rel_pt": "Matplotlib density plot of pT_pred / pT_truth - 1 for rollout candidates (wandb.Image). With frozen rollout EMA, this is the initial/reference model's pT bias distribution.",
        "diagnostics/reference_bias/profile/pt_delta_vs_truth_pt": "Profile plot: x-axis truth pT bin [GeV], y-axis mean delta pT = pT_pred - pT_truth [GeV] over all rollout candidates and the first two ν slots. Negative high-pT bins indicate tail shrink / dynamic-range compression.",
        "diagnostics/reweight/enabled": "1 when an RL-side event reweight is active this train step: dgpo.phase_space_reweight.enabled (if true, used) else dgpo.event_pt_reweight.enabled. Changes DGPO loss aggregation only, not reward ranking.",
        "diagnostics/reweight/threshold_pt": "Truth event-pT threshold [GeV] used by event_pt_reweight when phase_space_reweight is disabled; event_pt=max(pT_truth_nu1,pT_truth_nu2) before the threshold comparator.",
        "diagnostics/reweight/high_weight": "Raw high-pT event weight for event_pt_reweight (before batch mean normalization in the loss).",
        "diagnostics/reweight/high_fraction": "Fraction of valid events in this train batch with max truth neutrino pT >= threshold_pt (event_pt_reweight only).",
        "diagnostics/reweight/event_weight_raw_mean": "Mean raw per-event weight over valid events before batch-mean normalization inside the loss (phase_space or pT-threshold reweight).",
        "diagnostics/reweight/event_weight_raw_min": "Min raw per-event weight over valid events (phase_space_reweight only).",
        "diagnostics/reweight/event_weight_raw_max": "Max raw per-event weight over valid events (phase_space_reweight only).",
        "diagnostics/reweight/event_weight_norm_mean": "Mean valid-event multiplier after batch-mean normalization and max_weight capping (phase_space_reweight only).",
        "diagnostics/reweight/event_weight_norm_min": "Min valid-event multiplier after batch-mean normalization and max_weight capping (phase_space_reweight only).",
        "diagnostics/reweight/event_weight_norm_max": "Max valid-event multiplier after batch-mean normalization and max_weight capping; should be <= dgpo.phase_space_reweight.max_weight up to numerical tolerance.",
        "diagnostics/reweight/event_truth_pt_mean": "Mean max truth neutrino pT over valid events in this train batch (event_pt_reweight only).",
        "diagnostics/reweight/empty_or_low_count_fraction": "Fraction of valid truth neutrino slots whose histogram bin count is below min_count before flooring (phase_space_reweight only).",
        "diagnostics/reweight/valid_fraction": "Fraction of batch events passing get_event_valid_mask during reweight computation.",
        # --- train/grad (optimizer step; max over inner sub-steps when num_inner_epochs*num_train_timesteps>1) ---
        "train/grad/global_norm_pre_clip": "Total L2 norm of trainable gradients before clip_grad_norm_ (max over sub-steps in the batch). Compare to dgpo.grad_clip_norm in run config.",
        "train/grad/clip_active": "1.0 if any sub-step had pre-clip norm > dgpo.grad_clip_norm (clipping applied); else 0.0.",
        # --- parameter (scalars, every step; extend with more keys later) ---
        "parameter/w_e/mean": "Mean per-event gate w_e = sigmoid(M_e) in [0,1].",
        "parameter/w_e/std": "Std of w_e across events in the batch (population std).",
        "parameter/w_e/min": "Min w_e in the batch.",
        "parameter/w_e/max": "Max w_e in the batch.",
        "parameter/kl_weight/mean": "Mean reward-conditioned KL multiplier over candidate-event pairs when dgpo.reward_kl_weight.enabled=true.",
        "parameter/kl_weight/min": "Minimum reward-conditioned KL multiplier in the batch.",
        "parameter/kl_weight/max": "Maximum reward-conditioned KL multiplier in the batch.",
        # --- val (epoch-end) ---
        "val/reward/mean": "Mean combined reward over all valid (candidate × event) pairs. Used for top-k checkpoint selection.",
        "val/reward/median": "Global median of per-event reward across valid events (with val_K=1: single prediction per event; with val_K>1: best-of-K per event). Epoch x-axis.",
        "val/reward/p10": "10th percentile of per-event reward (val_K=1: single pred; val_K>1: best-of-K per event).",
        "val/reward/p30": "30th percentile.",
        "val/reward/p70": "70th percentile.",
        "val/reward/p90": "90th percentile.",
        "val/winrate": "Fraction of valid events where one current-policy sample beats one reference-policy sample on truth L2 distance. NaN if validation_compute_winrate=false.",
        "val_diagnostics/profile/pt_delta_vs_truth_pt": "Validation profile plot by truth-pT bin: selected-candidate mean delta pT = pT_pred - pT_truth, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/eta_delta_vs_truth_eta": "Validation profile plot by truth-eta bin: selected-candidate mean eta residual, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/pt/delta_mean": "Global validation mean pT residual, pT_pred - pT_truth, over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/pt/slope": "Linear fit slope of the validation binned mean delta-pT profile versus truth pT.",
        "val_diagnostics/profile/pt/zero_delta_truth": "Truth pT value where the fitted validation mean delta-pT profile crosses zero.",
        "val_diagnostics/profile/eta/delta_mean": "Global validation mean eta residual over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/eta/slope": "Linear fit slope of the validation binned mean delta-eta profile versus truth eta.",
        "val_diagnostics/profile/eta/zero_delta_truth": "Truth eta value where the fitted validation mean delta-eta profile crosses zero.",
        "val_diagnostics/profile/pt_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean pT residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean eta residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation pT-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation eta-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-pT zero-crossing where mean delta pT is zero.",
        "val_diagnostics/profile/eta_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-eta zero-crossing where mean delta eta is zero.",
        "val_diagnostics/profile/pt_zero_delta_vs_slope": "History plot with x-axis fitted pT-profile slope and y-axis fitted truth-pT zero-crossing.",
        "val_diagnostics/profile/eta_zero_delta_vs_slope": "History plot with x-axis fitted eta-profile slope and y-axis fitted truth-eta zero-crossing.",
        "val/response/reward_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event reward and y-axis current event reward, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val/response/pt_delta_mean_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event mean delta pT and y-axis current event mean delta pT, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val_neutrino/pt": "1D density overlay: truth vs current-policy vs frozen-reference prediction for pT [GeV] (original scale, expm1 of log1p(pT)) (wandb.Image); x-axis **epoch**. Current-policy histogram uses the same per-event candidate index rule as train_dist/* (truth-L2 argmin when rule_based mode is truth_distance).",
        "val_neutrino/eta": "Same three-way overlay for η; same candidate selection as val_neutrino/pt.",
        "val_neutrino/phi": "Same three-way overlay for φ [rad]; same candidate selection as val_neutrino/pt.",
        # --- train_dist (epoch end, accumulated over all training batches; own wandb panel) ---
        "train_dist/pt": "1D density overlay: truth vs best-of-K training prediction for pT [GeV] (original scale), accumulated over all training batches in the epoch (wandb.Image). x-axis **epoch**. With ``reward_config.rule_based.mode: truth_distance`` (and rule_based enabled), \"best\" = min truth L2 among K (same geometry as TruthDistanceReward); otherwise argmax of combined reward.",
        "train_dist/eta": "Same overlay for η (training); same candidate selection as train_dist/pt.",
        "train_dist/phi": "Same overlay for φ [rad] (training); same candidate selection as train_dist/pt.",
        "train_dist_k1/pt": "1D density overlay: truth vs candidate-0 training rollout prediction for pT [GeV], accumulated over all training batches in the epoch. This is a K=1 / single-sample proxy on the train rollout pool, separate from reward-best train_dist/*.",
        "train_dist_k1/eta": "Same overlay for η using candidate 0 as the train K=1 proxy.",
        "train_dist_k1/phi": "Same overlay for φ using candidate 0 as the train K=1 proxy.",
    }


def _dgpo_wandb_hyperparameter_definitions() -> dict[str, str]:
    """Explicit definitions for DGPO hyperparameters (visible in W&B Config).

    Explains what each parameter in the ``dgpo:`` and ``reward_config:`` sections does.
    """
    return {
        # --- dgpo: core RL hyperparameters ---
        "dgpo.beta": "beta_dgpo: Temperature parameter that scales the event-level gate logit M_e. Higher beta makes the gate more sensitive to the advantage-weighted velocity gap. M_e = (beta / K) * sum_over_candidates(advantage * Delta). Typical range: 0.1 to 1.0. Current value controls how aggressively events are up/down-weighted in the loss.",
        "dgpo.beta_kl": "Base weight on the KL anchor when beta_kl_schedule is disabled or as fallback. When beta_kl_schedule.enabled is true, piecewise-linear values override this along the schedule axis. loss_total = loss_main + beta_kl_effective * mean(MSE(current_velocity, reference_velocity)). Set to 0 to disable (unless schedule provides positive weights). Typical range: 0.0 to 0.1.",
        "dgpo.beta_kl_schedule": "Optional piecewise-linear schedule for the KL anchor weight. Use axis: global_step (recommended; resume-safe via checkpoint global_step) or epoch. When enabled: false, dgpo.beta_kl is used as a constant.",
        "dgpo.reward_kl_weight": "Optional reward-conditioned KL weighting. When enabled, each event maps its mean reward over K candidates to one KL multiplier between min_weight and max_weight: high-reward events get stronger reference anchoring, low-reward events get weaker KL.",
        "dgpo.grad_clip_norm": "Global L2 gradient clip for AdamW (torch.nn.utils.clip_grad_norm_). Compare train/grad/global_norm_pre_clip to this value.",
        "dgpo.advantage_positive_only": "If true: after computing per-event advantages (see dgpo.advantage_transform.mode), clamp negative values to 0. This means only better-than-average candidates contribute to the loss (no downweighting from worse candidates). If false: use the full advantages (both positive and negative). True = more conservative updates; false = stronger preference signal.",
        "dgpo.advantage_transform.mode": "How to map rewards (K, B) to advantages before dgpo.advantage_positive_only and adv_clip_max. 'zscore': per-event z-score over K (default). 'centered': subtract per-event K-candidate mean without dividing by group std. 'softmax_centered': advantages = K * softmax(reward / temperature, dim=0) - 1 (sums to zero per event; lower temperature = sharper focus on best candidates).",
        "dgpo.advantage_transform.temperature": "Positive scalar T for softmax_centered: logits = reward / T. Ignored when mode is zscore.",
        "dgpo.K": "Number of DDIM candidate samples generated per event **during training** (rollout + DGPO). Each event gets K neutrino reconstructions, and the reward function ranks them. Larger K = more candidates to choose from (better oracle performance) but slower generation. Typical values: 4-16.",
        "dgpo.validation_K": "Candidates per event during **validation** only (independent of training K). Default **1**: one current-policy DDIM sample per event; with validation_compute_winrate, one additional ref-policy DDIM per event for winrate. Does not advance the training global_step or train-panel x-axis.",
        "dgpo.num_ddim_steps": "Number of DDIM denoising steps (T_sample) used for online candidate generation during training. More steps = higher-quality samples but slower. Typical values: 20-100.",
        "dgpo.shared_noise": "If true: all K candidates for the same event use the same diffusion timestep t and noise sample eps (only the DDIM chain index varies). If false: each candidate gets independent t and eps. True = ablation control (reduces variance); false = more diverse candidates. Recommended: true.",
        "dgpo.accumulate_train_timesteps": "If true (reference-style): for each train batch, backprop (loss / M) for each of M = num_inner_epochs * num_train_timesteps policy-evaluation sub-steps, then one clipped AdamW update. Averages W&B loss diagnostics over sub-steps. If false (legacy): one AdamW update per sub-step, which can be much stronger on the same cached rollout. Defaults to false when the key is omitted.",
        "dgpo.use_ema_for_rollout": "If true: Phase 1 uses the fast rollout EMA shadow (decay ramps per dgpo.ema_rollout_*); validation uses the slow save EMA. Phase 2 policy eval always uses trainable weights. If false: trainable weights for generation and val.",
        "dgpo.update_ema_rollout": "If false: keep the fast rollout EMA frozen at the loaded initial policy. Use temporarily with diagnostics/reference_bias/* to check whether the starting/reference model already has kinematic bias.",
        "dgpo.event_pt_reweight": "Optional RL-side ablation that upweights high-truth-pT events in the DGPO loss aggregation only. It uses event_pt=max(pT_truth_nu1,pT_truth_nu2), applies low_weight/high_weight by threshold, and normalizes weights by batch mean to keep loss scale comparable. Ignored when dgpo.phase_space_reweight.enabled is true.",
        "dgpo.phase_space_reweight": "Truth-phase-space inverse-occupancy reweight for the DGPO loss only. Loads histogram counts from plot_training_truth_phase_space.py (training_truth_phase_space_histograms.npz); per slot weight ~ 1/max(count,min_count)^count_power, then mean/max/product across ν slots. max_weight caps the post-batch-mean event multiplier entering the DGPO loss.",
        "dgpo.diagnostic_profile_accumulate_steps": "Number of train batches to concatenate before logging accumulated diagnostics/reward_hacking/profile/*_delta_vs_truth_* images. Larger values stabilize sparse bins but update the W&B images less often.",
        "dgpo.log_every": "Log Python INFO messages (loss, reward, etc.) to the console every N optimizer steps. Does not affect wandb logging frequency (wandb logs every step when enabled). Typical: 1-10.",
        "dgpo.log_reward_dist_every": "Log reward/dist/overlap every N optimizer steps when wandb is enabled. Reward-hacking diagnostics are always logged while wandb is active.",
        "dgpo.validation_max_batches": "If set (e.g. 20): stop validation after this many batches per epoch (faster validation for debugging). If null: run full validation set. Typical: null for real training, 5-20 for smoke tests.",
        "dgpo.validation_compute_winrate": "If true: each validation batch generates one extra reference-policy DDIM sample to compute val/winrate. Adds ~50% validation time. If false: skip winrate (logged as NaN). Typical: false (cheaper validation).",
        "dgpo.validation_log_batches": "If true: log INFO messages for each validation batch (start time, DDIM wall time). Useful for monitoring long validation runs. Typical: true.",
        "dgpo.validation_tqdm_k_chains": "If true: show a tqdm progress bar over the K DDIM chains per validation batch. Typical: true (helps see validation progress).",
        "dgpo.validation_tqdm_ddim": "If true: show a tqdm progress bar for every DDIM step within each chain (very verbose). Typical: false (too much output).",
        "dgpo.local_kl_anchor.enabled": "If true: add slot-wise Gaussian-weighted fixed-reference velocity MSE (||v_θ−v_ref||²) around truth p_T = p0_ref. **Suppressed** when a nonempty ``dgpo.reference_reward_kl`` LUT is active (that path replaces this pT-slot anchor). Otherwise additive to ``beta_kl`` velocity KL, not a standalone replacement for it.",
        "dgpo.local_kl_anchor.baseline_weight": "Minimum slot weight w = baseline + amplitude·exp(−0.5((truth_pt−p0)/σ)²). Tunable floor far from p0_ref.",
        "dgpo.local_kl_anchor.amplitude": "Gaussian amplitude on top of baseline_weight for slots near p0_ref.",
        "dgpo.local_kl_anchor.sigma": "Gaussian width σ [GeV] in truth p_T for the extra local anchor.",
        "dgpo.local_kl_anchor.min_count": "Skip the local KL backward term if the masked neutrino-slot count in the minibatch is below this threshold.",
        "dgpo.local_kl_anchor.p0_ref": "Optional explicit truth-p_T crossing point [GeV]. If null, uses checkpoint dgpo_local_kl_anchor_p0_ref or fits from epoch −1 frozen-reference validation profile when p0_ref_from_epoch_minus_one_val is true.",
        "dgpo.local_kl_anchor.p0_ref_from_epoch_minus_one_val": "If true with p0_ref unset and no checkpoint scalar: freeze p0_ref from epoch −1 ref DDIM profile (same binning style as validation pT residual diagnostics).",
        "dgpo.reference_reward_kl.enabled": "If true: freeze per-event KL multipliers from epoch −1 ref-model training-shard traversal (baseline_K ComponentNormalizedTruthDistanceReward only); merge LUT across ranks and persist under checkpoint_key. Suppresses local_kl_anchor backward when LUT nonempty.",
        "dgpo.reference_reward_kl.baseline_K": "Reference DDIM candidate count used only for LUT construction (typically 10); independent from dgpo.K rollout size.",
        "dgpo.reference_reward_kl.weight_mode": "KL LUT weight shape: gaussian (default) or inverse_power.",
        "dgpo.reference_reward_kl.base_weight": "Gaussian-mode baseline KL multiplier.",
        "dgpo.reference_reward_kl.weight_scale": "Gaussian-mode peak height above base_weight; inverse_power front factor.",
        "dgpo.reference_reward_kl.sigma": "Gaussian-mode peak width in reference-reward units.",
        "dgpo.reference_reward_kl.eps": "Denominator floor for |r_ref| in inverse_power mode.",
        "dgpo.reference_reward_kl.require_event_key": "If true: require global_event_index (when synthetic synthesis is disabled) or raise.",
        "dgpo.reference_reward_kl.synthetic_event_key_if_missing": "If parquet lacks global_event_index, synthesize deterministic keys from visible ``batch['x']`` (quantized patch fingerprint).",
        "dgpo.reference_reward_kl.checkpoint_key": "Top-level Lightning/DGPO checkpoint dict key storing ``event_keys_int64``, rewards, kl weights tensors for resume.",
        # --- reward_config: reward function ---
        "reward_config.rule_based.enabled": "If true: include the truth-distance reward component (negative L2 between predicted and ground-truth neutrino momenta). This is the primary physics-based reward. Typical: true.",
        "reward_config.rule_based.weight": "Weight multiplier for the truth-distance reward. Combined reward sums weighted rule_based, pt_truth, log_pt_truth, w_mass, and w_projection components. Typical: 1.0.",
        "reward_config.w_mass.enabled": "If true: include TT2L W-mass consistency reward normalized by batch truth-W mean and std (additive with other reward sources). Typical: enable after verifying assignment targets exist in parquet.",
        "reward_config.w_mass.weight": "Multiplier on the raw W-mass reward before summing into the combined DGPO reward.",
        "reward_config.w_mass.eps": "Numerical stability added to truth_std in the W-mass normalization denominator.",
        "reward_config.w_projection.enabled": "If true: include TT2L W–pT projection reward toward PDG m_W at fixed neutrino (η, φ) (additive with other sources). Complements w_mass by scoring closeness to the W-mass manifold in log-pT space.",
        "reward_config.w_projection.weight": "Multiplier on the raw projection reward before summing into the combined DGPO reward.",
        "reward_config.w_projection.scale": "Normalization for squared Δ log1p(pT) terms: 'auto' uses invisible_std[0] (same as log_pt_truth), or set a positive float.",
        "reward_config.w_projection.eps": "Stability term: divide by scale**2 + eps.",
        "reward_config.w_projection.min_pt": "Lower physical bound (GeV) on projected neutrino pT (matches calibration utilities).",
        "reward_config.w_projection.max_pt": "Upper physical bound (GeV) on projected neutrino pT.",
    }


def _dgpo_wandb_publish_metric_docs() -> None:
    """Expose metric definitions in the W&B UI (Config, Summary, Notes, Artifact).

    W&B does not show definitions next to each chart; Config + Artifact are the supported surfaces.
    """
    import wandb

    run = wandb.run
    if run is None:
        return
    defs = _dgpo_wandb_metric_definition_map()
    param_defs = _dgpo_wandb_hyperparameter_definitions()
    try:
        wandb.config.update(
            {
                "dgpo_metric_definitions": defs,
                "dgpo_hyperparameter_definitions": param_defs,
                "dgpo_metrics_full_doc_repo_path": (
                    "RL/DGPO_neutrino/diagnostics/metrics_reference.md"
                ),
                "dgpo_dynamic_reward_keys_note": (
                    "Training metrics use W&B step=global_step; val/* and train_dist/* use epoch as x-axis. "
                    "Groups: reward/dist, reward/monitor, train/loss, train/grad, parameter/*, "
                    "components/*, diagnostics/reward_hacking/* (including all/ vs best/), "
                    "diagnostics/reference_bias/*; train_dist/*; val/reward/*, val/winrate, "
                    "val_diagnostics/*, val_neutrino/*."
                ),
            },
            allow_val_change=True,
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.config metric definitions failed: %s", e)
    try:
        run.summary["dgpo_how_to_read_metrics"] = (
            "Config → dgpo_metric_definitions (metrics) + dgpo_hyperparameter_definitions (params). "
            "Artifacts → dgpo-metrics-reference → metrics_reference.md (full doc)."
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.summary metric pointer failed: %s", e)
    try:
        run.notes = (
            "DGPO metric + hyperparameter definitions: open this run's **Config** "
            "(dgpo_metric_definitions, dgpo_hyperparameter_definitions) "
            "or **Artifacts** (dgpo-metrics-reference). "
            "Repo copy: RL/DGPO_neutrino/diagnostics/metrics_reference.md"
        )
    except Exception as e:
        _log.warning("[DGPO] wandb run notes failed: %s", e)
    md_path = Path(__file__).resolve().parent / "diagnostics" / "metrics_reference.md"
    if md_path.is_file():
        try:
            art = wandb.Artifact("dgpo-metrics-reference", type="documentation")
            art.add_file(str(md_path), name="metrics_reference.md")
            run.log_artifact(art)
        except Exception as e:
            _log.warning("[DGPO] wandb artifact for metrics_reference.md failed: %s", e)


def _wandb_is_media_value(v: Any) -> bool:
    """True for ``wandb`` loggable media types (histograms, images, etc.)."""
    mod = getattr(type(v), "__module__", "") or ""
    name = getattr(type(v), "__name__", "")
    return mod.startswith("wandb") and name in (
        "Histogram",
        "Image",
        "Plotly",
        "Video",
        "Html",
    )


def _wandb_train_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize logged keys: known prefixes pass through; bare keys get ``train/``.

    Passes through ``wandb.Histogram`` / ``wandb.Image`` values under ``reward/`` or ``val/``.
    Known prefixes ``train/``, ``val/``, ``reward/``, ``parameter/``, ``components/``,
    and ``diagnostics/`` are logged as-is. ``components/`` is the per-component reward
    breakdown panel; ``diagnostics/`` is for monitoring-only reward-hacking checks.
    Keys starting with ``_`` are internal (e.g. ``_kin_h_*`` histogram arrays) and are skipped.
    """
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if k.endswith("_hist") and isinstance(v, np.ndarray):
            try:
                import wandb

                out[k] = wandb.Histogram(v)
            except Exception:
                continue
        elif k.startswith(("train/", "val/", "reward/", "parameter/", "components/", "diagnostics/", "offset_anchor/", "local_kl_anchor/", "reference_reward_kl/")):
            out[k] = v
        elif _wandb_is_media_value(v):
            out[k] = v
        else:
            out[f"train/{k}"] = v
    return out


def _dgpo_wandb_yaml_section() -> tuple[Any | None, str]:
    """Resolve W&B settings the same way as ``evenet/train.py`` + ``WandbLogger``.

    Prefer ``logger.wandb`` when it defines a project (standard EveNet YAML). Otherwise
    use the top-level ``wandb:`` block (DGPO configs often keep it there alongside
    ``logger:`` for tensorboard-only fields).

    Returns:
        ``(section_dict, source_label)`` where ``source_label`` is ``\"logger.wandb\"``
        or ``\"wandb\"`` for logging.
    """
    gc = global_config._global_config
    logger = gc.get("logger")
    if isinstance(logger, dict):
        nested = logger.get("wandb")
        if isinstance(nested, dict) and nested.get("project") is not None:
            return nested, "logger.wandb"
    top = gc.get("wandb")
    if isinstance(top, dict) and len(top) > 0:
        return top, "wandb"
    return None, ""


def _wandb_sanitize_log_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop non-finite floats so one bad scalar does not invalidate the whole W&B step.

    Preserves ``wandb`` media objects (histograms, images) unchanged.
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        if _wandb_is_media_value(v):
            out[k] = v
            continue
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            out[k] = v
            continue
        if isinstance(v, float):
            if math.isfinite(v):
                out[k] = v
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
        except (TypeError, ValueError):
            pass
    return out


def _wandb_log_step(wandb_mod: Any, payload: dict[str, Any], *, step: int) -> None:
    """``wandb.log`` with finite scalars + pass-through ``wandb`` histogram/image objects."""
    clean = _wandb_sanitize_log_dict(payload)
    if not clean:
        return
    try:
        wandb_mod.log(clean, step=step)
    except Exception as e:
        _log.warning("[DGPO] wandb.log failed at step=%s: %s", step, e)


def _wandb_log_validation(
    wandb_mod: Any,
    val_metrics: dict[str, Any],
    *,
    epoch: int,
) -> None:
    """Log ``val/*`` only, with ``epoch`` as the custom x-axis (see ``wandb.define_metric`` in init).

    Does **not** pass ``step=`` so validation panels advance by **epoch**, not ``global_step``.
    """
    clean = _wandb_sanitize_log_dict(dict(val_metrics))
    if not clean:
        return
    clean["epoch"] = float(epoch)
    try:
        wandb_mod.log(clean)
    except Exception as e:
        _log.warning("[DGPO] wandb.log validation failed at epoch=%s: %s", epoch, e)


def _start_wandb_run(*, disable: bool = False) -> bool:
    """Initialize wandb like Lightning's ``WandbLogger`` in ``evenet/train.py``.

    Uses ``logger.wandb`` when present, else top-level ``wandb``. Applies
    ``Settings(start_method=\"thread\")`` for compatibility with Ray Train workers
    (forked subprocesses + threads do not mix well with the default ``fork`` method).
    """
    if disable:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        _log.info("[DGPO] WANDB_DISABLED set; skipping wandb.")
        return False
    try:
        import wandb
    except ImportError:
        _log.warning("[DGPO] wandb not installed; skipping experiment logging.")
        return False
    wb, wb_source = _dgpo_wandb_yaml_section()
    if wb is None:
        _log.info(
            "[DGPO] No wandb settings (add top-level ``wandb:`` or ``logger.wandb:``); "
            "skipping wandb.init."
        )
        return False
    project = wb.get("project")
    if project is None:
        _log.warning("[DGPO] wandb section missing ``project``; skipping wandb.init.")
        return False
    run_name = wb.get("run_name") or wb.get("name")
    tags = wb.get("tags") or []
    if not isinstance(tags, list):
        tags = list(tags)
    run_id = wb.get("id")
    init_kw: dict[str, Any] = {
        "project": str(project),
        "entity": wb.get("entity"),
        "name": run_name,
        "tags": tags,
        "id": run_id,
        "config": global_config.to_logger(),
    }
    try:
        init_kw["settings"] = wandb.Settings(start_method="thread")
    except Exception:
        pass
    wandb.init(**init_kw)
    _log.info(
        "[DGPO] wandb.init project=%s name=%s (config_key=%s)",
        project,
        run_name,
        wb_source or "unknown",
    )
    try:
        # Train metrics use ``wandb.log(..., step=global_step)``. Validation and training-distribution
        # metrics use ``epoch`` as the x-axis so DDIM work does not align to optimizer-step indices.
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val_diagnostics/*", step_metric="epoch")
        wandb.define_metric("val_neutrino/*", step_metric="epoch")
        wandb.define_metric("train_dist/*", step_metric="epoch")
        wandb.define_metric("offset_anchor/*", step_metric="epoch")
        wandb.define_metric("local_kl_anchor/*", step_metric="epoch")
    except Exception as e:
        _log.warning("[DGPO] wandb.define_metric(val/*) failed (val may share step with train): %s", e)
    _dgpo_wandb_publish_metric_docs()
    return True


def _finish_wandb_run(active: bool) -> None:
    if not active:
        return
    try:
        import wandb

        wandb.finish()
    except Exception as e:
        _log.warning("[DGPO] wandb.finish() failed: %s", e)


def _dgpo_save_last_ckpt(
    model: torch.nn.Module,
    ema_save: Any | None,
    optimizer: torch.optim.Optimizer,
    ref_model: torch.nn.Module,
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
    dgpo_offset_anchor_mu_ref: float | None = None,
    dgpo_offset_anchor_mu_ref_xyz: Tensor | None = None,
    dgpo_local_kl_anchor_p0_ref: float | None = None,
    dgpo_rrkl_ckpt_blob: dict[str, Tensor] | None = None,
    dgpo_rrkl_ckpt_key: str | None = None,
    dgpo_offset_anchor_dual_pt: float | None = None,
    dgpo_offset_anchor_z_ema_pt: float | None = None,
) -> None:
    """Write ``last.ckpt`` with live trainable weights in ``state_dict`` plus optional ``ema_state_dict``.

    Refuses to overwrite ``last.ckpt`` when any trainable parameter is non-finite, so a single
    bad batch cannot poison the resume state of a long-running DGPO job.
    """
    save_dir = global_config.options.Training.get("model_checkpoint_save_path", None)
    if not save_dir:
        _log.debug("[DGPO] model_checkpoint_save_path unset; skipping checkpoint save.")
        return
    core_for_check = _unwrap_core_evenet(model)
    bad = [n for n, p in core_for_check.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        _log.warning(
            "[DGPO] last.ckpt save SKIPPED at epoch=%s step=%s: %s non-finite trainable params "
            "(first: %s). Existing last.ckpt is preserved.",
            last_completed_epoch, global_step, len(bad), bad[:3],
        )
        return
    path = Path(str(save_dir)).expanduser().resolve() / "last.ckpt"
    save_lightning_compatible_checkpoint(
        path,
        model,
        ema_save,
        global_config,
        last_completed_epoch=last_completed_epoch,
        dgpo_next_epoch=dgpo_next_epoch,
        global_step=global_step,
        optimizer=optimizer,
        ref_model=ref_model,
        dgpo_offset_anchor_mu_ref=dgpo_offset_anchor_mu_ref,
        dgpo_offset_anchor_mu_ref_xyz=dgpo_offset_anchor_mu_ref_xyz,
        dgpo_local_kl_anchor_p0_ref=dgpo_local_kl_anchor_p0_ref,
        dgpo_rrkl_ckpt_blob=dgpo_rrkl_ckpt_blob,
        dgpo_rrkl_ckpt_key=dgpo_rrkl_ckpt_key,
        dgpo_offset_anchor_dual_pt=dgpo_offset_anchor_dual_pt,
        dgpo_offset_anchor_z_ema_pt=dgpo_offset_anchor_z_ema_pt,
    )


class _DgpoCheckpointTopK:
    """Keep the best ``val/reward/mean`` checkpoints (higher is better) under ``model_checkpoint_save_path``."""

    def __init__(self, save_dir: Path, top_k: int) -> None:
        self._save_dir = save_dir
        self._top_k = max(0, int(top_k))
        # (val_reward_mean, path_str) smallest reward first for easy pop
        self._worst_heap: list[tuple[float, str]] = []

    def maybe_save(
        self,
        *,
        val_reward_mean: float,
        last_completed_epoch: int,
        dgpo_next_epoch: int,
        global_step: int,
        model: torch.nn.Module,
        ema_save: Any | None,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module,
        dgpo_offset_anchor_mu_ref: float | None = None,
        dgpo_offset_anchor_mu_ref_xyz: Tensor | None = None,
        dgpo_local_kl_anchor_p0_ref: float | None = None,
        dgpo_rrkl_ckpt_blob: dict[str, Tensor] | None = None,
        dgpo_rrkl_ckpt_key: str | None = None,
        dgpo_offset_anchor_dual_pt: float | None = None,
        dgpo_offset_anchor_z_ema_pt: float | None = None,
    ) -> None:
        if self._top_k <= 0:
            return
        if not math.isfinite(val_reward_mean):
            _log.warning("[DGPO] val/reward/mean is non-finite; skipping top-k checkpoint.")
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"dgpo-top-val_reward_mean={val_reward_mean:.6f}-"
            f"next_ep={dgpo_next_epoch}-step={global_step}.ckpt"
        )
        path = self._save_dir / fname
        save_lightning_compatible_checkpoint(
            path,
            model,
            ema_save,
            global_config,
            last_completed_epoch=last_completed_epoch,
            dgpo_next_epoch=dgpo_next_epoch,
            global_step=global_step,
            optimizer=optimizer,
            ref_model=ref_model,
            dgpo_offset_anchor_mu_ref=dgpo_offset_anchor_mu_ref,
            dgpo_offset_anchor_mu_ref_xyz=dgpo_offset_anchor_mu_ref_xyz,
            dgpo_local_kl_anchor_p0_ref=dgpo_local_kl_anchor_p0_ref,
            dgpo_rrkl_ckpt_blob=dgpo_rrkl_ckpt_blob,
            dgpo_rrkl_ckpt_key=dgpo_rrkl_ckpt_key,
            dgpo_offset_anchor_dual_pt=dgpo_offset_anchor_dual_pt,
            dgpo_offset_anchor_z_ema_pt=dgpo_offset_anchor_z_ema_pt,
        )

        heapq.heappush(self._worst_heap, (val_reward_mean, str(path)))

        # Keep ``best.ckpt`` symlink pointing to the highest val/reward/mean seen so far.
        best_score, best_path_str = max(self._worst_heap, key=lambda x: x[0])
        best_link = self._save_dir / "best.ckpt"
        try:
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(Path(best_path_str).name)
            _log.info(
                "[DGPO] best.ckpt → %s (val/reward/mean=%.6f)",
                Path(best_path_str).name,
                best_score,
            )
        except OSError as e:
            _log.warning("[DGPO] Could not update best.ckpt symlink: %s", e)

        while len(self._worst_heap) > self._top_k:
            _worst_score, worst_path = heapq.heappop(self._worst_heap)
            wp = Path(worst_path)
            if wp.is_file():
                try:
                    wp.unlink()
                    _log.info(
                        "[DGPO] Removed checkpoint outside top-%s: %s (val/reward/mean=%.6f)",
                        self._top_k,
                        wp.name,
                        _worst_score,
                    )
                except OSError as e:
                    _log.warning("[DGPO] Failed to remove old checkpoint %s: %s", wp, e)


_VAL_KIN_NUM_BINS = 50


@torch.no_grad()
def _val_pred_truth_kin_flat(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``pt`` (GeV, original scale), ``eta``, ``phi`` for selected-candidate pred vs truth (same slots).

    The first invisible feature is ``log1p(pT)``; this function inverts that via ``expm1`` so the
    returned ``ppt`` / ``tpt`` arrays are in GeV (original physics scale), not log space.
    """
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if cartesian:
        t = batch_d["x_invisible_cartesian"]
        plp, pe, pp = cartesian_to_log_pt_eta_phi(pred[..., 0], pred[..., 1], pred[..., 2])
        tlp, te, tp = cartesian_to_log_pt_eta_phi(t[..., 0], t[..., 1], t[..., 2])
    else:
        t = batch_d["x_invisible"]
        plp, pe, pp = pred[..., 0], pred[..., 1], pred[..., 2]
        tlp, te, tp = t[..., 0], t[..., 1], t[..., 2]
    m = (mask > 0).reshape(B, N_nu)
    # Invert log1p to recover pT in GeV (original physics scale).
    ppt = np.expm1(plp[m].detach().float().cpu().numpy())
    pe = pe[m].detach().float().cpu().numpy()
    pp = pp[m].detach().float().cpu().numpy()
    tpt = np.expm1(tlp[m].detach().float().cpu().numpy())
    te = te[m].detach().float().cpu().numpy()
    tp = tp[m].detach().float().cpu().numpy()
    return ppt, pe, pp, tpt, te, tp


def _val_overlay_kin_figure(
    counts_truth: np.ndarray,
    counts_pred: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    *,
    pred_label: str = "Pred (val)",
    counts_ref: np.ndarray | None = None,
    ref_label: str = "Ref (frozen)",
    xlabel: str = "Value",
) -> Any:
    """1D density overlay (truth vs current-policy prediction, optionally also reference policy), EveNet-style, as ``wandb.Image``.

    Args:
        counts_truth: Histogram counts for truth.
        counts_pred: Histogram counts for current-policy prediction.
        bin_edges: Bin edges array (length = len(counts_truth) + 1).
        title: Figure title.
        pred_label: Legend label for the current-policy series.
        counts_ref: Optional histogram counts for the frozen reference policy.
        ref_label: Legend label for the reference series.
        xlabel: x-axis label.
    """
    import wandb

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = float(bin_edges[1] - bin_edges[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    nt = np.sum(counts_truth) * w + 1e-12
    npred = np.sum(counts_pred) * w + 1e-12
    ax.plot(
        centers,
        counts_truth / nt,
        label="Truth",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax.plot(
        centers,
        counts_pred / npred,
        label=pred_label,
        linewidth=2.0,
        marker="s",
        markersize=4,
    )
    if counts_ref is not None:
        nref = np.sum(counts_ref) * w + 1e-12
        ax.plot(
            centers,
            counts_ref / nref,
            label=ref_label,
            linewidth=2.0,
            marker="^",
            markersize=4,
            linestyle="--",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def _val_selected_delta_arrays(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Selected-candidate validation residual arrays for profile and response plots."""
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        slot_mask = xm.squeeze(-1).to(device=device) > 0
    else:
        slot_mask = xm.to(device=device) > 0
    slot_mask = slot_mask.reshape(B, N_nu)
    event_valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(B) > 0
    valid_slots = slot_mask & event_valid.unsqueeze(-1)

    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if cartesian:
        truth = batch_d["x_invisible_cartesian"].to(device=device, dtype=dtype)
        plp, pred_eta, _pred_phi = cartesian_to_log_pt_eta_phi(
            pred[..., 0], pred[..., 1], pred[..., 2]
        )
        tlp, truth_eta, _truth_phi = cartesian_to_log_pt_eta_phi(
            truth[..., 0], truth[..., 1], truth[..., 2]
        )
        pred_xyz = pred[:, :2, :3].contiguous()
        truth_xyz = truth[:, :2, :3].contiguous()
    else:
        truth = batch_d["x_invisible"].to(device=device, dtype=dtype)
        plp, pred_eta = pred[..., 0], pred[..., 1]
        pred_phi = pred[..., 2]
        tlp, truth_eta = truth[..., 0], truth[..., 1]
        truth_phi = truth[..., 2]
        pred_xyz = log_pt_eta_phi_to_cartesian(
            plp.clamp(-10.0, 10.0), pred_eta, pred_phi
        )[:, :2, :].contiguous()
        truth_xyz = log_pt_eta_phi_to_cartesian(
            tlp.clamp(-10.0, 10.0), truth_eta, truth_phi
        )[:, :2, :].contiguous()

    pred_pt = torch.expm1(plp.clamp(-10.0, 10.0))
    truth_pt = torch.expm1(tlp.clamp(-10.0, 10.0))
    delta_pt = pred_pt - truth_pt
    delta_eta = pred_eta - truth_eta
    delta_xyz = pred_xyz - truth_xyz

    slot_count = valid_slots.sum(dim=-1)
    valid_events_with_slots = slot_count > 0
    pt_delta_event_mean = (
        (delta_pt * valid_slots.to(delta_pt.dtype)).sum(dim=-1)
        / slot_count.clamp(min=1).to(delta_pt.dtype)
    )

    return {
        "pt_truth": truth_pt[valid_slots].detach().float().cpu().numpy(),
        "pt_delta": delta_pt[valid_slots].detach().float().cpu().numpy(),
        "px_delta": delta_xyz[..., 0][valid_slots].detach().float().cpu().numpy(),
        "py_delta": delta_xyz[..., 1][valid_slots].detach().float().cpu().numpy(),
        "pz_delta": delta_xyz[..., 2][valid_slots].detach().float().cpu().numpy(),
        "eta_truth": truth_eta[valid_slots].detach().float().cpu().numpy(),
        "eta_delta": delta_eta[valid_slots].detach().float().cpu().numpy(),
        "pt_delta_event_mean": pt_delta_event_mean[valid_events_with_slots]
        .detach()
        .float()
        .cpu()
        .numpy(),
    }


def _concat_np_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty numpy chunks into one float64 vector."""
    parts = [np.asarray(x, dtype=np.float64).reshape(-1) for x in chunks if x.size > 0]
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float64)


def _gather_val_array_dict(
    local_arrays: dict[str, np.ndarray],
    *,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    """Gather per-rank validation arrays to rank 0 and concatenate matching keys."""
    if world_size <= 1:
        return {
            k: np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in local_arrays.items()
        }
    if rank == 0:
        gathered: list[Any] = [None] * world_size
        dist.gather_object(local_arrays, object_gather_list=gathered, dst=0)
        keys = set(local_arrays)
        for part in gathered:
            if isinstance(part, dict):
                keys.update(part)
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            chunks = [
                np.asarray(part.get(key, np.array([], dtype=np.float64)), dtype=np.float64)
                .reshape(-1)
                for part in gathered
                if isinstance(part, dict)
            ]
            merged[key] = _concat_np_chunks(chunks)
        return merged
    dist.gather_object(local_arrays, dst=0)
    return {}


def _profile_fit_metrics(
    profile_name: str,
    truth_value: np.ndarray,
    delta_value: np.ndarray,
) -> tuple[float, float]:
    """Fit binned mean residual with ``delta_mean = slope * truth + intercept``."""
    bin_edges = _profile_bin_edges(profile_name, [truth_value])
    centers, means, _errors, counts = _binned_delta_profile(
        truth_value, delta_value, bin_edges=bin_edges
    )
    keep = np.isfinite(centers) & np.isfinite(means) & (counts > 0)
    if int(np.sum(keep)) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(centers[keep], means[keep], deg=1)
    slope_f = float(slope)
    intercept_f = float(intercept)
    zero = (
        float(-intercept_f / slope_f)
        if math.isfinite(slope_f) and abs(slope_f) > 1e-12
        else float("nan")
    )
    return slope_f, zero


def _validation_delta_profile_figure(
    truth_current: np.ndarray,
    delta_current: np.ndarray,
    *,
    profile_name: str,
    title: str,
    truth_initial: np.ndarray | None = None,
    delta_initial: np.ndarray | None = None,
) -> Any:
    """Validation profile plot of mean selected-candidate residual versus truth value."""
    import wandb

    initial_arrays = []
    if truth_initial is not None and delta_initial is not None:
        initial_arrays = [truth_initial]
    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(profile_name, [truth_current] + initial_arrays)
    centers, mean_cur, err_cur, counts = _binned_delta_profile(
        truth_current, delta_current, bin_edges=bin_edges
    )
    mean_init = err_init = None
    if truth_initial is not None and delta_initial is not None:
        _centers, mean_init, err_init, _counts = _binned_delta_profile(
            truth_initial, delta_initial, bin_edges=bin_edges
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if mean_init is not None and err_init is not None:
        keep_init = np.isfinite(mean_init)
        if np.any(keep_init):
            ax.errorbar(
                centers[keep_init],
                mean_init[keep_init],
                yerr=err_init[keep_init],
                fmt="o--",
                linewidth=1.6,
                markersize=4,
                capsize=2,
                color="#7f7f7f",
                label=f"initial validation {display}",
            )
    keep_cur = np.isfinite(mean_cur)
    if np.any(keep_cur):
        ax.errorbar(
            centers[keep_cur],
            mean_cur[keep_cur],
            yerr=err_cur[keep_cur],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            color="#1f77b4",
            label=f"current validation {display}",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_slope_history_figure(
    epochs: list[float],
    slopes: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: epoch number versus fitted validation residual-profile slope."""
    import wandb

    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    ep = np.asarray(epochs, dtype=np.float64)
    sl = np.asarray(slopes, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(sl)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], sl[keep], "o-", linewidth=1.8, markersize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Fitted {display} residual slope")
    ax.set_title(f"Validation {display} residual-profile slope over epochs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_epoch_history_figure(
    epochs: list[float],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    zero_line: bool = True,
) -> Any:
    """History plot with epoch on the x-axis and one validation diagnostic on y."""
    import wandb

    ep = np.asarray(epochs, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(val)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], val[keep], "o-", linewidth=1.8, markersize=4)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_zero_vs_slope_figure(
    slopes: list[float],
    zero_points: list[float],
    epochs: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: fitted slope versus truth value where mean residual crosses zero."""
    import wandb

    x_label, _y_label, display = _profile_axis_labels(profile_name)
    sl = np.asarray(slopes, dtype=np.float64)
    zp = np.asarray(zero_points, dtype=np.float64)
    ep = np.asarray(epochs, dtype=np.float64)
    keep = np.isfinite(sl) & np.isfinite(zp)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if np.any(keep):
        sc = ax.scatter(sl[keep], zp[keep], c=ep[keep], cmap="viridis", s=34)
        ax.plot(sl[keep], zp[keep], "-", linewidth=1.1, alpha=0.55)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Epoch")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(f"Fitted {display} residual slope")
    ax.set_ylabel(f"{x_label} where mean residual = 0")
    ax.set_title(f"Validation {display} zero-crossing versus slope")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _response_matrix_figure(
    initial: np.ndarray,
    current: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Any:
    """2D response matrix comparing event-level initial and current validation values."""
    import wandb

    x = np.asarray(initial, dtype=np.float64).reshape(-1)
    y = np.asarray(current, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)

    stacked = np.concatenate([x, y], axis=0)
    lo, hi = [float(v) for v in np.nanpercentile(stacked, [1.0, 99.0])]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
        lo, hi = center - 1.0, center + 1.0
    pad = max(0.05 * (hi - lo), 1e-6)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hist = ax.hist2d(x, y, bins=50, range=[[lo, hi], [lo, hi]], cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Events")
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_save: Any | None,
    val_loader: Any | None,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    val_K: int,
    num_ddim_steps: int,
    use_ema_for_rollout: bool,
    device: torch.device,
    dtype: torch.dtype,
    cartesian: bool,
    compute_winrate: bool,
    epoch: int | None = None,
    est_total_batches: int | None = None,
    val_log_batches: bool = True,
    val_tqdm_k_chains: bool = True,
    val_tqdm_ddim: bool = False,
    max_batches: int | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
    collect_dense_pt_residual_baseline: bool = False,
    collect_dense_xyz_residual_baseline: bool = False,
    offset_anchor_dense_pt_min: float = 20.0,
    offset_anchor_dense_pt_max: float = 80.0,
    collect_ref_pt_residual_for_local_kl: bool = False,
    local_kl_profile_min_total_slots: int = 50,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """One pass over the validation dataset; all tensors under ``torch.no_grad()``.

    Validation uses ``val_K`` candidates per event (default **1** in config), independent of training ``K``.
    Typical setup: **one** current-policy DDIM sample per event; if ``compute_winrate``, **one**
    reference-policy DDIM sample per event (no extra multi-candidate rollout for val).

    Under Ray Train, every rank receives its own validation shard via
    ``ray.train.get_dataset_shard("validation")``.  Each rank iterates its own shard;
    a cross-rank ``all_reduce(MIN)`` on the "has-more" flag keeps DDP collectives
    synchronized (loop terminates as soon as **any** rank exhausts its shard).
    Accumulators are all-reduced at the end so every rank returns identical aggregates.

    When ``use_ema_for_rollout`` is True and ``ema_save`` is set, candidate generation uses the
    save-EMA shadow; checkpoint ``state_dict`` still stores live trainable weights for resume.

    When ``val_log_batches`` is True, logs start/end timing per validation batch so long DDIM runs are
    not silent. ``val_tqdm_k_chains`` wraps the ``val_K`` sequential DDIM calls in a tqdm bar;
    ``val_tqdm_ddim`` adds an inner bar for each DDIM chain (verbose).

    If ``max_batches`` is set, stop after that many local batches per rank (partial val).

    When ``collect_dense_xyz_residual_baseline`` is True alongside the scalar baseline flag, each rank
    additionally aggregates global means of ``Δp_x``, ``Δp_y``, ``Δp_z`` under the same dense truth-``p_T``
    interval (using ``px_delta``/``py_delta``/``pz_delta`` from validation selection). Reduced means
    populate ``offset_anchor/baseline_mu_ref_{px,py,pz}`` for freezing ``mu_ref_xyz``.

    When ``collect_dense_pt_residual_baseline`` is True, each rank aggregates the global-open-interval
    mean ``pred_pt - truth_pt`` over masked entries where ``truth pt`` falls between
    ``offset_anchor_dense_pt_min`` and ``offset_anchor_dense_pt_max`` using the validation candidate
    selection arrays ``pt_truth``, ``pt_delta`` (broadcast over both ν slots). The reduced mean and
    total mask count populate ``offset_anchor/baseline_mu_ref`` for freezing ``dgpo.offset_anchor.mu_ref``.

    When ``collect_ref_pt_residual_for_local_kl`` is True, each rank collects slot-level ``(truth_pT,
    Δp_T)`` from the frozen-reference ``K=1`` DDIM rollout. After gathering, rank 0 fits the same
    binned linear profile as validation diagnostics and broadcasts ``p0_ref`` so every rank logs
    ``local_kl_anchor/p0_ref_fit`` (and fit diagnostics) for freezing ``dgpo.local_kl_anchor`` training.
    """
    is_rank0 = rank == 0
    model.eval()
    ref_model.eval()
    freeze_reference_model(ref_model)

    core = _unwrap_core_evenet(model)

    ep_str = f"epoch {epoch}" if epoch is not None else "val"
    if is_rank0:
        est_msg = (
            f"≈{est_total_batches} batches (ceil(val_events/batch_size))"
            if est_total_batches is not None and est_total_batches > 0
            else "unknown batch count (streaming)"
        )
        if max_batches is not None and max_batches > 0:
            est_msg = f"cap {max_batches} batches (partial val); full pass would be {est_msg}"
        if val_log_batches:
            _log.info("[DGPO] val: starting pass (%s, %s, %s GPUs).", ep_str, est_msg, world_size)

    t_epoch = time.perf_counter()
    n_val_batches = 0
    sum_r = 0.0
    cnt_r = 0
    sum_win = 0.0
    cnt_win = 0

    local_reward_chunks: list[np.ndarray] = []
    local_reward_event_chunks: list[np.ndarray] = []
    local_pt_delta_event_mean_chunks: list[np.ndarray] = []
    local_profile_chunks: dict[str, list[np.ndarray]] = {
        "pt_truth": [],
        "pt_delta": [],
        "eta_truth": [],
        "eta_delta": [],
    }
    local_lka_ref_pt_truth: list[np.ndarray] = []
    local_lka_ref_pt_delta: list[np.ndarray] = []
    # pT in GeV (original physics scale, after expm1 inversion of log1p).
    bin_pt_edges = np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
    bin_eta_edges = np.linspace(-4.0, 4.0, _VAL_KIN_NUM_BINS + 1)
    bin_phi_edges = np.linspace(-3.2, 3.2, _VAL_KIN_NUM_BINS + 1)
    # _p = current policy, _t = truth, _r = frozen reference policy
    h_pt_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_pt_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_pt_r = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_e_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_e_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_e_r = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_p_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_p_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
    h_p_r = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)

    oa_dense_sum = 0.0
    oa_dense_cnt = 0.0
    oa_dense_sum_x = 0.0
    oa_dense_sum_y = 0.0
    oa_dense_sum_z = 0.0

    val_iter = iter(val_loader) if val_loader is not None else None
    if val_iter is None:
        if is_rank0 and val_log_batches:
            _log.warning("[DGPO] val: rank=%s has no val shard; returning empty metrics.", rank)
        return {
            "val/reward/mean": float("nan"),
            "val/reward/median": float("nan"),
            "val/reward/p10": float("nan"),
            "val/reward/p30": float("nan"),
            "val/reward/p70": float("nan"),
            "val/reward/p90": float("nan"),
            "val/winrate": float("nan"),
        }

    batch_round = 0
    while True:
        batch_cpu, has_more = _next_batch_synced(
            val_iter, world_size=world_size, device=device
        )
        if not has_more or batch_cpu is None:
            break

        batch_round += 1
        n_val_batches += 1
        batch_d = batch_to_device(batch_cpu, device)
        B = int(batch_d["x"].shape[0])

        if is_rank0 and val_log_batches:
            batch_suffix = (
                f" {batch_round}/{est_total_batches}"
                if est_total_batches is not None and est_total_batches > 0
                else f" {batch_round}"
            )
            _log.info(
                "[DGPO] val round%s: B=%s×%s GPUs | current policy: val_K=%s DDIM chains (%s steps each)...",
                batch_suffix,
                B,
                world_size,
                val_K,
                num_ddim_steps,
            )

        buf: dict[str, Tensor] = {}
        if use_ema_for_rollout:
            if ema_save is not None:
                buf = _save_trainable_weights(model)
                ema_save.copy_to(core)
        t_gen = time.perf_counter()
        chain_desc = f"val DDIM ({ep_str})"
        try:
            candidates = generate_neutrino_candidates(
                core,
                batch_d,
                sampler,
                K=val_K,
                num_ddim_steps=num_ddim_steps,
                device=device,
                tqdm_k_chains=val_tqdm_k_chains and is_rank0,
                use_tqdm_ddim=val_tqdm_ddim and is_rank0,
                chain_progress_desc=chain_desc,
            )
        finally:
            if use_ema_for_rollout and buf and ema_save is not None:
                _restore_trainable_weights(model, buf)

        t_after_cur = time.perf_counter()
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: generation done in %.1fs.",
                batch_suffix,
                t_after_cur - t_gen,
            )

        rewards, _ = reward_agg.compute(candidates, batch_d)

        valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(-1)
        m_sel = valid > 0
        vb = m_sel

        k_sel = _kin_hist_candidate_indices_per_event(
            rewards, candidates, batch_d, cartesian=cartesian
        )

        if bool(vb.any().item()):
            if val_K == 1:
                r_per_event = rewards[0, vb]
            else:
                r_per_event = rewards[:, vb].max(dim=0).values
            sum_r += float(r_per_event.sum().detach().cpu().item())
            cnt_r += int(vb.sum().item())
            r_per_event_np = r_per_event.detach().float().cpu().numpy()
            local_reward_chunks.append(r_per_event_np)
            local_reward_event_chunks.append(r_per_event_np)
        selected_delta_arrays = _val_selected_delta_arrays(
            candidates,
            batch_d,
            k_sel,
            cartesian=cartesian,
            device=device,
            dtype=dtype,
        )
        for key in local_profile_chunks:
            local_profile_chunks[key].append(selected_delta_arrays[key])
        local_pt_delta_event_mean_chunks.append(
            selected_delta_arrays["pt_delta_event_mean"]
        )
        if collect_dense_pt_residual_baseline:
            pt_t_flat = np.asarray(
                selected_delta_arrays["pt_truth"], dtype=np.float64
            ).reshape(-1)
            pt_d_flat = np.asarray(
                selected_delta_arrays["pt_delta"], dtype=np.float64
            ).reshape(-1)
            if pt_t_flat.size == pt_d_flat.size:
                m = (
                    np.isfinite(pt_t_flat)
                    & np.isfinite(pt_d_flat)
                    & (pt_t_flat > float(offset_anchor_dense_pt_min))
                    & (pt_t_flat < float(offset_anchor_dense_pt_max))
                )
                if np.any(m):
                    oa_dense_sum += float(np.sum(pt_d_flat[m]))
                    oa_dense_cnt += float(np.sum(m))
                    if collect_dense_xyz_residual_baseline:
                        ox = np.asarray(
                            selected_delta_arrays["px_delta"], dtype=np.float64
                        ).reshape(-1)
                        oy = np.asarray(
                            selected_delta_arrays["py_delta"], dtype=np.float64
                        ).reshape(-1)
                        oz = np.asarray(
                            selected_delta_arrays["pz_delta"], dtype=np.float64
                        ).reshape(-1)
                        if ox.size == pt_t_flat.size == oy.size == oz.size:
                            oa_dense_sum_x += float(np.sum(ox[m]))
                            oa_dense_sum_y += float(np.sum(oy[m]))
                            oa_dense_sum_z += float(np.sum(oz[m]))
        ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
            candidates,
            batch_d,
            k_sel,
            cartesian=cartesian,
            device=device,
        )
        h_pt_p += np.histogram(ppt, bins=bin_pt_edges)[0]
        h_pt_t += np.histogram(tpt, bins=bin_pt_edges)[0]
        h_e_p += np.histogram(peta, bins=bin_eta_edges)[0]
        h_e_t += np.histogram(teta, bins=bin_eta_edges)[0]
        h_p_p += np.histogram(pphi, bins=bin_phi_edges)[0]
        h_p_t += np.histogram(tphi, bins=bin_phi_edges)[0]

        # Always run one ref-model DDIM pass (K=1) for kinematic overlay plots.
        # When compute_winrate is also True, the same r_one is reused for the winrate computation
        # at no additional cost.
        t_ref = time.perf_counter()
        if is_rank0 and val_log_batches:
            _log.info("[DGPO] val round%s: reference policy DDIM (K=1)...", batch_suffix)
        r_one = generate_neutrino_candidates(
            ref_model,
            batch_d,
            sampler,
            K=1,
            num_ddim_steps=num_ddim_steps,
            device=device,
            tqdm_k_chains=False,
            use_tqdm_ddim=val_tqdm_ddim and is_rank0,
            chain_progress_desc=f"val ref DDIM ({ep_str})",
        )
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: ref DDIM done in %.1fs.",
                batch_suffix,
                time.perf_counter() - t_ref,
            )

        # Collect reference kinematics for kinematic overlay plots.
        k_sel_ref = torch.zeros(B, dtype=torch.long, device=device)
        rpt, reta, rphi, _, _, _ = _val_pred_truth_kin_flat(
            r_one, batch_d, k_sel_ref, cartesian=cartesian, device=device
        )
        h_pt_r += np.histogram(rpt, bins=bin_pt_edges)[0]
        h_e_r += np.histogram(reta, bins=bin_eta_edges)[0]
        h_p_r += np.histogram(rphi, bins=bin_phi_edges)[0]

        if collect_ref_pt_residual_for_local_kl:
            ref_prof_arrays = _val_selected_delta_arrays(
                r_one,
                batch_d,
                k_sel_ref,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            local_lka_ref_pt_truth.append(ref_prof_arrays["pt_truth"])
            local_lka_ref_pt_delta.append(ref_prof_arrays["pt_delta"])

        if compute_winrate:
            d_cur = compute_truth_l2_distances_kb(
                candidates, batch_d, cartesian=cartesian, mask=None
            )[0]
            d_ref = compute_truth_l2_distances_kb(
                r_one, batch_d, cartesian=cartesian, mask=None
            )[0]
            wins = (d_cur < d_ref) & m_sel & torch.isfinite(d_cur) & torch.isfinite(d_ref)
            w = wins.float().sum()
            nw = m_sel.sum()
            sum_win += float(w.detach().cpu().item())
            cnt_win += int(nw.detach().cpu().item())

        if max_batches is not None and max_batches > 0 and batch_round >= max_batches:
            if is_rank0 and val_log_batches:
                _log.info(
                    "[DGPO] val: stopping early at validation_max_batches=%s (partial val metrics).",
                    max_batches,
                )
            break

    if is_rank0 and val_log_batches:
        _log.info(
            "[DGPO] val: finished %s batches (%s rounds × %s GPUs) in %.1fs.",
            n_val_batches * world_size,
            n_val_batches,
            world_size,
            time.perf_counter() - t_epoch,
        )

    # All-reduce accumulators so every rank has the global totals.
    if world_size > 1:
        acc = torch.tensor(
            [sum_r, cnt_r, sum_win, cnt_win],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        a = acc.cpu().tolist()
        sum_r, cnt_r = a[0], int(a[1])
        sum_win, cnt_win = a[2], int(a[3])

    hist_stack = np.stack([h_pt_p, h_pt_t, h_pt_r, h_e_p, h_e_t, h_e_r, h_p_p, h_p_t, h_p_r])
    t_hist = torch.from_numpy(hist_stack).to(device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t_hist, op=dist.ReduceOp.SUM)
    hist_merged = t_hist.cpu().numpy()
    h_pt_p, h_pt_t, h_pt_r, h_e_p, h_e_t, h_e_r, h_p_p, h_p_t, h_p_r = [hist_merged[i] for i in range(9)]

    baseline_mu_dense = float("nan")
    baseline_dense_mask_total = 0.0
    if collect_dense_pt_residual_baseline:
        acc_oa = torch.tensor(
            [oa_dense_sum, oa_dense_cnt], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(acc_oa, op=dist.ReduceOp.SUM)
        denom_o = float(acc_oa[1].item())
        baseline_dense_mask_total = denom_o
        if denom_o > 0.0:
            baseline_mu_dense = float(acc_oa[0].item() / denom_o)

    baseline_mu_dense_px = float("nan")
    baseline_mu_dense_py = float("nan")
    baseline_mu_dense_pz = float("nan")
    if collect_dense_pt_residual_baseline and collect_dense_xyz_residual_baseline:
        acc_xyz = torch.tensor(
            [oa_dense_sum_x, oa_dense_sum_y, oa_dense_sum_z, oa_dense_cnt],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(acc_xyz, op=dist.ReduceOp.SUM)
        denom_xyz = float(acc_xyz[3].item())
        if denom_xyz > 0.0:
            baseline_mu_dense_px = float(acc_xyz[0].item() / denom_xyz)
            baseline_mu_dense_py = float(acc_xyz[1].item() / denom_xyz)
            baseline_mu_dense_pz = float(acc_xyz[2].item() / denom_xyz)

    p10 = p30 = p50 = p70 = p90 = float("nan")
    if world_size > 1:
        if rank == 0:
            gathered: list[Any] = [None] * world_size
            dist.gather_object(
                local_reward_chunks,
                object_gather_list=gathered,
                dst=0,
            )
            merged_list: list[np.ndarray] = []
            for part in gathered:
                if part:
                    merged_list.extend(part)
            merged_r = (
                np.concatenate(merged_list, axis=0)
                if merged_list
                else np.array([], dtype=np.float64)
            )
            if merged_r.size > 0:
                p10, p30, p50, p70, p90 = [
                    float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
                ]
        else:
            dist.gather_object(local_reward_chunks, dst=0)
        pct_t = torch.tensor(
            [p10, p30, p50, p70, p90], dtype=torch.float64, device=device
        )
        dist.broadcast(pct_t, src=0)
        p10, p30, p50, p70, p90 = [float(x) for x in pct_t.cpu().tolist()]
    else:
        merged_list = local_reward_chunks
        merged_r = (
            np.concatenate(merged_list, axis=0)
            if merged_list
            else np.array([], dtype=np.float64)
        )
        if merged_r.size > 0:
            p10, p30, p50, p70, p90 = [
                float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
            ]

    def _mean(num: float, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    win_metric = _mean(sum_win, cnt_win) if compute_winrate else float("nan")

    local_state = {
        "reward": _concat_np_chunks(local_reward_event_chunks),
        "pt_delta_mean": _concat_np_chunks(local_pt_delta_event_mean_chunks),
        "pt_truth": _concat_np_chunks(local_profile_chunks["pt_truth"]),
        "pt_delta": _concat_np_chunks(local_profile_chunks["pt_delta"]),
        "eta_truth": _concat_np_chunks(local_profile_chunks["eta_truth"]),
        "eta_delta": _concat_np_chunks(local_profile_chunks["eta_delta"]),
    }
    profile_compare_local = {
        "pt_truth": local_state["pt_truth"],
        "pt_delta": local_state["pt_delta"],
        "eta_truth": local_state["eta_truth"],
        "eta_delta": local_state["eta_delta"],
    }
    if initial_state is not None:
        for key in ("pt_truth", "pt_delta", "eta_truth", "eta_delta"):
            profile_compare_local[f"initial_{key}"] = np.asarray(
                initial_state.get(key, np.array([], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
    profile_merged = _gather_val_array_dict(
        profile_compare_local, rank=rank, world_size=world_size
    )

    response_initial_state = initial_state
    if response_initial_state is None and epoch == -1:
        # The baseline pass should still create the W&B image series; later epochs
        # then update the same key with initial-vs-current heatmaps.
        response_initial_state = local_state

    response_merged: dict[str, np.ndarray] = {}
    if response_initial_state is not None:
        init_reward = np.asarray(
            response_initial_state.get("reward", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        init_pt_delta = np.asarray(
            response_initial_state.get("pt_delta_mean", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        n_reward = min(init_reward.size, local_state["reward"].size)
        n_pt = min(init_pt_delta.size, local_state["pt_delta_mean"].size)
        response_merged = _gather_val_array_dict(
            {
                "reward_initial": init_reward[:n_reward],
                "reward_current": local_state["reward"][:n_reward],
                "pt_delta_initial": init_pt_delta[:n_pt],
                "pt_delta_current": local_state["pt_delta_mean"][:n_pt],
            },
            rank=rank,
            world_size=world_size,
        )

    lka_pkg = torch.tensor(
        [float("nan"), float("nan"), float("nan"), float("nan")],
        dtype=torch.float64,
        device=device,
    )
    if collect_ref_pt_residual_for_local_kl:
        local_lka_merged_arrays = {
            "pt_truth": _concat_np_chunks(local_lka_ref_pt_truth),
            "pt_delta": _concat_np_chunks(local_lka_ref_pt_delta),
        }
        merged_lka_slots = _gather_val_array_dict(
            local_lka_merged_arrays, rank=rank, world_size=world_size
        )
        if rank == 0:
            tt = np.asarray(
                merged_lka_slots.get("pt_truth", np.array([], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            dd = np.asarray(
                merged_lka_slots.get("pt_delta", np.array([], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            p0_f, sl_f, ic_f, ns = fit_reference_pt_residual_profile_p0(
                tt,
                dd,
                min_nonempty_bins=2,
                min_total_slots=max(1, int(local_kl_profile_min_total_slots)),
            )
            lka_pkg[0] = float(p0_f)
            lka_pkg[1] = float(sl_f)
            lka_pkg[2] = float(ic_f)
            lka_pkg[3] = float(ns)
        if world_size > 1:
            dist.broadcast(lka_pkg, src=0)

    out: dict[str, Any] = {
        "val/reward/mean": _mean(sum_r, cnt_r),
        "val/reward/median": p50,
        "val/reward/p10": p10,
        "val/reward/p30": p30,
        "val/reward/p70": p70,
        "val/reward/p90": p90,
        "val/winrate": win_metric,
        "_val_initial_state": local_state,
    }
    if collect_dense_pt_residual_baseline:
        out["offset_anchor/baseline_mu_ref"] = baseline_mu_dense
        out["offset_anchor/baseline_mask_count_total"] = baseline_dense_mask_total
    if collect_dense_pt_residual_baseline and collect_dense_xyz_residual_baseline:
        out["offset_anchor/baseline_mu_ref_px"] = baseline_mu_dense_px
        out["offset_anchor/baseline_mu_ref_py"] = baseline_mu_dense_py
        out["offset_anchor/baseline_mu_ref_pz"] = baseline_mu_dense_pz
    if collect_ref_pt_residual_for_local_kl:
        out["local_kl_anchor/p0_ref_fit"] = float(lka_pkg[0].item())
        out["local_kl_anchor/fit_slope"] = float(lka_pkg[1].item())
        out["local_kl_anchor/fit_intercept"] = float(lka_pkg[2].item())
        out["local_kl_anchor/ref_profile_slot_count"] = float(lka_pkg[3].item())

    _val_kin_suffix = f"val: {val_K} candidate{'s' if val_K != 1 else ''} vs truth"
    _pred_lbl = "Pred (val)" if val_K == 1 else f"Pred (val, best-of-{val_K})"
    if is_rank0:
        for profile_name in ("pt", "eta"):
            truth_key = f"{profile_name}_truth"
            delta_key = f"{profile_name}_delta"
            truth_arr = profile_merged.get(truth_key, np.array([], dtype=np.float64))
            delta_arr = profile_merged.get(delta_key, np.array([], dtype=np.float64))
            slope, zero_point = _profile_fit_metrics(
                profile_name, truth_arr, delta_arr
            )
            finite_delta = delta_arr[np.isfinite(delta_arr)]
            delta_mean = float(np.mean(finite_delta)) if finite_delta.size > 0 else float("nan")
            out[f"val_diagnostics/profile/{profile_name}/delta_mean"] = delta_mean
            out[f"val_diagnostics/profile/{profile_name}/slope"] = slope
            out[f"val_diagnostics/profile/{profile_name}/zero_delta_truth"] = zero_point
            out[
                f"val_diagnostics/profile/{profile_name}_delta_vs_truth_{profile_name}"
            ] = _validation_delta_profile_figure(
                truth_arr,
                delta_arr,
                profile_name=profile_name,
                title=f"Validation {profile_name} residual vs truth {profile_name}",
                truth_initial=profile_merged.get(f"initial_{truth_key}"),
                delta_initial=profile_merged.get(f"initial_{delta_key}"),
            )
        if response_initial_state is not None:
            out["val/response/reward_initial_vs_current"] = (
                _response_matrix_figure(
                    response_merged.get("reward_initial", np.array([], dtype=np.float64)),
                    response_merged.get("reward_current", np.array([], dtype=np.float64)),
                    xlabel="Initial validation reward",
                    ylabel="Current validation reward",
                    title="Validation 2D correlation: initial reward vs current reward",
                )
            )
            out["val/response/pt_delta_mean_initial_vs_current"] = (
                _response_matrix_figure(
                    response_merged.get("pt_delta_initial", np.array([], dtype=np.float64)),
                    response_merged.get("pt_delta_current", np.array([], dtype=np.float64)),
                    xlabel="Initial event mean delta pT [GeV]",
                    ylabel="Current event mean delta pT [GeV]",
                    title="Validation 2D correlation: initial vs current event mean delta pT",
                )
            )
        out["val_neutrino/pt"] = _val_overlay_kin_figure(
            h_pt_t,
            h_pt_p,
            bin_pt_edges,
            f"Neutrino pT [GeV] ({_val_kin_suffix})",
            pred_label=_pred_lbl,
            counts_ref=h_pt_r,
            xlabel="pT [GeV]",
        )
        out["val_neutrino/eta"] = _val_overlay_kin_figure(
            h_e_t,
            h_e_p,
            bin_eta_edges,
            f"Neutrino η ({_val_kin_suffix})",
            pred_label=_pred_lbl,
            counts_ref=h_e_r,
            xlabel="η",
        )
        out["val_neutrino/phi"] = _val_overlay_kin_figure(
            h_p_t,
            h_p_p,
            bin_phi_edges,
            f"Neutrino φ ({_val_kin_suffix})",
            pred_label=_pred_lbl,
            counts_ref=h_p_r,
            xlabel="φ [rad]",
        )
    return out


def dgpo_train_loop(cfg: dict[str, Any]) -> None:
    """Per-worker DGPO training loop launched by ``ray.train.torch.TorchTrainer``.

    Each Ray Train worker runs this function in its own process.  Ray Train
    initialises the torch distributed process group and gives each worker its
    own per-rank Ray Data shard via ``ray.train.get_dataset_shard``.  This
    function:

    1. Resolves rank / world-size / device from the Ray Train context.
    2. Pulls its own train (and validation) shard.
    3. Builds the EveNet backbone, EMA shadows, reference policy, and DGPO optimizer.
    4. Iterates the DGPO algorithm in lock-step across ranks (``_next_batch_synced``
       all-reduces the ``has-more`` flag so collectives stay aligned).
    5. Runs validation, checkpoint top-K save, and ``last`` checkpoint on rank 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = ray.train.get_context()
    rank = int(ctx.get_world_rank())
    world_size = int(ctx.get_world_size())
    local_rank = int(ctx.get_local_rank())
    is_rank0 = rank == 0
    device = ray.train.torch.get_device()

    # Earliest possible visibility check: emitted by every worker before any I/O or model build,
    # so the user can see the actual world_size immediately and confirm multi-node DDP is live.
    _log.info(
        "[DGPO][boot] rank=%s/%s local_rank=%s host=%s device=%s",
        rank, world_size, local_rank, os.uname().nodename, device,
    )

    config_path = Path(str(cfg["config_path"])).resolve()
    max_steps: int | None = cfg.get("max_steps")
    wandb_flag = bool(cfg.get("wandb", True))
    total_events = int(cfg["total_events"])
    val_events_in = cfg.get("val_events", 0)
    val_events: int | None = int(val_events_in) if val_events_in else None

    global_config.load_yaml(config_path)
    platform_info = global_config.platform

    wandb_active = _start_wandb_run(disable=not wandb_flag) if is_rank0 else False

    # Per-rank Ray Data shards.  Ray Train assigns each worker a disjoint subset.
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("validation") if val_events else None

    batch_size = int(platform_info.batch_size)
    prefetch = int(getattr(platform_info, "prefetch_batches", 1))
    train_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
        # Mirror EveNet pretraining: enable in-shard random shuffling.
        "local_shuffle_buffer_size": batch_size * prefetch,
    }
    # Keep validation order deterministic so pre-DGPO and later response matrices
    # compare the same validation rows in the same per-rank order.
    val_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
    }

    bundle = load_evenet_model_for_dgpo(
        None,
        device,
        checkpoint_path=None,
        config=global_config,
        config_yaml_path=config_path,
    )
    eve_net = bundle.model

    ckpt_dict = None
    if bundle.checkpoint_path is not None:
        ckpt_dict = torch.load(
            str(bundle.checkpoint_path), map_location=device, weights_only=False
        )

    mu_ref_from_ckpt: float | None = None
    mu_ref_xyz_from_ckpt: tuple[float, float, float] | None = None
    if ckpt_dict is not None:
        raw_xyz_ck = ckpt_dict.get("dgpo_offset_anchor_mu_ref_xyz")
        if raw_xyz_ck is not None:
            if isinstance(raw_xyz_ck, torch.Tensor):
                flat_xyz = raw_xyz_ck.detach().cpu().reshape(-1).tolist()
            else:
                flat_xyz = [float(x) for x in list(raw_xyz_ck)]
            if (
                len(flat_xyz) >= 3
                and all(math.isfinite(float(flat_xyz[j])) for j in range(3))
            ):
                mu_ref_xyz_from_ckpt = (
                    float(flat_xyz[0]),
                    float(flat_xyz[1]),
                    float(flat_xyz[2]),
                )
        raw_mr = ckpt_dict.get("dgpo_offset_anchor_mu_ref")
        if raw_mr is not None:
            mrv = float(raw_mr)
            if math.isfinite(mrv):
                mu_ref_from_ckpt = mrv

    p0_ref_from_ckpt: float | None = None
    if ckpt_dict is not None:
        raw_p0_ck = ckpt_dict.get("dgpo_local_kl_anchor_p0_ref")
        if raw_p0_ck is not None:
            p0v = float(raw_p0_ck)
            if math.isfinite(p0v):
                p0_ref_from_ckpt = p0v

    dual_pt_from_ckpt: float | None = None
    z_ema_pt_from_ckpt: float | None = None
    if ckpt_dict is not None:
        raw_dual = ckpt_dict.get("dgpo_offset_anchor_dual_pt")
        if raw_dual is not None:
            if isinstance(raw_dual, torch.Tensor):
                dpv = float(raw_dual.detach().cpu().reshape(-1)[0])
            else:
                dpv = float(raw_dual)
            if math.isfinite(dpv):
                dual_pt_from_ckpt = dpv
        raw_zem = ckpt_dict.get("dgpo_offset_anchor_z_ema_pt")
        if raw_zem is not None:
            if isinstance(raw_zem, torch.Tensor):
                zmv = float(raw_zem.detach().cpu().reshape(-1)[0])
            else:
                zmv = float(raw_zem)
            if math.isfinite(zmv):
                z_ema_pt_from_ckpt = zmv

    eve_net.train()
    apply_component_freezes(eve_net, global_config)
    ref_model = make_reference_model(
        eve_net, global_config, bundle.normalization_dict, device, checkpoint=ckpt_dict
    )
    ema_save = make_ema(eve_net, global_config, checkpoint=ckpt_dict, device=device)
    ema_rollout = make_ema_rollout(eve_net, global_config)

    # Wrap the diffusion-vector forward in a thin nn.Module, then let Ray Train's
    # ``prepare_model`` install DDP with the right device + process-group config.
    fw = _DGPODDPForward(eve_net)
    if world_size > 1:
        model = ray.train.torch.prepare_model(
            fw,
            parallel_strategy_kwargs={"find_unused_parameters": True},
        )
    else:
        model = fw

    dtype = next(eve_net.parameters()).dtype
    sampler = DDIMSampler(device=device)
    reward_agg = build_reward_aggregator(
        eve_net, device, normalization_dict=bundle.normalization_dict
    )
    effective_batch = batch_size * world_size
    steps_per_epoch = max(1, math.ceil(total_events / effective_batch))
    train_opt_lr = global_config.options.Training
    warm_up_factor = float(train_opt_lr.get("learning_rate_warm_up_factor", 1.0))
    warmup_steps = max(1, math.ceil(warm_up_factor * steps_per_epoch))
    optimizer = build_optimizer(
        model,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        is_rank0=is_rank0,
    )

    start_epoch, global_step = parse_dgpo_resume_from_checkpoint(ckpt_dict)
    if ckpt_dict is not None and "dgpo_optimizer_state_dict" in ckpt_dict:
        try:
            optimizer.load_state_dict(ckpt_dict["dgpo_optimizer_state_dict"])
            if is_rank0:
                _log.info("[DGPO] Restored optimizer state from checkpoint.")
        except (ValueError, RuntimeError) as ex:
            if is_rank0:
                _log.warning(
                    "[DGPO] Could not load optimizer state (continuing fresh optimizer): %s", ex
                )

    dg = global_config.dgpo
    oa_block = _dgpo_cfg_get(dg, "offset_anchor", None)
    oa_collect_baseline = False
    oa_dense_min, oa_dense_max = 20.0, 80.0
    if oa_block is not None and bool(
        _dgpo_cfg_get(
            oa_block, "baseline_mu_ref_from_epoch_minus_one_val", False
        )
    ):
        oa_collect_baseline = True
        oa_dense_min = float(_dgpo_cfg_get(oa_block, "pt_min", 20.0))
        oa_dense_max = float(_dgpo_cfg_get(oa_block, "pt_max", 80.0))
    oa_collect_xyz_baseline = (
        bool(oa_collect_baseline)
        and oa_block is not None
        and str(_dgpo_cfg_get(oa_block, "mode", "pt_mean") or "pt_mean").strip().lower()
        == "xyz_mean"
    )
    lka_block = _dgpo_cfg_get(dg, "local_kl_anchor", None)
    lka_cfg_resolved_early = resolve_local_kl_anchor_train_config(dg)
    raw_yaml_p0_ref = (
        _dgpo_cfg_get(lka_block, "p0_ref", None) if lka_block is not None else None
    )
    yaml_p0_ref: float | None = None
    if raw_yaml_p0_ref is not None:
        try:
            py = float(raw_yaml_p0_ref)
            if math.isfinite(py):
                yaml_p0_ref = py
        except (TypeError, ValueError):
            yaml_p0_ref = None
    lka_use_epoch_minus_one_p0 = bool(
        _dgpo_cfg_get(lka_block, "p0_ref_from_epoch_minus_one_val", True)
    )
    lka_profile_min_slots = (
        int(_dgpo_cfg_get(lka_block, "min_count", 50))
        if lka_block is not None
        else int(lka_cfg_resolved_early.min_count)
    )
    collect_lka_ref_profile_at_baseline_val = (
        lka_cfg_resolved_early.enabled
        and lka_use_epoch_minus_one_p0
        and p0_ref_from_ckpt is None
        and yaml_p0_ref is None
    )
    _vm_raw = dg.get("validation_max_batches", None)
    val_max_batches: int | None = None
    if _vm_raw is not None:
        val_max_batches = int(_vm_raw)
        if val_max_batches <= 0:
            if is_rank0:
                _log.warning(
                    "[DGPO] validation_max_batches=%s is not positive; running full validation.",
                    _vm_raw,
                )
            val_max_batches = None
    K = int(dg.K)
    val_K = max(1, int(dg.get("validation_K", 1)))
    beta = float(dg.beta)
    advantage_positive_only = bool(dg.get("advantage_positive_only", False))
    _adv_tf = _dgpo_cfg_get(dg, "advantage_transform", None)
    if _adv_tf is None:
        advantage_mode = "zscore"
        advantage_temperature = 1.0
    else:
        advantage_mode = str(_dgpo_cfg_get(_adv_tf, "mode", "zscore")).strip().lower()
        advantage_temperature = float(_dgpo_cfg_get(_adv_tf, "temperature", 1.0))
    num_ddim = int(dg.num_ddim_steps)
    shared_noise = bool(dg.shared_noise)
    use_ema_rollout = bool(dg.use_ema_for_rollout)
    update_ema_rollout = bool(dg.get("update_ema_rollout", True))
    log_every = max(1, int(dg.get("log_every", 1)))
    log_reward_dist_every = max(1, int(dg.get("log_reward_dist_every", 5)))
    diagnostic_profile_accumulate_steps = max(
        1, int(dg.get("diagnostic_profile_accumulate_steps", 1))
    )
    num_inner_epochs = max(1, int(dg.get("num_inner_epochs", 1)))
    num_train_timesteps = max(1, int(dg.get("num_train_timesteps", 1)))
    accumulate_train_timesteps = bool(dg.get("accumulate_train_timesteps", False))
    _adv_raw = dg.get("adv_clip_max", None)
    adv_clip_max_cfg: float | None = float(_adv_raw) if _adv_raw is not None else None
    grad_clip_norm_cfg = float(dg.get("grad_clip_norm", GRAD_CLIP_NORM))
    _ppo_raw = dg.get("ppo_clip_range", None)
    ppo_clip_range_cfg: float | None = float(_ppo_raw) if _ppo_raw is not None else None
    policy_eval_t_min_cfg = float(dg.get("policy_eval_t_min", 0.0))
    policy_eval_t_max_cfg = float(dg.get("policy_eval_t_max", 1.0))

    save_dir_raw = global_config.options.Training.get("model_checkpoint_save_path", None)
    top_k_ckpt = int(global_config.options.Training.get("model_checkpoint_save_top_k", 5))
    ckpt_topk: _DgpoCheckpointTopK | None = None
    if save_dir_raw and is_rank0:
        ckpt_topk = _DgpoCheckpointTopK(
            Path(str(save_dir_raw)).expanduser().resolve(),
            top_k_ckpt,
        )

    epochs = int(global_config.options.Training.epochs)

    rrkl_cfg = resolve_reference_reward_kl_train_config(dg)
    dgpo_reference_reward_kl_store: ReferenceRewardKlStore | None = None
    if rrkl_cfg.enabled and ckpt_dict is not None:
        rr_blob_raw = ckpt_dict.get(rrkl_cfg.checkpoint_key)
        if rr_blob_raw is not None and isinstance(rr_blob_raw, dict):
            try:
                dgpo_reference_reward_kl_store = (
                    ReferenceRewardKlStore.from_checkpoint_payload(rr_blob_raw).with_weight_params(
                        eps=float(rrkl_cfg.eps),
                        weight_scale=float(rrkl_cfg.weight_scale),
                        weight_mode=str(rrkl_cfg.weight_mode),
                        sigma=float(rrkl_cfg.sigma),
                        base_weight=float(rrkl_cfg.base_weight),
                    )
                )
                if is_rank0:
                    _log.info(
                        "[DGPO][reference_reward_kl] Restored LUT from ckpt (%s keys).",
                        len(dgpo_reference_reward_kl_store),
                    )
            except Exception as ex:
                if is_rank0:
                    _log.warning(
                        "[DGPO][reference_reward_kl] Failed to restore from checkpoint (%s); "
                        "will rebuild at epoch −1 if needed.", ex,
                    )

    effective_offset_anchor_mu_ref: float | None = mu_ref_from_ckpt

    effective_offset_anchor_mu_ref_xyz: tuple[float, float, float] | None = (
        mu_ref_xyz_from_ckpt
    )
    effective_local_kl_anchor_p0_ref: float | None = p0_ref_from_ckpt
    if effective_local_kl_anchor_p0_ref is None:
        effective_local_kl_anchor_p0_ref = yaml_p0_ref

    def _local_kl_anchor_p0_ref_for_checkpoint(stored_p0: float | None) -> float | None:
        """Scalar ``p0_ref`` written when YAML ``dgpo.local_kl_anchor.enabled``."""
        resolved_lka = resolve_local_kl_anchor_train_config(global_config.dgpo)
        if not resolved_lka.enabled:
            return None
        if stored_p0 is None or not math.isfinite(float(stored_p0)):
            return None
        return float(stored_p0)

    if start_epoch > 0 or global_step > 0:
        if is_rank0:
            _log.info(
                "[DGPO] Resuming: start_epoch=%s global_step=%s (total epochs in config=%s).",
                start_epoch,
                global_step,
                epochs,
            )
    if start_epoch >= epochs:
        if is_rank0:
            _log.info(
                "[DGPO] start_epoch=%s >= epochs=%s; nothing to train. Check config or checkpoint.",
                start_epoch,
                epochs,
            )
        _finish_wandb_run(wandb_active)
        return

    if is_rank0:
        _log.info(
            "[DGPO] rank=%s/%s device=%s train_events≈%s val_events≈%s batch=%s train_K=%s val_K=%s "
            "ddim=%s inner_epochs=%s train_timesteps=%s accumulate_t=%s steps/epoch≈%s epochs=%s "
            "advantage=%s T=%s",
            rank,
            world_size,
            device,
            total_events,
            val_events if val_events is not None else 0,
            batch_size,
            K,
            val_K,
            num_ddim,
            num_inner_epochs,
            num_train_timesteps,
            accumulate_train_timesteps,
            steps_per_epoch,
            epochs,
            advantage_mode,
            advantage_temperature,
        )
        if use_ema_rollout and not update_ema_rollout:
            _log.info(
                "[DGPO] rollout EMA update is disabled; Phase-1 rollouts stay at "
                "the initial loaded policy for reference-bias diagnostics."
            )

    wandb_mod = None
    if wandb_active:
        import wandb as wandb_mod

    val_baseline_state: dict[str, np.ndarray] | None = None
    val_profile_history: dict[str, dict[str, list[float]]] = {
        name: {"epoch": [], "delta_mean": [], "slope": [], "zero": []}
        for name in ("pt", "eta")
    }

    def _append_validation_history_plots(
        val_metrics: dict[str, Any],
        *,
        epoch_value: int,
    ) -> None:
        if not is_rank0:
            return
        for profile_name, hist in val_profile_history.items():
            delta_mean = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/delta_mean",
                    float("nan"),
                )
            )
            slope = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/slope",
                    float("nan"),
                )
            )
            zero = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/zero_delta_truth",
                    float("nan"),
                )
            )
            if not (math.isfinite(slope) and math.isfinite(zero)):
                continue
            hist["epoch"].append(float(epoch_value))
            hist["delta_mean"].append(delta_mean)
            hist["slope"].append(slope)
            hist["zero"].append(zero)
            if wandb_mod is None:
                continue
            _x_label, y_label, display = _profile_axis_labels(profile_name)
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_delta_mean_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["delta_mean"],
                ylabel=y_label.replace("Mean ", "Global mean "),
                title=f"Validation {display} residual mean over epochs",
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_slope_vs_epoch"
            ] = _validation_slope_history_figure(
                hist["epoch"],
                hist["slope"],
                profile_name=profile_name,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_truth_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["zero"],
                ylabel=f"{_x_label} where mean residual = 0",
                title=f"Validation {display} zero-crossing over epochs",
                zero_line=False,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_vs_slope"
            ] = _validation_zero_vs_slope_figure(
                hist["slope"],
                hist["zero"],
                hist["epoch"],
                profile_name=profile_name,
            )

    profile_accum_suffixes = (
        "truth_all",
        "delta_all",
        "truth_best",
        "delta_best",
        "truth_oracle",
        "delta_oracle",
    )
    profile_accum: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            _diag_profile_raw_key(name, suffix): []
            for suffix in profile_accum_suffixes
        }
        for name in _DIAG_PROFILE_NAMES
    }
    profile_accum_batches: dict[str, int] = {name: 0 for name in _DIAG_PROFILE_NAMES}

    def _append_profile_accum(metrics: dict[str, Any]) -> None:
        if wandb_mod is None:
            return
        for profile_name, key_lists in profile_accum.items():
            have_all = all(
                isinstance(metrics.get(k), np.ndarray) and metrics[k].size > 0
                for k in key_lists
            )
            if not have_all:
                continue
            for k in key_lists:
                key_lists[k].append(metrics[k])
            profile_accum_batches[profile_name] += 1

    def _flush_profile_accum(*, step: int, force: bool = False) -> None:
        nonlocal profile_accum, profile_accum_batches
        if wandb_mod is None:
            return
        payload: dict[str, Any] = {}
        flushed_names: list[str] = []
        for profile_name, key_lists in profile_accum.items():
            batches = profile_accum_batches[profile_name]
            if batches <= 0:
                continue
            if not force and batches < diagnostic_profile_accumulate_steps:
                continue
            merged = {
                k: np.concatenate(v, axis=0) if v else np.array([], dtype=np.float64)
                for k, v in key_lists.items()
            }
            payload[_diag_profile_log_key(profile_name, accumulated=True)] = (
                _delta_selection_profiles_figure(
                    merged[_diag_profile_raw_key(profile_name, "truth_all")],
                    merged[_diag_profile_raw_key(profile_name, "delta_all")],
                    merged[_diag_profile_raw_key(profile_name, "truth_best")],
                    merged[_diag_profile_raw_key(profile_name, "delta_best")],
                    merged[_diag_profile_raw_key(profile_name, "truth_oracle")],
                    merged[_diag_profile_raw_key(profile_name, "delta_oracle")],
                    profile_name=profile_name,
                    title=_diag_profile_title(
                        profile_name,
                        accumulated_batches=batches,
                    )
                )
            )
            flushed_names.append(profile_name)
        if not payload:
            return
        try:
            _wandb_log_step(wandb_mod, payload, step=step)
        finally:
            for profile_name in flushed_names:
                profile_accum[profile_name] = {
                    _diag_profile_raw_key(profile_name, suffix): []
                    for suffix in profile_accum_suffixes
                }
                profile_accum_batches[profile_name] = 0

    def _barrier() -> None:
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

    # Following the EveNet ``train.py`` pattern: no rank-0-only synchronous setup
    # before the training loop.  All ranks proceed straight into ``fit``-style
    # iteration and hit the data pipeline simultaneously, avoiding NCCL barriers
    # that would otherwise busy-wait the GPU while rank 0 does cold-start work.
    ve_initial = int(val_events) if val_events is not None else 0
    if start_epoch == 0 and ve_initial > 0 and val_shard is not None:
        if is_rank0:
            _log.info(
                "[DGPO] val: running pre-DGPO baseline validation (epoch=-1) for response diagnostics."
            )
        val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
        est_val_batches = (
            max(1, math.ceil(ve_initial / effective_batch)) if ve_initial > 0 else None
        )
        initial_val_metrics = run_validation_epoch(
            model,
            ref_model,
            ema_save,
            val_loader,
            sampler,
            reward_agg,
            val_K=val_K,
            num_ddim_steps=num_ddim,
            use_ema_for_rollout=use_ema_rollout,
            device=device,
            dtype=dtype,
            cartesian=_truth_generation_cartesian(),
            compute_winrate=bool(dg.get("validation_compute_winrate", False)),
            epoch=-1,
            est_total_batches=est_val_batches,
            val_log_batches=bool(dg.get("validation_log_batches", True)),
            val_tqdm_k_chains=bool(dg.get("validation_tqdm_k_chains", True)),
            val_tqdm_ddim=bool(dg.get("validation_tqdm_ddim", False)),
            max_batches=val_max_batches,
            initial_state=None,
            collect_dense_pt_residual_baseline=oa_collect_baseline,
            collect_dense_xyz_residual_baseline=oa_collect_xyz_baseline,
            offset_anchor_dense_pt_min=oa_dense_min,
            offset_anchor_dense_pt_max=oa_dense_max,
            collect_ref_pt_residual_for_local_kl=collect_lka_ref_profile_at_baseline_val,
            local_kl_profile_min_total_slots=max(1, int(lka_profile_min_slots)),
            rank=rank,
            world_size=world_size,
        )
        maybe_initial_state = initial_val_metrics.get("_val_initial_state")
        if isinstance(maybe_initial_state, dict):
            val_baseline_state = maybe_initial_state

        baseline_metric = initial_val_metrics.get("offset_anchor/baseline_mu_ref")
        if (
            effective_offset_anchor_mu_ref is None
            and oa_collect_baseline
            and not oa_collect_xyz_baseline
        ):
            if baseline_metric is not None:
                try:
                    bm_flat = float(baseline_metric)
                except (TypeError, ValueError):
                    bm_flat = float("nan")
                if math.isfinite(bm_flat):
                    effective_offset_anchor_mu_ref = bm_flat
                    if is_rank0:
                        cnt_b = initial_val_metrics.get(
                            "offset_anchor/baseline_mask_count_total",
                            float("nan"),
                        )
                        _log.info(
                            "[DGPO][offset_anchor] mu_ref frozen from epoch=-1 val dense band "
                            "mean ΔpT=%.6g (masked_entries=%s, band=(%.6g, %.6g)).",
                            bm_flat,
                            cnt_b,
                            oa_dense_min,
                            oa_dense_max,
                        )

        if effective_offset_anchor_mu_ref_xyz is None and oa_collect_xyz_baseline:
            bx = initial_val_metrics.get("offset_anchor/baseline_mu_ref_px")
            by = initial_val_metrics.get("offset_anchor/baseline_mu_ref_py")
            bz = initial_val_metrics.get("offset_anchor/baseline_mu_ref_pz")
            try:
                tri = (float(bx), float(by), float(bz))
            except (TypeError, ValueError):
                tri = None
            if tri is not None and all(math.isfinite(v) for v in tri):
                effective_offset_anchor_mu_ref_xyz = tri
                if is_rank0:
                    cnt_b = initial_val_metrics.get(
                        "offset_anchor/baseline_mask_count_total",
                        float("nan"),
                    )
                    _log.info(
                        "[DGPO][offset_anchor] mu_ref_xyz frozen from epoch=-1 val dense band "
                        "means Δpx=%.6g Δpy=%.6g Δpz=%.6g (masked_entries=%s, band=(%.6g, %.6g)).",
                        tri[0],
                        tri[1],
                        tri[2],
                        cnt_b,
                        oa_dense_min,
                        oa_dense_max,
                    )
        p0_fit_metric = initial_val_metrics.get("local_kl_anchor/p0_ref_fit")
        if (
            collect_lka_ref_profile_at_baseline_val
            and effective_local_kl_anchor_p0_ref is None
            and p0_fit_metric is not None
        ):
            try:
                p0_fm = float(p0_fit_metric)
            except (TypeError, ValueError):
                p0_fm = float("nan")
            if math.isfinite(p0_fm):
                effective_local_kl_anchor_p0_ref = p0_fm
                if is_rank0:
                    ns_fm = initial_val_metrics.get(
                        "local_kl_anchor/ref_profile_slot_count", float("nan")
                    )
                    _log.info(
                        "[DGPO][local_kl_anchor] p0_ref frozen from epoch=-1 frozen-reference "
                        "validation profile crossing at truth pT≈%.6g (profile_slots=%s).",
                        p0_fm,
                        ns_fm,
                    )

        if is_rank0:
            _append_validation_history_plots(initial_val_metrics, epoch_value=-1)
            _log.info(
                "[DGPO] initial val r_mean=%.6f pt_slope=%.6g pt_zero=%.6g eta_slope=%.6g eta_zero=%.6g",
                initial_val_metrics["val/reward/mean"],
                initial_val_metrics.get("val_diagnostics/profile/pt/slope", float("nan")),
                initial_val_metrics.get(
                    "val_diagnostics/profile/pt/zero_delta_truth", float("nan")
                ),
                initial_val_metrics.get("val_diagnostics/profile/eta/slope", float("nan")),
                initial_val_metrics.get(
                    "val_diagnostics/profile/eta/zero_delta_truth", float("nan")
                ),
            )
            if wandb_mod is not None:
                _wandb_log_validation(wandb_mod, initial_val_metrics, epoch=-1)
        _barrier()
    elif start_epoch > 0 and ve_initial > 0 and is_rank0:
        _log.warning(
            "[DGPO] Response matrices need the pre-DGPO validation baseline; "
            "this run is resuming at start_epoch=%s, so val/response/* will be skipped.",
            start_epoch,
        )

    if rrkl_cfg.enabled:
        rebuild_rr = dgpo_reference_reward_kl_store is None or len(
            dgpo_reference_reward_kl_store
        ) == 0
        if rebuild_rr and start_epoch < epochs:
            if is_rank0:
                _log.info(
                    "[DGPO][reference_reward_kl] building frozen LUT (epoch=-1 train pass, "
                    "K_ref=%s, weight_mode=%s, eps=%.6g, weight_scale=%.6g, "
                    "sigma=%.6g, base_weight=%.6g).",
                    rrkl_cfg.baseline_K,
                    rrkl_cfg.weight_mode,
                    rrkl_cfg.eps,
                    rrkl_cfg.weight_scale,
                    rrkl_cfg.sigma,
                    rrkl_cfg.base_weight,
                )
            dgpo_reference_reward_kl_store = run_reference_reward_kl_training_baseline(
                train_shard=train_shard,
                ref_model=ref_model,
                sampler=sampler,
                normalization_dict=bundle.normalization_dict,
                rrkl_cfg=rrkl_cfg,
                num_ddim_steps=num_ddim,
                batch_size=batch_size,
                prefetch=prefetch,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=dtype,
                is_rank0=is_rank0,
            )
        _barrier()
        if wandb_active and wandb_mod is not None and is_rank0:
            st = dgpo_reference_reward_kl_store
            if st is not None and len(st) > 0:
                rlst = torch.tensor([t[0] for t in st.lut.values()], dtype=torch.float64)
                wlst = torch.tensor([t[1] for t in st.lut.values()], dtype=torch.float64)
                _wandb_log_step(
                    wandb_mod,
                    _wandb_sanitize_log_dict(
                        {
                            "epoch": float(-1),
                            "reference_reward_kl/lut_fill": float(len(st)),
                            "reference_reward_kl/ref_reward_mean_lut": float(rlst.mean()),
                            "reference_reward_kl/ref_reward_min_lut": float(rlst.min()),
                            "reference_reward_kl/ref_reward_max_lut": float(rlst.max()),
                            "reference_reward_kl/weight_mean_lut": float(wlst.mean()),
                            "reference_reward_kl/weight_min_lut": float(wlst.min()),
                            "reference_reward_kl/weight_max_lut": float(wlst.max()),
                        }
                    ),
                    step=int(global_step),
                )

    def _bundle_rrkl_ckpt() -> tuple[dict[str, Tensor] | None, str | None]:
        if not rrkl_cfg.enabled:
            return None, None
        st_r = dgpo_reference_reward_kl_store
        if st_r is None or len(st_r) == 0:
            return None, None
        return st_r.checkpoint_payload(ckpt_key_unused=""), rrkl_cfg.checkpoint_key

    def _oa_ckpt_mu_pair() -> tuple[float | None, Tensor | None]:
        return _dgpo_offset_anchor_checkpoint_fields(
            dg,
            bundle.normalization_dict,
            effective_mu_ref=effective_offset_anchor_mu_ref,
            effective_mu_ref_xyz=effective_offset_anchor_mu_ref_xyz,
        )

    offset_anchor_dual_state: dict[str, float] | None = None
    offset_anchor_dual_init_cfg = resolve_offset_anchor_train_config(
        dg,
        bundle.normalization_dict,
        stored_mu_ref=effective_offset_anchor_mu_ref,
        stored_mu_ref_xyz=effective_offset_anchor_mu_ref_xyz,
    )
    if offset_anchor_dual_init_cfg.dual_control.enabled:
        dp0 = float(dual_pt_from_ckpt) if dual_pt_from_ckpt is not None else 0.0
        zem0 = float(z_ema_pt_from_ckpt) if z_ema_pt_from_ckpt is not None else 0.0
        offset_anchor_dual_state = {
            "dual_pt": float(dp0),
            "z_ema_pt": float(zem0),
        }

    def dual_ckpt_scalars_for_save() -> tuple[float | None, float | None]:
        """Finite dual controller scalars persisted in Lightning-compatible DGPO checkpoints."""
        if offset_anchor_dual_state is None:
            return None, None
        dp = float(offset_anchor_dual_state.get("dual_pt", 0.0))
        zem = float(offset_anchor_dual_state.get("z_ema_pt", 0.0))
        return (
            dp if math.isfinite(dp) else None,
            zem if math.isfinite(zem) else None,
        )

    try:

        for epoch in range(start_epoch, epochs):
            # Each call to ``iter_torch_batches`` produces a fresh streaming generator
            # over this rank's shard.  ``local_shuffle_buffer_size`` (set in train_loader_cfg)
            # provides per-shard random shuffling each epoch.
            train_iter = train_shard.iter_torch_batches(**train_loader_cfg)
            train_it = iter(train_iter)

            # Epoch-level histogram accumulators for training-distribution plots (all batches).
            td_pt_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_pt_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_e_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_e_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_p_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_p_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_pt_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_pt_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_e_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_e_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_p_p = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)
            td_k1_p_t = np.zeros(_VAL_KIN_NUM_BINS, dtype=np.float64)

            stop_epoch = False
            while True:
                if max_steps is not None and global_step >= max_steps:
                    last_done = epoch - 1 if epoch > 0 else 0
                    if is_rank0:
                        rr_blob, rr_key = _bundle_rrkl_ckpt()
                        _oa_mr, _oa_mxyz = _oa_ckpt_mu_pair()
                        _dual_p, _dual_z = dual_ckpt_scalars_for_save()
                        _dgpo_save_last_ckpt(
                            model,
                            ema_save,
                            optimizer,
                            ref_model,
                            last_completed_epoch=last_done,
                            dgpo_next_epoch=epoch,
                            global_step=global_step,
                            dgpo_offset_anchor_mu_ref=_oa_mr,
                            dgpo_offset_anchor_mu_ref_xyz=_oa_mxyz,
                            dgpo_local_kl_anchor_p0_ref=_local_kl_anchor_p0_ref_for_checkpoint(
                                effective_local_kl_anchor_p0_ref
                            ),
                            dgpo_rrkl_ckpt_blob=rr_blob,
                            dgpo_rrkl_ckpt_key=rr_key,
                            dgpo_offset_anchor_dual_pt=_dual_p,
                            dgpo_offset_anchor_z_ema_pt=_dual_z,
                        )
                        _log.info("[DGPO] max_steps=%s reached; stopping.", max_steps)
                    _barrier()
                    return

                batch_cpu, has_more = _next_batch_synced(
                    train_it, world_size=world_size, device=device
                )
                if not has_more or batch_cpu is None:
                    stop_epoch = True
                    break

                batch_d = batch_to_device(batch_cpu, device)
                reward_dist_step = wandb_active and (
                    global_step % log_reward_dist_every == 0
                )
                diagnostic_dist_step = wandb_active
                beta_kl_current = _resolve_beta_kl_from_config(
                    dg, global_step=global_step, epoch=epoch
                )
                metrics = train_step(
                    model,
                    ref_model,
                    ema_rollout,
                    ema_save,
                    batch_d,
                    optimizer,
                    sampler,
                    reward_agg,
                    beta=beta,
                    beta_kl=beta_kl_current,
                    advantage_positive_only=advantage_positive_only,
                    advantage_mode=advantage_mode,
                    advantage_temperature=advantage_temperature,
                    K=K,
                    num_ddim_steps=num_ddim,
                    shared_noise=shared_noise,
                    use_ema_for_rollout=use_ema_rollout,
                    update_ema_rollout=update_ema_rollout,
                    global_step=global_step,
                    epoch=epoch,
                    device=device,
                    dtype=dtype,
                    log_reward_dist=reward_dist_step,
                    log_diagnostic_dist=diagnostic_dist_step,
                    num_inner_epochs=num_inner_epochs,
                    num_train_timesteps=num_train_timesteps,
                    adv_clip_max=adv_clip_max_cfg,
                    grad_clip_norm=grad_clip_norm_cfg,
                    ppo_clip_range=ppo_clip_range_cfg,
                    policy_eval_t_min=policy_eval_t_min_cfg,
                    policy_eval_t_max=policy_eval_t_max_cfg,
                    accumulate_train_timesteps=accumulate_train_timesteps,
                    normalization_dict=bundle.normalization_dict,
                    dgpo_offset_anchor_stored_mu_ref=effective_offset_anchor_mu_ref,
                    dgpo_offset_anchor_stored_mu_ref_xyz=effective_offset_anchor_mu_ref_xyz,
                    dgpo_local_kl_anchor_stored_p0_ref=effective_local_kl_anchor_p0_ref,
                    reference_reward_kl_store=dgpo_reference_reward_kl_store,
                    offset_anchor_dual_state=offset_anchor_dual_state,
                )
                if wandb_mod is not None:
                    payload = _wandb_train_payload(metrics)
                    payload["epoch"] = float(epoch)
                    _wandb_log_step(wandb_mod, payload, step=global_step)
                    _append_profile_accum(metrics)
                    _flush_profile_accum(step=global_step)

                td_pt_p += metrics["_kin_h_pt_p"]
                td_pt_t += metrics["_kin_h_pt_t"]
                td_e_p += metrics["_kin_h_e_p"]
                td_e_t += metrics["_kin_h_e_t"]
                td_p_p += metrics["_kin_h_p_p"]
                td_p_t += metrics["_kin_h_p_t"]
                td_k1_pt_p += metrics["_kin_h_pt_k1_p"]
                td_k1_pt_t += metrics["_kin_h_pt_k1_t"]
                td_k1_e_p += metrics["_kin_h_e_k1_p"]
                td_k1_e_t += metrics["_kin_h_e_k1_t"]
                td_k1_p_p += metrics["_kin_h_p_k1_p"]
                td_k1_p_t += metrics["_kin_h_p_k1_t"]

                if is_rank0 and global_step % log_every == 0:
                    _log.info(
                        "epoch=%s step=%s beta_kl=%.6g L_total=%.6f L_dgpo=%.6f L_kl=%.6f "
                        "L_cur=%.4f L_ref=%.4f delta=%.4f "
                        "r_best=%.4f r_med=%.4f gap=%.4f",
                        epoch,
                        global_step,
                        metrics["dgpo/beta_kl_current"],
                        metrics["train/loss/total"],
                        metrics["train/loss/dgpo"],
                        metrics["train/loss/kl"],
                        metrics["train/loss/L_cur"],
                        metrics["train/loss/L_ref"],
                        metrics["train/loss/delta"],
                        metrics["reward/monitor/best_of_k"],
                        metrics["reward/monitor/median"],
                        metrics["reward/monitor/mean_gap"],
                    )
                global_step += 1

            # --- Epoch-end: build training-distribution figures from accumulated histograms ---
            if wandb_mod is not None:
                _flush_profile_accum(step=max(global_step - 1, 0), force=True)

            if world_size > 1:
                td_stack = np.stack([
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ])
                td_hist_t = torch.from_numpy(td_stack).to(device=device, dtype=torch.float64)
                dist.all_reduce(td_hist_t, op=dist.ReduceOp.SUM)
                td_merged = td_hist_t.cpu().numpy()
                (
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ) = [td_merged[i] for i in range(12)]

            if is_rank0 and wandb_mod is not None:
                _td_bin_pt = np.linspace(0.0, 300.0, _VAL_KIN_NUM_BINS + 1)
                _td_bin_eta = np.linspace(-4.0, 4.0, _VAL_KIN_NUM_BINS + 1)
                _td_bin_phi = np.linspace(-3.2, 3.2, _VAL_KIN_NUM_BINS + 1)
                _rb = getattr(global_config.reward_config, "rule_based", None)
                _rb_on = _rb is not None and bool(getattr(_rb, "enabled", False))
                _rb_mode = str(getattr(_rb, "mode", "truth_distance")) if _rb is not None else ""
                if _rb_on and _rb_mode == "truth_distance":
                    _td_suffix = "train: truth-L2 best-of-K vs truth (all batches)"
                else:
                    _td_suffix = "train: reward best-of-K vs truth (all batches)"
                try:
                    td_log: dict[str, Any] = {
                        "train_dist/pt": _val_overlay_kin_figure(
                            td_pt_t, td_pt_p, _td_bin_pt,
                            f"Neutrino pT [GeV] ({_td_suffix})",
                            pred_label="Pred (train)", xlabel="pT [GeV]",
                        ),
                        "train_dist/eta": _val_overlay_kin_figure(
                            td_e_t, td_e_p, _td_bin_eta,
                            f"Neutrino η ({_td_suffix})",
                            pred_label="Pred (train)", xlabel="η",
                        ),
                        "train_dist/phi": _val_overlay_kin_figure(
                            td_p_t, td_p_p, _td_bin_phi,
                            f"Neutrino φ ({_td_suffix})",
                            pred_label="Pred (train)", xlabel="φ [rad]",
                        ),
                        "train_dist_k1/pt": _val_overlay_kin_figure(
                            td_k1_pt_t, td_k1_pt_p, _td_bin_pt,
                            "Neutrino pT [GeV] (train: candidate 0 / K=1 proxy vs truth, all batches)",
                            pred_label="Pred (train K=1 proxy)", xlabel="pT [GeV]",
                        ),
                        "train_dist_k1/eta": _val_overlay_kin_figure(
                            td_k1_e_t, td_k1_e_p, _td_bin_eta,
                            "Neutrino η (train: candidate 0 / K=1 proxy vs truth, all batches)",
                            pred_label="Pred (train K=1 proxy)", xlabel="η",
                        ),
                        "train_dist_k1/phi": _val_overlay_kin_figure(
                            td_k1_p_t, td_k1_p_p, _td_bin_phi,
                            "Neutrino φ (train: candidate 0 / K=1 proxy vs truth, all batches)",
                            pred_label="Pred (train K=1 proxy)", xlabel="φ [rad]",
                        ),
                        "epoch": float(epoch),
                    }
                    wandb_mod.log(td_log)
                except Exception as _e:
                    _log.warning("[DGPO] train_dist figures failed at epoch=%s: %s", epoch, _e)

            ve = int(val_events) if val_events is not None else 0
            if ve > 0 and val_shard is not None:
                if is_rank0:
                    _log.info(
                        "[DGPO] val: requesting val iterator (Ray read+preprocess may take a long time; "
                        "first DDIM batch logs after rows arrive).",
                    )
                val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
                est_val_batches = max(1, math.ceil(ve / effective_batch)) if ve > 0 else None
                val_metrics = run_validation_epoch(
                    model,
                    ref_model,
                    ema_save,
                    val_loader,
                    sampler,
                    reward_agg,
                    val_K=val_K,
                    num_ddim_steps=num_ddim,
                    use_ema_for_rollout=use_ema_rollout,
                    device=device,
                    dtype=dtype,
                    cartesian=_truth_generation_cartesian(),
                    compute_winrate=bool(dg.get("validation_compute_winrate", False)),
                    epoch=epoch,
                    est_total_batches=est_val_batches,
                    val_log_batches=bool(dg.get("validation_log_batches", True)),
                    val_tqdm_k_chains=bool(dg.get("validation_tqdm_k_chains", True)),
                    val_tqdm_ddim=bool(dg.get("validation_tqdm_ddim", False)),
                    max_batches=val_max_batches,
                    initial_state=val_baseline_state,
                    rank=rank,
                    world_size=world_size,
                )
                if is_rank0:
                    _append_validation_history_plots(val_metrics, epoch_value=epoch)
                    _log.info(
                        "[DGPO] val epoch=%s r_mean=%.6f r_med=%.6f p10=%.6f p90=%.6f winrate=%.4f",
                        epoch,
                        val_metrics["val/reward/mean"],
                        val_metrics["val/reward/median"],
                        val_metrics["val/reward/p10"],
                        val_metrics["val/reward/p90"],
                        val_metrics["val/winrate"],
                    )
                    if wandb_mod is not None:
                        _wandb_log_validation(wandb_mod, val_metrics, epoch=epoch)
                    if ckpt_topk is not None:
                        rb, rk = _bundle_rrkl_ckpt()
                        _oa_mr, _oa_mxyz = _oa_ckpt_mu_pair()
                        _dual_p, _dual_z = dual_ckpt_scalars_for_save()
                        ckpt_topk.maybe_save(
                            val_reward_mean=val_metrics["val/reward/mean"],
                            last_completed_epoch=epoch,
                            dgpo_next_epoch=epoch + 1,
                            global_step=global_step,
                            model=model,
                            ema_save=ema_save,
                            optimizer=optimizer,
                            ref_model=ref_model,
                            dgpo_offset_anchor_mu_ref=_oa_mr,
                            dgpo_offset_anchor_mu_ref_xyz=_oa_mxyz,
                            dgpo_local_kl_anchor_p0_ref=_local_kl_anchor_p0_ref_for_checkpoint(
                                effective_local_kl_anchor_p0_ref
                            ),
                            dgpo_rrkl_ckpt_blob=rb,
                            dgpo_rrkl_ckpt_key=rk,
                            dgpo_offset_anchor_dual_pt=_dual_p,
                            dgpo_offset_anchor_z_ema_pt=_dual_z,
                        )
            _barrier()

            if is_rank0:
                rb, rk = _bundle_rrkl_ckpt()
                _oa_mr, _oa_mxyz = _oa_ckpt_mu_pair()
                _dual_p, _dual_z = dual_ckpt_scalars_for_save()
                _dgpo_save_last_ckpt(
                    model,
                    ema_save,
                    optimizer,
                    ref_model,
                    last_completed_epoch=epoch,
                    dgpo_next_epoch=epoch + 1,
                    global_step=global_step,
                    dgpo_offset_anchor_mu_ref=_oa_mr,
                    dgpo_offset_anchor_mu_ref_xyz=_oa_mxyz,
                    dgpo_local_kl_anchor_p0_ref=_local_kl_anchor_p0_ref_for_checkpoint(
                        effective_local_kl_anchor_p0_ref
                    ),
                    dgpo_rrkl_ckpt_blob=rb,
                    dgpo_rrkl_ckpt_key=rk,
                    dgpo_offset_anchor_dual_pt=_dual_p,
                    dgpo_offset_anchor_z_ema_pt=_dual_z,
                )
            _barrier()

            # ``stop_epoch`` is set when this rank's shard ran dry; nothing else to do.
            del stop_epoch

        if is_rank0:
            _log.info("[DGPO] finished %s epochs (%s optimizer steps).", epochs, global_step)
    finally:
        _finish_wandb_run(wandb_active)


def main() -> None:
    """CLI entry point: build a Ray ``TorchTrainer`` and ``fit()`` across the cluster."""
    p = argparse.ArgumentParser(description="DGPO neutrino RL (Ray Train + DGPO loop)")
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML config (same merge rules as EveNet training)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (smoke test)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config)",
    )
    p.add_argument(
        "--ray-dir",
        type=str,
        default="~/ray_results",
        help="Ray Train RunConfig.storage_path",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    global_config.load_yaml(config_path)
    global_config.display()
    platform_info = global_config.platform
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    # Ray Train / Tune default is ``RAY_CHDIR_TO_TRIAL_DIR=1``: each worker ``chdir``s into a
    # per-trial directory under ``/tmp`` (or ``RunConfig.storage_path``), so *relative* paths in
    # YAML resolve there and ``torch.load("data/.../normalization.pt")`` fails. EveNet-private
    # cluster scripts typically export ``0`` here; match that unless the environment already set
    # it (e.g. ``RAY_CHDIR_TO_TRIAL_DIR=1`` to keep Ray's default trial-isolated cwd).
    os.environ.setdefault("RAY_CHDIR_TO_TRIAL_DIR", "0")

    existing_pp = os.environ.get("PYTHONPATH", "").strip()
    pythonpath_parts = [_REPO_ROOT, _EVENET_ROOT]
    if existing_pp:
        pythonpath_parts.append(existing_pp)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": ":".join(pythonpath_parts),
            "TORCH_NCCL_TIMEOUT": "180",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ[
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ],
            "RAY_CHDIR_TO_TRIAL_DIR": os.environ["RAY_CHDIR_TO_TRIAL_DIR"],
        },
    }
    if "WANDB_API_KEY" in os.environ:
        runtime_env["env_vars"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # ``address="auto"`` connects to a Ray cluster already started externally (e.g. Slurm
    # launcher scripts) instead of silently spinning up a fresh single-node cluster on the
    # head. ``RAY_ADDRESS`` takes precedence when present. Outside cluster jobs we fall back
    # to a local cluster.
    ray_addr_env = os.environ.get("RAY_ADDRESS")
    try:
        ray.init(
            address=ray_addr_env or "auto",
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except (ConnectionError, ValueError) as ex:
        _log.warning(
            "[DGPO][launch] No existing Ray cluster (%s); falling back to local init.",
            ex,
        )
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    # Wait for the expected number of Ray workers to join. Worker startup on shared
    # clusters can take tens of seconds; without this wait, ``trainer.fit()`` may see only
    # the head node and silently run on one node.
    expected_workers = int(platform_info.number_of_workers)
    expected_gpus_per_worker = float(dict(platform_info.resources_per_worker).get("GPU", 1))
    expected_gpus = float(expected_workers) * expected_gpus_per_worker
    wait_timeout_s = float(os.environ.get("DGPO_RAY_WAIT_S", "300"))
    poll_every = 5.0
    waited = 0.0
    while waited < wait_timeout_s:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        cur_nodes = len(ray.nodes())
        if cur_gpus >= expected_gpus:
            _log.info(
                "[DGPO][launch] Ray cluster ready: nodes=%s GPUs=%s (expected %s).",
                cur_nodes, cur_gpus, expected_gpus,
            )
            break
        _log.info(
            "[DGPO][launch] waiting for Ray workers... nodes=%s GPUs=%s/%s (%.0fs/%.0fs)",
            cur_nodes, cur_gpus, expected_gpus, waited, wait_timeout_s,
        )
        time.sleep(poll_every)
        waited += poll_every
    else:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        _log.warning(
            "[DGPO][launch] Timed out after %.0fs waiting for cluster: GPUs=%s (expected %s). "
            "Continuing — Ray Train may run with fewer workers or hang.",
            wait_timeout_s, cur_gpus, expected_gpus,
        )

    base_dir = Path(platform_info.data_parquet_dir)
    base_val_dir = (
        Path(platform_info.data_parquet_val_dir)
        if "data_parquet_val_dir" in platform_info
        else None
    )
    process_fn = make_process_fn(base_dir)
    train_ds, val_ds, total_events, val_events = prepare_datasets(
        base_dir=base_dir,
        process_event_batch_partial=process_fn,
        platform_info=platform_info,
        load_all_in_ram=False,
        base_val_dir=base_val_dir,
        predict=False,
    )

    datasets: dict[str, Any] = {"train": train_ds}
    if val_ds is not None and val_events:
        datasets["validation"] = val_ds

    scaling_config = ScalingConfig(
        num_workers=int(platform_info.number_of_workers),
        resources_per_worker=dict(platform_info.resources_per_worker),
        use_gpu=bool(platform_info.get("use_gpu", True)),
    )
    run_config = RunConfig(
        name="DGPO-Training",
        storage_path=args.ray_dir,
    )

    # Driver-side launch banner: visible on the head node before any worker spawns,
    # so a wrong cluster size is caught before the data pipeline starts.
    try:
        cluster_resources = ray.cluster_resources()
    except Exception:
        cluster_resources = {}
    _log.info(
        "[DGPO][launch] num_workers=%s resources_per_worker=%s use_gpu=%s "
        "cluster_GPUs=%s cluster_CPUs=%s nodes=%s",
        scaling_config.num_workers,
        scaling_config.resources_per_worker,
        scaling_config.use_gpu,
        cluster_resources.get("GPU"),
        cluster_resources.get("CPU"),
        len(ray.nodes()) if ray.is_initialized() else "?",
    )
    trainer_config = {
        "config_path": str(config_path),
        "max_steps": args.max_steps,
        "wandb": not args.no_wandb,
        "total_events": int(total_events),
        "val_events": int(val_events) if val_events else 0,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=dgpo_train_loop,
        train_loop_config=trainer_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets=datasets,
    )
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
