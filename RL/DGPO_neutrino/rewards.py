"""
Composable reward sources for DGPO neutrino RL: truth distance, W mass, PDG W mass-shell projection penalty, aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from RL.DGPO_neutrino.dgpo_utils import (
    MW_REF_GEV,
    _invariant_mass_gev_torch,
    _lepton_four_momentum_timestep_row,
    get_truth_assignment_indices,
    tt2l_w_mass_per_row_and_means,
)

# Invisible neutrino slots use the same 7-D sequential feature layout as jets; unused channels are masked/padded.
NU_FEAT_DIM = 7


def log_pt_eta_phi_to_cartesian(log_pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
    """Map ``(log1p(pT), η, φ)`` to ``(p_x, p_y, p_z)``.

    Matches ``preprocessing/preprocess.pt_eta_phi_to_cartesian`` and
    ``reasoning_analysis/RL/top_mass_ablation.to_cartesian`` (``pT = expm1(log_pt)``).

    Args:
        log_pt: Any shape; ``log1p(pT)`` storage.
        eta: Same shape as ``log_pt``.
        phi: Same shape as ``log_pt``.

    Returns:
        Tensor with shape ``(*log_pt.shape, 3)``, last dim ``[p_x, p_y, p_z]``.
    """
    pt = torch.expm1(log_pt)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack((px, py, pz), dim=-1)


def cartesian_to_log_pt_eta_phi(px: Tensor, py: Tensor, pz: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of :func:`log_pt_eta_phi_to_cartesian`: ``(p_x, p_y, p_z)`` → ``(log1p(pT), η, φ)``.

    Uses ``pT = sqrt(p_x^2 + p_y^2)``, ``φ = atan2(p_y, p_x)``, ``η = arcsinh(p_z / pT)`` (with ``η=0`` when ``pT`` is tiny).

    Args:
        px, py, pz: Same shape broadcastable tensors.

    Returns:
        Tuple ``(log_pt, eta, phi)`` each matching the broadcast shape of the inputs.
    """
    pt = torch.sqrt(px * px + py * py + 1e-12)
    log_pt = torch.log1p(pt)
    phi = torch.atan2(py, px)
    eta = torch.where(pt > 1e-8, torch.asinh(pz / pt), torch.zeros_like(pz))
    return log_pt, eta, phi


def invisible_kinematics_to_cartesian(x: Tensor) -> Tensor:
    """First three features per slot: ``log1p(pT), η, φ`` → Cartesian ``(p_x, p_y, p_z)``.

    Args:
        x: ``(..., F)`` with ``F >= 3``; indices ``[..., 0:3]`` are kinematics.

    Returns:
        ``(..., 3)`` Cartesian momentum.
    """
    if x.shape[-1] < 3:
        raise ValueError(f"need at least 3 features (log_pt, eta, phi), got F={x.shape[-1]}")
    return log_pt_eta_phi_to_cartesian(x[..., 0], x[..., 1], x[..., 2])


def get_event_valid_mask(
    batch: dict[str, Any],
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-event multiplicative mask for rewards: 1 = train/score, 0 = excluded.

    Combines (when present), in order:

    - ``batch['event_valid']`` ``(B,)`` or ``(B, 1)``: values > 0 keep the event.
    - ``batch['event_weight']`` ``(B,)`` or ``(B, 1)``: same convention as EveNet training
      (weight <= 0 excludes the event from loss aggregation).
    - ``batch['x_invisible_mask']``: events with no active neutrino slots (sum over slots == 0)
      are excluded so truth-distance is not applied without targets.

    If a key is missing or has a wrong leading dimension, it is skipped for that rule.
    """
    valid = torch.ones(B, device=device, dtype=dtype)

    ev = batch.get("event_valid")
    if isinstance(ev, Tensor):
        ev = ev.to(device=device, dtype=dtype).reshape(-1)
        if ev.numel() == B:
            valid = valid * (ev > 0).to(dtype)

    ew = batch.get("event_weight")
    if isinstance(ew, Tensor):
        ew = ew.to(device=device, dtype=dtype).reshape(-1)
        if ew.numel() == B:
            valid = valid * (ew > 1e-12).to(dtype)

    xm = batch.get("x_invisible_mask")
    if isinstance(xm, Tensor) and xm.shape[0] == B:
        m = xm.to(device=device, dtype=dtype)
        # Sum over all dims after batch — at least one valid neutrino slot.
        has_truth = m.reshape(B, -1).sum(dim=1) > 1e-12
        valid = valid * has_truth.to(dtype)

    return valid


def apply_event_valid_to_rewards(rewards_kb: Tensor, batch: dict[str, Any]) -> Tensor:
    """Multiply ``(K, B)`` rewards by per-event validity (see ``get_event_valid_mask``)."""
    if rewards_kb.dim() != 2:
        raise ValueError(f"expected rewards (K, B), got shape {tuple(rewards_kb.shape)}")
    K, B = rewards_kb.shape
    vm = get_event_valid_mask(batch, B, rewards_kb.device, rewards_kb.dtype).unsqueeze(0)
    return rewards_kb * vm


class BaseReward(ABC):
    """Abstract reward: maps (candidates, batch) to per-(candidate, event) scores.

    Implementations should return rewards that are already zeroed on invalid events via
    ``apply_event_valid_to_rewards`` (see built-in rewards).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging and breakdown dict keys."""

    @abstractmethod
    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return rewards of shape ``(K, B)`` for candidates ``(K, B, num_nu, F)``."""


class TruthDistanceReward(BaseReward):
    """Negative L2 distance to truth neutrinos (``pt_eta_phi`` → Cartesian, or raw Cartesian)."""

    def __init__(self, *, cartesian: bool = False) -> None:
        """Args:
            cartesian: If True, compare ``(p_x,p_y,p_z)`` directly using ``batch['x_invisible_cartesian']``
                (same space as DDIM output when ``TruthGeneration.cartesian: true``).
        """
        self._cartesian = bool(cartesian)

    @property
    def name(self) -> str:
        return "truth_distance"

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        """``r = -|| (p_cand - p_truth) ⊙ mask ||_2`` over neutrino slots.

        In ``pt_eta_phi`` mode, first three features per slot are ``log1p(pT)``, ``η``, ``φ``; reward
        uses Cartesian distance. In ``cartesian`` mode, truth is ``batch['x_invisible_cartesian']`` and
        candidates are already ``(p_x, p_y, p_z)`` per slot (DDIM denormalized output).

        Args:
            candidates: ``(K, B, N_nu, F)`` with ``F >= 3`` (kinematic) or ``F == 3`` (Cartesian).
            batch: ``x_invisible`` or ``x_invisible_cartesian`` depending on mode.
            mask: Optional ``(B, N_nu)`` or ``(B, N_nu, 1)``; if None, uses ``batch['x_invisible_mask']``.

        Returns:
            Tensor ``(K, B)``. Invalid events are zeroed (see ``get_event_valid_mask``).
        """
        K, B, N_nu, F = candidates.shape

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

        m_cand = m.unsqueeze(0)  # (1, B, N_nu, 1)

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "TruthDistanceReward(cartesian=True) requires batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3:
                raise ValueError(
                    f"x_invisible_cartesian must be (B, N_nu, F), got {tuple(truth.shape)}"
                )
            if truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
                raise ValueError(
                    f"truth shape {tuple(truth.shape)} incompatible with candidates {tuple(candidates.shape)}"
                )
            truth_kin = truth * m
            cand_kin = candidates * m_cand
            diff = cand_kin - truth_kin.unsqueeze(0)
            sq = diff.pow(2).sum(dim=(2, 3))
        else:
            if "x_invisible" not in batch:
                raise KeyError("TruthDistanceReward requires batch['x_invisible']")
            truth = batch["x_invisible"]
            if truth.dim() != 3:
                raise ValueError(f"x_invisible must be (B, N_nu, F), got {tuple(truth.shape)}")
            if truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
                raise ValueError(
                    f"truth shape {tuple(truth.shape)} incompatible with candidates {tuple(candidates.shape)}"
                )
            if F < 3:
                raise ValueError(f"TruthDistanceReward needs F >= 3 (log_pt, eta, phi), got F={F}")

            truth_kin = truth * m
            cand_kin = candidates * m_cand

            truth_c = invisible_kinematics_to_cartesian(truth_kin)  # (B, N_nu, 3)
            cand_c = invisible_kinematics_to_cartesian(
                cand_kin.reshape(K * B, N_nu, F)
            ).reshape(K, B, N_nu, 3)

            diff = cand_c - truth_c.unsqueeze(0)
            sq = diff.pow(2).sum(dim=(2, 3))

        dist = torch.sqrt(sq.clamp(min=0.0) + 1e-12)
        out = -dist
        return apply_event_valid_to_rewards(out, batch)


class ComponentNormalizedTruthDistanceReward(BaseReward):
    """Negative sum of per-component normalized squared errors over (nu1, nu2) × (px, py, pz).

    For each candidate, evaluate squared error separately for each of the six
    Cartesian components ``{nu1_px, nu1_py, nu1_pz, nu2_px, nu2_py, nu2_pz}``,
    normalize by configurable global scales, and return the negative sum::

        err_c = (pred_c - truth_c) ** 2 / (scale_c ** 2 + eps)
        reward = - sum_c err_c

    Always operates in Cartesian space: in ``pt_eta_phi`` mode (``cartesian=False``)
    both prediction and truth are mapped to ``(p_x, p_y, p_z)`` before the per-component
    error is computed (matches :class:`TruthDistanceReward` geometry).
    """

    COMPONENT_ORDER: tuple[str, ...] = (
        "nu1_px", "nu1_py", "nu1_pz",
        "nu2_px", "nu2_py", "nu2_pz",
    )

    def __init__(
        self,
        scales: dict[str, float] | tuple[float, ...] | list[float],
        *,
        cartesian: bool = False,
        eps: float = 1e-8,
    ) -> None:
        """Args:
            scales: Either a dict keyed by component name (see ``COMPONENT_ORDER``) or a
                length-6 sequence in that exact order. Used as ``scale_c`` in the
                normalized squared-error formula.
            cartesian: If True, candidates and ``batch['x_invisible_cartesian']`` are
                already ``(p_x, p_y, p_z)``. Otherwise both are converted from
                ``(log1p(pT), η, φ)`` first.
            eps: Numerical safety added inside the denominator (``scale^2 + eps``).
        """
        if isinstance(scales, dict):
            try:
                scales_list = [float(scales[k]) for k in self.COMPONENT_ORDER]
            except KeyError as exc:
                raise KeyError(
                    f"scales dict missing component {exc!s}; expected keys {self.COMPONENT_ORDER}"
                ) from exc
        else:
            if len(scales) != 6:
                raise ValueError(
                    f"scales must have 6 entries, one per component, got {len(scales)}"
                )
            scales_list = [float(s) for s in scales]

        self._scales: list[float] = scales_list
        self._cartesian = bool(cartesian)
        self._eps = float(eps)
        # Cache per-component normalized errors from the most recent compute() call,
        # keyed by component name. Values are detached tensors of shape (K, B) with
        # invalid events zeroed (consistent with the returned reward).
        self._last_components: dict[str, Tensor] | None = None
        self._last_component_deltas: dict[str, Tensor] | None = None
        self._last_component_truths: dict[str, Tensor] | None = None
        self._last_kinematic_deltas: dict[str, Tensor] | None = None

    @property
    def name(self) -> str:
        return "component_normalized_truth_distance"

    @property
    def scales(self) -> list[float]:
        """Return the configured per-component scales (length 6, in COMPONENT_ORDER)."""
        return list(self._scales)

    def last_component_errors(self) -> dict[str, Tensor] | None:
        """Per-component normalized squared errors ``(K, B)`` from the last ``compute()`` call.

        Returns ``None`` before any call. Each tensor is detached and zeroed on
        invalid events (matches the returned reward's masking).
        """
        return self._last_components

    def last_component_deltas(self) -> dict[str, Tensor] | None:
        """Raw Cartesian component residuals ``pred - truth`` in GeV, each ``(K, B)``."""
        return self._last_component_deltas

    def last_component_truths(self) -> dict[str, Tensor] | None:
        """Cartesian truth components in GeV, repeated to ``(K, B)`` for each candidate."""
        return self._last_component_truths

    def last_kinematic_deltas(self) -> dict[str, Tensor] | None:
        """Kinematic residuals ``pred - truth`` for first two neutrino slots, each ``(K, B, 2)``.

        Keys are ``pt``, ``log_pt``, ``eta``, and ``phi`` residuals plus ``truth_*`` values.
        The ``phi`` residual is wrapped to ``[-pi, pi]``.
        """
        return self._last_kinematic_deltas

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        K, B, N_nu, F = candidates.shape
        if N_nu < 2:
            raise ValueError(
                f"ComponentNormalizedTruthDistanceReward needs N_nu >= 2 (nu1, nu2), got {N_nu}"
            )

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

        m_cand = m.unsqueeze(0)  # (1, B, N_nu, 1)

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "ComponentNormalizedTruthDistanceReward(cartesian=True) requires "
                    "batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < 3:
                raise ValueError(
                    f"x_invisible_cartesian shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)} (need (B, N_nu, >=3))"
                )
            truth_c = (truth[..., :3] * m)  # (B, N_nu, 3)
            cand_c = (candidates[..., :3] * m_cand)  # (K, B, N_nu, 3)
            truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
                truth_c[..., 0],
                truth_c[..., 1],
                truth_c[..., 2],
            )
            cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
                cand_c[..., 0],
                cand_c[..., 1],
                cand_c[..., 2],
            )
        else:
            if "x_invisible" not in batch:
                raise KeyError(
                    "ComponentNormalizedTruthDistanceReward requires batch['x_invisible']"
                )
            truth = batch["x_invisible"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] != F:
                raise ValueError(
                    f"x_invisible shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            if F < 3:
                raise ValueError(
                    f"need F >= 3 (log_pt, eta, phi), got F={F}"
                )
            truth_kin = truth * m
            cand_kin = candidates * m_cand
            truth_c = invisible_kinematics_to_cartesian(truth_kin)  # (B, N_nu, 3)
            cand_c = invisible_kinematics_to_cartesian(
                cand_kin.reshape(K * B, N_nu, F)
            ).reshape(K, B, N_nu, 3)
            truth_log_pt = truth_kin[..., 0]
            truth_eta = truth_kin[..., 1]
            truth_phi = truth_kin[..., 2]
            cand_log_pt = cand_kin[..., 0]
            cand_eta = cand_kin[..., 1]
            cand_phi = cand_kin[..., 2]

        # Take only the first two slots (nu1, nu2) and flatten to 6 components.
        cand_six = cand_c[:, :, 0:2, :].reshape(K, B, 6)            # (K, B, 6)
        truth_six = truth_c[:, 0:2, :].reshape(B, 6).unsqueeze(0)   # (1, B, 6)
        delta_six = cand_six - truth_six

        scales = torch.tensor(
            self._scales, device=candidates.device, dtype=candidates.dtype
        )
        denom = scales.pow(2) + self._eps  # (6,)

        err_components = (cand_six - truth_six).pow(2) / denom  # (K, B, 6)
        total_err = err_components.sum(dim=-1)                  # (K, B)
        reward = -total_err
        reward = apply_event_valid_to_rewards(reward, batch)

        # Cache per-component errors with the same per-event mask applied.
        valid_kb = get_event_valid_mask(batch, B, reward.device, reward.dtype).unsqueeze(0)
        masked = (err_components * valid_kb.unsqueeze(-1)).detach()
        self._last_components = {
            cname: masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }
        delta_masked = (delta_six * valid_kb.unsqueeze(-1)).detach()
        self._last_component_deltas = {
            cname: delta_masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }
        truth_masked = (
            truth_six.expand_as(delta_six) * valid_kb.unsqueeze(-1)
        ).detach()
        self._last_component_truths = {
            cname: truth_masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }

        cand_log_pt2 = cand_log_pt[:, :, 0:2]
        cand_eta2 = cand_eta[:, :, 0:2]
        cand_phi2 = cand_phi[:, :, 0:2]
        truth_log_pt2 = truth_log_pt[:, 0:2].unsqueeze(0)
        truth_eta2 = truth_eta[:, 0:2].unsqueeze(0)
        truth_phi2 = truth_phi[:, 0:2].unsqueeze(0)
        delta_phi = torch.atan2(
            torch.sin(cand_phi2 - truth_phi2),
            torch.cos(cand_phi2 - truth_phi2),
        )
        kin_mask = valid_kb.unsqueeze(-1)
        cand_pt2 = torch.expm1(cand_log_pt2.clamp(-10.0, 10.0))
        truth_pt2 = torch.expm1(truth_log_pt2.clamp(-10.0, 10.0))
        delta_pt = cand_pt2 - truth_pt2
        rel_pt = delta_pt / truth_pt2.clamp(min=1e-6)
        self._last_kinematic_deltas = {
            "pt": delta_pt.mul(kin_mask).detach().contiguous(),
            "rel_pt": rel_pt.mul(kin_mask).detach().contiguous(),
            "truth_pt": truth_pt2.mul(kin_mask).detach().contiguous(),
            "truth_log_pt": truth_log_pt2.mul(kin_mask).detach().contiguous(),
            "truth_eta": truth_eta2.mul(kin_mask).detach().contiguous(),
            "truth_phi": truth_phi2.mul(kin_mask).detach().contiguous(),
            "log_pt": ((cand_log_pt2 - truth_log_pt2) * kin_mask).detach().contiguous(),
            "eta": ((cand_eta2 - truth_eta2) * kin_mask).detach().contiguous(),
            "phi": (delta_phi * kin_mask).detach().contiguous(),
        }
        return reward


def _signed_relative_denominator(truth: Tensor, eps: float) -> Tensor:
    """Return ``sign(truth) * max(abs(truth), eps)`` for stable relative residuals."""
    eps_t = truth.new_tensor(float(eps))
    return torch.where(
        truth.abs() >= eps_t,
        truth,
        torch.where(truth < 0.0, -eps_t, eps_t),
    )


class RelativeTruthReward(BaseReward):
    """Negative clipped relative Cartesian residual error without component normalization scales.

    For each of ``(nu1, nu2) x (px, py, pz)``, compute::

        denom_c = sign(truth_c) * max(abs(truth_c), eps)
        rel_c = clip((pred_c - truth_c) / denom_c, -clip, clip)
        reward = -sum_c rel_c ** 2

    Unlike :class:`ComponentNormalizedTruthDistanceReward`, this mode does not load
    or divide by global per-component standard deviations from ``normalization.pt``.
    The input prediction and truth tensors are not modified.
    """

    COMPONENT_ORDER: tuple[str, ...] = ComponentNormalizedTruthDistanceReward.COMPONENT_ORDER

    def __init__(
        self,
        *,
        cartesian: bool = False,
        eps: float = 1e-6,
        clip: float = 5.0,
    ) -> None:
        """Args:
            cartesian: If True, candidates and ``batch['x_invisible_cartesian']`` are
                already ``(p_x, p_y, p_z)``. Otherwise both are converted from
                ``(log1p(pT), eta, phi)`` first.
            eps: Minimum absolute denominator for near-zero truth components.
            clip: Absolute clamp applied to relative residuals before squaring.
        """
        if float(clip) <= 0.0:
            raise ValueError(f"clip must be positive, got {clip}")
        self._cartesian = bool(cartesian)
        self._eps = float(eps)
        self._clip = float(clip)
        self._last_components: dict[str, Tensor] | None = None
        self._last_component_deltas: dict[str, Tensor] | None = None
        self._last_component_truths: dict[str, Tensor] | None = None
        self._last_kinematic_deltas: dict[str, Tensor] | None = None

    @property
    def name(self) -> str:
        return "relative_reward"

    def last_component_errors(self) -> dict[str, Tensor] | None:
        """Per-component squared relative errors ``(K, B)`` from the last call."""
        return self._last_components

    def last_component_deltas(self) -> dict[str, Tensor] | None:
        """Raw Cartesian residuals ``pred - truth`` in GeV, each ``(K, B)``."""
        return self._last_component_deltas

    def last_component_truths(self) -> dict[str, Tensor] | None:
        """Cartesian truth components in GeV, repeated to ``(K, B)`` for each candidate."""
        return self._last_component_truths

    def last_kinematic_deltas(self) -> dict[str, Tensor] | None:
        """Kinematic residuals for first two neutrino slots, matching component-normalized diagnostics."""
        return self._last_kinematic_deltas

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        K, B, N_nu, F = candidates.shape
        if N_nu < 2:
            raise ValueError(
                f"RelativeTruthReward needs N_nu >= 2 (nu1, nu2), got {N_nu}"
            )

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

        m_cand = m.unsqueeze(0)  # (1, B, N_nu, 1)

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "RelativeTruthReward(cartesian=True) requires batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < 3:
                raise ValueError(
                    f"x_invisible_cartesian shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)} (need (B, N_nu, >=3))"
                )
            truth_c = truth[..., :3] * m
            cand_c = candidates[..., :3] * m_cand
            truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
                truth_c[..., 0],
                truth_c[..., 1],
                truth_c[..., 2],
            )
            cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
                cand_c[..., 0],
                cand_c[..., 1],
                cand_c[..., 2],
            )
        else:
            if "x_invisible" not in batch:
                raise KeyError("RelativeTruthReward requires batch['x_invisible']")
            truth = batch["x_invisible"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] != F:
                raise ValueError(
                    f"x_invisible shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            if F < 3:
                raise ValueError(f"need F >= 3 (log_pt, eta, phi), got F={F}")
            truth_kin = truth * m
            cand_kin = candidates * m_cand
            truth_c = invisible_kinematics_to_cartesian(truth_kin)
            cand_c = invisible_kinematics_to_cartesian(
                cand_kin.reshape(K * B, N_nu, F)
            ).reshape(K, B, N_nu, 3)
            truth_log_pt = truth_kin[..., 0]
            truth_eta = truth_kin[..., 1]
            truth_phi = truth_kin[..., 2]
            cand_log_pt = cand_kin[..., 0]
            cand_eta = cand_kin[..., 1]
            cand_phi = cand_kin[..., 2]

        cand_six = cand_c[:, :, 0:2, :].reshape(K, B, 6)
        truth_six = truth_c[:, 0:2, :].reshape(B, 6).unsqueeze(0)
        delta_six = cand_six - truth_six
        denom = _signed_relative_denominator(truth_six, self._eps)
        rel_components = (delta_six / denom).clamp(-self._clip, self._clip)
        err_components = rel_components.pow(2)

        reward = -err_components.sum(dim=-1)
        reward = apply_event_valid_to_rewards(reward, batch)

        valid_kb = get_event_valid_mask(batch, B, reward.device, reward.dtype).unsqueeze(0)
        masked = (err_components * valid_kb.unsqueeze(-1)).detach()
        self._last_components = {
            cname: masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }
        delta_masked = (delta_six * valid_kb.unsqueeze(-1)).detach()
        self._last_component_deltas = {
            cname: delta_masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }
        truth_masked = (
            truth_six.expand_as(delta_six) * valid_kb.unsqueeze(-1)
        ).detach()
        self._last_component_truths = {
            cname: truth_masked[..., i].contiguous()
            for i, cname in enumerate(self.COMPONENT_ORDER)
        }

        cand_log_pt2 = cand_log_pt[:, :, 0:2]
        cand_eta2 = cand_eta[:, :, 0:2]
        cand_phi2 = cand_phi[:, :, 0:2]
        truth_log_pt2 = truth_log_pt[:, 0:2].unsqueeze(0)
        truth_eta2 = truth_eta[:, 0:2].unsqueeze(0)
        truth_phi2 = truth_phi[:, 0:2].unsqueeze(0)
        delta_phi = torch.atan2(
            torch.sin(cand_phi2 - truth_phi2),
            torch.cos(cand_phi2 - truth_phi2),
        )
        kin_mask = valid_kb.unsqueeze(-1)
        cand_pt2 = torch.expm1(cand_log_pt2.clamp(-10.0, 10.0))
        truth_pt2 = torch.expm1(truth_log_pt2.clamp(-10.0, 10.0))
        delta_pt = cand_pt2 - truth_pt2
        rel_pt = delta_pt / truth_pt2.clamp(min=1e-6)
        self._last_kinematic_deltas = {
            "pt": delta_pt.mul(kin_mask).detach().contiguous(),
            "rel_pt": rel_pt.mul(kin_mask).detach().contiguous(),
            "truth_pt": truth_pt2.mul(kin_mask).detach().contiguous(),
            "truth_log_pt": truth_log_pt2.mul(kin_mask).detach().contiguous(),
            "truth_eta": truth_eta2.mul(kin_mask).detach().contiguous(),
            "truth_phi": truth_phi2.mul(kin_mask).detach().contiguous(),
            "log_pt": ((cand_log_pt2 - truth_log_pt2) * kin_mask).detach().contiguous(),
            "eta": ((cand_eta2 - truth_eta2) * kin_mask).detach().contiguous(),
            "phi": (delta_phi * kin_mask).detach().contiguous(),
        }
        return reward


class LogPtTruthReward(BaseReward):
    """Negative sum of per-neutrino squared errors in stored ``log1p(pT)`` space.

    Returns::

        reward = - sum_nu [(pred_log_pt - truth_log_pt)^2 / (scale^2 + eps)]

    Applies the same masking convention as other invisible rewards via
    ``apply_event_valid_to_rewards``. When ``TruthGeneration.cartesian`` is enabled,
    candidates and truth are ``(p_x,p_y,p_z)`` per slot and are inverted with
    :func:`cartesian_to_log_pt_eta_phi`.
    """

    def __init__(
        self,
        *,
        cartesian: bool = False,
        scale: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        """Args:
            cartesian: If True, use ``batch['x_invisible_cartesian']`` and invert to log-pT.
            scale: Denominator ``scale`` in ``(...) / scale`` (global, not per-neutrino).
            eps: Stability term added as ``scale^2 + eps`` in denominator.
        """
        if float(scale) <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._cartesian = bool(cartesian)
        self._scale = float(scale)
        self._eps = float(eps)

    @property
    def name(self) -> str:
        return "log_pt_truth"

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        K, B, N_nu, F = candidates.shape

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).squeeze(-1)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")
        if m.shape != (B, N_nu):
            raise ValueError(f"mask shape {tuple(m.shape)} expected ({B}, {N_nu})")

        denom = self._scale * self._scale + self._eps

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "LogPtTruthReward(cartesian=True) requires batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < 3:
                raise ValueError(
                    f"x_invisible_cartesian shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            px_p, py_p, pz_p = (
                candidates[..., 0],
                candidates[..., 1],
                candidates[..., 2],
            )
            px_t, py_t, pz_t = (
                truth[..., 0],
                truth[..., 1],
                truth[..., 2],
            )
            pred_log_pt = cartesian_to_log_pt_eta_phi(px_p, py_p, pz_p)[0]
            truth_log_pt = cartesian_to_log_pt_eta_phi(px_t, py_t, pz_t)[0].unsqueeze(0)
        else:
            if "x_invisible" not in batch:
                raise KeyError("LogPtTruthReward requires batch['x_invisible']")
            truth = batch["x_invisible"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu:
                raise ValueError(
                    f"x_invisible shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            if F < 1:
                raise ValueError(f"need F >= 1 (log_pt feature), got F={F}")
            pred_log_pt = candidates[..., 0]
            truth_log_pt = truth[..., 0].unsqueeze(0)

        diff = pred_log_pt - truth_log_pt
        sq = diff.pow(2) * m.unsqueeze(0)
        reward = -(sq.sum(dim=2) / denom)
        return apply_event_valid_to_rewards(reward, batch)


class PtTruthReward(BaseReward):
    """Negative sum of per-neutrino squared errors in linear pT [GeV].

    This complements Cartesian rewards when the marginal pT distribution drifts:
    log1p(pT) rewards emphasize the low-pT region, while this reward penalizes
    absolute pT shifts directly in the physical scale used by validation plots.
    """

    def __init__(
        self,
        *,
        cartesian: bool = False,
        scale: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        """Args:
            cartesian: If True, use ``sqrt(px^2 + py^2)`` from Cartesian candidates/truth.
            scale: Denominator scale in GeV for the normalized squared error.
            eps: Stability term added as ``scale^2 + eps`` in denominator.
        """
        if float(scale) <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._cartesian = bool(cartesian)
        self._scale = float(scale)
        self._eps = float(eps)

    @property
    def name(self) -> str:
        return "pt_truth"

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        K, B, N_nu, F = candidates.shape

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).squeeze(-1)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")
        if m.shape != (B, N_nu):
            raise ValueError(f"mask shape {tuple(m.shape)} expected ({B}, {N_nu})")

        denom = self._scale * self._scale + self._eps

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "PtTruthReward(cartesian=True) requires batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < 2:
                raise ValueError(
                    f"x_invisible_cartesian shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            pred_pt = torch.sqrt(candidates[..., 0].pow(2) + candidates[..., 1].pow(2) + 1e-12)
            truth_pt = torch.sqrt(truth[..., 0].pow(2) + truth[..., 1].pow(2) + 1e-12).unsqueeze(0)
        else:
            if "x_invisible" not in batch:
                raise KeyError("PtTruthReward requires batch['x_invisible']")
            truth = batch["x_invisible"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu:
                raise ValueError(
                    f"x_invisible shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            if F < 1:
                raise ValueError(f"need F >= 1 (log_pt feature), got F={F}")
            pred_pt = torch.expm1(candidates[..., 0].clamp(-20.0, 20.0))
            truth_pt = torch.expm1(truth[..., 0].clamp(-20.0, 20.0)).unsqueeze(0)

        diff = pred_pt - truth_pt
        sq = diff.pow(2) * m.unsqueeze(0)
        reward = -(sq.sum(dim=2) / denom)
        return apply_event_valid_to_rewards(reward, batch)


def _candidate_log_pt_eta_phi_rows(nu_rows: Tensor, *, cartesian: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Neutrino candidate rows ``(R, F)`` → clamped ``(log1p(pT), η, φ)`` for projection."""
    if cartesian:
        if nu_rows.shape[-1] < 3:
            raise ValueError(f"Cartesian neutrino rows need F >= 3, got F={nu_rows.shape[-1]}")
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(
            nu_rows[..., 0], nu_rows[..., 1], nu_rows[..., 2]
        )
    else:
        if nu_rows.shape[-1] < 3:
            raise ValueError(f"pt_eta_phi neutrino rows need F >= 3, got F={nu_rows.shape[-1]}")
        log_pt = nu_rows[..., 0]
        eta = nu_rows[..., 1]
        phi = nu_rows[..., 2]
    log_pt = log_pt.clamp(-10.0, 10.0)
    eta = eta.clamp(-8.0, 8.0)
    return log_pt, eta, phi


def _project_pt_w_mass_slot(
    lep_row: Tensor,
    log_pt_cand: Tensor,
    eta_nu: Tensor,
    phi_nu: Tensor,
    *,
    mw_gev: float,
    min_pt: float,
    max_pt: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Scalar ``pT`` projection onto PDG ``m_W`` shell with fixed neutrino direction (one flat row).

    Mirrors ``calibrate_neutrino_pt_w_mass`` in ``reasoning_analysis/calibration/neutrino_calibration_mw.py``.

    Returns:
        ``(pt_proj, pt_cand, valid)`` each ``(R,)``.
    """
    E, px, py, pz = _lepton_four_momentum_timestep_row(lep_row)
    m_lep = _invariant_mass_gev_torch(E, px, py, pz)
    dot = (
        E * torch.cosh(eta_nu)
        - px * torch.cos(phi_nu)
        - py * torch.sin(phi_nu)
        - pz * torch.sinh(eta_nu)
    )
    mw_sq = dot.new_tensor(float(mw_gev * mw_gev))
    numer = mw_sq - m_lep * m_lep
    denom = torch.where(dot.abs() < 1e-10, dot.new_tensor(1e-10), dot)
    pt_proj = numer / (2.0 * denom)
    pt_cand = torch.expm1(log_pt_cand.clamp(-10.0, 10.0))
    valid = (
        (dot > 1e-10)
        & (pt_proj >= float(min_pt))
        & (pt_proj <= float(max_pt))
        & torch.isfinite(pt_proj)
        & torch.isfinite(pt_cand)
    )
    return pt_proj, pt_cand, valid


class WMassProjectionReward(BaseReward):
    """TT2L additive reward penalizing ``log1p(pT)`` corrections needed to reach PDG ``m_W``."""

    def __init__(
        self,
        *,
        cartesian: bool,
        scale: float,
        eps: float = 1e-8,
        min_pt: float = 0.1,
        max_pt: float = 1000.0,
        mw_gev: float = MW_REF_GEV,
    ) -> None:
        """Args:
            cartesian: Same as ``TruthGeneration.cartesian`` for neutrino candidate rows.
            scale: Normalizes summed squared ``Δ log1p(pT)`` errors (use ``auto`` resolution in YAML).
            eps: Added as ``scale**2 + eps`` in the denominator.
            min_pt, max_pt: Physical projection bounds (GeV), matching calibration utilities.
            mw_gev: PDG-scale target mass (GeV).
        """
        if float(scale) <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._cartesian = bool(cartesian)
        self._scale = float(scale)
        self._eps = float(eps)
        self._min_pt = float(min_pt)
        self._max_pt = float(max_pt)
        self._mw_gev = float(mw_gev)
        self._last_reward_kb: Tensor | None = None
        self._last_logpt_delta_kb2: Tensor | None = None
        self._last_valid_kb2: Tensor | None = None

    @property
    def name(self) -> str:
        return "w_projection"

    def last_projection_tensors(
        self,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Detached ``(reward_kb, Δlog1p_pT_kb2, valid_kb2)`` from the last ``compute()``."""
        return self._last_reward_kb, self._last_logpt_delta_kb2, self._last_valid_kb2

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``(K, B)`` negative normalized squared ``Δ log1p(pT)`` projection penalty."""
        del mask  # parity with ``BaseReward`` API
        K, B, N_nu, F = candidates.shape
        if N_nu < 2:
            raise ValueError(f"WMassProjectionReward needs N_nu >= 2, got {N_nu}")

        device = candidates.device
        dtype = candidates.dtype

        if "x" not in batch or "x_mask" not in batch:
            raise KeyError("WMassProjectionReward requires batch['x'] and batch['x_mask']")

        x_bb = batch["x"].to(device=device, dtype=dtype)
        xm_bb = batch["x_mask"].to(device=device, dtype=dtype)
        assign_bb = get_truth_assignment_indices(batch).to(device=device)

        nu_kb = candidates[:, :, :2, :].reshape(K * B, 2, F).contiguous()
        rows = nu_kb.shape[0]
        idx = torch.arange(rows, device=device)
        x_rep = (
            x_bb.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, *x_bb.shape[1:])
        )
        xm_rep = (
            xm_bb.unsqueeze(0)
            .expand(K, -1, -1)
            .reshape(K * B, *xm_bb.shape[1:])
        )
        assign_rep = (
            assign_bb.unsqueeze(0)
            .expand(K, -1, -1, -1)
            .reshape(K * B, *assign_bb.shape[1:])
        )

        ai = assign_rep.long()
        lp0 = ai[:, 0, 1]
        lp1 = ai[:, 1, 1]
        valid_assign = (lp0 >= 0) & (lp1 >= 0)
        xm_dtype = xm_rep.to(dtype=dtype)
        valid_slot = (
            xm_dtype[idx, lp0.clamp(min=0)] > 0.5
        ) & (xm_dtype[idx, lp1.clamp(min=0)] > 0.5)
        valid_assign = valid_assign & valid_slot
        valid_assign = valid_assign & torch.isfinite(nu_kb).reshape(rows, -1).all(dim=1)

        lep0_row = x_rep[idx, lp0.clamp(min=0)]
        lep1_row = x_rep[idx, lp1.clamp(min=0)]

        nu0 = nu_kb[:, 0]
        nu1 = nu_kb[:, 1]
        lp0_log, eta0, phi0 = _candidate_log_pt_eta_phi_rows(nu0, cartesian=self._cartesian)
        lp1_log, eta1, phi1 = _candidate_log_pt_eta_phi_rows(nu1, cartesian=self._cartesian)

        pt_p0, pt_c0, v0 = _project_pt_w_mass_slot(
            lep0_row,
            lp0_log,
            eta0,
            phi0,
            mw_gev=self._mw_gev,
            min_pt=self._min_pt,
            max_pt=self._max_pt,
        )
        pt_p1, pt_c1, v1 = _project_pt_w_mass_slot(
            lep1_row,
            lp1_log,
            eta1,
            phi1,
            mw_gev=self._mw_gev,
            min_pt=self._min_pt,
            max_pt=self._max_pt,
        )

        d0 = torch.log1p(pt_p0.clamp(min=0.0)) - torch.log1p(pt_c0.clamp(min=0.0))
        d1 = torch.log1p(pt_p1.clamp(min=0.0)) - torch.log1p(pt_c1.clamp(min=0.0))
        sq = d0.pow(2) + d1.pow(2)
        both_valid = valid_assign & v0 & v1
        denom = self._scale * self._scale + self._eps
        r_flat = torch.where(both_valid, -(sq / denom), torch.zeros_like(sq))

        out = apply_event_valid_to_rewards(r_flat.reshape(K, B), batch)

        valid_kb2 = torch.stack((v0, v1), dim=-1).reshape(K, B, 2).detach()
        delta_kb2 = torch.stack((d0, d1), dim=-1).reshape(K, B, 2).detach()
        self._last_reward_kb = out.detach()
        self._last_logpt_delta_kb2 = delta_kb2
        self._last_valid_kb2 = valid_kb2

        return out


class WMassTruthNormalizedReward(BaseReward):
    """TT2L W-mass reward: negative squared residual normalized by batch truth-W mean and std."""

    def __init__(self, *, cartesian: bool, eps: float = 1e-8) -> None:
        """Args:
            cartesian: Same as ``TruthGeneration.cartesian``: neutrino rows are Cartesian
                ``(p_x,p_y,p_z)`` or ``(log1p(pT), η, φ)``.
            eps: Added to ``truth_std`` in the denominator for numerical stability.
        """
        self._cartesian = bool(cartesian)
        self._eps = float(eps)
        self._last_reward_kb: Tensor | None = None
        self._last_mp_kb: Tensor | None = None
        self._last_mm_kb: Tensor | None = None
        self._last_truth_mean: float | None = None
        self._last_truth_std: float | None = None

    @property
    def name(self) -> str:
        return "w_mass"

    def last_reward_tensors(self) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Detached ``(reward_kb, m_plus_kb, m_minus_kb)`` from the last ``compute()``, if any."""
        return self._last_reward_kb, self._last_mp_kb, self._last_mm_kb

    def last_truth_normalization(self) -> tuple[float | None, float | None]:
        """``(truth_mean, truth_std)`` in GeV used in the last ``compute()``, if any."""
        return self._last_truth_mean, self._last_truth_std

    def _truth_nu_and_keys(
        self, batch: dict[str, Any], B: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, str]:
        if self._cartesian:
            key = "x_invisible_cartesian"
            if key not in batch:
                raise KeyError(
                    "WMassTruthNormalizedReward(cartesian=True) requires batch['x_invisible_cartesian']"
                )
        else:
            key = "x_invisible"
            if key not in batch:
                raise KeyError("WMassTruthNormalizedReward requires batch['x_invisible']")
        nu = batch[key]
        if nu.dim() != 3 or nu.shape[0] != B:
            raise ValueError(f"{key} must be (B, N_nu, F), got {tuple(nu.shape)}")
        return nu.to(device=device, dtype=dtype), key

    def _resolve_truth_mean_std(
        self,
        nu_truth_bb: Tensor,
        x_bb: Tensor,
        xm_bb: Tensor,
        assign_bb: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return scalar ``truth_mean``, ``truth_std`` tensors (GeV)."""
        _, mp_t, mm_t = tt2l_w_mass_per_row_and_means(
            nu_truth_bb,
            x_bb,
            xm_bb,
            assign_bb,
            cartesian=self._cartesian,
        )
        fin_p = torch.isfinite(mp_t)
        fin_m = torch.isfinite(mm_t)
        parts = []
        if bool(fin_p.any().item()):
            parts.append(mp_t[fin_p])
        if bool(fin_m.any().item()):
            parts.append(mm_t[fin_m])
        if not parts:
            tm = mp_t.new_tensor(float(MW_REF_GEV))
            ts = mp_t.new_tensor(10.0)
            return tm, ts
        truth_vals = torch.cat(parts, dim=0)
        tm = truth_vals.mean()
        if truth_vals.numel() < 2:
            ts = truth_vals.new_tensor(10.0)
        else:
            ts = truth_vals.std(correction=0).clamp(min=self._eps)
        return tm.to(dtype=dtype), ts.to(dtype=dtype)

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``(K, B)`` negative normalized squared W-mass residual per candidate."""
        del mask  # parity with ``BaseReward`` API
        K, B, N_nu, F = candidates.shape
        if N_nu < 2:
            raise ValueError(f"WMassTruthNormalizedReward needs N_nu >= 2, got {N_nu}")

        device = candidates.device
        dtype = candidates.dtype

        if "x" not in batch or "x_mask" not in batch:
            raise KeyError("WMassTruthNormalizedReward requires batch['x'] and batch['x_mask']")

        x_bb = batch["x"].to(device=device, dtype=dtype)
        xm_bb = batch["x_mask"].to(device=device, dtype=dtype)
        assign_bb = get_truth_assignment_indices(batch).to(device=device)

        nu_truth, _ = self._truth_nu_and_keys(batch, B, device, dtype)
        nu_truth_bb = nu_truth[:, :2, :].contiguous()

        truth_mean, truth_std = self._resolve_truth_mean_std(
            nu_truth_bb, x_bb, xm_bb, assign_bb, device=device, dtype=dtype
        )
        denom = truth_std + self._eps

        nu_kb = candidates[:, :, :2, :].reshape(K * B, 2, F).contiguous()
        x_rep = (
            x_bb.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, *x_bb.shape[1:])
        )
        xm_rep = (
            xm_bb.unsqueeze(0)
            .expand(K, -1, -1)
            .reshape(K * B, *xm_bb.shape[1:])
        )
        assign_rep = (
            assign_bb.unsqueeze(0)
            .expand(K, -1, -1, -1)
            .reshape(K * B, *assign_bb.shape[1:])
        )

        _lw, mp_flat, mm_flat = tt2l_w_mass_per_row_and_means(
            nu_kb,
            x_rep,
            xm_rep,
            assign_rep,
            cartesian=self._cartesian,
        )
        mp_kb = mp_flat.reshape(K, B)
        mm_kb = mm_flat.reshape(K, B)

        valid_mass = torch.isfinite(mp_kb) & torch.isfinite(mm_kb)
        r = -(
            ((mp_kb - truth_mean) / denom).pow(2)
            + ((mm_kb - truth_mean) / denom).pow(2)
        )
        r = torch.where(valid_mass, r, torch.zeros_like(r))

        out = apply_event_valid_to_rewards(r, batch)

        self._last_reward_kb = out.detach()
        self._last_mp_kb = mp_kb.detach()
        self._last_mm_kb = mm_kb.detach()
        self._last_truth_mean = float(truth_mean.detach().cpu())
        self._last_truth_std = float(truth_std.detach().cpu())

        return out


def compute_truth_l2_distances_kb(
    candidates: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
    mask: Tensor | None = None,
) -> Tensor:
    """Per-candidate masked L2 distance to truth neutrinos, same geometry as :class:`TruthDistanceReward`.

    Returns positive distances ``(K, B)``. Invalid events (see :func:`get_event_valid_mask`) are set to
    ``nan`` so downstream means can skip them with ``nanmean``.
    """
    K, B, N_nu, F = candidates.shape

    if mask is None:
        if "x_invisible_mask" in batch:
            mask = batch["x_invisible_mask"]
        else:
            mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

    if mask.dim() == 2:
        m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
    elif mask.dim() == 3 and mask.shape[-1] == 1:
        m = mask.to(dtype=candidates.dtype, device=candidates.device)
    else:
        raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

    m_cand = m.unsqueeze(0)

    if cartesian:
        if "x_invisible_cartesian" not in batch:
            raise KeyError("cartesian=True requires batch['x_invisible_cartesian']")
        truth = batch["x_invisible_cartesian"]
        if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
            raise ValueError(
                f"x_invisible_cartesian shape incompatible with candidates {tuple(candidates.shape)}"
            )
        truth_kin = truth * m
        cand_kin = candidates * m_cand
        diff = cand_kin - truth_kin.unsqueeze(0)
        sq = diff.pow(2).sum(dim=(2, 3))
    else:
        if "x_invisible" not in batch:
            raise KeyError("cartesian=False requires batch['x_invisible']")
        truth = batch["x_invisible"]
        if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
            raise ValueError(f"x_invisible shape incompatible with candidates {tuple(candidates.shape)}")
        if F < 3:
            raise ValueError(f"need F >= 3 for pt_eta_phi mode, got F={F}")
        truth_kin = truth * m
        cand_kin = candidates * m_cand
        truth_c = invisible_kinematics_to_cartesian(truth_kin)
        cand_c = invisible_kinematics_to_cartesian(
            cand_kin.reshape(K * B, N_nu, F)
        ).reshape(K, B, N_nu, 3)
        diff = cand_c - truth_c.unsqueeze(0)
        sq = diff.pow(2).sum(dim=(2, 3))

    dist = torch.sqrt(sq.clamp(min=0.0) + 1e-12)
    valid_1d = get_event_valid_mask(batch, B, dist.device, dist.dtype).unsqueeze(0).expand(K, -1)
    dist = torch.where(valid_1d > 0, dist, torch.full_like(dist, float("nan")))
    return dist


class RewardAggregator:
    """Weighted sum of several ``BaseReward`` sources."""

    def __init__(self) -> None:
        self.sources: list[tuple[BaseReward, float]] = []

    def add(self, reward: BaseReward, weight: float) -> None:
        """Register a reward with a scalar multiplier."""
        self.sources.append((reward, float(weight)))

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return ``(total_reward, breakdown)`` with each tensor ``(K, B)``.

        Per-source rewards already zero invalid events (via ``apply_event_valid_to_rewards`` in each
        ``BaseReward`` implementation).
        """
        if not self.sources:
            raise RuntimeError("RewardAggregator has no sources; call add() first.")
        total: Tensor | float = 0.0
        breakdown: dict[str, Tensor] = {}
        for reward, w in self.sources:
            r = reward.compute(candidates, batch, mask=mask)
            breakdown[reward.name] = r
            total = total + w * r
        if not isinstance(total, Tensor):
            raise RuntimeError("Aggregator produced non-tensor total.")
        return total, breakdown


__all__ = [
    "BaseReward",
    "ComponentNormalizedTruthDistanceReward",
    "LogPtTruthReward",
    "RelativeTruthReward",
    "WMassProjectionReward",
    "WMassTruthNormalizedReward",
    "NU_FEAT_DIM",
    "RewardAggregator",
    "TruthDistanceReward",
    "apply_event_valid_to_rewards",
    "compute_truth_l2_distances_kb",
    "get_event_valid_mask",
    "cartesian_to_log_pt_eta_phi",
    "invisible_kinematics_to_cartesian",
    "log_pt_eta_phi_to_cartesian",
]
