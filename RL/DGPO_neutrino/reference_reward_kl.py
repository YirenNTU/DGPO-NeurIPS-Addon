"""Frozen per-event reward / KL multipliers (LUT) for DGPO.

Minimal implementation restored so ``dgpo_trainer`` imports resolve. Full LUT construction from
training shards is only stubbed; enable ``dgpo.reference_reward_kl`` in YAML only after extending
:func:`run_reference_reward_kl_training_baseline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_log = logging.getLogger(__name__)


@dataclass
class ReferenceRewardKlTrainConfig:
    enabled: bool = False
    baseline_K: int = 10
    checkpoint_key: str = "dgpo_reference_reward_kl"
    eps: float = 1e-8
    weight_scale: float = 1.0
    weight_mode: str = "gaussian"
    sigma: float = 1.0
    base_weight: float = 1.0
    require_event_key: bool = False
    synthetic_event_key_if_missing: bool = True


def resolve_reference_reward_kl_train_config(dg: Any) -> ReferenceRewardKlTrainConfig:
    if dg is None:
        return ReferenceRewardKlTrainConfig()
    block = dg.get("reference_reward_kl") if hasattr(dg, "get") else getattr(dg, "reference_reward_kl", None)
    if block is None:
        return ReferenceRewardKlTrainConfig()
    g = block.get if hasattr(block, "get") else lambda k, d=None: getattr(block, k, d)
    return ReferenceRewardKlTrainConfig(
        enabled=bool(g("enabled", False)),
        baseline_K=int(g("baseline_K", 10)),
        checkpoint_key=str(g("checkpoint_key", "dgpo_reference_reward_kl")),
        eps=float(g("eps", 1e-8)),
        weight_scale=float(g("weight_scale", 1.0)),
        weight_mode=str(g("weight_mode", "gaussian")),
        sigma=float(g("sigma", 1.0)),
        base_weight=float(g("base_weight", 1.0)),
        require_event_key=bool(g("require_event_key", False)),
        synthetic_event_key_if_missing=bool(g("synthetic_event_key_if_missing", True)),
    )


class ReferenceRewardKlStore:
    """Maps ``(key0, key1)`` int pair → ``(reference_reward, kl_weight)``."""

    def __init__(self, lut: dict[tuple[int, int], tuple[float, float]] | None = None) -> None:
        self.lut = lut or {}

    def __len__(self) -> int:
        return len(self.lut)

    def with_weight_params(
        self,
        *,
        eps: float,
        weight_scale: float,
        weight_mode: str,
        sigma: float,
        base_weight: float,
    ) -> ReferenceRewardKlStore:
        del eps, weight_scale, weight_mode, sigma, base_weight
        return self

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> ReferenceRewardKlStore:
        try:
            k0 = payload.get("event_keys_k0")
            k1 = payload.get("event_keys_k1")
            rr = payload.get("ref_rewards")
            ww = payload.get("kl_weights")
            if k0 is None or k1 is None or rr is None or ww is None:
                return cls({})
            k0t = k0.detach().cpu().long().reshape(-1)
            k1t = k1.detach().cpu().long().reshape(-1)
            rrf = rr.detach().cpu().float().reshape(-1)
            wwf = ww.detach().cpu().float().reshape(-1)
            n = min(k0t.numel(), k1t.numel(), rrf.numel(), wwf.numel())
            lut: dict[tuple[int, int], tuple[float, float]] = {}
            for i in range(n):
                lut[(int(k0t[i].item()), int(k1t[i].item()))] = (
                    float(rrf[i].item()),
                    float(wwf[i].item()),
                )
            return cls(lut)
        except Exception as ex:
            _log.warning("[reference_reward_kl] from_checkpoint_payload failed: %s", ex)
            return cls({})

    def checkpoint_payload(self, ckpt_key_unused: str = "") -> dict[str, Tensor]:
        del ckpt_key_unused
        if not self.lut:
            return {}
        k0 = [k[0] for k in self.lut]
        k1 = [k[1] for k in self.lut]
        rr = [v[0] for v in self.lut.values()]
        ww = [v[1] for v in self.lut.values()]
        return {
            "event_keys_k0": torch.tensor(k0, dtype=torch.long),
            "event_keys_k1": torch.tensor(k1, dtype=torch.long),
            "ref_rewards": torch.tensor(rr, dtype=torch.float32),
            "kl_weights": torch.tensor(ww, dtype=torch.float32),
        }

    def lookup_weights(
        self,
        ek0: Tensor,
        ek1: Tensor,
        valid_b: Tensor,
    ) -> tuple[Tensor, float]:
        B = int(ek0.shape[0])
        device = ek0.device
        dtype = torch.float32
        out = torch.ones(B, device=device, dtype=dtype)
        miss = 0
        vb = valid_b.reshape(-1) > 0
        for i in range(B):
            if i < vb.numel() and not bool(vb[i].item()):
                continue
            key = (int(ek0[i].detach().cpu()), int(ek1[i].detach().cpu()))
            if key in self.lut:
                out[i] = float(self.lut[key][1])
            else:
                miss += 1
        miss_frac = float(miss) / float(max(B, 1))
        return out, miss_frac

    def lookup_reference_rewards(self, ek0: Tensor, ek1: Tensor) -> Tensor:
        B = int(ek0.shape[0])
        device = ek0.device
        out = torch.full((B,), float("nan"), device=device, dtype=torch.float32)
        for i in range(B):
            key = (int(ek0[i].detach().cpu()), int(ek1[i].detach().cpu()))
            if key in self.lut:
                out[i] = float(self.lut[key][0])
        return out


def multiply_event_and_row_kl_weights(
    row_weights: Tensor | None,
    kl_weights: Tensor | None,
) -> Tensor | None:
    if row_weights is None:
        return kl_weights
    if kl_weights is None:
        return row_weights
    rw = row_weights.to(device=kl_weights.device, dtype=kl_weights.dtype)
    return rw * kl_weights


def event_key_pair_columns_from_batch(
    batch: dict[str, Any],
    rrkl_cfg: ReferenceRewardKlTrainConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, float]:
    """Return two int64 keys per row; synthetic indices if event columns are absent."""
    del rrkl_cfg
    B = int(batch["x"].shape[0])
    idx = torch.arange(B, device=device, dtype=torch.long)
    return idx, idx + 713, 0.0


def run_reference_reward_kl_training_baseline(
    *,
    train_shard: Any,
    ref_model: Any,
    sampler: Any,
    normalization_dict: Any,
    rrkl_cfg: ReferenceRewardKlTrainConfig,
    num_ddim_steps: int,
    batch_size: int,
    prefetch: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
    is_rank0: bool,
) -> ReferenceRewardKlStore:
    del (
        train_shard,
        ref_model,
        sampler,
        normalization_dict,
        num_ddim_steps,
        batch_size,
        prefetch,
        rank,
        world_size,
        device,
        dtype,
    )
    if rrkl_cfg.enabled and is_rank0:
        _log.warning(
            "[reference_reward_kl] LUT build is not implemented in this minimal stub; "
            "training continues with an empty LUT (reference_reward_kl has no effect)."
        )
    return ReferenceRewardKlStore({})
