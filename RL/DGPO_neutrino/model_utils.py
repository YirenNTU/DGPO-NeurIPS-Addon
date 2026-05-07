"""Load EveNet for DGPO neutrino RL: checkpoint, frozen reference clone, EMA."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from evenet.control.global_config import Config
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.ema import EMA
from evenet.utilities.tool import safe_load_state

_log = logging.getLogger(__name__)


def build_evenet_model_from_training_config(
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
) -> EveNetModel:
    """Local builder mirroring ``scripts/engine.py:EveNetEngine.configure_model``.

    The slim ``evenet/`` core no longer ships this helper, so we reproduce the
    exact ``EveNetModel(...)`` wiring that the Lightning engine uses. Kept here
    (not in ``evenet/``) so the add-on stays self-contained against the slim
    ``evenet`` submodule.
    """
    cc = config.options.Training.Components
    return EveNetModel(
        config=config,
        device=device,
        classification=cc.Classification.include,
        regression=cc.Regression.include,
        global_generation=cc.GlobalGeneration.include,
        point_cloud_generation=cc.ReconGeneration.include,
        neutrino_generation=cc.TruthGeneration.include,
        assignment=cc.Assignment.include,
        segmentation=cc.Segmentation.include,
        normalization_dict=normalization_dict,
    )


@dataclass(frozen=True)
class EvenetForDGPO:
    """Artifacts returned by :func:`load_evenet_model_for_dgpo`."""

    model: EveNetModel
    config: Config
    normalization_dict: dict[str, Any]
    checkpoint_path: Path | None


def load_training_config(config_path: str | Path) -> Config:
    """Load merged EveNet YAML (including ``dgpo`` / ``reward_config``) into a fresh :class:`Config`."""
    path = Path(config_path).resolve()
    cfg = Config()
    cfg.load_yaml(path)
    return cfg


def resolve_normalization_file_path(
    config: Config, *, config_yaml_path: str | Path | None = None
) -> Path:
    """Resolve ``options.Dataset.normalization_file`` to an absolute :class:`Path`.

    Relative paths are interpreted relative to the directory containing the merged
    training YAML (``config.yaml``), matching the comment in ``RL/DGPO_neutrino/config.yaml``.

    **Why this exists:** upstream Ray Train defaults ``RAY_CHDIR_TO_TRIAL_DIR`` to ``1``,
    which moves each worker's cwd into a per-trial directory (often under ``/tmp``), so a
    relative ``normalization_file`` no longer resolves next to the checkout. EveNet-private
    site jobs usually set ``RAY_CHDIR_TO_TRIAL_DIR=0``; :func:`~RL.DGPO_neutrino.dgpo_trainer.main`
    does the same before ``ray.init``. Resolving against ``config_yaml_path`` remains the
    robust fix when cwd is not ``RL/DGPO_neutrino/`` (e.g. launch from repo root).
    """
    raw = config.options.Dataset.normalization_file
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_yaml_path is None:
        raise ValueError(
            "options.Dataset.normalization_file is relative (%r) but config_yaml_path was not "
            "passed. Under Ray Train, cwd may be a per-trial directory (see "
            "RAY_CHDIR_TO_TRIAL_DIR); pass config_yaml_path= to resolve next to config.yaml."
            % (raw,)
        )
    base = Path(config_yaml_path).expanduser().resolve().parent
    return (base / path).resolve()


def load_normalization_dict(
    config: Config, *, config_yaml_path: str | Path | None = None
) -> dict[str, Any]:
    """Load ``options.Dataset.normalization_file`` (same as ``dgpo_sanity_check.load_normalization_dict``)."""
    path = resolve_normalization_file_path(config, config_yaml_path=config_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"normalization_file not found: {path} (YAML value={config.options.Dataset.normalization_file!r})"
        )
    normalization_dict: dict[str, Any] = torch.load(str(path), weights_only=False)
    _log.info("[DGPO/model] normalization_file=%s", path)
    return normalization_dict


def resolve_checkpoint_path(
    config: Config,
    checkpoint_path: str | Path | None,
) -> Path | None:
    """Prefer explicit path; else YAML paths in EveNet order (resume ckpt before pretrain-only)."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        return p if p.is_file() else None
    tr = config.options.Training
    for key in ("model_checkpoint_load_path", "pretrain_model_load_path"):
        raw = getattr(tr, key, None)
        if not raw:
            continue
        p = Path(str(raw)).expanduser().resolve()
        if p.is_file():
            return p
    return None


def load_weights_like_configure_model(
    model: EveNetModel,
    ckpt_path: Path,
    device: torch.device,
    config: Config,
    *,
    for_dgpo_training: bool = False,
) -> dict[str, Any]:
    """Load Lightning checkpoint: respect EMA replace flags like ``EveNetEngine.configure_model``.

    When ``for_dgpo_training`` is True the EMA ``replace_model_after_load`` flag is **ignored**:
    DGPO resumes from ``state_dict`` and restores the EMA shadow separately via :func:`make_ema`.
    For DGPO checkpoints saved by current code, ``state_dict`` is the live trainable model.
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    ema_enable = bool(ema_cfg.get("enable", False))
    ema_replace = bool(ema_cfg.get("replace_model_after_load", False))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    is_dgpo_ckpt = int(ckpt.get("dgpo_checkpoint_version", 0)) >= 1

    if for_dgpo_training and is_dgpo_ckpt:
        safe_load_state(model, ckpt["state_dict"])
    elif ema_enable and "ema_state_dict" in ckpt and ema_replace:
        safe_load_state(model, ckpt["ema_state_dict"])
    else:
        safe_load_state(model, ckpt["state_dict"])
    return ckpt


def freeze_reference_model(model: nn.Module) -> None:
    """``eval()`` and disable gradients (reference policy in DGPO)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _debug_verify_component_freeze(
    model: EveNetModel, logical_name: str, freeze_cfg: Any
) -> None:
    """Emit debug logs when YAML freeze did not take effect as expected."""
    ftype = freeze_cfg.get("type", "none")
    if ftype == "none":
        return

    head = getattr(model, logical_name, None)
    if head is None:
        _log.debug(
            "[DGPO/model] freeze: component %r missing on model (YAML type=%r); no parameters frozen.",
            logical_name,
            ftype,
        )
        return

    if ftype == "full":
        n_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
        if n_train:
            _log.debug(
                "[DGPO/model] freeze: %r declared type=full but %s parameters still require_grad.",
                logical_name,
                f"{n_train:,}",
            )
        return

    if ftype == "partial":
        components = freeze_cfg.get("partial_freeze_components", None) or []
        if not components:
            _log.debug(
                "[DGPO/model] freeze: %r type=partial but partial_freeze_components is empty.",
                logical_name,
            )
            return
        named = dict(head.named_modules())
        unknown = [c for c in components if c not in named]
        if unknown:
            _log.debug(
                "[DGPO/model] freeze: %r partial_freeze_components not found on module: %s",
                logical_name,
                unknown,
            )
        for c in components:
            if c not in named:
                continue
            sub = named[c]
            n_sub = sum(p.numel() for p in sub.parameters() if p.requires_grad)
            if n_sub:
                _log.debug(
                    "[DGPO/model] freeze: %r partial subtree %r still has %s trainable parameters.",
                    logical_name,
                    c,
                    f"{n_sub:,}",
                )
        return

    if ftype == "random":
        params = list(head.parameters())
        if not params:
            _log.debug(
                "[DGPO/model] freeze: %r type=random but submodule has no parameters.",
                logical_name,
            )


def apply_component_freezes(model: EveNetModel, config: Config) -> None:
    """
    Apply ``options.Training.Components.<Name>.freeze`` via :meth:`EveNetModel.freeze_module`
    (same contract as Lightning ``EveNetEngine.configure_model``).

    Component names in YAML (``GlobalEmbedding``, ``PET``, ``TruthGeneration``, etc.) match
    :class:`EveNetModel` attributes used by ``freeze_module``.
    """
    cc = config.options.Training.Components
    applied: list[str] = []
    for name in cc:
        freeze_cfg = cc[name].get("freeze", None)
        if freeze_cfg is None:
            continue
        model.freeze_module(name, freeze_cfg)
        _debug_verify_component_freeze(model, name, freeze_cfg)
        ftype = freeze_cfg.get("type", "none")
        if ftype != "none":
            applied.append(f"{name}({ftype})")

    n_train = count_trainable_params(model)
    _log.info(
        "[DGPO/model] Component freeze from YAML: %s | trainable params: %s",
        applied if applied else "(none)",
        f"{n_train:,}",
    )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_reference_model_frozen(model_ref: EveNetModel, *, where: str) -> None:
    """Hard guard: reference must have no trainable parameters."""
    if count_trainable_params(model_ref) != 0:
        raise RuntimeError(f"[{where}] model_ref expected 0 trainable params.")
    n_grad = sum(1 for p in model_ref.parameters() if p.grad is not None)
    if n_grad != 0:
        raise RuntimeError(f"[{where}] model_ref has gradients on {n_grad} tensors.")


def build_evenet_on_device(
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
) -> EveNetModel:
    """Instantiate ``EveNetModel`` and move modules/buffers to ``device``."""
    model = build_evenet_model_from_training_config(config, normalization_dict, device)
    return model.to(device)


def load_evenet_model_for_dgpo(
    config_path: str | Path | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    *,
    config: Config | None = None,
    config_yaml_path: str | Path | None = None,
) -> EvenetForDGPO:
    """Load config, normalization, build ``EveNetModel``, optionally load checkpoint weights.

    Pass either ``config_path`` or a pre-populated ``config`` (e.g. ``global_config`` after
    ``load_yaml``) so dataset prep and model loading share the same merged YAML.

    When ``normalization_file`` in YAML is a **relative** path, pass ``config_yaml_path`` to the
    merged ``config.yaml`` so it resolves next to that file (required under Ray Train, where cwd
    is not the repo).

    Checkpoint selection matches ``RL/DGPO_sanity/dgpo_sanity_check.py`` (EMA swap when configured).
    """
    yaml_anchor: Path | None = None
    if config is None:
        if config_path is None:
            raise ValueError("load_evenet_model_for_dgpo requires config_path or config=")
        yaml_anchor = Path(config_path).expanduser().resolve()
        config = load_training_config(yaml_anchor)
    elif config_yaml_path is not None:
        yaml_anchor = Path(config_yaml_path).expanduser().resolve()
    elif config_path is not None:
        yaml_anchor = Path(config_path).expanduser().resolve()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization_dict = load_normalization_dict(config, config_yaml_path=yaml_anchor)
    model = build_evenet_on_device(config, normalization_dict, device)

    resolved = resolve_checkpoint_path(config, checkpoint_path)
    if resolved is not None:
        _log.info("[DGPO/model] Loading weights from %s", resolved)
        load_weights_like_configure_model(model, resolved, device, config, for_dgpo_training=True)
    else:
        _log.warning(
            "[DGPO/model] No checkpoint (pass checkpoint_path= or set model_checkpoint_load_path / "
            "pretrain_model_load_path); model is randomly initialized."
        )

    return EvenetForDGPO(
        model=model,
        config=config,
        normalization_dict=normalization_dict,
        checkpoint_path=resolved,
    )


def make_reference_model(
    current_model: EveNetModel,
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
    checkpoint: dict[str, Any] | None = None,
) -> EveNetModel:
    """Frozen policy reference for the DGPO objective.

    On **first run** (no DGPO checkpoint, or supervised-only ckpt): ``ref_model`` gets the same
    weights as ``current_model`` (= the pretrained init). This is correct because both start equal.

    On **resume** from a DGPO checkpoint that contains ``dgpo_ref_state_dict``: the original
    reference weights are restored so the anchor stays fixed across sessions.

    Uses rebuild + :meth:`load_state_dict` because ``EveNetModel`` is not ``deepcopy``-safe
    (see ``make_model_ref_match_cur`` in ``RL/DGPO_sanity/dgpo_sanity_check.py``).
    """
    model_ref = build_evenet_on_device(config, normalization_dict, device)
    if (
        checkpoint is not None
        and int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1
        and "dgpo_ref_state_dict" in checkpoint
    ):
        safe_load_state(model_ref, checkpoint["dgpo_ref_state_dict"])
        _log.info("[DGPO/model] Loaded ref_model from dgpo_ref_state_dict (fixed anchor).")
    else:
        model_ref.load_state_dict(current_model.state_dict())
        _log.info("[DGPO/model] Initialized ref_model from current model weights (first run).")
    freeze_reference_model(model_ref)
    assert_reference_model_frozen(model_ref, where="make_reference_model")
    return model_ref


def make_ema(
    model: EveNetModel,
    config: Config,
    checkpoint: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> EMA | None:
    """Build :class:`~evenet.utilities.ema.EMA` and optionally load ``ema_state_dict`` from a Lightning ckpt.

    Returns ``None`` when ``options.Training.EMA.enable`` is false (same gating as :class:`~evenet.engine.EveNetEngine`).
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    if not bool(ema_cfg.get("enable", False)):
        return None
    decay = float(ema_cfg.get("decay", 0.999))
    ema = EMA(model, decay=decay)
    if checkpoint is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
    return ema


def make_ema_rollout(model: EveNetModel, config: Config) -> EMA | None:
    """Build a second EMA used only for Phase-1 rollout; decay is set each step via ``update(..., decay_=...)``.

    Not loaded from or written to checkpoints; re-initialized from current trainable weights on resume.
    Returns ``None`` when ``options.Training.EMA.enable`` is false.
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    if not bool(ema_cfg.get("enable", False)):
        return None
    return EMA(model, decay=0.0)


def build_lightning_compatible_checkpoint(
    model: nn.Module,
    ema: EMA | None,
    config: Config,
) -> dict[str, Any]:
    """Build a Lightning-style DGPO checkpoint payload.

    ``state_dict`` uses the live trainable model weights so DGPO resume matches the optimizer
    state. ``ema_state_dict`` separately holds the save-EMA shadow when EMA is enabled.
    """
    orig_model = model
    if isinstance(orig_model, nn.parallel.DistributedDataParallel):
        orig_model = orig_model.module
    _inner = getattr(orig_model, "eve_net", None)
    if isinstance(_inner, nn.Module):
        orig_model = _inner
    if hasattr(orig_model, "_orig_mod"):
        orig_model = orig_model._orig_mod
    ema_cfg = config.options.Training.get("EMA", None) or {}
    ema_enabled = bool(ema_cfg.get("enable", False))

    checkpoint: dict[str, Any] = {}
    checkpoint["state_dict"] = {f"model.{k}": v for k, v in orig_model.state_dict().items()}

    if ema_enabled and ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
        _log.info("[DGPO/model] Saved live state_dict plus separate ema_state_dict.")

    return checkpoint


def save_lightning_compatible_checkpoint(
    path: Path | str,
    model: nn.Module,
    ema: EMA | None,
    config: Config,
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
    optimizer: Optimizer | None = None,
    ref_model: nn.Module | None = None,
    dgpo_offset_anchor_mu_ref: float | None = None,
    dgpo_offset_anchor_mu_ref_xyz: torch.Tensor | None = None,
    dgpo_local_kl_anchor_p0_ref: float | None = None,
    dgpo_rrkl_ckpt_blob: dict[str, Any] | None = None,
    dgpo_rrkl_ckpt_key: str | None = None,
    dgpo_offset_anchor_dual_pt: float | None = None,
    dgpo_offset_anchor_z_ema_pt: float | None = None,
) -> None:
    """Write a ``.ckpt`` file using the same tensor layout as Lightning + EveNetEngine.

    ``last_completed_epoch`` is the last fully finished training epoch index (0-based).
    ``dgpo_next_epoch`` is the next epoch index the loop should run (equals
    ``last_completed_epoch + 1`` after a full epoch; can equal ``last_completed_epoch`` when
    saving mid-epoch, e.g. ``--max-steps`` interrupt).

    ``ref_model`` — frozen reference policy. Its ``state_dict`` is saved as
    ``dgpo_ref_state_dict`` so the anchor survives across resume sessions.
    """
    out_path = Path(path).expanduser().resolve()
    payload = build_lightning_compatible_checkpoint(model, ema, config)
    payload["epoch"] = int(last_completed_epoch)
    payload["global_step"] = int(global_step)
    payload["dgpo_checkpoint_version"] = 1
    try:
        import lightning
        payload["pytorch-lightning_version"] = lightning.__version__
    except Exception:
        payload["pytorch-lightning_version"] = "2.0.0"
    payload["dgpo_next_epoch"] = int(dgpo_next_epoch)
    if optimizer is not None:
        payload["dgpo_optimizer_state_dict"] = optimizer.state_dict()
    if ref_model is not None:
        orig_ref = ref_model
        if isinstance(orig_ref, nn.parallel.DistributedDataParallel):
            orig_ref = orig_ref.module
        _ir = getattr(orig_ref, "eve_net", None)
        if isinstance(_ir, nn.Module):
            orig_ref = _ir
        if hasattr(orig_ref, "_orig_mod"):
            orig_ref = orig_ref._orig_mod
        payload["dgpo_ref_state_dict"] = {
            f"model.{k}": v for k, v in orig_ref.state_dict().items()
        }
    if dgpo_offset_anchor_mu_ref is not None and math.isfinite(
        float(dgpo_offset_anchor_mu_ref)
    ):
        payload["dgpo_offset_anchor_mu_ref"] = float(dgpo_offset_anchor_mu_ref)
    if (
        dgpo_offset_anchor_mu_ref_xyz is not None
        and isinstance(dgpo_offset_anchor_mu_ref_xyz, torch.Tensor)
        and int(dgpo_offset_anchor_mu_ref_xyz.numel()) == 3
        and torch.isfinite(dgpo_offset_anchor_mu_ref_xyz.flatten()[:3]).all()
    ):
        payload["dgpo_offset_anchor_mu_ref_xyz"] = (
            dgpo_offset_anchor_mu_ref_xyz.detach().cpu().flatten()[:3].to(torch.float32).clone()
        )
    if dgpo_local_kl_anchor_p0_ref is not None and math.isfinite(
        float(dgpo_local_kl_anchor_p0_ref)
    ):
        payload["dgpo_local_kl_anchor_p0_ref"] = float(dgpo_local_kl_anchor_p0_ref)
    if (
        dgpo_rrkl_ckpt_blob is not None
        and dgpo_rrkl_ckpt_key is not None
        and str(dgpo_rrkl_ckpt_key).strip() != ""
    ):
        payload[str(dgpo_rrkl_ckpt_key)] = dgpo_rrkl_ckpt_blob
    if dgpo_offset_anchor_dual_pt is not None and math.isfinite(float(dgpo_offset_anchor_dual_pt)):
        payload["dgpo_offset_anchor_dual_pt"] = float(dgpo_offset_anchor_dual_pt)
    if dgpo_offset_anchor_z_ema_pt is not None and math.isfinite(
        float(dgpo_offset_anchor_z_ema_pt)
    ):
        payload["dgpo_offset_anchor_z_ema_pt"] = float(dgpo_offset_anchor_z_ema_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    _log.info(
        "[DGPO/model] Wrote checkpoint %s (next_epoch=%s step=%s)",
        out_path,
        dgpo_next_epoch,
        global_step,
    )


def is_lightning_trainer_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Return True if ``checkpoint`` looks like a full PyTorch Lightning trainer state (not DGPO-only)."""
    if int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1:
        return False
    keys = set(checkpoint.keys())
    return bool(
        "pytorch-lightning_version" in keys
        or "optimizer_states" in keys
        or "lr_schedulers" in keys
    )


def parse_dgpo_resume_from_checkpoint(checkpoint: dict[str, Any] | None) -> tuple[int, int]:
    """Return ``(start_epoch, global_step)`` for the DGPO training loop.

    Loads weights separately via :func:`load_evenet_model_for_dgpo`. Here we only interpret
    scheduling counters so supervised Lightning ckpts do not accidentally set a huge start epoch:
    those are detected via :func:`is_lightning_trainer_checkpoint` and reset to ``(0, 0)``.
    """
    if not checkpoint:
        return 0, 0
    gs = int(checkpoint.get("global_step", 0))
    if int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1:
        if "dgpo_next_epoch" not in checkpoint:
            _log.warning(
                "[DGPO/model] Checkpoint has dgpo_checkpoint_version but no dgpo_next_epoch; "
                "starting from epoch 0."
            )
            return 0, gs
        return int(checkpoint["dgpo_next_epoch"]), gs
    if is_lightning_trainer_checkpoint(checkpoint):
        return 0, 0
    if "dgpo_next_epoch" in checkpoint:
        return int(checkpoint["dgpo_next_epoch"]), gs
    ep = int(checkpoint.get("epoch", -1))
    return ep + 1, gs


