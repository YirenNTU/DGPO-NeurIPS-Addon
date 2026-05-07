#!/usr/bin/env python3
r"""NeurIPS paper: self-contained overlay-only unfolding & reconstruction.

This file vendors the TT2L multi-model unfolding helpers directly so it can be
run as a single script. It produces only the overlay PNGs for the NeurIPS paper
prediction stacks.

Usage (from the DGPO-NeurIPS-Addon repo root):

  python RL/Unfolding/run_neurips_neutrino_unfolding.py \\
      --model nominal=/path/to/nominal.pt \\
      --model ablation=/path/to/ablation.pt \\
      --output_dir outputs/unfolding_neutrino

  SVD panels require ROOT/PyROOT plus a built RooUnfold library. Build per
  docs/roounfold.md, then source RooUnfold/build/setup.sh in each new shell.
  Optional: set ROOUNFOLD_LIB_PATH to the built lib prefix (no extension).
"""

from __future__ import annotations


import argparse
import os
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import vector

vector.register_awkward()

_HERE = Path(__file__).resolve()
# Layout: ``parents[2]`` = ``DGPO-NeurIPS-Addon/`` (this repo, holds ``RL.*``,
# ``event_selection.*``, ``shared.*``); EveNet core lives in the nested
# ``EveNet-Full/`` companion checkout next to it (provides ``evenet.*`` /
# ``preprocessing.*``). See top-level README "Install" for the clone recipe.
_REPO_ROOT = _HERE.parents[2]
_EVENET_ROOT = _REPO_ROOT / "EveNet-Full"
for _p in (_REPO_ROOT, _EVENET_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from event_selection.event_cuts import (
    POINTCLOUD_KEY,
    compute_event_mask,
    get_value,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prediction paths: use CLI --model KEY=PATH (--model repeatable).
# ---------------------------------------------------------------------------
BASELINE_MODEL_KEY = "baseline"

MODEL_STYLES: Dict[str, Dict[str, str]] = {
    # tex_short: used in ratio-axis label (must be LaTeX-safe, no spaces)
    "E2E_pretrain": {"color": "blue", "label": "E2E pretrain", "tex_short": "E2E"},
    "DGPO_RL": {"color": "red", "label": "DGPO RL", "tex_short": "DGPO"},
    "E2E_scratch": {"color": "green", "label": "E2E scratch", "tex_short": "E2E-scratch"},
    # NeurIPS ablation checkpoints (short keys match common filenames)
    "nominal_anchor": {"color": "C0", "label": "Nominal anchor", "tex_short": "nom."},
    "nominal": {"color": "C0", "label": "Nominal", "tex_short": "nom."},
    "ablation_no_anchor": {"color": "C1", "label": "No anchor", "tex_short": "no-anch."},
    "ablation_scratch": {"color": "C2", "label": "From scratch", "tex_short": "scratch"},
    "ablation_5_candidates": {"color": "C3", "label": "5 candidates", "tex_short": "5cand."},
    "nu2flow": {"color": "C4", "label": "nu2flow", "tex_short": "n2f"},
}

# Campaign tag prepended to all figure suptitles.
FIGURE_SUPTITLE_PREFIX = "neutrino_TT2L"


def _analysis_suptitle(body: str) -> str:
    """Return suptitle string with the analysis campaign prefix."""
    return f"{FIGURE_SUPTITLE_PREFIX} — {body}"

KINEMATICS = ["log_pt", "eta", "phi"]
ASSIGN_MASK_PATH = "assignment_target_mask/TT2L"
# Ground-truth b/ℓ indices per top (same as top_reconstruction.py)
ASSIGN_TARGET_PATH = "assignment_target/TT2L"
# Predicted assignment (process key in prediction batches)
ASSIGN_PRED_PROCESS = "TT2L"

# Truth-assigned reco objects saved by the nu2flow / TT2L prediction configs.
EXTRA_RECO_OBJECT_KEYS: Dict[str, Dict[str, str]] = {
    "t1_b": {
        "pt": "EXTRA/t1/b/pt",
        "eta": "EXTRA/t1/b/eta",
        "phi": "EXTRA/t1/b/phi",
        "energy": "EXTRA/t1/b/energy",
    },
    "t1_l": {
        "pt": "EXTRA/t1/l/pt",
        "eta": "EXTRA/t1/l/eta",
        "phi": "EXTRA/t1/l/phi",
        "energy": "EXTRA/t1/l/energy",
    },
    "t2_b": {
        "pt": "EXTRA/t2/b/pt",
        "eta": "EXTRA/t2/b/eta",
        "phi": "EXTRA/t2/b/phi",
        "energy": "EXTRA/t2/b/energy",
    },
    "t2_l": {
        "pt": "EXTRA/t2/l/pt",
        "eta": "EXTRA/t2/l/eta",
        "phi": "EXTRA/t2/l/phi",
        "energy": "EXTRA/t2/l/energy",
    },
}
EXTRA_TRUTH_TOP_KEYS: Dict[str, Dict[str, str]] = {
    "t1": {
        "pt": "EXTRA/truth_t1/t/pt",
        "eta": "EXTRA/truth_t1/t/eta",
        "phi": "EXTRA/truth_t1/t/phi",
        "mass": "EXTRA/truth_t1/t/mass",
    },
    "t2": {
        "pt": "EXTRA/truth_t2/t/pt",
        "eta": "EXTRA/truth_t2/t/eta",
        "phi": "EXTRA/truth_t2/t/phi",
        "mass": "EXTRA/truth_t2/t/mass",
    },
}

# Reconstructed-top histogram ranges (both tops merged: 2 entries / event)
TOP_KINEMATICS_PLOT = {
    "pt": {"range": (0.0, 450.0), "bins": 50, "label": r"$p_T^{t,\mathrm{reco}}$", "unit": "GeV"},
    "eta": {"range": (-4.0, 4.0), "bins": 40, "label": r"$\eta^{t,\mathrm{reco}}$", "unit": ""},
    "phi": {
        "range": (-np.pi, np.pi),
        "bins": 32,
        "label": r"$\phi^{t,\mathrm{reco}}$",
        "unit": "rad",
    },
    "mass": {"range": (100.0, 240.0), "bins": 50, "label": r"$m^{t,\mathrm{reco}}$", "unit": "GeV"},
}

# Reconstructed W (ℓ+ν): both W merged (2 entries / event), calibration MW convention
W_KINEMATICS_PLOT = {
    "pt": {"range": (0.0, 200.0), "bins": 50, "label": r"$p_T^{W,\mathrm{reco}}$", "unit": "GeV"},
    "eta": {"range": (-4.0, 4.0), "bins": 40, "label": r"$\eta^{W,\mathrm{reco}}$", "unit": ""},
    "phi": {
        "range": (-np.pi, np.pi),
        "bins": 32,
        "label": r"$\phi^{W,\mathrm{reco}}$",
        "unit": "rad",
    },
    "mass": {"range": (40.0, 120.0), "bins": 50, "label": r"$m^{W,\mathrm{reco}}$", "unit": "GeV"},
}

BINNING_CONFIG = {
    # Merged neutrinos (used by plot_neutrino_kinematics_overlay).
    "pt": {"range": (0, 150), "num_bins": 30, "label": r"$p_T^{\nu}$", "unit": "GeV"},
    "eta": {"range": (-4, 4), "num_bins": 20, "label": r"$\eta^{\nu}$", "unit": ""},
    "phi": {"range": (-np.pi, np.pi), "num_bins": 16, "label": r"$\phi^{\nu}$", "unit": "rad"},
    # Split per charge for unfolding: nu (from t, paired with l+).
    "pt_nu":  {"range": (0, 150), "num_bins": 30, "label": r"$p_T^{\nu}$ (from $t$)",  "unit": "GeV"},
    "eta_nu": {"range": (-4, 4),  "num_bins": 20, "label": r"$\eta^{\nu}$ (from $t$)", "unit": ""},
    "phi_nu": {"range": (-np.pi, np.pi), "num_bins": 16, "label": r"$\phi^{\nu}$ (from $t$)", "unit": "rad"},
    # Split per charge for unfolding: nubar (from t-bar, paired with l-).
    "pt_nubar":  {"range": (0, 150), "num_bins": 30, "label": r"$p_T^{\bar\nu}$ (from $\bar t$)",  "unit": "GeV"},
    "eta_nubar": {"range": (-4, 4),  "num_bins": 20, "label": r"$\eta^{\bar\nu}$ (from $\bar t$)", "unit": ""},
    "phi_nubar": {"range": (-np.pi, np.pi), "num_bins": 16, "label": r"$\phi^{\bar\nu}$ (from $\bar t$)", "unit": "rad"},
}

# Six unfolding plots per model: pt/eta/phi x {nu (from t), nubar (from t-bar)}.
# Charge ordering follows EVENT.TT2L event_info (t1 = top with l+, t2 = anti-top with l-),
# so neutrino slot 0 = nu, slot 1 = nubar.
FEATURES_TO_ANALYZE = [
    "pt_nu",  "eta_nu",  "phi_nu",
    "pt_nubar", "eta_nubar", "phi_nubar",
]
SVD_KREG = 5
EPSILON = 1e-10

# ---------------------------------------------------------------------------
# RooUnfold: load via ROOUNFOLD_LIB_PATH, then standard repo-relative builds.
# ---------------------------------------------------------------------------
def _roounfold_candidates() -> List[str]:
    """Return loader paths passed to ROOT.gSystem.Load without file extension.

    Search order:
      1. ``$ROOUNFOLD_LIB_PATH`` (explicit override; no extension)
      2. RooUnfold built directly under the ``DGPO-NeurIPS-Addon/`` repo root —
         this is the layout the README's Step 1c / Step 4 recipe produces.
      3. Same paths under ``DGPO-NeurIPS-Addon/EveNet-Full/`` (fallback for
         older layouts that nested RooUnfold inside the EveNet checkout).
      4. System-wide ``libRooUnfold``.
    """
    extra = os.environ.get("ROOUNFOLD_LIB_PATH", "").strip()
    repo_root = Path(__file__).resolve().parents[2]
    evenet_root = repo_root / "EveNet-Full"
    cands: List[str] = []
    for base in (repo_root, evenet_root):
        cands.extend(
            [
                str(base / "external" / "RooUnfold" / "build" / "libRooUnfold"),
                str(base / "RooUnfold" / "build-conda" / "libRooUnfold"),
                str(base / "RooUnfold" / "build" / "libRooUnfold"),
            ]
        )
    cands.append("libRooUnfold")
    if extra:
        cands.insert(0, extra)
    return cands


_ROOUNFOLD_CANDIDATES = _roounfold_candidates()
ROOUNFOLD_AVAILABLE = False

try:
    import ROOT

    for _candidate in _ROOUNFOLD_CANDIDATES:
        if ROOT.gSystem.Load(_candidate) >= 0:
            ROOUNFOLD_AVAILABLE = True
            logger.info("RooUnfold loaded from: %s", _candidate)
            break
    if not ROOUNFOLD_AVAILABLE:
        logger.warning("RooUnfold library not found in any candidate path")
except ImportError:
    logger.warning("ROOT not available - RooUnfold disabled")
except Exception as e:
    logger.warning("Failed to load RooUnfold: %s", e)


def decode_log(x: np.ndarray) -> np.ndarray:
    return np.expm1(x)


def to_numpy(x: Any) -> np.ndarray:
    try:
        return x.to_numpy()
    except Exception:
        return np.asarray(x)


def neutrino_from_log_kinematics(
    log_pt: np.ndarray, eta: np.ndarray, phi: np.ndarray
) -> Any:
    pt = decode_log(log_pt)
    return vector.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": np.zeros_like(pt),
        }
    )


def extract_neutrino_features(nu1: Any, nu2: Any) -> Dict[str, np.ndarray]:
    """Return neutrino kinematics in both merged and per-charge form.

    nu1 = neutrino from t1 slot (paired with l+, i.e. nu).
    nu2 = neutrino from t2 slot (paired with l-, i.e. nubar).

    Merged keys ("pt", "eta", "phi") are length-2N arrays kept for the
    kinematics overlay plot. Split keys ("pt_nu", ..., "phi_nubar") are
    per-event length-N arrays consumed by the per-charge unfolding loop.
    """
    pt1, eta1, phi1 = to_numpy(nu1.pt), to_numpy(nu1.eta), to_numpy(nu1.phi)
    pt2, eta2, phi2 = to_numpy(nu2.pt), to_numpy(nu2.eta), to_numpy(nu2.phi)
    return {
        "pt": np.concatenate([pt1, pt2]),
        "eta": np.concatenate([eta1, eta2]),
        "phi": np.concatenate([phi1, phi2]),
        "pt_nu":  pt1,  "eta_nu":  eta1, "phi_nu":  phi1,
        "pt_nubar": pt2, "eta_nubar": eta2, "phi_nubar": phi2,
    }


def _pc_row_to_4vec(pc_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Point-cloud row: logE, logPt, eta, phi (first 4 cols) → E, px, py, pz."""
    pc_row = np.asarray(pc_row)
    if pc_row.shape[-1] > 4:
        pc_row = pc_row[..., :4]
    log_e, log_pt, eta, phi = pc_row[..., 0], pc_row[..., 1], pc_row[..., 2], pc_row[..., 3]
    energy = decode_log(log_e)
    pt = decode_log(log_pt)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return energy, px, py, pz


def _nu_to_4vec(log_pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Massless neutrino: E = pt * cosh(eta)."""
    pt = decode_log(log_pt)
    energy = pt * np.cosh(eta)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return energy, px, py, pz


def _add4(
    a: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    b: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _four_momentum_to_pt_eta_phi(
    energy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.sqrt(np.maximum(px * px + py * py, 0.0))
    phi = np.arctan2(py, px)
    eta = np.arcsinh(np.divide(pz, np.maximum(pt, EPSILON)))
    return pt, eta, phi


def _four_momentum_to_mass(
    energy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> np.ndarray:
    """Invariant mass: sqrt(max(E^2 - |p|^2, 0))."""
    m2 = energy * energy - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2, 0.0))


def _extra_particle_to_4vec(features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
    """Truth-assigned reco object stored as pt/eta/phi/energy in EXTRA columns."""
    pt = np.asarray(features["pt"])
    eta = np.asarray(features["eta"])
    phi = np.asarray(features["phi"])
    energy = np.asarray(features["energy"])
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return energy, px, py, pz


def merged_top_from_extra(
    extra_reco: Dict[str, Dict[str, np.ndarray]],
    nu_log_pt: np.ndarray,
    nu_eta: np.ndarray,
    nu_phi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Reconstruct truth-assigned tops from EXTRA b/l objects plus predicted or target neutrinos."""
    b1 = _extra_particle_to_4vec(extra_reco["t1_b"])
    l1 = _extra_particle_to_4vec(extra_reco["t1_l"])
    b2 = _extra_particle_to_4vec(extra_reco["t2_b"])
    l2 = _extra_particle_to_4vec(extra_reco["t2_l"])
    nu1 = _nu_to_4vec(nu_log_pt[:, 0], nu_eta[:, 0], nu_phi[:, 0])
    nu2 = _nu_to_4vec(nu_log_pt[:, 1], nu_eta[:, 1], nu_phi[:, 1])

    t1 = _add4(_add4(b1, l1), nu1)
    t2 = _add4(_add4(b2, l2), nu2)
    pt1, eta1, phi1 = _four_momentum_to_pt_eta_phi(*t1)
    pt2, eta2, phi2 = _four_momentum_to_pt_eta_phi(*t2)
    m1 = _four_momentum_to_mass(*t1)
    m2 = _four_momentum_to_mass(*t2)

    return {
        "pt": np.concatenate([pt1, pt2]),
        "eta": np.concatenate([eta1, eta2]),
        "phi": np.concatenate([phi1, phi2]),
        "mass": np.concatenate([m1, m2]),
    }


def merged_w_from_extra(
    extra_reco: Dict[str, Dict[str, np.ndarray]],
    nu_log_pt: np.ndarray,
    nu_eta: np.ndarray,
    nu_phi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Reconstruct truth-assigned W candidates from EXTRA lepton objects plus neutrinos."""
    l1 = _extra_particle_to_4vec(extra_reco["t1_l"])
    l2 = _extra_particle_to_4vec(extra_reco["t2_l"])
    nu1 = _nu_to_4vec(nu_log_pt[:, 0], nu_eta[:, 0], nu_phi[:, 0])
    nu2 = _nu_to_4vec(nu_log_pt[:, 1], nu_eta[:, 1], nu_phi[:, 1])

    w1 = _add4(l1, nu1)
    w2 = _add4(l2, nu2)
    pt1, eta1, phi1 = _four_momentum_to_pt_eta_phi(*w1)
    pt2, eta2, phi2 = _four_momentum_to_pt_eta_phi(*w2)
    m1 = _four_momentum_to_mass(*w1)
    m2 = _four_momentum_to_mass(*w2)

    return {
        "pt": np.concatenate([pt1, pt2]),
        "eta": np.concatenate([eta1, eta2]),
        "phi": np.concatenate([phi1, phi2]),
        "mass": np.concatenate([m1, m2]),
    }


def merged_truth_top_from_extra(
    truth_top: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Return parton-level top ground truth from EXTRA/truth_t*/t/* columns."""
    return {
        "pt": np.concatenate([truth_top["t1"]["pt"], truth_top["t2"]["pt"]]),
        "eta": np.concatenate([truth_top["t1"]["eta"], truth_top["t2"]["eta"]]),
        "phi": np.concatenate([truth_top["t1"]["phi"], truth_top["t2"]["phi"]]),
        "mass": np.concatenate([truth_top["t1"]["mass"], truth_top["t2"]["mass"]]),
    }


def merged_top_pt_eta_phi(
    pc_all: np.ndarray,
    assign0_idx: np.ndarray,
    assign1_idx: np.ndarray,
    nu_log_pt: np.ndarray,
    nu_eta: np.ndarray,
    nu_phi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct both tops as b + l + nu and return merged pt, eta, phi, mass (length 2 * N).

    Args:
        pc_all: (N, num_particles, n_features) point cloud.
        assign0_idx, assign1_idx: (N, 2) jet/lepton slot indices [b, l] per top (truth or pred.).
        nu_*: (N, 2) per-event neutrinos (log_pt, eta, phi).
    """
    n_ev = pc_all.shape[0]
    idx = np.arange(n_ev)
    b1 = _pc_row_to_4vec(pc_all[idx, assign0_idx[:, 0], :])
    l1 = _pc_row_to_4vec(pc_all[idx, assign0_idx[:, 1], :])
    b2 = _pc_row_to_4vec(pc_all[idx, assign1_idx[:, 0], :])
    l2 = _pc_row_to_4vec(pc_all[idx, assign1_idx[:, 1], :])
    nu1 = _nu_to_4vec(nu_log_pt[:, 0], nu_eta[:, 0], nu_phi[:, 0])
    nu2 = _nu_to_4vec(nu_log_pt[:, 1], nu_eta[:, 1], nu_phi[:, 1])

    t1 = _add4(_add4(b1, l1), nu1)
    t2 = _add4(_add4(b2, l2), nu2)

    pt1, eta1, phi1 = _four_momentum_to_pt_eta_phi(*t1)
    pt2, eta2, phi2 = _four_momentum_to_pt_eta_phi(*t2)

    m1 = _four_momentum_to_mass(*t1)
    m2 = _four_momentum_to_mass(*t2)

    return {
        "pt": np.concatenate([pt1, pt2]),
        "eta": np.concatenate([eta1, eta2]),
        "phi": np.concatenate([phi1, phi2]),
        "mass": np.concatenate([m1, m2]),
    }


def merged_w_pt_eta_phi(
    pc_all: np.ndarray,
    assign0_idx: np.ndarray,
    assign1_idx: np.ndarray,
    nu_log_pt: np.ndarray,
    nu_eta: np.ndarray,
    nu_phi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Reconstruct both $W \\to \\ell\\nu$ as $\\ell$ + $\\nu$ (same convention as calibration MW).

    Lepton four-vectors are read from the point cloud at the **assignment** lepton slot: index
    **1** in each ``assign{{0,1}}_idx`` row ``[b, \\ell]`` (TT2L ``assignment_target`` /
    ``assignment_prediction`` schema). Neutrinos are ``nu_*[:, 0]`` / ``nu_*[:, 1]``. Returns
    merged arrays of length ``2 * N`` (first top, then second top), same layout as
    :func:`merged_top_pt_eta_phi`.
    """
    n_ev = pc_all.shape[0]
    idx = np.arange(n_ev)
    l1 = _pc_row_to_4vec(pc_all[idx, assign0_idx[:, 1], :])
    l2 = _pc_row_to_4vec(pc_all[idx, assign1_idx[:, 1], :])
    nu1 = _nu_to_4vec(nu_log_pt[:, 0], nu_eta[:, 0], nu_phi[:, 0])
    nu2 = _nu_to_4vec(nu_log_pt[:, 1], nu_eta[:, 1], nu_phi[:, 1])

    w1 = _add4(l1, nu1)
    w2 = _add4(l2, nu2)

    pt1, eta1, phi1 = _four_momentum_to_pt_eta_phi(*w1)
    pt2, eta2, phi2 = _four_momentum_to_pt_eta_phi(*w2)
    m1 = _four_momentum_to_mass(*w1)
    m2 = _four_momentum_to_mass(*w2)

    return {
        "pt": np.concatenate([pt1, pt2]),
        "eta": np.concatenate([eta1, eta2]),
        "phi": np.concatenate([phi1, phi2]),
        "mass": np.concatenate([m1, m2]),
    }


def _assignment_pred_from_batch(batch: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (best_indices top0, top1) or None if ``assignment_prediction`` is absent."""
    if "assignment_prediction" not in batch:
        return None
    ap = batch["assignment_prediction"]
    proc = ASSIGN_PRED_PROCESS if ASSIGN_PRED_PROCESS in ap else next(iter(ap.keys()))
    bi = ap[proc]["best_indices"]
    return bi[0].cpu(), bi[1].cpu()


def _event_keys_from_gei(gei: torch.Tensor) -> List[Tuple[int, ...]]:
    g = gei.detach().cpu().numpy()
    if g.ndim == 1:
        return [(int(x),) for x in g]
    return [tuple(int(x) for x in row) for row in g]


def _take_first_candidate(t: torch.Tensor) -> torch.Tensor:
    """Drop a leading K dimension if present, taking candidate 0.

    EveNet predict stores ``(B, 2)`` per kinematic feature. Some baselines
    (e.g. nu2flow with multi-sample output) store ``(K, B, 2)``; this helper
    collapses K → candidate 0 unconditionally so the rest of the pipeline only
    sees the K=1 layout.
    """
    if t.dim() >= 3:
        return t[0]
    return t


def _coerce_neutrino_feat_dict(
    feat: Dict[str, Any],
    *,
    role: str,
    filepath: str,
) -> Dict[str, torch.Tensor]:
    """Map batch ``neutrinos`` predict/target feature dict to ``KINEMATICS`` tensors.

    EveNet prediction usually stores ``log_pt`` (``log(1+p_T)``, see :func:`decode_log`).
    Some checkpoints / baselines store linear ``pt`` or cartesian ``px, py, pz`` instead.
    Multi-candidate predictions (leading K dim) are collapsed to candidate 0 via
    :func:`_take_first_candidate`.
    """
    if all(k in feat for k in KINEMATICS):
        return {k: _take_first_candidate(feat[k]) for k in KINEMATICS}
    if all(k in feat for k in ("pt", "eta", "phi")):
        pt = _take_first_candidate(feat["pt"])
        return {
            "log_pt": torch.log1p(pt.clamp(min=0.0)),
            "eta": _take_first_candidate(feat["eta"]),
            "phi": _take_first_candidate(feat["phi"]),
        }
    if all(k in feat for k in ("px", "py", "pz")):
        px = _take_first_candidate(feat["px"])
        py = _take_first_candidate(feat["py"])
        pz = _take_first_candidate(feat["pz"])
        pt = torch.sqrt(px * px + py * py + 1e-12)
        p = torch.sqrt(px * px + py * py + pz * pz + 1e-12)
        eta = 0.5 * torch.log((p + pz) / (p - pz).clamp(min=1e-12))
        phi = torch.atan2(py, px)
        return {
            "log_pt": torch.log1p(pt.clamp(min=0.0)),
            "eta": eta,
            "phi": phi,
        }
    raise ValueError(
        f"{role} neutrino features in {filepath} have keys {sorted(feat.keys())}; "
        f"expected {KINEMATICS}, (pt, eta, phi), or (px, py, pz)."
    )


def _batch_has_key(batch: Dict[str, Any], key: str) -> bool:
    """Return True for either flat ``EXTRA/foo`` keys or nested ``EXTRA/foo`` paths."""
    if key in batch:
        return True
    try:
        get_value(batch, key)
    except (KeyError, TypeError):
        return False
    return True


def _batch_get_tensor(batch: Dict[str, Any], key: str) -> torch.Tensor:
    value = batch[key] if key in batch else get_value(batch, key)
    return torch.as_tensor(value).cpu()


def _has_all_extra_keys(batch: Dict[str, Any], spec: Dict[str, Dict[str, str]]) -> bool:
    return all(_batch_has_key(batch, key) for obj in spec.values() for key in obj.values())


def load_tt2l_stacked(filepath: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Load TT2L prediction .pt (list of batch dicts) and stack per-event arrays.

    Returns dict with nu_pred, nu_targ, assignment mask, optional assignment_target and
    assignment_prediction (per-top (N,2) indices), point_cloud, gei or None, and path.
    """
    if verbose:
        logger.info("Loading %s", filepath)

    data = torch.load(filepath, map_location="cpu", weights_only=False)
    batches = data if isinstance(data, list) else [data]

    nu_pred: Dict[str, List[torch.Tensor]] = {k: [] for k in KINEMATICS}
    nu_targ: Dict[str, List[torch.Tensor]] = {k: [] for k in KINEMATICS}
    assign0_all: List[torch.Tensor] = []
    assign1_all: List[torch.Tensor] = []
    point_cloud_all: List[torch.Tensor] = []
    gei_all: List[torch.Tensor] = []
    gei_present: Optional[bool] = None
    assign0_tgt_all: List[torch.Tensor] = []
    assign1_tgt_all: List[torch.Tensor] = []
    assign_target_present: Optional[bool] = None
    assign0_pred_all: List[torch.Tensor] = []
    assign1_pred_all: List[torch.Tensor] = []
    assign_pred_present: Optional[bool] = None
    extra_reco_all: Dict[str, Dict[str, List[torch.Tensor]]] = {
        obj: {comp: [] for comp in keys} for obj, keys in EXTRA_RECO_OBJECT_KEYS.items()
    }
    extra_reco_present: Optional[bool] = None
    truth_top_all: Dict[str, Dict[str, List[torch.Tensor]]] = {
        top: {comp: [] for comp in keys} for top, keys in EXTRA_TRUTH_TOP_KEYS.items()
    }
    truth_top_present: Optional[bool] = None

    for batch in batches:
        if "neutrinos" not in batch:
            raise ValueError(f"Batch missing 'neutrinos' in {filepath}")
        pred = _coerce_neutrino_feat_dict(
            batch["neutrinos"]["predict"], role="predict", filepath=filepath
        )
        targ = _coerce_neutrino_feat_dict(
            batch["neutrinos"]["target"], role="target", filepath=filepath
        )
        for k in KINEMATICS:
            nu_pred[k].append(pred[k].cpu())
            nu_targ[k].append(targ[k].cpu())

        msk = get_value(batch, ASSIGN_MASK_PATH)
        assign0_all.append(msk[0].cpu())
        assign1_all.append(msk[1].cpu())

        if POINTCLOUD_KEY not in batch:
            raise ValueError(
                f"Batch missing '{POINTCLOUD_KEY}' (needed for reco cuts) in {filepath}"
            )
        point_cloud_all.append(batch[POINTCLOUD_KEY].cpu())

        has_gei = "global_event_index" in batch
        if gei_present is None:
            gei_present = has_gei
        elif gei_present != has_gei:
            raise ValueError(
                f"Inconsistent global_event_index presence across batches in {filepath}"
            )
        if has_gei:
            gei_all.append(batch["global_event_index"].cpu())

        has_at = "assignment_target" in batch
        if assign_target_present is None:
            assign_target_present = has_at
        elif assign_target_present != has_at:
            raise ValueError(
                f"Inconsistent assignment_target presence across batches in {filepath}"
            )
        if has_at:
            tgt = get_value(batch, ASSIGN_TARGET_PATH)
            assign0_tgt_all.append(tgt[0].cpu())
            assign1_tgt_all.append(tgt[1].cpu())

        ap = _assignment_pred_from_batch(batch)
        if assign_pred_present is None:
            assign_pred_present = ap is not None
        elif assign_pred_present != (ap is not None):
            raise ValueError(
                f"Inconsistent assignment_prediction presence across batches in {filepath}"
            )
        if ap is not None:
            assign0_pred_all.append(ap[0])
            assign1_pred_all.append(ap[1])

        has_extra_reco = _has_all_extra_keys(batch, EXTRA_RECO_OBJECT_KEYS)
        if extra_reco_present is None:
            extra_reco_present = has_extra_reco
        elif extra_reco_present != has_extra_reco:
            raise ValueError(
                f"Inconsistent truth-assigned EXTRA reco object presence across batches in {filepath}"
            )
        if has_extra_reco:
            for obj, keys in EXTRA_RECO_OBJECT_KEYS.items():
                for comp, extra_key in keys.items():
                    extra_reco_all[obj][comp].append(_batch_get_tensor(batch, extra_key))

        has_truth_top = _has_all_extra_keys(batch, EXTRA_TRUTH_TOP_KEYS)
        if truth_top_present is None:
            truth_top_present = has_truth_top
        elif truth_top_present != has_truth_top:
            raise ValueError(
                f"Inconsistent EXTRA truth top presence across batches in {filepath}"
            )
        if has_truth_top:
            for top, keys in EXTRA_TRUTH_TOP_KEYS.items():
                for comp, extra_key in keys.items():
                    truth_top_all[top][comp].append(_batch_get_tensor(batch, extra_key))

    assign0 = torch.cat(assign0_all, dim=0)
    assign1 = torch.cat(assign1_all, dim=0)
    point_cloud = torch.cat(point_cloud_all, dim=0)
    nu_pred_np = {k: torch.cat(nu_pred[k], dim=0).numpy() for k in KINEMATICS}
    nu_targ_np = {k: torch.cat(nu_targ[k], dim=0).numpy() for k in KINEMATICS}
    gei_tensor: Optional[torch.Tensor]
    if gei_present:
        gei_tensor = torch.cat(gei_all, dim=0)
    else:
        gei_tensor = None

    assign0_tgt_np: Optional[np.ndarray]
    assign1_tgt_np: Optional[np.ndarray]
    if assign_target_present:
        assign0_tgt_np = torch.cat(assign0_tgt_all, dim=0).numpy().astype(np.int64)
        assign1_tgt_np = torch.cat(assign1_tgt_all, dim=0).numpy().astype(np.int64)
    else:
        assign0_tgt_np = None
        assign1_tgt_np = None

    assign0_pred_np: Optional[np.ndarray]
    assign1_pred_np: Optional[np.ndarray]
    if assign_pred_present:
        assign0_pred_np = torch.cat(assign0_pred_all, dim=0).numpy().astype(np.int64)
        assign1_pred_np = torch.cat(assign1_pred_all, dim=0).numpy().astype(np.int64)
    else:
        assign0_pred_np = None
        assign1_pred_np = None

    extra_reco_np: Optional[Dict[str, Dict[str, np.ndarray]]]
    if extra_reco_present:
        extra_reco_np = {
            obj: {
                comp: torch.cat(values, dim=0).numpy()
                for comp, values in comps.items()
            }
            for obj, comps in extra_reco_all.items()
        }
    else:
        extra_reco_np = None

    truth_top_np: Optional[Dict[str, Dict[str, np.ndarray]]]
    if truth_top_present:
        truth_top_np = {
            top: {
                comp: torch.cat(values, dim=0).numpy()
                for comp, values in comps.items()
            }
            for top, comps in truth_top_all.items()
        }
    else:
        truth_top_np = None

    if verbose:
        n = assign0.shape[0]
        logger.info(
            "  Stacked N=%d events; gei=%s; assignment_target=%s; assignment_prediction=%s; extra_reco=%s; truth_top=%s",
            n,
            "yes" if gei_tensor is not None else "no",
            "yes" if assign_target_present else "no",
            "yes" if assign_pred_present else "no",
            "yes" if extra_reco_present else "no",
            "yes" if truth_top_present else "no",
        )

    return {
        "nu_pred": nu_pred_np,
        "nu_targ": nu_targ_np,
        "assign0": assign0.numpy().astype(bool),
        "assign1": assign1.numpy().astype(bool),
        "point_cloud": point_cloud.numpy(),
        "gei": gei_tensor,
        "assign0_tgt": assign0_tgt_np,
        "assign1_tgt": assign1_tgt_np,
        "assign0_pred": assign0_pred_np,
        "assign1_pred": assign1_pred_np,
        "extra_reco": extra_reco_np,
        "truth_top": truth_top_np,
        "path": filepath,
    }


def _align_indices(raw_a: Dict[str, Any], raw_b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return index arrays into raw_a and raw_b for the same physical events."""
    ga, gb = raw_a["gei"], raw_b["gei"]
    if ga is not None and gb is not None:
        keys_a = _event_keys_from_gei(ga)
        keys_b = _event_keys_from_gei(gb)
        map_a = {k: i for i, k in enumerate(keys_a)}
        map_b = {k: i for i, k in enumerate(keys_b)}
        common = sorted(set(map_a.keys()) & set(map_b.keys()))
        if not common:
            raise ValueError("No overlapping global_event_index between the two prediction files")
        idx_a = np.array([map_a[k] for k in common], dtype=np.int64)
        idx_b = np.array([map_b[k] for k in common], dtype=np.int64)
        logger.info(
            "Aligned on global_event_index: %d shared events (file A had %d, B had %d)",
            len(common),
            len(keys_a),
            len(keys_b),
        )
        return idx_a, idx_b

    if ga is not None or gb is not None:
        raise ValueError(
            "global_event_index present in only one file; cannot align safely. "
            "Re-run prediction with global_event_index in both, or use matched ordering."
        )

    na = raw_a["assign0"].shape[0]
    nb = raw_b["assign0"].shape[0]
    if na != nb:
        raise ValueError(
            f"Same-length alignment requires equal event counts; got N_A={na}, N_B={nb}. "
            "Add global_event_index to both .pt outputs to intersect subsets."
        )
    idx = np.arange(na, dtype=np.int64)
    logger.info("Row-aligned comparison: N=%d (no global_event_index)", na)
    return idx, idx


def _slice_raw(raw: Dict[str, Any], idx: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "nu_pred": {k: raw["nu_pred"][k][idx] for k in KINEMATICS},
        "nu_targ": {k: raw["nu_targ"][k][idx] for k in KINEMATICS},
        "assign0": raw["assign0"][idx],
        "assign1": raw["assign1"][idx],
        "point_cloud": raw["point_cloud"][idx],
        "path": raw["path"],
    }
    if raw.get("assign0_tgt") is not None:
        out["assign0_tgt"] = raw["assign0_tgt"][idx]
        out["assign1_tgt"] = raw["assign1_tgt"][idx]
    else:
        out["assign0_tgt"] = None
        out["assign1_tgt"] = None
    if raw.get("assign0_pred") is not None:
        out["assign0_pred"] = raw["assign0_pred"][idx]
        out["assign1_pred"] = raw["assign1_pred"][idx]
    else:
        out["assign0_pred"] = None
        out["assign1_pred"] = None
    if raw.get("extra_reco") is not None:
        out["extra_reco"] = {
            obj: {comp: arr[idx] for comp, arr in comps.items()}
            for obj, comps in raw["extra_reco"].items()
        }
    else:
        out["extra_reco"] = None
    if raw.get("truth_top") is not None:
        out["truth_top"] = {
            top: {comp: arr[idx] for comp, arr in comps.items()}
            for top, comps in raw["truth_top"].items()
        }
    else:
        out["truth_top"] = None
    return out


def _assert_truth_agreement(slices: Dict[str, Dict[str, Any]], atol: float = 1e-4) -> None:
    names = list(slices.keys())
    ref = names[0]
    for other in names[1:]:
        for k in KINEMATICS:
            a = slices[ref]["nu_targ"][k]
            b = slices[other]["nu_targ"][k]
            if not np.allclose(a, b, rtol=0.0, atol=atol):
                logger.warning(
                    "Truth neutrino %s differs between %s and %s beyond atol=%g (max abs diff %g)",
                    k,
                    ref,
                    other,
                    atol,
                    float(np.max(np.abs(a - b))),
                )


def _assert_assignment_target_agreement(slices: Dict[str, Dict[str, Any]]) -> None:
    """Warn if ground-truth assignment indices differ between aligned prediction files."""
    names = list(slices.keys())
    ref = names[0]
    a0_ref = slices[ref].get("assign0_tgt")
    if a0_ref is None:
        return
    for other in names[1:]:
        a0 = slices[other].get("assign0_tgt")
        a1 = slices[other].get("assign1_tgt")
        if a0 is None or a1 is None:
            continue
        if not np.array_equal(a0_ref, a0) or not np.array_equal(
            slices[ref]["assign1_tgt"], a1
        ):
            logger.warning(
                "assignment_target indices differ between %s and %s — using each file's own target",
                ref,
                other,
            )


def compute_tt2l_event_keep(
    slice_dict: Dict[str, Any],
    apply_recon_cuts: bool,
    verbose: bool,
) -> np.ndarray:
    """Assignment + valid neutrino + optional reco cuts (same chain as event_cuts)."""
    return compute_event_mask(
        slice_dict["assign0"],
        slice_dict["assign1"],
        slice_dict["nu_pred"]["log_pt"],
        point_cloud=slice_dict["point_cloud"] if apply_recon_cuts else None,
        apply_recon_cuts=apply_recon_cuts,
        verbose=verbose,
    )


def extract_features_after_keep(
    slice_dict: Dict[str, Any],
    keep: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build merged truth/recon feature dicts after a boolean event mask."""
    nu_pred = {k: slice_dict["nu_pred"][k][keep] for k in KINEMATICS}
    nu_targ = {k: slice_dict["nu_targ"][k][keep] for k in KINEMATICS}

    nu1_recon = neutrino_from_log_kinematics(
        nu_pred["log_pt"][:, 0], nu_pred["eta"][:, 0], nu_pred["phi"][:, 0]
    )
    nu2_recon = neutrino_from_log_kinematics(
        nu_pred["log_pt"][:, 1], nu_pred["eta"][:, 1], nu_pred["phi"][:, 1]
    )
    nu1_truth = neutrino_from_log_kinematics(
        nu_targ["log_pt"][:, 0], nu_targ["eta"][:, 0], nu_targ["phi"][:, 0]
    )
    nu2_truth = neutrino_from_log_kinematics(
        nu_targ["log_pt"][:, 1], nu_targ["eta"][:, 1], nu_targ["phi"][:, 1]
    )

    truth_features = extract_neutrino_features(nu1_truth, nu2_truth)
    recon_features = extract_neutrino_features(nu1_recon, nu2_recon)
    return truth_features, recon_features


def load_aligned_model_features(
    model_paths: Dict[str, str],
    apply_recon_cuts: bool = True,
    verbose: bool = True,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    int,
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, int],
]:
    """Load TT2L .pt files and apply per-model event cuts independently.

    Each model uses its own full event set (no GEI intersection).  This maximises
    statistical power: unfolding compares the *distributions* produced by each model,
    not event-by-event correspondences.  Truth features from the baseline model are
    returned for the shared Poisson reference; each model's own truth is also returned
    for use in its own unfolding response matrix.

    Returns:
        model_recon: {model_name: {pt, eta, phi}} neutrino-level merged arrays
        model_truth: {model_name: {pt, eta, phi}} per-model truth arrays (after own mask)
        truth_features: truth merged arrays from the baseline model (after its own mask)
        n_events: baseline model event count after cuts (informational)
        top_kinematics: {model_name: {"truth_nu_truth_assign", "pred_nu_truth_assign", "w"}}
            with merged pt/eta/phi (and ``mass`` for tops); ``truth_top_ground`` is added when
            saved parton-level truth tops are available. ``w`` holds the same two curves for
            $W\\to\\ell\\nu$ (lepton from truth assignment + $\\nu$). Prefer truth-assigned
            ``EXTRA/t*/{b,l}`` objects; fall back to ``assignment_target`` only for older files.
    """
    raws = {name: load_tt2l_stacked(path, verbose=verbose) for name, path in model_paths.items()}
    names = list(model_paths.keys())
    if len(names) < 1:
        raise ValueError("model_paths must contain at least one model")

    model_recon: Dict[str, Dict[str, np.ndarray]] = {}
    model_truth: Dict[str, Dict[str, np.ndarray]] = {}
    top_kinematics: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    n_events_per_model: Dict[str, int] = {}

    for name in names:
        raw = raws[name]
        idx = np.arange(raw["assign0"].shape[0], dtype=np.int64)
        sl = _slice_raw(raw, idx)

        keep = compute_tt2l_event_keep(sl, apply_recon_cuts=apply_recon_cuts, verbose=verbose)
        n_ev = int(keep.sum())
        n_events_per_model[name] = n_ev
        logger.info("Model %s: %d events after cuts (out of %d)", name, n_ev, idx.shape[0])

        truth_f, recon_f = extract_features_after_keep(sl, keep)
        model_recon[name] = recon_f
        model_truth[name] = truth_f

        nu_p = {k: sl["nu_pred"][k][keep] for k in KINEMATICS}
        nu_t = {k: sl["nu_targ"][k][keep] for k in KINEMATICS}
        extra_reco = sl.get("extra_reco")
        truth_top = sl.get("truth_top")
        if extra_reco is not None:
            # Prefer truth-assigned EXTRA objects. This matches the nu2flow notebook and
            # does not require assignment_target to be present in the prediction file.
            er_k = {
                obj: {comp: arr[keep] for comp, arr in comps.items()}
                for obj, comps in extra_reco.items()
            }
            top_kinematics[name] = {
                "truth_nu_truth_assign": merged_top_from_extra(
                    er_k, nu_t["log_pt"], nu_t["eta"], nu_t["phi"]
                ),
                "pred_nu_truth_assign": merged_top_from_extra(
                    er_k, nu_p["log_pt"], nu_p["eta"], nu_p["phi"]
                ),
                "w": {
                    "truth_nu_truth_assign": merged_w_from_extra(
                        er_k, nu_t["log_pt"], nu_t["eta"], nu_t["phi"]
                    ),
                    "pred_nu_truth_assign": merged_w_from_extra(
                        er_k, nu_p["log_pt"], nu_p["eta"], nu_p["phi"]
                    ),
                },
            }
            if truth_top is not None:
                tt_k = {
                    top: {comp: arr[keep] for comp, arr in comps.items()}
                    for top, comps in truth_top.items()
                }
                top_kinematics[name]["truth_top_ground"] = merged_truth_top_from_extra(tt_k)
        elif sl.get("assign0_tgt") is not None:
            # Fallback for older files without EXTRA objects: use assignment_target as
            # the truth assignment to read b/lepton rows from the point cloud.
            pc_k = sl["point_cloud"][keep]
            a0t = sl["assign0_tgt"][keep]
            a1t = sl["assign1_tgt"][keep]
            if verbose and sl.get("assign0_pred") is not None:
                logger.info(
                    "Model %s: ignoring assignment_prediction; using truth assignment for reco curves.",
                    name,
                )
            top_kinematics[name] = {
                "truth_nu_truth_assign": merged_top_pt_eta_phi(
                    pc_k, a0t, a1t, nu_t["log_pt"], nu_t["eta"], nu_t["phi"]
                ),
                "pred_nu_truth_assign": merged_top_pt_eta_phi(
                    pc_k, a0t, a1t, nu_p["log_pt"], nu_p["eta"], nu_p["phi"]
                ),
                "w": {
                    "truth_nu_truth_assign": merged_w_pt_eta_phi(
                        pc_k, a0t, a1t, nu_t["log_pt"], nu_t["eta"], nu_t["phi"]
                    ),
                    "pred_nu_truth_assign": merged_w_pt_eta_phi(
                        pc_k, a0t, a1t, nu_p["log_pt"], nu_p["eta"], nu_p["phi"]
                    ),
                },
            }
        elif verbose:
            logger.warning(
                "Skip top kinematics for %s: missing truth-assigned EXTRA objects and assignment_target in %s",
                name,
                sl["path"],
            )

    baseline_name = names[0]
    truth_ref = model_truth[baseline_name]
    n_events = n_events_per_model[baseline_name]
    return model_recon, model_truth, truth_ref, n_events, top_kinematics, n_events_per_model


# ---------------------------------------------------------------------------
# RooUnfold (same as scoring_head_unfolding / model_uncertainty_comparison)
# ---------------------------------------------------------------------------
def build_roounfold_response(
    truth_data: np.ndarray,
    recon_data: np.ndarray,
    num_bins: int,
    bin_range: Tuple[float, float],
) -> Any:
    if not ROOUNFOLD_AVAILABLE:
        raise RuntimeError("RooUnfold is not available")
    response = ROOT.RooUnfoldResponse(num_bins, bin_range[0], bin_range[1])
    for truth_val, recon_val in zip(truth_data, recon_data):
        if not (np.isnan(truth_val) or np.isnan(recon_val)):
            response.Fill(recon_val, truth_val)
    return response


def build_roounfold_histogram(
    data: np.ndarray,
    num_bins: int,
    bin_range: Tuple[float, float],
    name: str = "hist",
) -> Any:
    if not ROOUNFOLD_AVAILABLE:
        raise RuntimeError("RooUnfold is not available")
    hist = ROOT.TH1D(name, name, num_bins, bin_range[0], bin_range[1])
    for val in data:
        if not np.isnan(val):
            hist.Fill(val)
    return hist


def extract_response_matrix(response: Any, num_bins: int) -> np.ndarray:
    h_response = response.Hresponse()
    if h_response is None:
        return np.eye(num_bins)
    matrix = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        for j in range(num_bins):
            matrix[i, j] = h_response.GetBinContent(i + 1, j + 1)
    return matrix


def calculate_unfolded_uncertainty(
    truth_data: np.ndarray,
    recon_data: np.ndarray,
    num_bins: int,
    bin_range: Tuple[float, float],
    kreg: int = SVD_KREG,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ROOUNFOLD_AVAILABLE:
        raise RuntimeError("RooUnfold is not available")

    response = build_roounfold_response(truth_data, recon_data, num_bins, bin_range)
    h_measure = build_roounfold_histogram(recon_data, num_bins, bin_range, "h_measure")
    response_matrix = extract_response_matrix(response, num_bins)

    unfold = ROOT.RooUnfoldSvd(response, h_measure, kreg)
    h_unfold = unfold.Hunfold(3)

    sigma = np.array([h_unfold.GetBinError(i + 1) for i in range(num_bins)])
    unfolded_counts = np.array([h_unfold.GetBinContent(i + 1) for i in range(num_bins)])

    if verbose:
        logger.info("      mean sigma=%.4f sum counts=%.0f", float(np.mean(sigma)), float(np.sum(unfolded_counts)))

    return sigma, unfolded_counts, response_matrix


def plot_neutrino_kinematics_overlay(
    model_recon: Dict[str, Dict[str, np.ndarray]],
    model_truth: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """One 2×3 figure: predicted neutrino pt/eta/phi overlaid with truth, split by charge.

    Top row: nu (slot 0, paired with l+, i.e. from t).
    Bottom row: nubar (slot 1, paired with l-, i.e. from t-bar).
    Each panel shows truth (black dashed) and each model's prediction (colour-coded).

    Args:
        model_recon: {model_name: {pt_nu, eta_nu, phi_nu, pt_nubar, eta_nubar, phi_nubar, ...}}
            reconstructed neutrino arrays (split keys are length-N per event).
        model_truth: same schema as model_recon, per-model truth arrays.
        output_dir: directory for the output PNG.
        model_styles: per-model color/label dict.
        model_order: draw order.
        n_events_per_model: {model_name: event_count} for the figure title.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    NU_PLOT_CFG = {
        "pt":  {"range": (0.0, 150.0), "bins": 30, "label_template": r"$p_T^{{{sym}}}$",  "unit": "GeV"},
        "eta": {"range": (-4.0,  4.0),  "bins": 20, "label_template": r"$\eta^{{{sym}}}$", "unit": ""},
        "phi": {"range": (-np.pi, np.pi), "bins": 16, "label_template": r"$\phi^{{{sym}}}$", "unit": "rad"},
    }
    kin_order = ["pt", "eta", "phi"]
    # (row label, key suffix, latex symbol used in axis label / legend)
    charge_rows = [
        ("nu",    r"\nu",      r"$\nu$ (from $t$)"),
        ("nubar", r"\bar\nu",  r"$\bar\nu$ (from $\bar t$)"),
    ]

    # Use the first available model's truth as the shared truth curve.
    first_key = next((k for k in model_order if k in model_truth), None)
    if first_key is None:
        logger.warning("plot_neutrino_kinematics_overlay: no models with truth data; skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row_idx, (charge_suffix, sym, charge_legend) in enumerate(charge_rows):
        for col_idx, kin_key in enumerate(kin_order):
            ax = axes[row_idx, col_idx]
            cfg = NU_PLOT_CFG[kin_key]
            br = cfg["range"]
            nb = cfg["bins"]
            bins = np.linspace(br[0], br[1], nb + 1)
            base_label = cfg["label_template"].format(sym=sym)
            xlabel = f"{base_label} [{cfg['unit']}]" if cfg["unit"] else base_label

            split_key = f"{kin_key}_{charge_suffix}"  # e.g. pt_nu / pt_nubar

            truth_vals = model_truth[first_key].get(split_key)
            if truth_vals is None:
                logger.warning(
                    "plot_neutrino_kinematics_overlay: missing %s in truth dict; skipping panel.",
                    split_key,
                )
                continue
            ax.hist(
                truth_vals,
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                linestyle="--",
                density=True,
                color="black",
                label=rf"Truth {charge_legend}",
            )

            for model_name in model_order:
                if model_name not in model_recon:
                    continue
                recon_vals = model_recon[model_name].get(split_key)
                if recon_vals is None:
                    continue
                style = model_styles.get(model_name, {"color": "C0", "label": model_name})
                ax.hist(
                    recon_vals,
                    bins=bins,
                    range=br,
                    histtype="step",
                    linewidth=2,
                    density=True,
                    color=style.get("color", "C0"),
                    label=rf"Pred. {charge_legend} ({style.get('label', model_name)})",
                )

            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
            ax.set_ylabel("Density", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    if n_events_per_model:
        counts_str = ", ".join(
            f"{model_styles.get(m, {}).get('label', m)}: {n_events_per_model[m]:,}"
            for m in model_order if m in n_events_per_model
        )
        title_events = counts_str
    else:
        title_events = ""
    fig.suptitle(
        _analysis_suptitle(
            rf"Neutrino kinematics — predicted vs truth, split by charge  "
            rf"(top: $\nu$ from $t$, bottom: $\bar\nu$ from $\bar t$; {title_events})"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_dir) / "tt2l_neutrino_kinematics_overlay.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_single_model_vs_poisson(
    sigma: np.ndarray,
    poisson_baseline: np.ndarray,
    bin_edges: np.ndarray,
    feature_name: str,
    output_dir: str,
    truth_counts: np.ndarray,
    n_events: int,
    model_key: str,
    style: Dict[str, str],
    n_events_this_model: Optional[int] = None,
) -> None:
    """
    One figure per model: unfolded σ/N (with N = truth bin counts) vs Poisson 1/√N only.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = BINNING_CONFIG.get(feature_name, {})
    feat_label = config.get("label", feature_name)
    feat_unit = config.get("unit", "")

    n_safe = np.maximum(truth_counts.astype(float), EPSILON)
    rel_unc = sigma / n_safe

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.fill_between(
        bin_edges[:-1],
        0,
        poisson_baseline,
        step="post",
        color="lightgray",
        alpha=0.5,
        label=r"Poisson ($1/\sqrt{N}$, $N$ = truth)",
    )
    ax.step(bin_edges[:-1], poisson_baseline, where="post", color="gray", linewidth=2)
    ax.hlines(poisson_baseline[-1], bin_edges[-2], bin_edges[-1], color="gray", linewidth=2)

    color = style.get("color", "C0")
    label = style.get("label", model_key)
    ax.step(
        bin_edges[:-1],
        rel_unc,
        where="post",
        color=color,
        linewidth=2.5,
        label=rf"Unfolded $\sigma / N$ ({label})",
    )
    ax.hlines(rel_unc[-1], bin_edges[-2], bin_edges[-1], color=color, linewidth=2.5)

    xlabel = f"{feat_label} [{feat_unit}]" if feat_unit else feat_label
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Relative Uncertainty $\sigma / N$", fontsize=12, fontweight="bold")
    ax.set_title(f"Unfolding uncertainty vs Poisson — {label}", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    finite_vals = [
        np.nanmax(poisson_baseline[np.isfinite(poisson_baseline)]),
        np.nanmax(rel_unc[np.isfinite(rel_unc)]),
    ]
    finite_vals = [v for v in finite_vals if np.isfinite(v) and v > 0]
    if finite_vals:
        ax.set_ylim(0, max(finite_vals) * 1.2)

    n_display = n_events_this_model if n_events_this_model is not None else n_events
    fig.suptitle(
        _analysis_suptitle(f"{feat_label}  (events after cuts: {n_display:,})"),
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    safe_key = model_key.replace(" ", "_").replace("/", "-")
    output_path = Path(output_dir) / f"tt2l_unfolding_{safe_key}_{feature_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_unfolding_overlay(
    model_results: Dict[str, Dict[str, np.ndarray]],
    poisson_baseline: np.ndarray,
    bin_edges: np.ndarray,
    feature_name: str,
    output_dir: str,
    truth_counts: np.ndarray,
    n_events: int,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    baseline_key: str,
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """Overlay all models' unfolded σ/N on one figure with the shared Poisson baseline.

    Args:
        model_results: {model_name: {"sigma": ..., "counts": ...}}.
        poisson_baseline: 1/√N truth reference, shape (num_bins,).
        bin_edges: histogram bin edges, shape (num_bins + 1,).
        feature_name: key in BINNING_CONFIG.
        output_dir: where to write the PNG.
        truth_counts: integer truth counts per bin.
        n_events: baseline model event count (fallback when n_events_per_model absent).
        model_styles: per-model color/label dict.
        model_order: draw order.
        baseline_key: first model key (used in σ-ratio label).
        n_events_per_model: {model_name: event_count} to show per-model counts in title.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = BINNING_CONFIG.get(feature_name, {})
    feat_label = config.get("label", feature_name)
    feat_unit = config.get("unit", "")
    n_safe = np.maximum(truth_counts.astype(float), EPSILON)

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
    )
    ax_main, ax_ratio = axes

    # Poisson background
    ax_main.fill_between(
        bin_edges[:-1], 0, poisson_baseline,
        step="post", color="lightgray", alpha=0.5,
        label=r"Poisson ($1/\sqrt{N}$, $N$ = truth)",
    )
    ax_main.step(bin_edges[:-1], poisson_baseline, where="post", color="gray", linewidth=2)
    ax_main.hlines(poisson_baseline[-1], bin_edges[-2], bin_edges[-1], color="gray", linewidth=2)

    baseline_sigma: Optional[np.ndarray] = None
    for model_name in model_order:
        if model_name not in model_results:
            continue
        mr = model_results[model_name]
        sigma = mr["sigma"]
        # Use each model's own truth counts for its σ/N so the reference is self-consistent.
        own_tc = mr.get("truth_counts")
        own_n_safe = np.maximum(own_tc.astype(float), EPSILON) if own_tc is not None else n_safe
        rel_unc = sigma / own_n_safe
        style = model_styles.get(model_name, {"color": "C0", "label": model_name})
        color = style.get("color", "C0")
        label = style.get("label", model_name)

        ax_main.step(
            bin_edges[:-1], rel_unc, where="post",
            color=color, linewidth=2.5,
            label=rf"Unfolded $\sigma / N$ ({label})",
        )
        ax_main.hlines(rel_unc[-1], bin_edges[-2], bin_edges[-1], color=color, linewidth=2.5)

        if model_name == baseline_key:
            baseline_sigma = sigma

    # Ratio panel: each model σ relative to baseline
    ax_ratio.axhline(1.0, color="gray", linewidth=1.8, linestyle="--")
    if baseline_sigma is not None:
        base_safe = np.where(baseline_sigma > EPSILON, baseline_sigma, np.nan)
        for model_name in model_order:
            if model_name == baseline_key or model_name not in model_results:
                continue
            sigma = model_results[model_name]["sigma"]
            ratio = sigma / base_safe
            style = model_styles.get(model_name, {"color": "C0", "label": model_name})
            color = style.get("color", "C0")
            label = style.get("label", model_name)
            baseline_label = model_styles.get(baseline_key, {}).get("tex_short", baseline_key)
            ax_ratio.step(
                bin_edges[:-1], ratio, where="post",
                color=color, linewidth=2.0,
                label=rf"$\sigma_{{{label}}} / \sigma_{{{baseline_label}}}$",
            )
            ax_ratio.hlines(ratio[-1], bin_edges[-2], bin_edges[-1], color=color, linewidth=2.0)

    xlabel = f"{feat_label} [{feat_unit}]" if feat_unit else feat_label
    ax_ratio.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax_ratio.set_ylabel("Ratio", fontsize=11, fontweight="bold")
    ax_ratio.legend(fontsize=8, loc="upper right", frameon=True)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.set_xlim(bin_edges[0], bin_edges[-1])

    ax_main.set_ylabel(r"Relative Uncertainty $\sigma / N$", fontsize=12, fontweight="bold")
    ax_main.legend(loc="upper right", fontsize=9, frameon=True)
    ax_main.grid(True, alpha=0.3)

    all_vals = [poisson_baseline]
    for mn, mr in model_results.items():
        own_tc = mr.get("truth_counts")
        own_ns = np.maximum(own_tc.astype(float), EPSILON) if own_tc is not None else n_safe
        all_vals.append(mr["sigma"] / own_ns)
    finite_max = max(
        (np.nanmax(v[np.isfinite(v)]) for v in all_vals if np.any(np.isfinite(v))),
        default=1.0,
    )
    ax_main.set_ylim(0, finite_max * 1.25)

    if n_events_per_model:
        counts_str = ", ".join(
            f"{model_styles.get(m, {}).get('label', m)}: {n_events_per_model[m]:,}"
            for m in model_order if m in n_events_per_model
        )
        title_events = f"events after cuts — {counts_str}"
    else:
        title_events = f"events after cuts: {n_events:,}"
    fig.suptitle(
        _analysis_suptitle(rf"Unfolding overlay — {feat_label}  ({title_events})"),
        fontsize=12,
        fontweight="bold",
    )

    output_path = Path(output_dir) / f"tt2l_unfolding_overlay_{feature_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_reconstructed_top_kinematics_overlay(
    top_kinematics: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: str,
    n_events: int,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """One 1×4 overlay figure: truth (black dashed) + all models' pred. ν + truth assign."""
    if not top_kinematics:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kin_order = ["pt", "eta", "phi", "mass"]

    # Use truth curve from the first available model (shared truth)
    first_key = next((k for k in model_order if k in top_kinematics), None)
    if first_key is None:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, kin_key in zip(axes, kin_order):
        cfg = TOP_KINEMATICS_PLOT[kin_key]
        br = cfg["range"]
        nb = cfg["bins"]
        bins = np.linspace(br[0], br[1], nb + 1)
        xlabel = cfg["label"]
        if cfg["unit"]:
            xlabel = f"{xlabel} [{cfg['unit']}]"

        # Truth once (black dashed): parton-level top truth when saved, otherwise target-nu reco.
        gt_curve = _top_truth_reference(top_kinematics[first_key])
        truth_label = (
            r"Truth top"
            if top_kinematics[first_key].get("truth_top_ground") is not None
            else r"Target $\nu$ + truth assign."
        )
        ax.hist(
            gt_curve[kin_key],
            bins=bins, range=br,
            histtype="step", linewidth=2, linestyle="--",
            density=True, color="black",
            label=truth_label,
        )

        for model_name in model_order:
            if model_name not in top_kinematics:
                continue
            pred_curve = top_kinematics[model_name]["pred_nu_truth_assign"]
            style = model_styles.get(model_name, {"color": "C0", "label": model_name})
            color = style.get("color", "C0")
            label = style.get("label", model_name)
            ax.hist(
                pred_curve[kin_key],
                bins=bins, range=br,
                histtype="step", linewidth=2,
                density=True, color=color,
                label=rf"Pred. $\nu$ + truth assign. ({label})",
            )

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    if n_events_per_model:
        counts_str = ", ".join(
            f"{model_styles.get(m, {}).get('label', m)}: {n_events_per_model[m]:,}"
            for m in model_order if m in n_events_per_model
        )
        top_title_events = counts_str
    else:
        top_title_events = f"N={n_events:,}"
    fig.suptitle(
        _analysis_suptitle(
            rf"Reconstructed top overlay — truth $\nu$+truth assign. vs models"
            f" ({top_title_events}, 2 tops / event)"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = Path(output_dir) / "tt2l_reconstructed_top_kinematics_overlay.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_reconstructed_top_kinematics(
    top_kinematics: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: str,
    n_events: int,
    model_styles: Dict[str, Dict[str, str]],
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """
    One 1×4 figure per model: (1) target ν + truth assign. vs (2) pred. ν + truth assign.
    """
    if not top_kinematics:
        logger.warning(
            "No top kinematics to plot (need truth-assigned EXTRA objects or assignment_target)."
        )
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kin_order = ["pt", "eta", "phi", "mass"]

    for model_key, tk in top_kinematics.items():
        gt_curve = _top_truth_reference(tk)
        pred_curve = tk["pred_nu_truth_assign"]
        style = model_styles.get(model_key, {"color": "C0", "label": model_key})
        color = style.get("color", "C0")
        label = style.get("label", model_key)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, kin_key in zip(axes, kin_order):
            cfg = TOP_KINEMATICS_PLOT[kin_key]
            br = cfg["range"]
            nb = cfg["bins"]
            bins = np.linspace(br[0], br[1], nb + 1)
            xlabel = cfg["label"]
            if cfg["unit"]:
                xlabel = f"{xlabel} [{cfg['unit']}]"

            ax.hist(
                gt_curve[kin_key],
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                linestyle="--",
                density=True,
                color="black",
                label=(
                    r"Truth top"
                    if tk.get("truth_top_ground") is not None
                    else r"Target $\nu$ + truth assign."
                ),
            )
            ax.hist(
                pred_curve[kin_key],
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                density=True,
                color=color,
                label=rf"Pred. $\nu$ + truth assign. ({label})",
            )
            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
            ax.set_ylabel("Density", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        n_this = (n_events_per_model or {}).get(model_key, n_events)
        fig.suptitle(
            _analysis_suptitle(
                rf"Reconstructed top: truth $\nu$+truth assign. vs pred. $\nu$+truth assign. — {label} "
                f"(N={n_this:,} events, 2 tops / event)"
            ),
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        safe_key = model_key.replace(" ", "_").replace("/", "-")
        output_path = Path(output_dir) / f"tt2l_reconstructed_top_kinematics_{safe_key}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", output_path)


def _split_merged_top_kinematics(
    merged: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split ``merged_top_pt_eta_phi`` output (length ``2 * N`` per key) into top vs anti-top.

    Ordering matches ``merged_top_pt_eta_phi``: first ``N`` entries are top 1 ($t$),
    next ``N`` are top 2 ($\\bar{t}$).
    """
    out: Dict[str, Dict[str, np.ndarray]] = {"t": {}, "tbar": {}}
    n_tot = None
    for key, arr in merged.items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"expected 1D array for {key!r}, got shape {arr.shape}")
        if arr.size % 2 != 0:
            raise ValueError(f"expected even length for merged top kinematics, got {arr.size}")
        n = arr.size // 2
        if n_tot is None:
            n_tot = n
        elif n != n_tot:
            raise ValueError(f"inconsistent merged lengths: {n_tot} vs {n}")
        out["t"][key] = arr[:n]
        out["tbar"][key] = arr[n:]
    return out["t"], out["tbar"]


def _top_truth_reference(tk: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Use parton-level top truth when available; otherwise fall back to target-nu reco."""
    return tk.get("truth_top_ground") or tk["truth_nu_truth_assign"]


def plot_mass_resolution_overlay(
    top_kinematics: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    *,
    particle: str,
    output_filename: str,
    suptitle_prefix: str,
    n_events_per_model: Optional[Dict[str, int]] = None,
    rel_range: Tuple[float, float] = (-0.4, 0.4),
    n_bins: int = 60,
    baseline_key: Optional[str] = None,
) -> None:
    """Mass resolution $(m_{\\mathrm{reco}} - m_{\\mathrm{truth}}) / m_{\\mathrm{truth}}$: top or $W$.

    ``particle`` is ``\"top\"`` (reconstructed top $b+\\ell+\\nu$) or ``\"w\"`` ($W\\to\\ell\\nu$
    from lepton + neutrino, same as ``reasoning_analysis/calibration/neutrino_calibration_mw``).

    Two columns: first hadronic top vs second ($t$ / $\\bar{t}$ side). Main row: Events; lower
    row: density ratio vs baseline model.
    """
    if not top_kinematics:
        return
    if particle not in ("top", "w"):
        raise ValueError(f"particle must be 'top' or 'w', got {particle!r}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nbin = n_bins
    r = rel_range
    bin_edges = np.linspace(r[0], r[1], nbin + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    xlabel = r"$(m_{\mathrm{reco}} - m_{\mathrm{truth}}) / m_{\mathrm{truth}}$"

    base = baseline_key if baseline_key is not None else BASELINE_MODEL_KEY
    log_tag = f"plot_mass_resolution_overlay({particle})"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.09},
    )
    ax_t, ax_tb = axes[0]
    ax_rt, ax_rtb = axes[1]

    def _model_tb_bundle(model_name: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        tk = top_kinematics.get(model_name)
        if tk is None:
            return None, None
        if particle == "w":
            wk = tk.get("w")
            if wk is None:
                return None, None
            tm = wk.get("truth_nu_truth_assign")
            pm = wk.get("pred_nu_truth_assign")
            return tm, pm
        return _top_truth_reference(tk), tk.get("pred_nu_truth_assign")

    def _relative_mass_residual(
        model_name: str,
        which: str,
    ) -> Optional[np.ndarray]:
        tm, pm = _model_tb_bundle(model_name)
        if tm is None or pm is None or tm.get("mass") is None or pm.get("mass") is None:
            return None
        truth_t, truth_tb = _split_merged_top_kinematics(tm)
        pred_t, pred_tb = _split_merged_top_kinematics(pm)
        m_true = truth_t["mass"] if which == "t" else truth_tb["mass"]
        m_rec = pred_t["mass"] if which == "t" else pred_tb["mass"]
        if m_true.shape != m_rec.shape:
            logger.warning(
                "%s: shape mismatch for %s / %s: truth %s pred %s",
                log_tag,
                model_name,
                which,
                m_true.shape,
                m_rec.shape,
            )
            return None
        return (m_rec - m_true) / (m_true + 1e-6)

    def _densities_for_top(which: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for model_name in model_order:
            rel = _relative_mass_residual(model_name, which)
            if rel is None:
                continue
            d, _ = np.histogram(rel, bins=bin_edges, density=True)
            out[model_name] = d
        return out

    def _panel_title_main(which: str) -> str:
        if particle == "top":
            if which == "t":
                return rf"$t$ quark: mass resolution"
            return rf"$\bar{{t}}$ quark: mass resolution"
        if which == "t":
            return rf"$W$ boson: mass resolution ($t$ side)"
        return rf"$W$ boson: mass resolution ($\bar{{t}}$ side)"

    def _draw_main(ax: Any, which: str) -> None:
        title = _panel_title_main(which)
        for model_name in model_order:
            rel = _relative_mass_residual(model_name, which)
            if rel is None:
                continue
            style = model_styles.get(model_name, {"color": "C0", "label": model_name})
            color = style.get("color", "C0")
            label = style.get("label", model_name)
            ax.hist(
                rel,
                bins=nbin,
                range=r,
                histtype="step",
                color=color,
                label=label,
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("Events")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelbottom=False)

    def _draw_ratio(ax: Any, which: str) -> None:
        dens = _densities_for_top(which)
        if not dens:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        if base in dens:
            b_ref = base
        else:
            b_ref = next((m for m in model_order if m in dens), None)
            if b_ref is None:
                return

        b_d = dens[b_ref]
        b_short = model_styles.get(b_ref, {}).get("tex_short", b_ref)
        base_safe = np.where(b_d > EPSILON, b_d, np.nan)
        ax.axhline(
            1.0,
            color="gray",
            linewidth=1.8,
            linestyle="--",
            label=f"baseline PDF ({b_short})",
        )
        for model_name in model_order:
            if model_name not in dens or model_name == b_ref:
                continue
            ratio = dens[model_name] / base_safe
            style = model_styles.get(model_name, {"color": "C0", "label": model_name})
            color = style.get("color", "C0")
            short = style.get("tex_short", model_name)
            ax.step(
                bin_centers,
                ratio,
                where="mid",
                color=color,
                linewidth=2.0,
                label=f"{short} / {b_short}",
            )
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel("Ratio", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(r[0], r[1])

    _draw_main(ax_t, "t")
    _draw_main(ax_tb, "tbar")
    _draw_ratio(ax_rt, "t")
    _draw_ratio(ax_rtb, "tbar")

    if n_events_per_model:
        counts_str = ", ".join(
            f"{model_styles.get(m, {}).get('label', m)}: {n_events_per_model[m]:,}"
            for m in model_order if m in n_events_per_model
        )
        title_events = counts_str
    else:
        title_events = ""

    fig.suptitle(
        _analysis_suptitle(
            f"{suptitle_prefix} ({title_events})  —  lower: density ratio vs {base}"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    output_path = Path(output_dir) / output_filename
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_top_mass_overlay(
    top_kinematics: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    n_events_per_model: Optional[Dict[str, int]] = None,
    rel_range: Tuple[float, float] = (-0.4, 0.4),
    n_bins: int = 60,
    baseline_key: Optional[str] = None,
) -> None:
    """Relative top mass resolution (see :func:`plot_mass_resolution_overlay`)."""
    plot_mass_resolution_overlay(
        top_kinematics,
        output_dir,
        model_styles,
        model_order,
        particle="top",
        output_filename="tt2l_top_mass_overlay.png",
        suptitle_prefix="Top mass resolution",
        n_events_per_model=n_events_per_model,
        rel_range=rel_range,
        n_bins=n_bins,
        baseline_key=baseline_key,
    )


def plot_w_mass_overlay(
    top_kinematics: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    n_events_per_model: Optional[Dict[str, int]] = None,
    rel_range: Tuple[float, float] = (-0.4, 0.4),
    n_bins: int = 60,
    baseline_key: Optional[str] = None,
) -> None:
    """$W \\to \\ell\\nu$ mass resolution (lepton + $\\nu$; calibration MW convention)."""
    plot_mass_resolution_overlay(
        top_kinematics,
        output_dir,
        model_styles,
        model_order,
        particle="w",
        output_filename="tt2l_w_mass_overlay.png",
        suptitle_prefix="W boson mass resolution",
        n_events_per_model=n_events_per_model,
        rel_range=rel_range,
        n_bins=n_bins,
        baseline_key=baseline_key,
    )


def plot_w_kinematics_overlay(
    top_kinematics: Dict[str, Dict[str, Any]],
    output_dir: str,
    n_events: int,
    model_styles: Dict[str, Dict[str, str]],
    model_order: List[str],
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """One 1×4 overlay: absolute $W$ kinematics (truth $\\nu$+truth assign. vs models' pred.)."""
    if not top_kinematics:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kin_order = ["pt", "eta", "phi", "mass"]

    first_key = next((k for k in model_order if k in top_kinematics), None)
    if first_key is None:
        return
    w0 = top_kinematics[first_key].get("w")
    if w0 is None:
        logger.warning("No W kinematics in top_kinematics (missing 'w' bundle).")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, kin_key in zip(axes, kin_order):
        cfg = W_KINEMATICS_PLOT[kin_key]
        br = cfg["range"]
        nb = cfg["bins"]
        bins = np.linspace(br[0], br[1], nb + 1)
        xlabel = cfg["label"]
        if cfg["unit"]:
            xlabel = f"{xlabel} [{cfg['unit']}]"

        gt_curve = w0["truth_nu_truth_assign"]
        ax.hist(
            gt_curve[kin_key],
            bins=bins,
            range=br,
            histtype="step",
            linewidth=2,
            linestyle="--",
            density=True,
            color="black",
            label=r"Target $\nu$ + truth assign.",
        )

        for model_name in model_order:
            if model_name not in top_kinematics:
                continue
            wk = top_kinematics[model_name].get("w")
            if wk is None:
                continue
            pred_curve = wk["pred_nu_truth_assign"]
            style = model_styles.get(model_name, {"color": "C0", "label": model_name})
            color = style.get("color", "C0")
            label = style.get("label", model_name)
            ax.hist(
                pred_curve[kin_key],
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                density=True,
                color=color,
                label=rf"Pred. $\nu$ + truth assign. ({label})",
            )

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    if n_events_per_model:
        counts_str = ", ".join(
            f"{model_styles.get(m, {}).get('label', m)}: {n_events_per_model[m]:,}"
            for m in model_order if m in n_events_per_model
        )
        w_title_events = counts_str
    else:
        w_title_events = f"N={n_events:,}"
    fig.suptitle(
        _analysis_suptitle(
            rf"Reconstructed $W\to\ell\nu$ overlay (lepton from assignment) — "
            rf"truth $\nu$+truth assign. vs models ({w_title_events}, 2 $W$ / event)"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = Path(output_dir) / "tt2l_w_kinematics_overlay.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_w_reconstructed_kinematics(
    top_kinematics: Dict[str, Dict[str, Any]],
    output_dir: str,
    n_events: int,
    model_styles: Dict[str, Dict[str, str]],
    n_events_per_model: Optional[Dict[str, int]] = None,
) -> None:
    """One 1×4 figure per model: absolute $W$ kinematics, target $\\nu$+truth vs pred.+truth."""
    if not top_kinematics:
        logger.warning("No top kinematics for W plots.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kin_order = ["pt", "eta", "phi", "mass"]

    for model_key, tk in top_kinematics.items():
        wk = tk.get("w")
        if wk is None:
            continue
        gt_curve = wk["truth_nu_truth_assign"]
        pred_curve = wk["pred_nu_truth_assign"]
        style = model_styles.get(model_key, {"color": "C0", "label": model_key})
        color = style.get("color", "C0")
        label = style.get("label", model_key)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, kin_key in zip(axes, kin_order):
            cfg = W_KINEMATICS_PLOT[kin_key]
            br = cfg["range"]
            nb = cfg["bins"]
            bins = np.linspace(br[0], br[1], nb + 1)
            xlabel = cfg["label"]
            if cfg["unit"]:
                xlabel = f"{xlabel} [{cfg['unit']}]"

            ax.hist(
                gt_curve[kin_key],
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                linestyle="--",
                density=True,
                color="black",
                label=r"Target $\nu$ + truth assign.",
            )
            ax.hist(
                pred_curve[kin_key],
                bins=bins,
                range=br,
                histtype="step",
                linewidth=2,
                density=True,
                color=color,
                label=rf"Pred. $\nu$ + truth assign. ({label})",
            )
            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
            ax.set_ylabel("Density", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        n_this = (n_events_per_model or {}).get(model_key, n_events)
        fig.suptitle(
            _analysis_suptitle(
                rf"Reconstructed $W\to\ell\nu$: target $\nu$+truth assign. vs pred. $\nu$+truth assign. "
                rf"— {label} (N={n_this:,} events, 2 $W$ / event; lepton from assignment)"
            ),
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        safe_key = model_key.replace(" ", "_").replace("/", "-")
        output_path = Path(output_dir) / f"tt2l_w_reconstructed_kinematics_{safe_key}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", output_path)


def run_two_model_unfolding(
    model_paths: Dict[str, str],
    output_dir: str,
    apply_recon_cuts: bool = True,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """End-to-end: load, cut, top kinematic plots, unfold, per-model plots.

    Any number of models is allowed (dict insertion order sets overlay order; first
    key is the baseline for shared truth / ratio panels).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_order = list(model_paths.keys())
    baseline_key = model_order[0]

    model_recon, model_truth, truth_features, n_events, top_kinematics, n_events_per_model = (
        load_aligned_model_features(
            model_paths,
            apply_recon_cuts=apply_recon_cuts,
            verbose=verbose,
        )
    )

    plot_reconstructed_top_kinematics(
        top_kinematics, output_dir, n_events, MODEL_STYLES,
        n_events_per_model=n_events_per_model,
    )
    plot_reconstructed_top_kinematics_overlay(
        top_kinematics, output_dir, n_events, MODEL_STYLES,
        model_order=model_order, n_events_per_model=n_events_per_model,
    )
    plot_top_mass_overlay(
        top_kinematics=top_kinematics,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
        baseline_key=baseline_key,
    )
    plot_w_mass_overlay(
        top_kinematics=top_kinematics,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
        baseline_key=baseline_key,
    )
    plot_w_reconstructed_kinematics(
        top_kinematics,
        output_dir,
        n_events,
        MODEL_STYLES,
        n_events_per_model=n_events_per_model,
    )
    plot_w_kinematics_overlay(
        top_kinematics,
        output_dir,
        n_events,
        MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
    )
    plot_neutrino_kinematics_overlay(
        model_recon=model_recon,
        model_truth=model_truth,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
    )

    if not ROOUNFOLD_AVAILABLE:
        logger.error("RooUnfold not available; skipping unfolding plots.")
        return {"top_kinematics": top_kinematics}

    all_results: Dict[str, Any] = {"top_kinematics": top_kinematics}

    for feat_name in FEATURES_TO_ANALYZE:
        logger.info("Feature %s", feat_name)
        config = BINNING_CONFIG[feat_name]
        num_bins = config["num_bins"]
        bin_range = config["range"]
        # Shared Poisson baseline uses baseline model's truth distribution.
        baseline_truth_data = truth_features[feat_name]

        bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        model_results: Dict[str, Dict[str, Any]] = {}
        for model_name in model_order:
            # Each model uses its own truth for the unfolding response matrix and
            # its own Poisson baseline so the reference is self-consistent.
            own_truth_data = model_truth[model_name][feat_name]
            recon_data = model_recon[model_name][feat_name]
            own_truth_counts, _ = np.histogram(own_truth_data, bins=num_bins, range=bin_range)
            own_truth_counts_safe = np.maximum(own_truth_counts.astype(float), EPSILON)
            own_poisson = 1.0 / np.sqrt(own_truth_counts_safe)
            try:
                sigma, unfolded_counts, response_matrix = calculate_unfolded_uncertainty(
                    own_truth_data,
                    recon_data,
                    num_bins,
                    bin_range,
                    kreg=SVD_KREG,
                    verbose=verbose,
                )
                model_results[model_name] = {
                    "sigma": sigma,
                    "counts": unfolded_counts,
                    "response_matrix": response_matrix,
                    "truth_counts": own_truth_counts,
                    "poisson_baseline": own_poisson,
                }
            except Exception as e:
                logger.exception("Unfolding failed for %s / %s: %s", model_name, feat_name, e)

        if not model_results:
            continue

        # Shared overlay baseline: use the E2E (baseline) model's truth distribution
        # so the single reference line in the overlay is well-defined.
        baseline_truth_counts, _ = np.histogram(baseline_truth_data, bins=num_bins, range=bin_range)
        baseline_truth_counts_safe = np.maximum(baseline_truth_counts.astype(float), EPSILON)
        poisson_baseline_shared = 1.0 / np.sqrt(baseline_truth_counts_safe)

        for model_name in model_order:
            if model_name not in model_results:
                continue
            mr = model_results[model_name]
            plot_single_model_vs_poisson(
                sigma=mr["sigma"],
                poisson_baseline=mr["poisson_baseline"],
                bin_edges=bin_edges,
                feature_name=feat_name,
                output_dir=output_dir,
                truth_counts=mr["truth_counts"],
                n_events=n_events,
                model_key=model_name,
                style=MODEL_STYLES.get(model_name, {"color": "C0", "label": model_name}),
                n_events_this_model=n_events_per_model.get(model_name),
            )

        plot_unfolding_overlay(
            model_results=model_results,
            poisson_baseline=poisson_baseline_shared,
            bin_edges=bin_edges,
            feature_name=feat_name,
            output_dir=output_dir,
            truth_counts=baseline_truth_counts,
            n_events=n_events,
            model_styles=MODEL_STYLES,
            model_order=model_order,
            baseline_key=baseline_key,
            n_events_per_model=n_events_per_model,
        )
        all_results[feat_name] = {"model_results": model_results}

    return all_results


# Optional fallback prediction files. Insertion order = overlay order; first
# entry is the baseline used for the shared truth / Poisson reference.
MODEL_PATHS: Dict[str, str] = {}

OUTPUT_DIR = "outputs/unfolding_neutrino"


def _parse_model_key_path(spec: str) -> tuple[str, str]:
    """Parse a KEY=PATH model specification from the CLI."""
    if "=" not in spec:
        raise ValueError("model entries must have the form KEY=PATH")
    key, path = spec.split("=", 1)
    key = key.strip()
    path = path.strip()
    if not key or not path:
        raise ValueError("model entries must have non-empty KEY and PATH")
    return key, path


def run_overlays_only(
    model_paths: Dict[str, str],
    output_dir: str,
    apply_recon_cuts: bool = True,
    verbose: bool = True,
) -> None:
    """Generate only the overlay PNGs (no per-model figures).

    Mirrors the overlay subset of the built-in multi-model unfolding pipeline.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_order = list(model_paths.keys())
    baseline_key = model_order[0]

    model_recon, model_truth, truth_features, n_events, top_kinematics, n_events_per_model = (
        load_aligned_model_features(
            model_paths,
            apply_recon_cuts=apply_recon_cuts,
            verbose=verbose,
        )
    )

    plot_reconstructed_top_kinematics_overlay(
        top_kinematics, output_dir, n_events, MODEL_STYLES,
        model_order=model_order, n_events_per_model=n_events_per_model,
    )
    plot_top_mass_overlay(
        top_kinematics=top_kinematics,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
        baseline_key=baseline_key,
    )
    plot_w_mass_overlay(
        top_kinematics=top_kinematics,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
        baseline_key=baseline_key,
    )
    plot_w_kinematics_overlay(
        top_kinematics,
        output_dir,
        n_events,
        MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
    )
    plot_neutrino_kinematics_overlay(
        model_recon=model_recon,
        model_truth=model_truth,
        output_dir=output_dir,
        model_styles=MODEL_STYLES,
        model_order=model_order,
        n_events_per_model=n_events_per_model,
    )

    if not ROOUNFOLD_AVAILABLE:
        logger.error("RooUnfold not available; skipping unfolding overlays.")
        return

    for feat_name in FEATURES_TO_ANALYZE:
        logger.info("Feature %s", feat_name)
        config = BINNING_CONFIG[feat_name]
        num_bins = config["num_bins"]
        bin_range = config["range"]
        bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        model_results: Dict[str, Dict[str, object]] = {}
        for model_name in model_order:
            own_truth_data = model_truth[model_name][feat_name]
            recon_data = model_recon[model_name][feat_name]
            own_truth_counts, _ = np.histogram(own_truth_data, bins=num_bins, range=bin_range)
            own_truth_counts_safe = np.maximum(own_truth_counts.astype(float), EPSILON)
            own_poisson = 1.0 / np.sqrt(own_truth_counts_safe)
            try:
                sigma, unfolded_counts, response_matrix = calculate_unfolded_uncertainty(
                    own_truth_data,
                    recon_data,
                    num_bins,
                    bin_range,
                    kreg=SVD_KREG,
                    verbose=verbose,
                )
                model_results[model_name] = {
                    "sigma": sigma,
                    "counts": unfolded_counts,
                    "response_matrix": response_matrix,
                    "truth_counts": own_truth_counts,
                    "poisson_baseline": own_poisson,
                }
            except Exception as e:
                logger.exception("Unfolding failed for %s / %s: %s", model_name, feat_name, e)

        if not model_results:
            continue

        baseline_truth_data = truth_features[feat_name]
        baseline_truth_counts, _ = np.histogram(
            baseline_truth_data, bins=num_bins, range=bin_range
        )
        baseline_truth_counts_safe = np.maximum(baseline_truth_counts.astype(float), EPSILON)
        poisson_baseline_shared = 1.0 / np.sqrt(baseline_truth_counts_safe)

        plot_unfolding_overlay(
            model_results=model_results,
            poisson_baseline=poisson_baseline_shared,
            bin_edges=bin_edges,
            feature_name=feat_name,
            output_dir=output_dir,
            truth_counts=baseline_truth_counts,
            n_events=n_events,
            model_styles=MODEL_STYLES,
            model_order=model_order,
            baseline_key=baseline_key,
            n_events_per_model=n_events_per_model,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS TT2L neutrino unfolding overlays from saved prediction tensors."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="KEY=PATH",
        help=(
            "Logical model name and prediction .pt path. Repeat once per model; "
            "the first model is the ratio baseline."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--no_recon_cuts",
        action="store_true",
        help="Skip base reconstruction cuts while keeping required validity masks.",
    )
    args = parser.parse_args()

    if args.model:
        model_paths: Dict[str, str] = {}
        for spec in args.model:
            try:
                key, path = _parse_model_key_path(spec)
            except ValueError as exc:
                parser.error(str(exc))
            model_paths[key] = path
    else:
        model_paths = dict(MODEL_PATHS)
    if not model_paths:
        parser.error("Provide at least one --model KEY=PATH entry.")

    logger.info("Output directory: %s", args.output_dir)
    logger.info("Models: %s", model_paths)
    run_overlays_only(
        model_paths=model_paths,
        output_dir=args.output_dir,
        apply_recon_cuts=not args.no_recon_cuts,
        verbose=True,
    )


if __name__ == "__main__":
    main()
