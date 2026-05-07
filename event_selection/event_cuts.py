"""
Shared event cuts for TT2L analysis.

Exports compute_base_recon_cut, compute_event_mask, and related utilities
used by neutrino_calibration and RL scripts to exclude bad events.
"""

import numpy as np
from typing import Tuple, Dict, Optional

# Point cloud key in prediction batches
POINTCLOUD_KEY = "full_input_point_cloud"

# Point cloud feature indices
PC_IDX_PT = 1       # pt (log-transformed, decode with expm1)
PC_IDX_ETA = 2      # eta
PC_IDX_PHI = 3      # phi
PC_IDX_BTAG = 4     # btag (>0.5 for b-jet)
PC_IDX_ISLEPTON = 5  # isLepton (>0.5 for lepton)
PC_IDX_CHARGE = 6   # charge

# Reconstruction cut thresholds (TT2L analysis selection)
LEPTON_PT_MIN = 25.0   # GeV
LEPTON_ETA_MAX = 2.47
BJET_PT_MIN = 25.0     # GeV
BJET_ETA_MAX = 2.5


def get_value(d: dict, key: str):
    """Helper to get nested dictionary value using '/' separator."""
    out = d
    for k in key.split("/"):
        out = out[k]
    return out


def compute_base_recon_cut(point_cloud) -> Tuple[np.ndarray, Dict]:
    """
    Compute TT2L base reconstruction cut from point cloud.

    Cuts: num_bjet>0, num_lepton==2, total_charge==0,
    lepton pT>25, |lepton eta|<2.47, opposite sign leptons,
    b-jet pT>25, |b-jet eta|<2.5.

    Returns:
        base_recon_cut: bool array [N], True = pass
        cut_stats: dict with per-cut statistics
    """
    if hasattr(point_cloud, "numpy"):
        pc = point_cloud.numpy()
    else:
        pc = np.asarray(point_cloud)

    N = pc.shape[0]
    pt = np.expm1(pc[:, :, PC_IDX_PT])
    eta = pc[:, :, PC_IDX_ETA]
    btag = pc[:, :, PC_IDX_BTAG]
    isLepton = pc[:, :, PC_IDX_ISLEPTON]
    charge = pc[:, :, PC_IDX_CHARGE]

    num_lepton = isLepton.sum(axis=1).astype(np.int32)
    num_bjet = (btag > 0.5).sum(axis=1).astype(np.int32)
    total_charge = charge.sum(axis=1).astype(np.int32)

    cut_num_bjet = num_bjet > 0
    cut_num_lepton = num_lepton == 2
    cut_total_charge = total_charge == 0

    lepton_mask = isLepton > 0.5
    bjet_mask = btag > 0.5

    lepton_pt = np.where(lepton_mask, pt, np.nan)
    lepton_eta = np.where(lepton_mask, eta, np.nan)
    lepton_charge = np.where(lepton_mask, charge, np.nan)

    lepton_pt_min = np.nanmin(lepton_pt, axis=1)
    lepton_eta_max = np.nanmax(np.abs(lepton_eta), axis=1)
    lepton_charge_sum = np.nansum(lepton_charge, axis=1)

    cut_lepton_pt = lepton_pt_min > LEPTON_PT_MIN
    cut_lepton_eta = lepton_eta_max < LEPTON_ETA_MAX
    cut_lepton_charge = np.abs(lepton_charge_sum) < 0.5

    bjet_pt = np.where(bjet_mask, pt, np.nan)
    bjet_eta = np.where(bjet_mask, eta, np.nan)
    bjet_pt_min = np.nanmin(bjet_pt, axis=1)
    bjet_eta_max = np.nanmax(np.abs(bjet_eta), axis=1)

    cut_bjet_pt = np.where(num_bjet > 0, bjet_pt_min > BJET_PT_MIN, False)
    cut_bjet_eta = np.where(num_bjet > 0, bjet_eta_max < BJET_ETA_MAX, False)

    base_recon_cut = (
        cut_num_bjet
        & cut_num_lepton
        & cut_total_charge
        & cut_lepton_pt
        & cut_lepton_eta
        & cut_lepton_charge
        & cut_bjet_pt
        & cut_bjet_eta
    )

    cut_stats = {
        "total_events": N,
        "num_bjet_gt_0": cut_num_bjet.sum(),
        "num_lepton_eq_2": cut_num_lepton.sum(),
        "total_charge_eq_0": cut_total_charge.sum(),
        "lepton_pt_gt_25": cut_lepton_pt.sum(),
        "lepton_eta_lt_2p47": cut_lepton_eta.sum(),
        "lepton_opposite_charge": cut_lepton_charge.sum(),
        "bjet_pt_gt_25": cut_bjet_pt.sum(),
        "bjet_eta_lt_2p5": cut_bjet_eta.sum(),
        "pass_all_cuts": base_recon_cut.sum(),
    }

    return base_recon_cut, cut_stats


def print_cut_flow(cut_stats: Dict, prefix: str = "  ") -> None:
    """Print cut flow table."""
    total = cut_stats["total_events"]
    print(f"{prefix}Cut Flow Table:")
    print(f"{prefix}{'='*60}")
    print(f"{prefix}{'Cut':<35} {'N Events':>10} {'Efficiency':>12}")
    print(f"{prefix}{'-'*60}")
    print(f"{prefix}{'Total events':<35} {total:>10} {'100.00%':>12}")

    cuts = [
        ("num_bjet > 0", "num_bjet_gt_0"),
        ("num_lepton == 2", "num_lepton_eq_2"),
        ("total_charge == 0", "total_charge_eq_0"),
        ("lepton pT > 25 GeV", "lepton_pt_gt_25"),
        ("|lepton eta| < 2.47", "lepton_eta_lt_2p47"),
        ("opposite lepton charge", "lepton_opposite_charge"),
        ("b-jet pT > 25 GeV", "bjet_pt_gt_25"),
        ("|b-jet eta| < 2.5", "bjet_eta_lt_2p5"),
    ]

    for cut_name, key in cuts:
        n = cut_stats[key]
        eff = 100 * n / total if total > 0 else 0
        print(f"{prefix}{cut_name:<35} {n:>10} {eff:>11.2f}%")

    print(f"{prefix}{'-'*60}")
    n_pass = cut_stats["pass_all_cuts"]
    eff = 100 * n_pass / total if total > 0 else 0
    print(f"{prefix}{'Pass all cuts (base_recon_cut)':<35} {n_pass:>10} {eff:>11.2f}%")
    print(f"{prefix}{'='*60}")


def compute_event_mask(
    assign0_msk: np.ndarray,
    assign1_msk: np.ndarray,
    nu_pred_log_pt: np.ndarray,
    point_cloud: Optional[np.ndarray] = None,
    apply_recon_cuts: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute combined event mask to exclude bad events.

    Applies:
      1. Assignment mask (assign0 & assign1)
      2. Valid neutrino (non-zero predictions)
      3. Base reconstruction cut (when point_cloud available)

    Args:
        assign0_msk: [N] bool
        assign1_msk: [N] bool
        nu_pred_log_pt: [N, 2] log-transformed pT (use first/best sample)
        point_cloud: [N, num_particles, features] or None
        apply_recon_cuts: whether to apply base_recon_cut when point_cloud exists
        verbose: print cut flow

    Returns:
        keep: [N] bool, True = pass all cuts
    """
    keep = (assign0_msk == True) & (assign1_msk == True)

    keep_event = (np.abs(nu_pred_log_pt) > 0).all(axis=1)
    keep = keep & keep_event

    if apply_recon_cuts and point_cloud is not None:
        if verbose:
            print("\n[INFO] Applying base_recon_cut (TT2L analysis selection)...")
        pc_subset = point_cloud[keep]
        recon_cut_mask, cut_stats = compute_base_recon_cut(pc_subset)
        if verbose:
            print_cut_flow(cut_stats, prefix="  ")
        # recon_cut_mask applies to the subset; map back to full indices
        keep_indices = np.where(keep)[0]
        keep[keep_indices[~recon_cut_mask]] = False

    elif apply_recon_cuts and point_cloud is None and verbose:
        print(f"[WARNING] Cannot apply recon cuts - '{POINTCLOUD_KEY}' not found")

    return keep
