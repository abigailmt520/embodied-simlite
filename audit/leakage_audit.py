# -*- coding: utf-8 -*-
"""
leakage_audit.py  ——  Phase3 · contract-layer "mutual-information leakage audit" (C_I audit, a principled generalization of C1)
==================================================================================
Invariant: the mutual information between the legitimate report channel (odom) and truth is upper-bounded by its **noise budget**;
        leakage = I(odom; truth) exceeds that noise-budget bound. Symmetric with the physics layer's "energy-conservation residual"
        (physics: ΔE conserved; contract: I(report;truth) ≤ noise budget).

🔴 Engineering judgment (correction to Opus's original design, see docs/Phase3 §3 "design pitfalls"):
  the original design "I(odom_pos;truth_pos) ≤ ½log(1+Var/σ²)" does not hold for **absolute position** — odom is the **cumulative integral**
  of dead reckoning; at episode start odom≡truth, then drifts cumulatively; absolute-position MI is dominated by the shared
  trajectory correlation (not the noise channel), and is near-infinite after reset. **The correct approach: estimate MI on per-step displacement increments (Δpos)**
  (a memoryless channel). The noise-budget bound uses the MI of the **truth increments re-noised by the declared noise model** (a Monte-Carlo operationalization),
  with the closed-form ½log(1+SNR) only as a reference (its AWGN assumption holds only approximately for multiplicative slip noise).

Criterion (C_I audit, mirroring the C1/EC gating): within a sliding window estimate I(Δodom;Δtruth); over "the noise-budget bound + margin"
        and persisting → flag RED + localize.

Reliability (INV-E): KSG saturates and is biased for high MI (high SNR / near-determinism); so for the platform-deployed slip=0.05
        (odom very accurate, bound ~3.3 nats approaching the KSG reliable ceiling) detection power is limited (the L-3 rotation leak is missed);
        in the estimable regime slip≈0.3 all three leaks are cleanly detected. See run_leakage_audit.py measurements + docs.
"""

import numpy as np

from mi_estimator import ksg_mi, increments

MARGIN_NATS = 0.4     # the RED margin above the noise-budget bound (calibrated to the clean / re-noise MC variance)
K_PERSIST = 2         # the consecutive-over-bound window-count threshold (mirrors C3 STALE_TOL)


def _result(name, desc, ok, detail, locator=None):
    return {"check": name, "desc": desc, "status": "GREEN" if ok else "RED",
            "ok": bool(ok), "detail": detail, "locator": locator}


# ---- noise-budget bound: the MI of truth increments re-noised by the declared noise model (operationalization) + closed-form reference ----
def noise_budget_bound(d_truth, slip, n_mc=5, seed=0):
    """Re-noise the truth increments by the declared slip model, estimate its I(Δtruth;Δodom_legit) as the noise-budget bound (nats).

    Declared model (the increment approximation of embodied_env:494-495): Δodom = (1+slip)·Δtruth + slip·|Δtruth|·N(0,I).
    Returns (budget_mean, budget_std, closed_form_ref).
    """
    rng = np.random.default_rng(seed)
    mag = np.linalg.norm(d_truth, axis=1, keepdims=True)
    mis = []
    for _ in range(n_mc):
        legit = (1.0 + slip) * d_truth + slip * mag * rng.normal(0, 1, d_truth.shape)
        mis.append(ksg_mi(d_truth, legit))
    # closed-form reference: ½log(1+SNR), SNR=Var(Δtruth)/Var(noise), noise≈slip·|Δtruth| (per-component, approximate)
    var_truth = float(np.mean(np.var(d_truth, axis=0)))
    var_noise = float(slip ** 2 * np.mean(mag ** 2))
    closed = 0.5 * np.log(1.0 + var_truth / max(var_noise, 1e-12)) * d_truth.shape[1]
    return float(np.mean(mis)), float(np.std(mis)), float(closed)


def estimate_leakage_mi(truth_traj, odom_traj):
    """Estimate I(Δodom; Δtruth) on a (truth_pos, odom_pos) trajectory (full sample, nats)."""
    dT, dO = increments(truth_traj), increments(odom_traj)
    return ksg_mi(dT, dO)


# ====================================================================
# C_I audit: sliding-window I(Δodom;Δtruth) over "the noise-budget bound + margin" and persisting → flag RED
# ====================================================================
def ci_audit(truth_traj, odom_traj, slip, window=500, stride=250, margin=MARGIN_NATS):
    """Run the C_I audit on a (truth_pos, odom_pos) trajectory. Returns result (incl. full-sample MI, bound, per-window)."""
    dT, dO = increments(truth_traj), increments(odom_traj)
    N = dT.shape[0]
    budget, b_std, closed = noise_budget_bound(dT, slip)
    thresh = budget + margin
    mi_full = ksg_mi(dT, dO)

    # sliding window + persistence gate
    over_run = 0
    flagged_from = None
    win_mis = []
    for s in range(0, max(1, N - window + 1), stride):
        wm = ksg_mi(dT[s:s + window], dO[s:s + window])
        win_mis.append((s, round(wm, 3)))
        if wm > thresh:
            if over_run == 0:
                flagged_from = s
            over_run += 1
        else:
            over_run = 0
    # degenerate to a full-sample verdict when there are too few windows
    persistent = over_run >= K_PERSIST or (len(win_mis) < K_PERSIST + 1 and mi_full > thresh)
    leak = (mi_full > thresh) and persistent

    loc = {"mi_full_nats": round(mi_full, 3), "budget_nats": round(budget, 3),
           "budget_std": round(b_std, 3), "threshold_nats": round(thresh, 3),
           "closed_form_ref_nats": round(closed, 3), "excess_nats": round(mi_full - budget, 3),
           "n_increments": N, "flagged_window_from": flagged_from}
    if leak:
        return _result("CI_MI_LEAKAGE", "I(odom;truth) within the noise-budget bound", False,
                       f"mutual information I(Δodom;Δtruth)={mi_full:.3f} nats over the noise-budget bound {budget:.3f}"
                       f"(+margin {margin}) {thresh:.3f}; over by {mi_full-budget:+.3f} nats (leak)", loc)
    return _result("CI_MI_LEAKAGE", "I(odom;truth) within the noise-budget bound", True,
                   f"mutual information I={mi_full:.3f} nats ≤ noise-budget bound {budget:.3f}(+margin) {thresh:.3f}"
                   f" (legitimate, no leak)", loc)


# ====================================================================
# leakage injectors (contract layer, hidden in the odom report; test-only to prove it catches fakes, never enters production)
#   act on (truth_traj, odom_traj_orig) → return the leaked odom_traj.
# ====================================================================
def inject_l1_full_leak(truth_traj, odom_traj):
    """L-1 full leak: odom secretly returns truth (Δodom=Δtruth). Error≈0, C1 can also catch it."""
    return np.asarray(truth_traj, dtype=np.float64).copy()


def inject_l2_partial_leak(truth_traj, odom_traj, shrink=0.25):
    """L-2 partial leak: odom returns truth + noise smaller than declared (each step's error shrunk to shrink×).
    There is still systematic drift (error nonzero) → C1 easily misses; but information over budget → C_I catches."""
    T = np.asarray(truth_traj, np.float64); O = np.asarray(odom_traj, np.float64)
    dT, dO = increments(T), increments(O)
    err = dO - dT
    dO_new = dT + shrink * err               # error shrunk → noise smaller than declared → partial leak
    out = np.empty_like(T); out[0] = T[0]
    out[1:] = T[0] + np.cumsum(dO_new, axis=0)
    return out


def inject_l3_privileged(truth_traj, odom_traj, deg=35.0):
    """L-3 privileged deterministic transform: odom increment = rotated truth increment (a deterministic invertible function).
    The position error is **large** (C1 sees a large error → mistakes it as legitimate) but is fully determined by truth → I=∞ → C_I catches.
    Replicates "odom returns a privileged feature (a deterministic function of truth) instead of a noisy observation"."""
    T = np.asarray(truth_traj, np.float64)
    dT = increments(T)
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dO_new = dT @ R.T                        # deterministic rotation (no noise)
    out = np.empty_like(T); out[0] = T[0]
    out[1:] = T[0] + np.cumsum(dO_new, axis=0)
    return out


LEAK_INJECTORS = {
    "L-1_full_leak": inject_l1_full_leak,
    "L-2_partial_leak": inject_l2_partial_leak,
    "L-3_privileged": inject_l3_privileged,
}
LEAK_DESCRIPTIONS = {
    "L-1_full_leak": "odom secretly = truth (full leak, error≈0, C1 also catches)",
    "L-2_partial_leak": "odom = truth + noise smaller than declared (partial leak, still drifts → C1 easily misses)",
    "L-3_privileged": "odom = a deterministic rotation of truth (privileged transform, large error but deterministic → C1 mistakes it as legitimate)",
}
