# -*- coding: utf-8 -*-
"""
run_baseline_comparison.py  ——  S-A external baseline: general anomaly detectors vs the structured oracle
========================================================================================================
Question: do off-the-shelf anomaly detectors (IsolationForest, LOF), given increasingly privileged
features (B1 report-only -> B2 +truth/residual/velocity -> B3 +per-point map), catch the *irreducible*
relational fault (d<xi Scenario B: both truth and report in free space, only the displacement segment
crosses the wall)?  Hypothesis (NOT a foregone conclusion): they catch the gross spatial faults but miss
the relational one — because none of B1/B2/B3 contains the pairwise segment-crossing feature, which *is*
the oracle.  We report the actual per-cell rates; if a baseline catches d<xi Scenario B, or misses a gross
fault, we say so.

Design (locked after recon, S-A step 1):
  * detectors = IsolationForest + LOF(novelty)   (One-Class SVM dropped: probe showed gamma-degeneracy at
    gamma=1.0 and notable cross-fit instability; LOF is stable. See meta + the step-1 report.)
  * Scenario B = the d<xi irreducible version (residual ||o-x|| <= xi, overlapping honest), NOT the
    large-residual coverage_matrix.scenario_b().
  * comparison = spatial-fault family only {healthy(=FP), Scenario A phantom, Scenario B d<xi,
    CF-2 penetration, L-1/L-2 leak}.  Non-spatial faults (seq/stall, data_tamper, P-1 energy) are a
    different feature channel — reported as a note, not as baseline failures.
  * per-SCENARIO matched honest/fault (same region+geometry, fault off vs on); StandardScaler fit per
    scenario (never pool raw coordinates across geometries).
  * per-frame anomaly flag -> per-trajectory stat = fraction of anomalous frames -> threshold = a
    persistence gate, calibrated on a held-out honest split to (a) fixed 5% per-trajectory FP and
    (b) oracle-matched (~0%) FP.  This mirrors the relational oracle's persist_frac mechanism.

Reproducible; writes audit/baseline_comparison.json (+ .png).  Imports repo generators/oracles read-only.
Run:  conda run -n base python audit/run_baseline_comparison.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodied_env import EmbodiedNavEnv as Env                                  # noqa: E402
from relational_oracle import (physics_oracle, relational_oracle,              # noqa: E402
                               dist_point_aabb, _point_in_aabb)
import run_scenB_irreducibility as RS                                          # noqa: E402
from leakage_audit import inject_l1_full_leak, inject_l2_partial_leak, ci_audit  # noqa: E402

import sklearn                                                                 # noqa: E402
from sklearn.preprocessing import StandardScaler                              # noqa: E402
from sklearn.ensemble import IsolationForest                                  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor                              # noqa: E402

# ---- constants ----
MAZE = Env.MAZE_WALLS; RAD = Env.ROBOT_RADIUS
WALL_THIN = RS.WALL_THIN; XI = RS.XI; PERSIST = RS.PERSIST; Nnw = RS.N
SEED0 = 20260630
N_HON_TRAIN, N_HON_HOLD, N_FAULT = 50, 30, 30
CONTAM = 0.05            # per-frame native contamination for IF/LOF
FP_TARGET = 0.05         # fixed per-trajectory FP operating point


# ====================================================================
# generators (per-scenario matched honest vs fault; same region+geometry)
# ====================================================================
def run_maze(seed, start, action, steps, fault=None, slip=0.30):
    e = Env(slip=slip, control_mode="A", map_type="maze"); e.reset(seed=seed)
    e.pos = np.array(start, float); e.theta = 0.0; e.v_act = 0.0; e.w_act = 0.0
    e.odom_pos = e.pos.copy(); e.odom_theta = 0.0
    if fault is not None:
        e.physics_fault = fault
    t = [e.pos.copy()]; o = [e.odom_pos.copy()]; a = np.array(action, np.float32)
    for _ in range(steps):
        e.step(a); t.append(e.pos.copy()); o.append(e.odom_pos.copy())
    return np.array(t), np.array(o)


def hnw(seed):                       # near-wall honest (residual <= xi)
    return RS.healthy_near_wall(seed)


def scenB_fault(seed):               # d<xi Scenario B with per-seed noise; residual <= xi, persistent crossing
    rng = np.random.default_rng(10_000 + seed)
    y = 10.0 + 3.0 * np.arange(Nnw) / (Nnw - 1)
    truth = np.column_stack([np.full(Nnw, 29.90), y])
    eps = np.zeros((Nnw, 2)); a, s = 0.85, 0.12
    for i in range(1, Nnw):
        eps[i] = a * eps[i - 1] + rng.normal(0, s, 2)
    r = eps + np.array([0.45, 0.0])
    nr = np.linalg.norm(r, axis=1); over = nr > XI
    r[over] = r[over] * (XI / nr[over])[:, None]
    return truth, truth + r


def _leak_base(seed):
    return run_maze(seed, [20.0, 20.0], [0.85, 0.25], 300, None)


SCEN = {
    "ScenarioB_dxi": dict(walls=WALL_THIN, rad=0.0, kind="relational",
                          honest=lambda s: hnw(s), fault=lambda s: scenB_fault(s)),
    "ScenarioA_phantom": dict(walls=MAZE, rad=RAD, kind="physics",
                              honest=lambda s: run_maze(s, [26.5, 20.0], [1, 0], 40, None),
                              fault=lambda s: run_maze(s, [26.5, 20.0], [1, 0], 40, {"phantom_walls": [4]})),
    "CF2_penetration": dict(walls=MAZE, rad=RAD, kind="physics",
                            honest=lambda s: run_maze(s, [37.0, 20.0], [1, 0], 40, None),
                            fault=lambda s: run_maze(s, [37.0, 20.0], [1, 0], 40, {"skip_pushout": True})),
    "L1_full_leak": dict(walls=MAZE, rad=RAD, kind="ci",
                         honest=lambda s: _leak_base(s),
                         fault=lambda s: (lambda t, o: (t, inject_l1_full_leak(t, o)))(*_leak_base(s))),
    "L2_partial_leak": dict(walls=MAZE, rad=RAD, kind="ci",
                            honest=lambda s: _leak_base(s),
                            fault=lambda s: (lambda t, o: (t, inject_l2_partial_leak(t, o, 0.25)))(*_leak_base(s))),
}
SCEN_LABEL = {"ScenarioB_dxi": "Scenario B (d<xi, relational)", "ScenarioA_phantom": "Scenario A (phantom wall)",
              "CF2_penetration": "CF-2 (penetration)", "L1_full_leak": "L-1 (full leak)", "L2_partial_leak": "L-2 (partial leak)"}


# ====================================================================
# features  (per frame; B1 report-only -> B2 +truth/residual/vel -> B3 +per-point map)
# ====================================================================
def _clear(p, walls, rad):
    return min(dist_point_aabb(p, w) for w in walls) - rad


def _free(p, walls, rad):
    return 0.0 if any(_point_in_aabb(p, w, rad) for w in walls) else 1.0


def features(truth, odom, walls, rad, level):
    truth = np.asarray(truth, float); odom = np.asarray(odom, float); n = len(odom)
    do = np.diff(odom, axis=0, prepend=odom[:1]); dx = np.diff(truth, axis=0, prepend=truth[:1])
    ho = np.arctan2(do[:, 1], do[:, 0]); dh = (np.diff(ho, prepend=ho[:1]) + np.pi) % (2 * np.pi) - np.pi
    r = odom - truth; dr = np.diff(r, axis=0, prepend=r[:1])
    B1 = np.column_stack([odom[:, 0], odom[:, 1], do[:, 0], do[:, 1], np.linalg.norm(do, axis=1), dh])
    if level == "B1":
        return B1
    B2 = np.column_stack([B1, truth[:, 0], truth[:, 1], r[:, 0], r[:, 1], np.linalg.norm(r, axis=1),
                          dx[:, 0], dx[:, 1], np.linalg.norm(dx, axis=1), np.linalg.norm(dr, axis=1)])
    if level == "B2":
        return B2
    mp = np.array([[_clear(odom[i], walls, rad), _clear(truth[i], walls, rad),
                    _free(odom[i], walls, rad), _free(truth[i], walls, rad)] for i in range(n)])
    return np.column_stack([B2, mp])


# ====================================================================
# oracle (M5 channels recomputed on the SAME generated inputs): physics(truth-in-wall) | relational(joint) | C_I
# ====================================================================
def oracle_catch(truth, odom, walls, rad, kind):
    if not physics_oracle(truth, walls, rad)["ok"]:
        return True                                            # truth enters a wall (EC5'/physics)
    if not relational_oracle(truth, odom, walls, rad, persist_frac=PERSIST)["ok"]:
        return True                                            # displacement segment crosses (joint)
    if kind == "ci":
        try:
            if not ci_audit(np.asarray(truth, float), np.asarray(odom, float), 0.30)["ok"]:
                return True                                    # mutual-information leakage (contract C_I)
        except Exception:
            pass
    return False


# ====================================================================
# detectors + per-trajectory persistence aggregation + calibration
# ====================================================================
def make_detector(name, Xtr):
    if name == "IF":
        return IsolationForest(contamination=CONTAM, random_state=0, n_estimators=200).fit(Xtr)
    return LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONTAM).fit(Xtr)


def traj_stats(det, scaler, trajs, walls, rad, level):
    """per-trajectory anomaly statistic = fraction of frames the detector flags."""
    out = []
    for (t, o) in trajs:
        X = scaler.transform(features(t, o, walls, rad, level))
        out.append(float(np.mean(det.predict(X) == -1)))
    return np.array(out)


def evaluate():
    meta = {"sklearn": sklearn.__version__, "numpy": np.__version__, "seed0": SEED0,
            "detectors": ["IsolationForest", "LOF(novelty)"], "features": ["B1", "B2", "B3"],
            "n_honest_train": N_HON_TRAIN, "n_honest_holdout": N_HON_HOLD, "n_fault": N_FAULT,
            "per_frame_contamination": CONTAM, "fp_target": FP_TARGET, "persist_mechanism": "fraction-of-flagged-frames > gate",
            "ocsvm_note": "One-Class SVM dropped after stability probe (gamma=1.0 degenerate; cross-fit fault-flag 0.33-0.46); using LOF."}
    results = {}
    print("=" * 110)
    print("  S-A external baseline: IsolationForest + LOF (B1/B2/B3) vs structured oracle  | sklearn", sklearn.__version__)
    print("=" * 110)
    for sk, cfg in SCEN.items():
        walls, rad, kind = cfg["walls"], cfg["rad"], cfg["kind"]
        hon_tr = [cfg["honest"](SEED0 + i) for i in range(N_HON_TRAIN)]
        hon_ho = [cfg["honest"](SEED0 + 1000 + i) for i in range(N_HON_HOLD)]
        flt = [cfg["fault"](SEED0 + 2000 + i) for i in range(N_FAULT)]
        # oracle column (recomputed per seed on same inputs)
        orc_fault = float(np.mean([oracle_catch(t, o, walls, rad, kind) for (t, o) in flt]))
        orc_fp = float(np.mean([oracle_catch(t, o, walls, rad, kind) for (t, o) in hon_ho]))
        row = {"oracle": {"detect": orc_fault, "honest_fp": orc_fp}}
        for level in ("B1", "B2", "B3"):
            Xtr = np.vstack([features(t, o, walls, rad, level) for (t, o) in hon_tr])
            scaler = StandardScaler().fit(Xtr); Xs = scaler.transform(Xtr)
            for dname in ("IF", "LOF"):
                det = make_detector(dname, Xs)
                s_ho = traj_stats(det, scaler, hon_ho, walls, rad, level)
                s_fl = traj_stats(det, scaler, flt, walls, rad, level)
                gate5 = float(np.percentile(s_ho, 100 * (1 - FP_TARGET)))      # ~5% honest exceed
                gate0 = float(s_ho.max())                                       # ~0% honest exceed (oracle-matched)
                row[f"{dname}-{level}"] = {
                    "detect_fp5": float(np.mean(s_fl > gate5)),
                    "fp_at_fp5": float(np.mean(s_ho > gate5)),
                    "detect_fp0": float(np.mean(s_fl > gate0)),
                    "fp_at_fp0": float(np.mean(s_ho > gate0)),
                }
        results[sk] = row
        print(f"\n  [{SCEN_LABEL[sk]}]  (kind={kind})  oracle detect={orc_fault:.0%} (honest FP {orc_fp:.0%})")
        for dname in ("IF", "LOF"):
            cells = "  ".join(f"{lvl}:{row[f'{dname}-{lvl}']['detect_fp5']:.0%}" for lvl in ("B1", "B2", "B3"))
            print(f"      {dname:<4} detect@FP5%  {cells}")
    return meta, results


def make_figure(meta, results, path):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = ["oracle"] + [f"{d}-{l}" for d in ("IF", "LOF") for l in ("B1", "B2", "B3")]
    rows = list(SCEN.keys())
    M = np.array([[results[r]["oracle"]["detect"] if c == "oracle" else results[r][c]["detect_fp5"] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([SCEN_LABEL[r] for r in rows], fontsize=9)
    ax.axvline(0.5, color="k", lw=1.5)
    ax.set_xlabel("detection rate over %d fault seeds @ fixed 5%% per-trajectory FP" % N_FAULT)
    fig.colorbar(im, ax=ax, fraction=0.025, label="detection rate")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    print("\n  [OK] figure:", path)


def main():
    meta, results = evaluate()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "baseline_comparison.json")
    json.dump({"meta": meta, "results": results,
               "nonspatial_note": "seq-freeze / disconnect-as-live / data-tamper / P-1 energy are a DIFFERENT feature channel "
               "(sequence/link metadata, checksum, energy ledger) — outside B1/B2/B3 spatial features; the oracle catches them via "
               "the contract-seq / code-integrity / physics-energy layers (see coverage_matrix.json M5). Not a baseline failure."},
              open(out, "w"), indent=2)
    print("\n  [OK] data:", out)
    try:
        make_figure(meta, results, os.path.join(here, "baseline_comparison.png"))
    except Exception as e:
        print("  [WARN] figure skipped:", e)


if __name__ == "__main__":
    main()
