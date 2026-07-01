# -*- coding: utf-8 -*-
"""
scenB_controlled_validity.py  ——  S-A step 3a: construct a controlled irreducible Scenario B and
STRICTLY validate that NO per-frame / marginal feature separates honest from fault (only the
window-level segment-crossing persistence should differ).  This step does NOT run detectors.

Construction (natural generalization of the persistence-gate scenario, Fig 4 (2)):
  * thin wall M (AABB, thickness d < xi), a finite obstacle the robot goes around -> truth roams BOTH sides.
  * honest: truth hugs the wall on side A or side B (clearance c), odom = truth + ISOTROPIC AR1 noise
    (||eps|| up to xi).  Because d<xi, noise occasionally pushes o_t transiently across the wall.
  * fault: truth roams the SAME way (both sides), but odom is PERSISTENTLY mirrored to the opposite
    free side (displacement crosses every frame); the crossing residual is built to be as small as the
    geometry allows + the SAME isotropic noise, so per-frame it should look like a normal honest residual.
The ONLY intended difference: honest crosses transiently (low frac), fault crosses persistently (high frac).

Validity = for EVERY marginal feature, is honest vs fault separable?  We report, per feature:
  - honest vs fault summary (mean/sd/percentiles),
  - 1-D separability AUC (0.5 = indistinguishable; >~0.65 = a single feature can separate => FAIL),
  - % of fault frames inside honest's bulk (central 50%, [p25,p75]).
If ANY feature is separable -> we FLAG it and stop for discussion (do not proceed to detectors).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relational_oracle import dist_point_aabb, _point_in_aabb, relational_oracle
from sklearn.metrics import roc_auc_score

# ---- geometry / noise params (repo values; d and c are the knobs) ----
XI = 0.50
D = 0.20                                  # wall thickness (repo WALL_THIN). d < xi.
WALLC = 30.10                             # wall center x
WALL = [(WALLC - D / 2, WALLC + D / 2, 9.0, 16.0)]   # AABB; finite in y so the robot can go around
A_EDGE, B_EDGE = WALLC - D / 2, WALLC + D / 2        # 30.0 / 30.2
N = 60
CLEAR = 0.05                              # truth clearance from the near wall edge (hugging, both sides)
S_NOISE = 0.22                            # isotropic odom noise scale (AR1), clipped to ||eps|| <= xi
AR = 0.85
RAD = 0.0                                 # point estimate (repo)


def _iso_noise(rng):
    eps = np.zeros((N, 2))
    for i in range(1, N):
        eps[i] = AR * eps[i - 1] + rng.normal(0, S_NOISE, 2)
    nr = np.linalg.norm(eps, axis=1); over = nr > XI
    eps[over] = eps[over] * (XI / nr[over])[:, None]
    return eps


def _truth(rng, side):
    y = 9.5 + 6.0 * np.arange(N) / (N - 1)
    jit = np.zeros(N)
    for i in range(1, N):
        jit[i] = 0.9 * jit[i - 1] + rng.normal(0, 0.01)
    if side == "A":
        x = (A_EDGE - CLEAR) - np.abs(jit)        # stay on side A (x < A_EDGE), free
    else:
        x = (B_EDGE + CLEAR) + np.abs(jit)        # stay on side B (x > B_EDGE), free
    return np.column_stack([x, y])


def honest(seed):
    rng = np.random.default_rng(seed)
    side = "A" if seed % 2 == 0 else "B"
    t = _truth(rng, side)
    return t, t + _iso_noise(rng)               # odom = truth + isotropic noise


def fault(seed):
    rng = np.random.default_rng(100000 + seed)
    side = "A" if seed % 2 == 0 else "B"
    t = _truth(rng, side)
    # mirror the truth to the opposite free side at the SAME clearance, then add the SAME isotropic noise
    o_base = t.copy()
    if side == "A":
        o_base[:, 0] = (B_EDGE + CLEAR) + (A_EDGE - CLEAR - t[:, 0])   # opposite side, mirror clearance
    else:
        o_base[:, 0] = (A_EDGE - CLEAR) - (t[:, 0] - (B_EDGE + CLEAR))
    return t, o_base + _iso_noise(rng)


# ---- features (per frame) ----
def _clear(p):
    return min(dist_point_aabb(p, w) for w in WALL) - RAD


def _free(p):
    return 0.0 if any(_point_in_aabb(p, w, RAD) for w in WALL) else 1.0


def frame_feats(t, o):
    r = o - t
    return {
        "o_x": o[:, 0], "o_y": o[:, 1], "x_x": t[:, 0], "x_y": t[:, 1],
        "||r||": np.linalg.norm(r, axis=1), "r_x": r[:, 0], "r_y": r[:, 1],
        "r_angle": np.arctan2(r[:, 1], r[:, 0]),
        "clear_o": np.array([_clear(p) for p in o]), "clear_x": np.array([_clear(p) for p in t]),
        "infree_o": np.array([_free(p) for p in o]), "infree_x": np.array([_free(p) for p in t]),
    }


def collect(gen, n, s0):
    feats = {}
    cross_frac = []
    for i in range(n):
        t, o = gen(s0 + i)
        ff = frame_feats(t, o)
        for k, v in ff.items():
            feats.setdefault(k, []).append(v)
        cross_frac.append(relational_oracle(t, o, WALL, RAD, persist_frac=0.0)["frac"])
    return {k: np.concatenate(v) for k, v in feats.items()}, np.array(cross_frac)


def main():
    print("=" * 96)
    print(f"  Controlled Scenario B validity | wall d={D} (<xi={XI}), clearance={CLEAR}, noise s={S_NOISE}, N={N}")
    print("=" * 96)
    H, hf = collect(honest, 80, 7000)
    F, ff = collect(fault, 30, 9000)
    print(f"  relational crossing-frac: honest mean={hf.mean():.2f} max={hf.max():.2f} | fault mean={ff.mean():.2f} min={ff.min():.2f}")
    print(f"  (oracle persistence gate {0.5} separates if honest.max < 0.5 < fault.min)\n")
    print(f"  {'feature':<10}{'honest mean/sd':<22}{'fault mean/sd':<22}{'AUC':<7}{'%fault in honest bulk':<22}{'verdict'}")
    flagged = []
    order = ["o_x", "x_x", "||r||", "r_x", "r_y", "r_angle", "clear_o", "clear_x", "infree_o", "infree_x", "o_y", "x_y"]
    for k in order:
        h, f = H[k], F[k]
        y = np.r_[np.zeros(len(h)), np.ones(len(f))]; sc = np.r_[h, f]
        try:
            auc = roc_auc_score(y, sc); auc = max(auc, 1 - auc)
        except Exception:
            auc = float("nan")
        q25, q75 = np.percentile(h, 25), np.percentile(h, 75)
        in_bulk = float(np.mean((f >= q25) & (f <= q75)))
        sep = auc > 0.65
        if sep:
            flagged.append((k, auc))
        print(f"  {k:<10}{f'{h.mean():+.3f}/{h.std():.3f}':<22}{f'{f.mean():+.3f}/{f.std():.3f}':<22}{auc:<7.3f}{in_bulk*100:<22.0f}{'<<< SEPARABLE' if sep else 'overlap'}")
    print()
    if flagged:
        print("  RESULT: FAIL — these marginal features separate honest from fault (AUC>0.65):")
        for k, a in flagged:
            print(f"     - {k}  (AUC={a:.2f})")
    else:
        print("  RESULT: PASS — no single marginal feature separates honest from fault.")
    # small figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = ["o_x", "r_x", "||r||", "clear_x", "infree_o", "r_angle"]
        fig, axs = plt.subplots(2, 3, figsize=(13, 6))
        for ax, k in zip(axs.ravel(), keys):
            ax.hist(H[k], bins=40, density=True, alpha=0.55, label="honest", color="#2a7")
            ax.hist(F[k], bins=40, density=True, alpha=0.55, label="fault", color="#d33")
            ax.set_title(k); ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenB_controlled_validity.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); print("\n  [fig]", out)
    except Exception as e:
        print("  [fig skipped]", e)


if __name__ == "__main__":
    main()
