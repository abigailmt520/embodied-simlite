# -*- coding: utf-8 -*-
"""
record_fork.py  ——  Phase 0 · DoD-1 true-fork evidence recorder (instrumentation, not functional code)
=================================================================================
Purpose: drive embodied_env headlessly over an episode-length (500-step) trajectory, under both
      slip=0 (before / control) and slip=SLIP_FACTOR (after / real slip), recording per frame
      the truth (Truth) and odometry (Odom) poses, computing the cumulative position error, exporting a CSV and plotting the before/after
      error-vs-time curve.

Methodology (why reproducible, why not a hardcoded fake curve, per INV-2):
  - Controller: an open-loop "fixed-curvature" action [v=0.6, w=0.25], constant throughout → the truth trajectory is a bounded circle,
    reproducible and policy-independent (the standard way to characterize drift: drive a known trajectory, compare truth vs dead reckoning).
  - Span a fixed 500 steps, **no early reset on terminated/truncated**: a reset re-calibrates the odometry and
    zeros the error, which cannot show the continuous accumulation curve; the fixed-curvature truth is bounded and will not run away from ignoring collisions.
  - before/after use **the same random seed + the same controller**, the only variable is the slip factor (change one variable at a time).
  - all error comes from the env's internal real slip-integration process (seeded self.np_random); this script injects no numbers.

Artifacts:
  diagnostics/fork_before.csv      slip=0 per-frame data (error constant 0.000000)
  diagnostics/fork_after.csv       slip=0.05 per-frame data (error nonzero, grows with time)
  diagnostics/fork_error_curve.png before/after error-vs-time comparison curve

Run: python diagnostics/record_fork.py
"""

import csv
import os
import sys

import numpy as np

# allow running directly from the diagnostics/ subdirectory (add the repo root to the import path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodied_env import EmbodiedNavEnv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
STEPS = EmbodiedNavEnv.MAX_STEPS          # one episode length = 500 steps
ACTION = np.array([0.6, 0.25], dtype=np.float32)  # open-loop fixed curvature: real v=0.6 m/s, w=0.375 rad/s


def record(slip):
    """Run a fixed STEPS-step trajectory, returning the per-frame records list[dict]."""
    env = EmbodiedNavEnv(slip=slip)
    env.reset(seed=SEED)
    rows = []
    for i in range(STEPS):
        env.step(ACTION)
        err_xy = float(np.linalg.norm(env.pos - env.odom_pos))
        dyaw = (env.theta - env.odom_theta + np.pi) % (2 * np.pi) - np.pi
        rows.append({
            "step": env.step_count,
            "t_s": round(env.step_count * env.DT, 3),
            "truth_x": round(float(env.pos[0]), 6),
            "truth_y": round(float(env.pos[1]), 6),
            "truth_theta": round(float(env.theta), 6),
            "odom_x": round(float(env.odom_pos[0]), 6),
            "odom_y": round(float(env.odom_pos[1]), 6),
            "odom_theta": round(float(env.odom_theta), 6),
            "err_xy": round(err_xy, 6),
            "err_yaw_deg": round(float(np.degrees(dyaw)), 4),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def metrics(rows):
    err = np.array([r["err_xy"] for r in rows])
    return {
        "ATE_RMSE": float(np.sqrt(np.mean(err ** 2))),
        "final_err": float(err[-1]),
        "max_err": float(err.max()),
        "mean_err": float(err.mean()),
        "final_yaw_drift_deg": float(rows[-1]["err_yaw_deg"]),
    }


def main():
    before = record(slip=0.0)
    after = record(slip=EmbodiedNavEnv.SLIP_FACTOR)

    write_csv(before, os.path.join(HERE, "fork_before.csv"))
    write_csv(after, os.path.join(HERE, "fork_after.csv"))

    mb, ma = metrics(before), metrics(after)
    print("=" * 68)
    print(f"  true-fork record · seed={SEED} · steps={STEPS} · action={ACTION.tolist()} (open-loop fixed curvature)")
    print("=" * 68)
    print(f"  {'metric':<22}{'before(slip=0)':>20}{'after(slip=0.05)':>22}")
    print(f"  {'ATE_RMSE (m)':<22}{mb['ATE_RMSE']:>20.6f}{ma['ATE_RMSE']:>22.6f}")
    print(f"  {'final position error (m)':<24}{mb['final_err']:>20.6f}{ma['final_err']:>22.6f}")
    print(f"  {'max position error (m)':<24}{mb['max_err']:>20.6f}{ma['max_err']:>22.6f}")
    print(f"  {'final heading drift (deg)':<25}{mb['final_yaw_drift_deg']:>20.4f}{ma['final_yaw_drift_deg']:>22.4f}")
    print("=" * 68)
    # sample a few frames: after's err_xy visibly grows monotonically, before stays 0
    print("  sampled frames (step | before.err_xy | after.err_xy):")
    for k in (50, 100, 200, 300, 400, 500):
        b = before[k - 1]["err_xy"]
        a = after[k - 1]["err_xy"]
        print(f"    step {k:>4} | {b:>12.6f} | {a:>12.6f}")

    # ---- plotting ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tb = [r["t_s"] for r in before]
        eb = [r["err_xy"] for r in before]
        ta = [r["t_s"] for r in after]
        ea = [r["err_xy"] for r in after]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tb, eb, label="before  (slip=0.00) — Truth ≡ Odom, err ≡ 0", color="#2c7", lw=2)
        ax.plot(ta, ea, label="after   (slip=0.05) — real drift, err grows", color="#d33", lw=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cumulative position error |Truth - Odom|  (m)")
        ax.set_title("Embodied-SimLite | Stage-0 Truth-vs-Odom true fork (DoD-1)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        out = os.path.join(HERE, "fork_error_curve.png")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\n  [OK] error curve saved: {out}")
    except Exception as e:  # degrade to CSV-only when matplotlib is missing (PRD allows PNG or a plottable CSV)
        print(f"\n  [WARN] plotting skipped ({e}); CSV generated, you can plot it yourself.")

    print(f"  [OK] CSV: {os.path.join(HERE, 'fork_before.csv')}")
    print(f"  [OK] CSV: {os.path.join(HERE, 'fork_after.csv')}")


if __name__ == "__main__":
    main()
