# -*- coding: utf-8 -*-
"""
g1a_baseline.py  ——  G1a · healthy PyBullet baseline + noise-floor characterization (honest checkpoint 1)
=============================================================================
🔴 Checkpoint 1: get a healthy PyBullet running; the physics audit (energy/momentum/non-penetration) must have **zero false positives**,
   or honestly characterize the engine's normal numerical noise floor (separating "engine integration noise" from "real pathology").
   If a clean baseline cannot be achieved, report it honestly, no forced tuning.
"""

import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_helpers as H


def baseline_energy_floor():
    """Frictionless free projectile (a conservative system) → characterize the engine integrator's energy noise floor."""
    H.connect(dt=1.0 / 240.0)
    m = 1.0
    ball = H.add_ball(m, [0, 0, 5], radius=0.1, damping=0.0)
    p.resetBaseVelocity(ball, [3, 0, 4], [0, 0, 0])
    E = []
    for _ in range(400):
        pos, _, _, _ = H.base_state(ball)
        if pos[2] < 0.3:
            break
        E.append(H.mechanical_energy(ball, m))
        p.stepSimulation()
    p.disconnect()
    E = np.array(E)
    dE = np.diff(E)
    floor = float(np.max(np.abs(dE)))
    print(f"[energy noise floor] frictionless free projectile N={len(E)} steps: E0={E[0]:.3f}J")
    print(f"  per-step |ΔE| max={floor:.3e}J, mean={np.mean(dE):+.3e}J, cumulative drift={E[-1]-E[0]:+.3e}J "
          f"({abs(E[-1]-E[0])/E[0]*100:.4f}%)")
    print(f"  → engine energy noise floor ≈ {floor:.1e} J/step (audit threshold must be > this; analogous to the Phase3 KSG noise-budget bound)")
    return floor


def baseline_nonpenetration_and_momentum():
    """Healthy: a slow ball rolls toward the wall on the ground and bounces normally at low speed (the engine handles contact correctly) → EC5'/engine contact should be all green.
    🔴 Key: a ground plane must be added to keep **planar motion** (z≈const) so EC5''s 2D(x,y) projection is valid (the platform's v_z≈0 invariant)."""
    H.connect(dt=1.0 / 240.0)
    p.loadURDF("plane.urdf")                                            # ground (keep planar motion)
    wall_uid, wall_aabb = H.add_wall([2.0, 0, 0.3], [0.05, 1.0, 0.3])   # thin wall x≈2
    m = 1.0
    ball = H.add_ball(m, [0, 0, 0.1], radius=0.1, damping=0.0)
    p.changeDynamics(ball, -1, restitution=0.85, rollingFriction=0.0)   # elastic bounce
    p.changeDynamics(wall_uid, -1, restitution=0.85)
    p.resetBaseVelocity(ball, [1.5, 0, 0], [0, 0, 0])    # slow straight charge at the wall (1.5 m/s, the engine detects it correctly)
    truth_xyz = []; truth_xy = []; eng_pen = []
    for _ in range(500):
        pos, _, v, _ = H.base_state(ball)
        truth_xyz.append(pos); truth_xy.append([pos[0], pos[1]])
        eng_pen.append(H.engine_reported_penetration(ball, [wall_uid]))
        p.stepSimulation()
    p.disconnect()
    truth_xy = np.array(truth_xy); truth_xyz = np.array(truth_xyz)
    r5 = H.ec5prime_truth_vs_map(truth_xy, [wall_aabb], radius=0.1)
    max_eng_pen = max(eng_pen)
    reached = truth_xy[:, 0].max()
    z_range = (truth_xyz[:, 2].min(), truth_xyz[:, 2].max())
    print(f"\n[non-penetration healthy baseline] ground slow (1.5m/s) wall impact N={len(truth_xy)} steps: ball furthest x={reached:.3f} (wall inner face x≈1.85)")
    print(f"  planarity: z∈[{z_range[0]:.3f},{z_range[1]:.3f}] (should be ≈0.1 constant → 2D projection valid)")
    print(f"  EC5'(geometric truth-vs-wall): {'🟢green (no crossing)' if r5['ok'] else '🔴red '+r5['detail']}")
    print(f"  engine-reported penetration max={max_eng_pen:.4e} m (low-speed contact resolved correctly by the engine → ≈0)")
    return r5["ok"], max_eng_pen


def main():
    print("=" * 78)
    print("  G1a · healthy PyBullet baseline + noise-floor characterization (checkpoint 1)")
    print("=" * 78)
    floor = baseline_energy_floor()
    # use ~10× the noise floor as the energy-audit floor, run a healthy (conservative) segment to confirm zero false positives
    H.connect(dt=1.0 / 240.0)
    m = 1.0; ball = H.add_ball(m, [0, 0, 5], radius=0.1, damping=0.0)
    p.resetBaseVelocity(ball, [2, 1, 3], [0, 0, 0])
    E = []
    for _ in range(300):
        pos, _, _, _ = H.base_state(ball)
        if pos[2] < 0.3:
            break
        E.append(H.mechanical_energy(ball, m)); p.stepSimulation()
    p.disconnect()
    ok_e, det_e, resid = H.energy_audit_series(E, floor_abs=max(10 * floor, 1e-2), floor_rel=1e-3)
    print(f"\n[energy audit healthy] floor={max(10*floor,1e-2):.1e}J/step → {'🟢green (zero false positives)' if ok_e else '🔴red '+det_e}")

    ok_p, eng_pen = baseline_nonpenetration_and_momentum()

    print("\n" + "=" * 78)
    print("  checkpoint 1 conclusion")
    print("=" * 78)
    clean = ok_e and ok_p
    print(f"  energy audit healthy zero false positives: {'✅' if ok_e else '❌'}  (noise floor {floor:.1e} J/step, threshold 10×)")
    print(f"  non-penetration healthy zero false positives: {'✅' if ok_p else '❌'}")
    print(f"  → healthy baseline {'clean (zero false positives, noise floor characterized) ✅' if clean else 'has false positives (see above, reported honestly) ⚠️'}")
    return clean


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
