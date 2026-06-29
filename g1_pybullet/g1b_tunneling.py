# -*- coding: utf-8 -*-
"""
g1b_tunneling.py  ——  G1b · trigger PyBullet's native high-speed tunneling, test whether the physics audit catches it (non-circular · crown jewel)
============================================================================================
🔴 Non-circular: only set **inducing conditions** (thin wall + high speed + default step, no CCD), letting PyBullet's own discrete collision
   detection miss the fast object → the truth really crosses the wall. **No hand-set fault state**. Test whether the physics audit (EC5' point/swept) catches it.
Checkpoint 2: honestly report catch or miss; distinguish inducing conditions from a hand-set fault.
"""

import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_helpers as H


def run(speed, wall_half_thick=0.02, dt=1.0 / 120.0, ccd=False):
    """Charge straight at the thin wall on the ground at `speed`. Returns (truth_xy, wall_aabb, eng_pen_max, tunneled)."""
    H.connect(dt=dt)
    p.loadURDF("plane.urdf")
    wall_uid, wall_aabb = H.add_wall([2.0, 0, 0.3], [wall_half_thick, 1.0, 0.3])
    m = 1.0
    ball = H.add_ball(m, [0, 0, 0.1], radius=0.1, damping=0.0)
    p.changeDynamics(ball, -1, restitution=0.9, rollingFriction=0.0)
    if ccd:
        p.changeDynamics(ball, -1, ccdSweptSphereRadius=0.1)   # enable CCD as a control (off by default)
    p.resetBaseVelocity(ball, [speed, 0, 0], [0, 0, 0])
    truth_xy = []; eng_pen = 0.0
    for _ in range(400):
        pos, _, _, _ = H.base_state(ball)
        truth_xy.append([pos[0], pos[1]])
        eng_pen = max(eng_pen, H.engine_reported_penetration(ball, [wall_uid]))
        if pos[0] > 3.5:
            break
        p.stepSimulation()
    p.disconnect()
    truth_xy = np.array(truth_xy)
    tunneled = truth_xy[:, 0].max() > 2.5          # reaching the far side of the wall = it tunneled through
    return truth_xy, wall_aabb, eng_pen, tunneled


def main():
    print("=" * 80)
    print("  G1b · trigger PyBullet's native high-speed tunneling → test the physics audit (checkpoint 2)")
    print("=" * 80)
    R = 0.1; DT = 1.0 / 120.0
    print(f"\n  wall: x∈[1.98,2.02] (half-thickness 0.02m), step 1/120s. Inducing condition = high speed (per-step displacement ≫ wall thickness) + default no CCD\n")
    print(f"  {'speed(m/s)':<11}{'step disp':<11}{'pass?':<9}{'engine penetration':<20}{'EC5 point':<12}{'EC5 swept':<12}")
    rows = {}
    for speed in (50.0, 200.0, 400.0):
        truth_xy, wall_aabb, eng_pen, tunneled = run(speed, dt=DT)
        disp = speed * DT
        r_pt = H.ec5prime_truth_vs_map(truth_xy, [wall_aabb], R)        # per-frame point check (body radius)
        r_sw = H.ec5prime_swept(truth_xy, [wall_aabb], 0.0)            # swept: center-vs-wall-core crossing
        rows[speed] = {"tunneled": tunneled, "eng_pen": eng_pen,
                       "ec5_point_red": not r_pt["ok"], "ec5_swept_red": not r_sw["ok"]}
        print(f"  {speed:<11}{disp:<11.4f}{'yes🔴' if tunneled else 'no(blocked)':<9}"
              f"{eng_pen:<20.2e}{'🔴red' if not r_pt['ok'] else '🟢green':<12}{'🔴red' if not r_sw['ok'] else '🟢green':<12}")

    print("\n" + "=" * 80)
    print("  checkpoint 2 conclusion (PyBullet's own high-speed tunneling → can the physics audit catch it)")
    print("=" * 80)
    fast = rows[200.0]
    if fast["tunneled"] and fast["eng_pen"] < 1e-9:
        print(f"  ✅ inducement succeeds and is **clean**: at 200 m/s the ball **really crosses the wall**, **engine-reported penetration=0** —")
        print(f"     PyBullet's own discrete collision detection **completely misses it** (not a hand-set fault, it is the engine's real numerical pathology).")
        if fast["ec5_swept_red"] and not fast["ec5_point_red"]:
            print(f"  ✅ the physics audit **catches it**: the **swept EC5' flags RED**; while the per-frame point EC5' **also misses** (the ball jumps over the whole wall in one step, no frame inside)")
            print(f"     — same cause as the engine miss (discrete sampling). So high-speed tunneling **requires swept** segment detection to catch.")
            print(f"  🔴 non-circular confirmation: the fault comes from PyBullet's own numerical tunneling; our audit logic (swept EC5') caught **a real pathology of an engine we did not build**.")
        elif fast["ec5_swept_red"]:
            print(f"  ✅ the physics audit catches it: both swept and point EC5' flag RED. Non-circular confirmation.")
        else:
            print(f"  ❌ the physics audit **misses it**: neither point nor swept EC5' caught it → honest negative result, the framework did not generalize to this pathology.")
    elif fast["tunneled"]:
        print(f"  ⚠️ tunneled but the engine still reports partial penetration ({fast['eng_pen']:.1e}); EC5' swept={'red' if fast['ec5_swept_red'] else 'green'}.")
    else:
        print(f"  ⚠️ 200 m/s did not tunnel → needs stronger inducement; flagged honestly.")
    catch = fast["tunneled"] and fast["ec5_swept_red"] and fast["eng_pen"] < 1e-9
    return catch


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
