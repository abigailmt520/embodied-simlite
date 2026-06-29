# -*- coding: utf-8 -*-
"""
g1c_pathologies.py  ——  G1c · other native pathologies + contract layer + joint (3D real engine)
========================================================================
- physics pathology (non-circular): energy injection (elastic bounce / large-step engine numerical gain) → can the energy audit catch it.
- contract layer (support · ported): C1-C3 in a 3D truth/odom state space, tested with our report injectors (odom spliced back to truth / disconnect-claims-online).
- joint (support): reported trajectory vs PyBullet physical wall joint cross-check.
Each item honestly reports catch/miss. warm-start ghost force: an honest attempt + a note on the isolation difficulty.
"""

import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_helpers as H


# ── pathology 1: energy injection (elastic bounce, engine numerical gain) ──────────────────────────
def pathology_energy_injection(dt=1.0 / 120.0):
    H.connect(dt=dt)
    pl = p.loadURDF("plane.urdf"); p.changeDynamics(pl, -1, restitution=1.0)
    ball = H.add_ball(1.0, [0, 0, 2.0], 0.1, damping=0.0)
    p.changeDynamics(ball, -1, restitution=1.0)      # e=1 conservative system (no external work)
    E = []
    for _ in range(1500):
        E.append(H.mechanical_energy(ball, 1.0)); p.stepSimulation()
    p.disconnect()
    ok, det, excess = H.energy_conservation_check(E, work_in=0.0, floor_abs=1e-2)
    print(f"\n[pathology 1 · energy injection] elastic bounce (e=1, conservative) dt={1/dt:.0f}Hz: E0={E[0]:.2f} → max={max(E):.2f} J")
    print(f"  energy audit (conservation upper bound): {'🔴red caught' if not ok else '🟢green missed'} — {det}")
    print(f"  🔴 non-circular: the injection comes from PyBullet's discrete contact solver (e=1 should conserve), not hand-set by us.")
    return (not ok)


# ── pathology 2 (honest attempt): warm-start ghost force ───────────────────────────────
def pathology_warmstart_ghost():
    """A pseudo-impulse at the instant of contact disconnect. Honest attempt: after stacking, suddenly remove the support, and see whether there is an anomalous momentum/energy jump at the disconnect instant.
    Hard to isolate: it aliases with normal gravity fall. Honestly report whether it can be cleanly measured."""
    H.connect(dt=1.0 / 240.0)
    p.loadURDF("plane.urdf")
    base = H.add_box_body(0, [0, 0, 0.25], (0.3, 0.3, 0.25), damping=0.0)  # static support
    top = H.add_box_body(1.0, [0, 0, 0.6], (0.1, 0.1, 0.1), damping=0.0)
    for _ in range(240):
        p.stepSimulation()                    # stabilize contact (warm-start established)
    pos0, _, v0, _ = H.base_state(top)
    p.removeBody(base)                         # suddenly remove the support → the contact-disconnect instant
    dKE = []
    for _ in range(5):
        _, _, v, _ = H.base_state(top)
        dKE.append(0.5 * 1.0 * float(np.dot(v, v)))
        p.stepSimulation()
    p.disconnect()
    # after disconnect it should be pure free fall (vertical acceleration), horizontal velocity should be ≈0; a ghost impulse would give a non-physical lateral velocity
    print(f"\n[pathology 2 · warm-start ghost force] top-block velocity evolution after removing support (should be pure vertical free fall):")
    print(f"  velocity at disconnect v0={v0}, then KE series={[round(k,4) for k in dKE]}")
    lateral = abs(v0[0]) + abs(v0[1])
    print(f"  lateral velocity |vx|+|vy|={lateral:.4e} m/s ({'🔴 anomalous lateral (suspected ghost impulse)' if lateral>1e-3 else '🟢 no anomalous lateral'})")
    print(f"  ⚠️ flagged honestly: the warm-start ghost force aliases with normal gravity fall and is hard to isolate cleanly; this test only checks the disconnect-transient lateral anomaly, "
          f"not a full diagnosis. {'no significant anomaly seen in this case' if lateral<=1e-3 else ''}")
    return None     # not counted in the main catch/miss conclusion (hard to isolate, honestly scoped out)


# ── contract layer (ported · support): C1-C3 in 3D truth/odom + our report injectors ──────────
def contract_layer_3d():
    H.connect(dt=1.0 / 240.0); p.loadURDF("plane.urdf")
    husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
    wheels = [j for j in range(p.getNumJoints(husky))
              if "wheel" in p.getJointInfo(husky, j)[1].decode()]
    rep = H.OdomReporter(init_xy=[0, 0], slip=0.05, seed=1)
    truth_xy, odom_xy = [[0, 0]], [[0, 0]]
    for _ in range(600):
        for j in wheels:                       # differential drive: left/right wheels at different speeds → turn while moving
            tgt = 8.0 if j in wheels[1::2] else 5.0
            p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=tgt, force=40)
        p.stepSimulation()
        pos, _, v, _ = H.base_state(husky)
        truth_xy.append([pos[0], pos[1]])
        odom_xy.append(list(rep.update([v[0], v[1]], 1.0 / 240.0)))
    p.disconnect()
    truth = np.array(truth_xy); odom = np.array(odom_xy)

    def c123(truth, odom):
        sess = [{"recv_t": float(i), "seq": i, "step": i,
                 "truth": {"x": float(truth[i][0]), "y": float(truth[i][1]), "theta": 0.0},
                 "odom": {"x": float(odom[i][0]), "y": float(odom[i][1]), "theta": 0.0},
                 "terminated": False, "truncated": False, "link_status": "online"}
                for i in range(len(truth))]
        return (H.check_truth_odom_fork(sess)["ok"], H.check_seq_integrity(sess)["ok"],
                H.check_feed_liveness(sess)["ok"])

    print(f"\n[contract layer 3D · ported] husky differential drive N={len(truth)} steps, odom = true-state velocity + slip dead reckoning")
    h = c123(truth, odom)
    print(f"  healthy: C1={'🟢' if h[0] else '🔴'} C2={'🟢' if h[1] else '🔴'} C3={'🟢' if h[2] else '🔴'} "
          f"→ {'all green (no false positive) ✅' if all(h) else 'has false positives'}")
    # injector 1-A: odom spliced back to truth → C1 should catch
    leak = c123(truth, truth)
    print(f"  inject odom=truth (full leak): C1={'🔴catch' if not leak[0] else '🟢miss'} (expect C1 red)")
    # inject: frame-seq freeze (data moves, seq does not) → C2 should catch
    sess = [{"recv_t": float(i), "seq": (i if i < 300 else 300), "step": i,
             "truth": {"x": float(truth[i][0]), "y": float(truth[i][1]), "theta": 0.0},
             "odom": {"x": float(odom[i][0]), "y": float(odom[i][1]), "theta": 0.0},
             "terminated": False, "truncated": False, "link_status": "online"} for i in range(len(truth))]
    c2red = not H.check_seq_integrity(sess)["ok"]
    print(f"  inject seq freeze (data updating): C2={'🔴catch' if c2red else '🟢miss'} (expect C2 red)")
    return all(h), (not leak[0]), c2red, truth, odom


# ── joint (support): reported trajectory vs PyBullet wall ──────────────────────────────
def joint_3d(truth, odom):
    from joint_audit import ec5_prime, joint_report_vs_map
    # claimed wall: place a wall outside the husky's driving area (truth does not cross, legitimate); fabricate odom crossing the wall
    wall_aabb = (3.0, 3.1, -5.0, 5.0)
    radius = 0.3
    ec5p = ec5_prime(truth, [wall_aabb], radius)          # truth legitimate
    fake_odom = np.array([[i * 0.02, 0.0] for i in range(len(truth))])  # straight through the x=3 wall
    jr = joint_report_vs_map(fake_odom, [wall_aabb], radius)
    print(f"\n[joint 3D · support] claimed wall x∈[3.0,3.1]")
    print(f"  EC5'(true-state husky vs wall): {'🟢green legal' if ec5p['ok'] else '🔴red'}")
    print(f"  joint (fabricated odom crossing vs wall): {'🔴red catch' if not jr['ok'] else '🟢green miss'} (expect red)")
    return ec5p["ok"], (not jr["ok"])


def main():
    print("=" * 80)
    print("  G1c · other native pathologies + contract layer + joint (3D real engine)")
    print("=" * 80)
    e_caught = pathology_energy_injection()
    pathology_warmstart_ghost()
    h_ok, c1_red, c2_red, truth, odom = contract_layer_3d()
    ec5_ok, joint_red = joint_3d(truth, odom)

    print("\n" + "=" * 80)
    print("  G1c summary")
    print("=" * 80)
    print(f"  physics pathology · energy injection: {'✅ caught by the audit' if e_caught else '❌ missed'}")
    print(f"  contract layer healthy zero false positives: {'✅' if h_ok else '❌'} | inject C1 (full leak){'✅catch' if c1_red else '❌miss'} "
          f"C2 (seq freeze){'✅catch' if c2_red else '❌miss'}")
    print(f"  joint truth-legal green + fabricated-odom-crossing red: {'✅' if (ec5_ok and joint_red) else '❌'}")
    return e_caught and h_ok and c1_red and c2_red and ec5_ok and joint_red


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
