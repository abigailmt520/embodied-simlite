# -*- coding: utf-8 -*-
"""
run_coupling_test.py  ——  Phase4 · dual-state coupling stress test (stress-testing the paper's core claim itself with anti-self-deception discipline)
====================================================================================
Goal: construct a self-deception such that (a) the contract-layer audit run alone = pass, (b) the physics-layer audit run alone = pass,
      but (c) the report×physics joint cross-check = catch. Only this proves "dual-state coupling", as opposed to two side-by-side audits.

🔴 Honest distinction (most important): whether a wall-crossing self-deception constitutes **true coupling** is judged by truth_vs_map:
    - truth_vs_map red → the truth really crosses → **catchable within physics alone (truth vs map)** → **not true coupling** (merely an EC5 implementation gap).
    - truth_vs_map green and odom_vs_map red → truth legitimate, only the report illegitimate → catchable neither within physics nor by contract self-consistency → **true coupling**.

Two scenarios:
  Scenario A "literal phantom wall": collision detection drops one wall → **the truth really crosses**. Expect physics/contract each pass, but truth_vs_map red
        → honestly judged "not true coupling, catchable within physics (EC5 gap)". Never tune parameters to force coupling.
  Scenario B "true-coupling variant": **the truth is honest (blocked by the wall, legitimate)**, but odom is fabricated as a self-consistent wall-crossing track.
        Expect truth_vs_map green, odom_vs_map red → true coupling (catchable only by report × physical-map).

Run: python audit/run_coupling_test.py
Artifact: audit/coupling_summary.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodied_env import EmbodiedNavEnv                                   # noqa: E402
from audit_suite import run_suite, format_suite, coupling_label          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WALLS = EmbodiedNavEnv.MAZE_WALLS
RADIUS = EmbodiedNavEnv.ROBOT_RADIUS
PHANTOM_IDX = 4          # the phantom-dropped wall: the death-corridor left wall (28,29,5,33)
THIN_WALL = [(30.0, 30.2, 9.0, 16.0)]   # thin wall d=0.20 < ξ (for the Scenario B2 displacement-crossing · topological-irreducibility wiring regression)


def _energy_row(env):
    st = env.get_render_state(); e = st["energy"]
    return {"step": st["step"], "seq": st["seq"], "E_kin": e["E_kin"], "dE": e["dE"],
            "W_act": e["W_act"], "D_damp": e["D_damp"],
            "E_contact_decl": e["E_contact_decl"], "E_contact_act": e["E_contact_act"],
            "penetration": e["penetration"], "v_act": st["v_act"], "w_act": st["w_act"]}


def run_layers(truth, odom, eledger, slip, tag):
    """Run the resident three-layer suite (incl. physics-layer EC5'), returning each layer's verdict + the coupling decision."""
    res = run_suite(truth, odom, eledger, WALLS, RADIUS, slip,
                    EmbodiedNavEnv.V_PHYS_MAX, EmbodiedNavEnv.W_PHYS_MAX)
    verdict = coupling_label(res)
    print(f"\n{'='*78}\n  {tag}\n{'='*78}")
    print(format_suite(res))
    print(f"  ── coupling verdict ──────► {verdict}")
    return {"physics_passed": res["physics"]["ok"], "contract_passed": res["contract"]["ok"],
            "ec5_prime_ok": res["physics"]["ec5_prime_ok"], "joint_ok": res["joint"]["ok"],
            "verdict": verdict}


# ====================================================================
# Scenario A: literal "phantom wall" — the truth really crosses
# ====================================================================
def scenario_a():
    env = EmbodiedNavEnv(slip=0.05, control_mode="A", map_type="maze")
    env.reset(seed=11)
    env.pos = np.array([26.5, 20.0]); env.theta = 0.0; env.v_act = 0.0; env.w_act = 0.0
    env.odom_pos = env.pos.copy(); env.odom_theta = 0.0
    env.physics_fault = {"phantom_walls": [PHANTOM_IDX]}   # collision detection drops the death-corridor left wall
    truth, odom, eledger = [env.pos.copy()], [env.odom_pos.copy()], []
    for _ in range(40):
        env.step(np.array([1.0, 0.0], dtype=np.float32))   # charge +x, passing through the [28,29] wall
        truth.append(env.pos.copy()); odom.append(env.odom_pos.copy()); eledger.append(_energy_row(env))
    return np.array(truth), np.array(odom), eledger


# ====================================================================
# Scenario B: true-coupling variant — truth honest (blocked by the wall), odom fabricated as a self-consistent wall-crossing track
# ====================================================================
def scenario_b():
    # truth: runs honestly (bounces off the wall, legitimate), recording the truth trajectory and its honest energy ledger
    env = EmbodiedNavEnv(slip=0.05, control_mode="A", map_type="maze")
    env.reset(seed=11)
    env.pos = np.array([26.5, 20.0]); env.theta = 0.0; env.v_act = 0.0; env.w_act = 0.0
    env.odom_pos = env.pos.copy(); env.odom_theta = 0.0
    # no phantom injection → the truth is blocked by the [28,29] wall (legitimate)
    truth, eledger = [env.pos.copy()], []
    for _ in range(40):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        truth.append(env.pos.copy()); eledger.append(_energy_row(env))
    truth = np.array(truth)
    # odom: fabricate a self-consistent "straight wall-crossing" track (constant +x increment, crossing the [28,29] wall).
    #   internally self-consistent dead reckoning (constant-velocity straight line) → contract C1-3/C_I cannot see it; but it crosses the claimed wall → odom_vs_map red.
    n = len(truth)
    odom = np.array([[26.5 + i * 0.12, 20.0] for i in range(n)])   # straight from 26.5 to ~31
    return truth, odom, eledger


# ====================================================================
# Scenario B2: reconfigured "displacement crosses wall" true coupling (topologically irreducible) — naive point check misses, relational catches
#   truth free on side A, odom free on side B (**not inside the wall**), ‖o-x‖≤ξ, only the displacement segment crosses the thin wall (d=0.20<ξ).
#   see docs/ScenB-Irreducibility.md. Verifies "irreducibility is wired into the resident audit_suite".
# ====================================================================
def scenario_b2():
    # take a healthy env run as the **clean energy ledger** (physics-layer EC1-EC5 green); the geometry is synthetic (point estimate radius=0).
    env = EmbodiedNavEnv(slip=0.05, control_mode="A", map_type="maze")
    env.reset(seed=7)
    env.pos = np.array([20.0, 20.0]); env.theta = 0.0; env.v_act = 0.0; env.w_act = 0.0
    env.odom_pos = env.pos.copy(); env.odom_theta = 0.0
    rng = np.random.default_rng(7)
    eledger = []
    for _ in range(60):
        env.step(np.array([0.8, (rng.random() - 0.5) * 0.6], dtype=np.float32))
        eledger.append(_energy_row(env))
    n = len(eledger) + 1
    y = 10.0 + 3.0 * np.arange(n) / (n - 1)
    truth = np.column_stack([np.full(n, 29.90), y]) + np.random.default_rng(2).normal(0, 0.02, (n, 2))  # free on side A (honest)
    odom = np.column_stack([np.full(n, 30.35), y]) + np.random.default_rng(3).normal(0, 0.03, (n, 2))   # free on side B (not inside the wall)
    return truth, odom, eledger


def main():
    print("Phase4b · dual-state coupling stress test (incl. physics-layer EC5' + resident joint layer) — finishing the criterion separation")
    ta, oa, la_led = scenario_a()
    rA = run_layers(ta, oa, la_led, 0.05, "Scenario A · literal phantom wall (truth really crosses)")
    a_ok = (rA["verdict"] == "PHYSICS_INTERNAL") and (not rA["ec5_prime_ok"]) and (not rA["physics_passed"])
    if a_ok:
        print("  ► honest verdict: **not true coupling**. EC5' (within-physics truth-vs-claimed-map) flags RED single-handedly → catchable within physics alone,")
        print("    no joint needed. Nails down 'Scenario A = single-layer gap (now plugged by EC5'), not coupling'.")

    tb, ob, lb_led = scenario_b()
    rB = run_layers(tb, ob, lb_led, 0.05, "Scenario B · true-coupling variant (truth honest, odom fabricates the crossing)")
    # true coupling iff: physics (incl. EC5') passes + contract passes + EC5' green (did not stand in for joint) + only joint red
    b_ok = (rB["verdict"] == "TRUE_COUPLING" and rB["physics_passed"] and rB["contract_passed"]
            and rB["ec5_prime_ok"] and (not rB["joint_ok"]))
    if b_ok:
        print("  ► honest verdict: **true coupling**. physics (incl. EC5') 🟢 + contract 🟢 + **EC5' green (truth legitimate, did not stand in for joint)**,")
        print("    only joint (report×physics) red → catchable only by the joint. Nails down 'Scenario B = true coupling, only the joint catches'.")

    # —— Scenario B2: displacement-crossing true coupling (topologically irreducible) — verifies "irreducibility is wired into the resident audit_suite" ——
    #   C_I (the slip-noise MI channel) does not apply to this synthetic geometric construction → run_ci=False; the contract layer uses C1/C2/C3.
    tb2, ob2, lb2 = scenario_b2()
    res2 = run_suite(tb2, ob2, lb2, THIN_WALL, 0.0, 0.05,
                     EmbodiedNavEnv.V_PHYS_MAX, EmbodiedNavEnv.W_PHYS_MAX, run_ci=False)
    v2 = coupling_label(res2)
    jp2 = res2["joint"]["check"]["ok"]; jr2 = res2["joint"]["relational"]["ok"]
    print(f"\n{'='*78}\n  Scenario B2 · displacement-crossing true coupling (topologically irreducible, d=0.20<ξ) — irreducibility wired into the resident-suite regression\n{'='*78}")
    print(format_suite(res2))
    print(f"  ── coupling verdict ──────► {v2}")
    b2_ok = (v2 == "TRUE_COUPLING" and res2["physics"]["ok"] and res2["contract"]["ok"]
             and res2["physics"]["ec5_prime_ok"] and jp2 and (not jr2))
    if b2_ok:
        print("  ► wiring confirmed: physics 🟢 + contract 🟢 + EC5' 🟢 + **naive point check 🟢 (misses, o_t in free space)**, only **relational 🔴 catches** → TRUE_COUPLING.")
        print("    The resident audit_suite now covers the 'displacement-crosses-the-wall' topologically-irreducible self-deception (map-equipped naive point check misses, relational covers it).")
    else:
        print(f"  ► ⚠️ wiring anomaly (reported honestly): verdict={v2} naive-point-check ok={jp2} relational ok={jr2}")

    print(f"\n{'='*78}\n  conclusion (criterion separation, each doing its job + irreducibility wiring)\n{'='*78}")
    print(f"  Scenario A : {rA['verdict']}  EC5'={'🔴' if not rA['ec5_prime_ok'] else '🟢'} joint={'🔴' if not rA['joint_ok'] else '🟢'}"
          f"  → physics-layer EC5' single-layer catch (not coupling)")
    print(f"  Scenario B : {rB['verdict']}  EC5'={'🟢' if rB['ec5_prime_ok'] else '🔴'} joint={'🔴' if not rB['joint_ok'] else '🟢'}"
          f"  → joint naive point check catches (o_t inside wall · reducible form)")
    print(f"  Scenario B2: {v2}  EC5'={'🟢' if res2['physics']['ec5_prime_ok'] else '🔴'} "
          f"naive-point-check={'🟢 miss' if jp2 else '🔴'} relational={'🔴 catch' if not jr2 else '🟢'}"
          f"  → only the relational catches (displacement-crosses-wall · topologically irreducible)")
    clean = a_ok and b_ok and b2_ok
    print(f"\n  criterion separation clean + irreducibility wired: {'✅ A=EC5'+chr(39)+' single-layer / B=joint point check / B2=only relational' if clean else '⚠️ see analysis above'}")

    json.dump({"scenario_A": rA, "scenario_B": rB,
               "scenario_B2": {"verdict": v2, "naive_point_ok": bool(jp2), "relational_ok": bool(jr2),
                               "physics_ok": bool(res2["physics"]["ok"]), "contract_ok": bool(res2["contract"]["ok"]),
                               "ec5_prime_ok": bool(res2["physics"]["ec5_prime_ok"]), "irreducible_integrated": bool(b2_ok)},
               "criteria_separated": bool(clean)},
              open(os.path.join(HERE, "coupling_summary.json"), "w"), indent=2, ensure_ascii=False)
    return clean


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
