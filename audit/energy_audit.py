# -*- coding: utf-8 -*-
"""
energy_audit.py  ——  Phase1a · physics-layer energy audit (dynamics-fidelity self-deception detection)
=====================================================================
The **physics-layer** counterpart of the contract layer's integrity_audit.py (C1/C2/C3). Input a per-frame energy-ledger stream,
output the RED/GREEN + localization of the three checks EC1/EC2/EC3. The audit itself is first proven able to "catch fakes" (see run_physics_audit.py).

The auditable conserved quantity = mechanical energy E = ½m·v² + ½I·w². Under action space A the system is non-isolated (actuator injection + damping dissipation),
so we audit "energy-budget self-consistency" rather than absolute conservation:
        each step should have  ΔE ≈ W_act − D_damp   (W_act=actuator net work, D_damp=damping dissipation)

Three checks (per-frame ledger contract, list[dict]):
    EC1 ENERGY_BUDGET   —— the budget residual r=ΔE−(W_act−D_damp) must be within the numeric floor. Catches anything breaking energy-ledger self-consistency.
    EC2 NO_FREE_ENERGY  —— the kinetic-energy increment must not exceed actuator work (ΔE≤W_act, damping only dissipates). Catches "energy created from nothing".
    EC3 ACTUATOR_BOUND  —— the actual velocity must not exceed the actuator physical limit. Catches "velocity out of bounds / KE jump".

Ledger-frame record contract:
    {"step","seq","E_kin","dE","W_act","D_damp","v_act","w_act"}

Criterion (conservative + persistence gating, mirroring the contract layer's C1 path gating / C3 STALE_TOL anti-jitter):
    a residual / over-energy must be **over the floor for ≥ K_PERSIST consecutive steps** to flag RED, avoiding single-step numerical-transient false positives.
Pure stdlib, no env/torch dependency; can run independently on any ledger session (incl. offline json).
"""

# ---- numeric floor (clean residual at machine precision ~1e-16; fault residual ~1e-2..1e-1, very wide separation) ----
EPS_ABS = 1e-6        # absolute energy floor (J)
EPS_REL = 1e-3        # relative floor: floor = EPS_REL·(|W_act|+|D_damp|+|ΔE|) + EPS_ABS
K_PERSIST = 3         # consecutive-over-floor frame threshold (anti single-step transient; mimics C3 STALE_TOL)

# actuator physical speed limit (aligned with embodied_env.V_PHYS_MAX / W_PHYS_MAX; overridable in audit_session)
V_PHYS_MAX = 1.25     # = MAX_LIN_VEL * 1.25
W_PHYS_MAX = 1.875    # = MAX_ANG_VEL * 1.25


PEN_FLOOR = 1e-4      # non-penetration residual floor (m): after 2-iteration pushout the residual penetration should be far below this


def _floor(f):
    return EPS_REL * (abs(f["W_act"]) + abs(f["D_damp"]) + abs(f["dE"])
                      + abs(f.get("E_contact_decl", 0.0))) + EPS_ABS


def _result(name, desc, ok, detail, locator=None):
    return {"check": name, "desc": desc, "status": "GREEN" if ok else "RED",
            "ok": bool(ok), "detail": detail, "locator": locator}


# ====================================================================
# EC1 · energy-budget residual (catches anything breaking ΔE=W_act−D_damp self-consistency)
# ====================================================================
def check_energy_budget(session):
    run = 0
    run_start = None
    worst = 0.0
    for i, f in enumerate(session):
        # including the (declared) collision dissipation term: ΔE should = W_act − D_damp − E_contact_decl (degenerates with E_contact=0 when no collision)
        r = f["dE"] - (f["W_act"] - f["D_damp"] - f.get("E_contact_decl", 0.0))
        if abs(r) > _floor(f):
            if run == 0:
                run_start = i
            run += 1
            if abs(r) > abs(worst):
                worst = r
            if run >= K_PERSIST:
                return _result(
                    "EC1_ENERGY_BUDGET", "energy budget: is ΔE=W_act−D_damp self-consistent", False,
                    f"budget residual over the floor for {run} consecutive frames (from step={session[run_start]['step']}); "
                    f"max residual r={worst:.3e} J (clean should be ~1e-16)",
                    locator={"first_step": session[run_start]["step"],
                             "first_seq": session[run_start]["seq"],
                             "max_residual_J": round(worst, 6),
                             "floor_J": round(_floor(session[run_start]), 9)})
        else:
            run = 0
    return _result("EC1_ENERGY_BUDGET", "energy budget: is ΔE=W_act−D_damp self-consistent", True,
                   "all frame budget residuals within the numeric floor (ΔE and W_act−D_damp self-consistent)")


# ====================================================================
# EC2 · no free energy (ΔE ≤ W_act, damping only dissipates and never adds energy → catches "energy creation")
# ====================================================================
def check_no_free_energy(session):
    run = 0
    run_start = None
    worst = 0.0
    for i, f in enumerate(session):
        excess = f["dE"] - f["W_act"]    # >0 means the KE increment exceeds actuator work (energy from nothing)
        if excess > _floor(f):
            if run == 0:
                run_start = i
            run += 1
            worst = max(worst, excess)
            if run >= K_PERSIST:
                return _result(
                    "EC2_NO_FREE_ENERGY", "KE increment not exceeding actuator work (no free energy)", False,
                    f"KE created from nothing for {run} consecutive frames (from step={session[run_start]['step']}); "
                    f"max excess ΔE−W_act={worst:.3e} J — energy created without work (violates the 2nd law)",
                    locator={"first_step": session[run_start]["step"],
                             "first_seq": session[run_start]["seq"],
                             "max_excess_J": round(worst, 6)})
        else:
            run = 0
    return _result("EC2_NO_FREE_ENERGY", "KE increment not exceeding actuator work (no free energy)", True,
                   "no free energy: KE increment always ≤ actuator work (damping only dissipates)")


# ====================================================================
# EC3 · actuator speed limit (catches "velocity over the physical limit")
# ====================================================================
def check_actuator_bound(session, v_max=V_PHYS_MAX, w_max=W_PHYS_MAX):
    run = 0
    run_start = None
    worst = 0.0
    for i, f in enumerate(session):
        over = max(abs(f["v_act"]) - v_max, abs(f["w_act"]) - w_max)
        if over > 0:
            if run == 0:
                run_start = i
            run += 1
            worst = max(worst, over)
            if run >= K_PERSIST:
                return _result(
                    "EC3_ACTUATOR_BOUND", "actual velocity within the actuator physical bound", False,
                    f"velocity over the bound for {run} consecutive frames (from step={session[run_start]['step']}); "
                    f"max overshoot {worst:.3f} (v_max={v_max}, w_max={w_max})",
                    locator={"first_step": session[run_start]["step"],
                             "first_seq": session[run_start]["seq"],
                             "max_overshoot": round(worst, 4),
                             "v_act": round(session[run_start]["v_act"], 4)})
        else:
            run = 0
    return _result("EC3_ACTUATOR_BOUND", "actual velocity within the actuator physical bound", True,
                   "both actual linear/angular velocity within the actuator physical bound")


# ====================================================================
# EC4 · collision energy non-negativity (a collision only dissipates, never adds energy: E_contact_act ≥ 0 → catches "over-bounce energy gain")
# ====================================================================
def check_collision_nonneg(session):
    run = 0
    run_start = None
    worst = 0.0
    for i, f in enumerate(session):
        ec = f.get("E_contact_act", 0.0)        # actual collision kinetic-energy change (KE_before−KE_after)
        if ec < -_floor(f):                     # <0 means the collision creates energy from nothing (violates the 2nd law)
            if run == 0:
                run_start = i
            run += 1
            worst = min(worst, ec)
            if run >= 1:                         # collisions are sparse: a single-frame energy gain is flagged (no persistence required)
                return _result(
                    "EC4_COLLISION_NONNEG", "collision creates no energy (E_contact≥0)", False,
                    f"collision created energy @step={session[run_start]['step']}: E_contact_act={worst:.3e} J<0 "
                    f"— bounce adds energy (violates collision energy non-negativity)",
                    locator={"first_step": session[run_start]["step"],
                             "first_seq": session[run_start]["seq"],
                             "min_E_contact_J": round(worst, 6)})
        else:
            run = 0
    return _result("EC4_COLLISION_NONNEG", "collision creates no energy (E_contact≥0)", True,
                   "no collision energy creation: all contacts have E_contact_act ≥ 0 (collisions only dissipate)")


# ====================================================================
# EC5 · non-penetration (after resolution the robot must not overlap a wall: penetration ≤ floor → catches "penetration not corrected")
# ====================================================================
def check_non_penetration(session):
    worst = 0.0
    worst_i = None
    for i, f in enumerate(session):
        pen = f.get("penetration", 0.0)
        if pen > PEN_FLOOR and pen > worst:
            worst, worst_i = pen, i
    if worst_i is not None:
        return _result(
            "EC5_NON_PENETRATION", "no residual penetration after resolution", False,
            f"residual penetration @step={session[worst_i]['step']}: penetration={worst:.4f} m > {PEN_FLOOR} "
            f"— penetration not corrected but claimed resolved",
            locator={"first_step": session[worst_i]["step"], "first_seq": session[worst_i]["seq"],
                     "max_penetration_m": round(worst, 5)})
    return _result("EC5_NON_PENETRATION", "no residual penetration after resolution", True,
                   "no residual penetration: after collision pushout the robot never overlaps a wall")


CHECKS = [check_energy_budget, check_no_free_energy, check_actuator_bound,
          check_collision_nonneg, check_non_penetration]


def audit_session(session, v_max=V_PHYS_MAX, w_max=W_PHYS_MAX, with_collision=False):
    """Run the energy/collision audit on a ledger session.
    When with_collision=True, additionally run EC4 (collision non-negativity) / EC5 (non-penetration) (the ledger must contain E_contact_act/penetration fields).
    """
    results = [
        check_energy_budget(session),
        check_no_free_energy(session),
        check_actuator_bound(session, v_max=v_max, w_max=w_max),
    ]
    if with_collision:
        results.append(check_collision_nonneg(session))
        results.append(check_non_penetration(session))
    passed = all(r["ok"] for r in results)
    return {"passed": passed, "verdict": "GREEN" if passed else "RED", "checks": results}


def format_report(audit, title=""):
    lines = []
    if title:
        lines.append(title)
    overall = "🟢 ALL GREEN (GREEN)" if audit["passed"] else "🔴 physics self-deception detected (RED)"
    lines.append(f"  energy audit verdict: {overall}")
    for r in audit["checks"]:
        mark = "🟢 GREEN" if r["ok"] else "🔴 RED  "
        lines.append(f"    [{mark}] {r['check']:<20} {r['desc']}")
        lines.append(f"             └─ {r['detail']}")
        if not r["ok"] and r["locator"]:
            lines.append(f"             └─ locator: {r['locator']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage: python energy_audit.py <energy_session.json>")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        sess = json.load(fh)
    res = audit_session(sess)
    print(format_report(res, title=f"== energy audit {sys.argv[1]} =="))
    sys.exit(0 if res["passed"] else 2)
