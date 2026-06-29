# -*- coding: utf-8 -*-
"""
audit_suite.py  ——  resident three-layer audit suite (contract / physics / joint)
==========================================================
Fixes each layer's audit into a parallel resident suite, for unified runtime invocation:

  contract layer (report self-consistency) : C1 true-fork / C2 frame-seq monotonicity / C3 disconnect-freezes / C_I mutual-info leakage
  physics layer (physics self-consistency)  : EC1 energy budget / EC2 no free energy / EC3 actuator bound /
                                 EC4 collision non-negativity / EC5 non-penetration (ledger) / **EC5' truth-vs-map (within-physics geometric recompute)**
  joint layer (report×physics cross-state)  : ① JOINT odom-vs-claimed-map (naive point check · reducible)
                                 ② RELATIONAL seg(truth,odom) crosses wall · both endpoints free (**topologically irreducible**)

Layer responsibility separation (Phase4 + this round's EC5' completion):
  - EC5' (within physics) catches "truth really crosses the wall" — incl. the EC5 gap where collision detection drops a wall (phantom wall) making ledger penetration=0.
  - JOINT (cross-state) catches "truth legitimate but odom fabricates the crossing" — true coupling, catchable neither within physics nor by contract self-consistency.
  Criterion separation: truth_vs_map(EC5') red ⇒ catchable within a single physics layer (not coupling); EC5' green ∧ JOINT red ⇒ true coupling.
"""

import numpy as np

from energy_audit import audit_session as _energy_audit
from integrity_audit import (check_truth_odom_fork, check_seq_integrity, check_feed_liveness)
from leakage_audit import ci_audit
from joint_audit import ec5_prime, joint_report_vs_map
from relational_oracle import relational_oracle

# relational through-cross persistence-gate threshold: validated by docs/ScenB-Irreducibility.md (e) + G5 Part D,
# within a gated short window healthy is 0 false positives (δ≪clearance), so folding it into the resident suite introduces no false positives.
JOINT_PERSIST_FRAC = 0.5


def _c1c2c3_session(truth, odom):
    return [{"recv_t": float(i), "seq": i, "step": i,
             "truth": {"x": float(truth[i][0]), "y": float(truth[i][1]), "theta": 0.0},
             "odom": {"x": float(odom[i][0]), "y": float(odom[i][1]), "theta": 0.0},
             "terminated": False, "truncated": False, "link_status": "online"}
            for i in range(len(truth))]


def run_suite(truth_traj, odom_traj, ledger, walls, radius, slip,
              v_max, w_max, run_ci=True):
    """Run the resident three-layer suite, returning {contract, physics, joint} per-layer results + overall verdict.

    ledger: per-frame energy/collision ledger (for EC1-EC5). walls/radius: claimed-map geometry (for EC5'/JOINT).
    """
    truth = np.asarray(truth_traj, dtype=np.float64)
    odom = np.asarray(odom_traj, dtype=np.float64)

    # —— contract layer ——
    sess = _c1c2c3_session(truth, odom)
    contract_checks = [check_truth_odom_fork(sess), check_seq_integrity(sess),
                       check_feed_liveness(sess)]
    if run_ci:
        contract_checks.append(ci_audit(truth, odom, slip))
    contract_ok = all(c["ok"] for c in contract_checks)

    # —— physics layer: EC1-EC5 (ledger) + EC5' (within-physics truth-vs-map geometry) ——
    ec15 = _energy_audit(ledger, v_max=v_max, w_max=w_max, with_collision=True)
    ec5p = ec5_prime(truth, walls, radius)
    physics_checks = ec15["checks"] + [ec5p]
    physics_ok = ec15["passed"] and ec5p["ok"]

    # —— joint layer (report×physics cross-state · two complementary paths) ——
    #   ① naive point check joint_report_vs_map: catches "odom inside the wall" — reducible (the map-equipped contract also catches it).
    #   ② relational relational_oracle (through-cross + persistence gate): catches "truth/report land in different free connected regions,
    #      displacement segment crosses the wall" — **topologically irreducible** (both endpoints free, the map-equipped contract also misses; see docs/ScenB-Irreducibility.md).
    #   the joint layer is red ⟺ either path is red (complementary coverage: the old form o_t-inside-wall + the new form displacement-crosses-wall).
    joint_point = joint_report_vs_map(odom, walls, radius)
    joint_rel = relational_oracle(truth, odom, walls, radius, persist_frac=JOINT_PERSIST_FRAC)
    joint_ok = joint_point["ok"] and joint_rel["ok"]

    return {
        "contract": {"ok": contract_ok, "checks": contract_checks},
        "physics": {"ok": physics_ok, "checks": physics_checks,
                    "ec5_prime_ok": ec5p["ok"], "ec5_prime": ec5p},
        "joint": {"ok": joint_ok, "check": joint_point, "relational": joint_rel},
    }


def format_suite(res, title=""):
    lines = [title] if title else []
    def layer(tag, ok):
        return f"  {tag:<30}: {'🟢 all pass' if ok else '🔴 has red'}"
    lines.append(layer("contract C1/C2/C3+C_I", res["contract"]["ok"]))
    for c in res["contract"]["checks"]:
        lines.append(f"      [{'🟢' if c['ok'] else '🔴'}] {c['check']}")
    lines.append(layer("physics EC1-EC5+EC5'", res["physics"]["ok"]))
    for c in res["physics"]["checks"]:
        mk = '🟢' if c['ok'] else '🔴'
        lines.append(f"      [{mk}] {c['check']}"
                     + ("" if c["ok"] else f"  └ {c['detail']}"))
    lines.append(layer("joint JOINT(report×physics)", res["joint"]["ok"]))
    jp = res["joint"]["check"]
    lines.append(f"      [{'🟢' if jp['ok'] else '🔴'}] {jp['check']} (naive point check · reducible)"
                 + ("" if jp["ok"] else f"  └ {jp['detail']}"))
    jr = res["joint"].get("relational")
    if jr is not None:
        lines.append(f"      [{'🟢' if jr['ok'] else '🔴'}] {jr['check']} (relational · topologically irreducible)"
                     + ("" if jr["ok"] else f"  └ {jr['detail']}"))
    return "\n".join(lines)


def coupling_label(res):
    """Give the coupling verdict from the three-layer result. EC5' red = catchable within physics (not coupling); EC5' green ∧ JOINT red = true coupling."""
    ec5p_ok = res["physics"]["ec5_prime_ok"]
    joint_ok = res["joint"]["ok"]
    if not ec5p_ok:
        return "PHYSICS_INTERNAL"     # truth really crosses the wall: catchable within a single physics layer (EC5') → not true coupling
    if not joint_ok:
        return "TRUE_COUPLING"        # truth legitimate, only the report illegitimate → catchable only by the joint → true coupling
    return "NO_VIOLATION"
