# -*- coding: utf-8 -*-
"""
joint_audit.py  ——  Phase4 · report×physics joint cross-check (trajectory vs claimed physical map)
==================================================================================
Invariant (cross-state): every position on a reported trajectory must be legally reachable within the **claimed physical map** —
        the robot circle must not fall inside any claimed wall's AABB. A violation = reporting a "physically impossible to reach" position.

🔴 Key distinction (this module serves the coupling stress test, honesty first):
    this check can act on two kinds of trajectory, which distinguish "true coupling" from "merely an EC5 implementation gap":
      - truth_vs_map (truth trajectory vs map): a pure **within-physics** check (both truth and map are on the physics side),
        no report channel needed. If it flags red, the self-deception is **catchable within the physics layer alone** (should belong to a complete EC5),
        and **does not constitute report×physics true coupling**.
      - odom_vs_map (reported trajectory vs map): a **report×physics cross-state** check (odom from contract, map from physics).
    Necessary-and-sufficient characterization of true coupling: **truth_vs_map green (physical truth legitimate) while odom_vs_map red (report illegitimate)** —
    catchable neither within physics nor by contract self-consistency, only by the report × physical-map joint.

Pure geometry, no env/torch dependency; consumes (trajectory positions, claimed walls AABB, radius).
"""

import math

PEN_FLOOR = 1e-3       # trajectory-map penetration red threshold (m): beyond it means "inside the wall"


def _circle_aabb_pen(pos, walls, radius):
    """The max penetration depth of circle center pos against walls (positive when inside a wall)."""
    max_pen = 0.0
    for wx1, wx2, wy1, wy2 in walls:
        cx = min(max(pos[0], wx1), wx2)
        cy = min(max(pos[1], wy1), wy2)
        d = math.hypot(pos[0] - cx, pos[1] - cy)
        if d < radius:
            max_pen = max(max_pen, radius - d)
    return max_pen


def traj_vs_map(traj, walls, radius, label="traj"):
    """Run the "trajectory-map non-penetration" check on a trajectory, returning result."""
    worst, worst_i = 0.0, None
    n_illegal = 0
    for i, p in enumerate(traj):
        pen = _circle_aabb_pen(p, walls, radius)
        if pen > PEN_FLOOR:
            n_illegal += 1
            if pen > worst:
                worst, worst_i = pen, i
    name = f"JOINT_TRAJ_VS_MAP[{label}]"
    desc = f"is the {label} trajectory legally reachable within the claimed map throughout"
    if worst_i is not None:
        return {"check": name, "desc": desc, "status": "RED", "ok": False,
                "detail": f"{label} reported position falls inside a claimed wall: {n_illegal} illegal frames, deepest penetration "
                          f"{worst:.3f} m @frame{worst_i} (physically impossible to legally reach)",
                "locator": {"first_illegal_frame": worst_i, "n_illegal_frames": n_illegal,
                            "max_penetration_m": round(worst, 4),
                            "pos": [round(float(traj[worst_i][0]), 3), round(float(traj[worst_i][1]), 3)]}}
    return {"check": name, "desc": desc, "status": "GREEN", "ok": True,
            "detail": f"{label} trajectory is legal within the claimed map throughout (no wall entry)", "locator": None}


def ec5_prime(truth_traj, walls, radius):
    """EC5' (physics layer): within-physics "truth-vs-claimed-map geometry" recomputation of non-penetration.

    **Does not trust the ledger's penetration field** — judges independently using the truth position against the **claimed full map** geometry.
    So it catches the EC5 gap where "collision detection drops a wall (phantom wall) making ledger penetration=0" (the truth really inside the wall).
    A pure within-physics check (both truth and map on the physics side), no report channel needed.
    """
    r = traj_vs_map(truth_traj, walls, radius, "truth")
    r["check"] = "EC5P_TRUTH_MAP"
    r["desc"] = "EC5': truth position not inside any claimed-map wall (within-physics geometric recompute, does not trust the ledger)"
    return r


def joint_report_vs_map(odom_traj, walls, radius):
    """Joint layer (report×physics): reported trajectory (odom) -vs- claimed-map geometry non-penetration.

    odom from contract, map from physics → a cross-state check. Catches the true coupling "truth legitimate but the report fabricates a wall-crossing track"
    (this kind is catchable neither within physics (truth legitimate, EC5' green) nor by contract self-consistency (odom self-consistent, C1-3/C_I green)).
    """
    r = traj_vs_map(odom_traj, walls, radius, "odom")
    r["check"] = "JOINT_ODOM_MAP"
    r["desc"] = "joint: reported (odom) position not inside any claimed-map wall (report×physics cross-state)"
    return r


def coupling_verdict(truth_traj, odom_traj, walls, radius):
    """Coupling stress-test core verdict: returns (truth_vs_map, odom_vs_map, verdict_str).

    verdict:
      'TRUE_COUPLING'  —— truth_vs_map green and odom_vs_map red: catchable only by the report×physics joint (true coupling).
      'PHYSICS_INTERNAL' —— truth_vs_map red: catchable within physics alone (truth vs map) → not true coupling (EC5 gap).
      'NO_VIOLATION'   —— both green: no wall-crossing violation.
    """
    tv = traj_vs_map(truth_traj, walls, radius, "truth")
    ov = traj_vs_map(odom_traj, walls, radius, "odom")
    if not tv["ok"]:
        verdict = "PHYSICS_INTERNAL"
    elif not ov["ok"]:
        verdict = "TRUE_COUPLING"
    else:
        verdict = "NO_VIOLATION"
    return tv, ov, verdict
