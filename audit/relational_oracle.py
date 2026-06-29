# -*- coding: utf-8 -*-
"""
relational_oracle.py  ——  dual-state-coupling "topological irreducibility" oracle (Scenario B defense §5)
================================================================================
Background: a reviewer attacks the "dual-state coupling is irreducible" claim — arguing that Scenario B can be caught just by "giving the map to the contract layer and checking whether o_t is inside the wall",
      which would make the relational (joint) layer redundant. Our defense is **topological irreducibility**: a genuine relational fault
      cannot be caught even by the **map-equipped contract layer** alone — the fault lies in the **displacement vector o_t−x_t crossing the wall**, not in either endpoint being inside the wall.

Five oracles (evaluated in parallel on the same (truth x, odom o, claimed map M)):
  ① physics_oracle      : x_t vs M        —— is the truth in free space (within physics).
  ② contract_noise_oracle: ‖o_t−x_t‖ ≤ ξ  —— is the report within the sensor noise budget (contract-noise self-consistency).
  ③ map_point_oracle    : o_t ∈ M?        —— 🔴**map-equipped contract baseline (M6a)**: is the reported point inside the wall (single-point check).
  ④ map_continuity_oracle: seg(o_{t-1},o_t)∩M —— 🔴**stronger report-only baseline (M6b)**: does the **reported track itself** cross the wall.
  ⑤ relational_oracle   : seg(x_t,o_t)∩M  —— **relational**: does the truth→report **displacement segment** cross the rigid wall (this alone is the (x,o,M) ternary joint).

Necessary-and-sufficient irreducibility: ①②③④ all pass (each single projection legitimate) and only ⑤ rejects ⇒ the fault lives only in the truth↔report cross relation,
            and no "single-endpoint/single-projection" check (incl. the map-equipped one) can decompose it.

Key precondition: **wall thickness d < noise budget ξ** — otherwise o_t cannot be placed in the free space on the other side of the wall within the noise.

Pure geometry, stdlib + numpy, no env/torch dependency. Wall AABB format (xmin, xmax, ymin, ymax).
"""

import math

import numpy as np

PEN_FLOOR = 1e-3       # point-in-wall decision floor (m)


# --------------------------------------------------------------------
# geometric primitives
# --------------------------------------------------------------------
def _point_in_aabb(p, w, radius=0.0):
    """Whether point p (a circle inflated by radius) falls inside the wall AABB.

    radius=0 (point estimate): whether the point is **strictly inside the AABB** (incl. the AABB containment test, meaningful even for a zero-radius circle).
    radius>0: the center is inside the AABB, or the center-to-AABB nearest distance < radius (circle overlaps the wall).
    """
    xmin, xmax, ymin, ymax = w
    if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
        return True                                   # the center/point is inside the wall
    if radius <= 0.0:
        return False
    cx = min(max(p[0], xmin), xmax)
    cy = min(max(p[1], ymin), ymax)
    return math.hypot(p[0] - cx, p[1] - cy) < radius - PEN_FLOOR


def _seg_crosses_aabb(p0, p1, w, radius=0.0):
    """Whether segment p0→p1 (with the wall inflated by radius) intersects the wall AABB (slab/Liang-Barsky method)."""
    xmin, xmax, ymin, ymax = w[0] - radius, w[1] + radius, w[2] - radius, w[3] + radius
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    tmin, tmax = 0.0, 1.0
    for lo, hi, o, dd in ((xmin, xmax, p0[0], dx), (ymin, ymax, p0[1], dy)):
        if abs(dd) < 1e-12:
            if o < lo or o > hi:
                return False
        else:
            t1, t2 = (lo - o) / dd, (hi - o) / dd
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1); tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _any_point_in(traj, walls, radius):
    n = sum(1 for p in traj if any(_point_in_aabb(p, w, radius) for w in walls))
    return n


def dist_point_aabb(p, w):
    """The Euclidean distance from point p to the wall AABB (xmin,xmax,ymin,ymax) (0 if the point is inside the AABB)."""
    dx = max(w[0] - p[0], 0.0, p[0] - w[1])
    dy = max(w[2] - p[1], 0.0, p[1] - w[3])
    return math.hypot(dx, dy)


def clearance(p, walls, radius=0.0):
    """The **free clearance** from point p (robot circle radius radius) to the nearest wall (touching the wall=0, larger is safer).

    Empirical-soundness sufficient condition: if the drift δ(t)=‖o_t−x_t‖ < clearance(x_t), then o_t lies inside the wall-free open ball
    B(x_t, clearance) → o_t and x_t are in the same free connected region, and the displacement segment stays inside the ball without crossing a wall →
    the relational oracle is **provably free of false positives**. So δ ≪ clearance is direct evidence of empirical soundness.
    """
    return min(dist_point_aabb(p, w) for w in walls) - radius


def _result(check, ok, detail, **extra):
    r = {"check": check, "status": "GREEN" if ok else "RED", "ok": bool(ok), "detail": detail}
    r.update(extra)
    return r


# --------------------------------------------------------------------
# five oracles
# --------------------------------------------------------------------
def physics_oracle(truth, walls, radius=0.0):
    """① within physics: whether the truth position is in free space throughout (not inside a wall)."""
    n = _any_point_in(np.asarray(truth, float), walls, radius)
    return _result("PHYSICS[x∈free]", n == 0,
                   f"truth-in-wall frames={n} (should be 0=legal)", n_illegal=n)


def contract_noise_oracle(truth, odom, xi):
    """② contract-noise self-consistency: max_t ‖o_t−x_t‖ ≤ ξ (the report is within the sensor noise budget)."""
    t = np.asarray(truth, float); o = np.asarray(odom, float)
    e = np.linalg.norm(o - t, axis=1)
    emax = float(e.max())
    return _result("CONTRACT_NOISE[‖o-x‖≤ξ]", emax <= xi,
                   f"max‖o-x‖={emax:.3f} {'≤' if emax <= xi else '>'} ξ={xi:.3f}",
                   max_err=emax, xi=float(xi))


def map_point_oracle(odom, walls, radius=0.0):
    """③ 🔴 map-equipped contract baseline (M6a): whether the reported point o_t is inside the wall (single-point map check)."""
    n = _any_point_in(np.asarray(odom, float), walls, radius)
    return _result("MAP_POINT[o∈M? · M6a]", n == 0,
                   f"report-point-in-wall frames={n} (>0 to catch; =0=miss)", n_illegal=n)


def map_continuity_oracle(odom, walls, radius=0.0):
    """④ 🔴 stronger report-only baseline (M6b): whether the **reported track itself** segment seg(o_{t-1},o_t) crosses the wall."""
    o = np.asarray(odom, float)
    cross = [i for i in range(len(o) - 1)
             if any(_seg_crosses_aabb(o[i], o[i + 1], w, radius) for w in walls)]
    return _result("MAP_CONTINUITY[seg(o,o)∩M · M6b]", len(cross) == 0,
                   f"report-track self-crossing segments={len(cross)} (>0 to catch; =0=miss)", n_cross=len(cross))


def relational_oracle(truth, odom, walls, radius=0.0, persist_frac=0.0):
    """⑤ relational: whether seg(x_t,o_t) **crosses** the rigid wall (the truth→report displacement segment; the (x,o,M) ternary joint).

    **through-crossing** = both endpoints are in free space (neither x_t nor o_t inside the wall) **while the displacement segment crosses the wall** —
    i.e. truth and report land in **different free connected regions** separated by a rigid wall (topological dislocation). This is the genuine relational fault:
      - both endpoints free → physics(x) and the map-equipped contract(o∈M?) each pass;
      - only "the displacement segment connecting the two free regions crosses the wall" catches it → (x,o,M) ternary, indecomposable.
    (only "o inside the wall" belongs to map_point; only x inside the wall belongs to physics — so here both endpoints are required to be free, extracting the pure relational component.)

    persist_frac: the minimum crossing-frame fraction required to reject (separating "persistent crossing = real fault frac→1" from
                  "occasional noise crossing = healthy frac small"; =0 means a single frame is flagged, as a pure-geometry irreducibility witness).
    """
    t = np.asarray(truth, float); o = np.asarray(odom, float)
    n = len(t)
    cross = []
    for i in range(n):
        x_free = not any(_point_in_aabb(t[i], w, radius) for w in walls)
        o_free = not any(_point_in_aabb(o[i], w, radius) for w in walls)
        seg = any(_seg_crosses_aabb(t[i], o[i], w, radius) for w in walls)
        if x_free and o_free and seg:                 # both endpoints free + displacement segment crosses = topological dislocation
            cross.append(i)
    frac = len(cross) / n if n else 0.0
    reject = frac > persist_frac
    first = cross[0] if cross else None
    return _result("RELATIONAL[seg(x,o) crosses M · both endpoints free]", not reject,
                   f"displacement-seg crossing (both endpoints free) frames={len(cross)}/{n} (frac={frac:.2f}"
                   + (f" >thr {persist_frac:.2f}→catch" if reject else f" ≤thr {persist_frac:.2f}→pass")
                   + ")", n_cross=len(cross), frac=float(frac), first_cross=first)


def five_oracles(truth, odom, walls, xi, radius=0.0, persist_frac=0.0):
    """Run all five oracles on the same (truth, odom, walls), returning {name: result} + the irreducibility verdict."""
    res = {
        "physics": physics_oracle(truth, walls, radius),
        "contract_noise": contract_noise_oracle(truth, odom, xi),
        "map_point": map_point_oracle(odom, walls, radius),       # M6a
        "map_continuity": map_continuity_oracle(odom, walls, radius),  # M6b
        "relational": relational_oracle(truth, odom, walls, radius, persist_frac),
    }
    # irreducible: ①②③④ pass and only ⑤ rejects
    singles_pass = (res["physics"]["ok"] and res["contract_noise"]["ok"]
                    and res["map_point"]["ok"] and res["map_continuity"]["ok"])
    only_relational = singles_pass and (not res["relational"]["ok"])
    if only_relational:
        verdict = "IRREDUCIBLE_RELATIONAL"          # only the relational catches → topologically irreducible (defense holds)
    elif (not res["map_point"]["ok"]) or (not res["map_continuity"]["ok"]):
        verdict = "REDUCIBLE_BY_MAP_CONTRACT"        # the map-equipped contract also catches → reducible (the claim needs revision)
    elif res["relational"]["ok"]:
        verdict = "NO_VIOLATION"                     # all five oracles pass
    else:
        verdict = "OTHER"
    res["verdict"] = verdict
    return res


def format_five(res, title=""):
    order = [("① physics x∈free", "physics"), ("② contract noise ‖o-x‖≤ξ", "contract_noise"),
             ("③ map-equipped contract o∈M? (M6a)", "map_point"), ("④ report-only seg(o,o)∩M (M6b)", "map_continuity"),
             ("⑤ relational seg(x,o)∩M", "relational")]
    lines = [title] if title else []
    for label, key in order:
        r = res[key]
        mark = "🟢 pass" if r["ok"] else "🔴 reject"
        lines.append(f"    [{mark}] {label:<34} └ {r['detail']}")
    lines.append(f"    ── verdict ──► {res['verdict']}")
    return "\n".join(lines)
