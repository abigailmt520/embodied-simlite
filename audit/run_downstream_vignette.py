# -*- coding: utf-8 -*-
"""
run_downstream_vignette.py  ——  S-B downstream-consequence vignette (deterministic A* planner)
=============================================================================================
Point (lightweight Motivation / RQ0 vignette): a navigation planner that CONSUMES the twin's
reported map inherits the twin's self-deception.  The self-deceptive map = true geometry MINUS the
thin wall W (the same d<xi barrier the twin self-deceives about in Scenario B / tunneling).  A fixed,
deterministic A* faithfully plans the shortest path on whatever map it is given; on the deceptive map
it plans an ILLEGAL shortcut straight through the impassable wall.  The auditor gates the plan at the
consumption boundary using the SAME segment / non-penetration predicate as the relational oracle
(`relational_oracle._seg_crosses_aabb`), checked against the TRUE geometry -> the illegal plan is
FLAGGED and rejected, cutting the "poison chain".

HONEST FRAMING (do not drift): A* does NOT learn or exploit.  The illegal path is a CONSEQUENCE of
consuming the corrupted map (propagation / inheritance), not learned exploitation.  Learned
specification-gaming is a separate literature we cite, not demonstrate.  Overlap with RQ3 is expected
and stated: same thin-wall fabrication, same crossing predicate; the new angle is the downstream
CONSUMER failure.

Reproducible: deterministic grid A* (fixed resolution, fixed neighbour order, fixed heap tie-break).
Read-only imports of repo modules (embodied_env, relational_oracle, run_scenB_irreducibility); does not
modify them.
Run:  conda run -n base python audit/run_downstream_vignette.py
"""
import json
import math
import os
import sys
import heapq

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodied_env import EmbodiedNavEnv as Env                         # noqa: E402  (constants only, read-only)
import run_scenB_irreducibility as RS                                  # noqa: E402  (WALL_THIN, XI)
from relational_oracle import _seg_crosses_aabb, dist_point_aabb       # noqa: E402  (SAME predicate as the oracle)

# ---- geometry (reuse the Scenario-B / RQ3 thin wall W) ----
W = tuple(RS.WALL_THIN[0])                # (xmin, xmax, ymin, ymax) = (30.0, 30.2, 9.0, 16.0)
D = round(W[1] - W[0], 4)                 # 0.20
XI = RS.XI                                # 0.50
RAD = Env.ROBOT_RADIUS                    # 0.20
ARENA = (27.0, 34.0, 5.0, 20.0)           # xlo, xhi, ylo, yhi (free space above/below W to go around)
START = (28.5, 12.5)                      # side A (x < 30.0)
GOAL = (31.5, 12.5)                       # side B (x > 30.2); straight S->G crosses W
RES = 0.05                                # grid cell (<= 0.10 so d=0.20 wall spans >= 2 cells)
PLAN_MARGIN = 0.07                         # planner keeps an extra safety margin (plan radius = RAD+margin);
PLAN_RAD = RAD + PLAN_MARGIN               #   the auditor still checks at the true footprint RAD=0.20.


def _blocked(cx, cy, walls, radius):
    """cell-center (world) blocked if outside arena or footprint(radius) overlaps any wall."""
    if not (ARENA[0] <= cx <= ARENA[1] and ARENA[2] <= cy <= ARENA[3]):
        return True
    p = (cx, cy)
    return any(dist_point_aabb(p, w) < radius for w in walls)


def _grid():
    nx = int(round((ARENA[1] - ARENA[0]) / RES)) + 1
    ny = int(round((ARENA[3] - ARENA[2]) / RES)) + 1
    return nx, ny


def _c2w(i, j):
    return (ARENA[0] + i * RES, ARENA[2] + j * RES)


def _w2c(x, y):
    return (int(round((x - ARENA[0]) / RES)), int(round((y - ARENA[2]) / RES)))


# fixed neighbour order -> determinism
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def astar(walls):
    nx, ny = _grid()
    s = _w2c(*START); g = _w2c(*GOAL)
    bset = {}

    def blocked(i, j):
        if (i, j) not in bset:
            bset[(i, j)] = _blocked(*_c2w(i, j), walls, PLAN_RAD)
        return bset[(i, j)]

    def h(i, j):
        return math.hypot(i - g[0], j - g[1]) * RES

    openh = [(h(*s), 0, s)]; came = {}; gc = {s: 0.0}; cnt = 1; closed = set()
    while openh:
        _, _, c = heapq.heappop(openh)
        if c == g:
            break
        if c in closed:
            continue
        closed.add(c)
        ci, cj = c
        for dx, dy in _NB:
            ni, nj = ci + dx, cj + dy
            if blocked(ni, nj) or (ni, nj) in closed:
                continue
            if dx != 0 and dy != 0:                       # no diagonal corner-cutting through a wall
                if blocked(ci + dx, cj) or blocked(ci, cj + dy):
                    continue
            ng = gc[c] + math.hypot(dx, dy) * RES
            if (ni, nj) not in gc or ng < gc[(ni, nj)] - 1e-12:
                gc[(ni, nj)] = ng; came[(ni, nj)] = c
                heapq.heappush(openh, (ng + h(ni, nj), cnt, (ni, nj))); cnt += 1
    if g not in gc:
        return None, None
    path = [g]
    while path[-1] != s:
        path.append(came[path[-1]])
    path.reverse()
    return [list(_c2w(i, j)) for (i, j) in path], gc[g]


def _simplify(path):
    """collinear-merge to readable waypoints (keeps the polyline identical)."""
    if len(path) < 3:
        return path
    out = [path[0]]
    for k in range(1, len(path) - 1):
        a, b, c = out[-1], path[k], path[k + 1]
        # keep b only if direction changes
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-9:
            out.append(b)
    out.append(path[-1])
    return [[round(x, 3), round(y, 3)] for x, y in out]


def crosses_true_wall(path):
    """auditor: does the planned polyline (footprint r) traverse the TRUE wall W? (SAME oracle predicate)"""
    segs = [i for i in range(len(path) - 1) if _seg_crosses_aabb(path[i], path[i + 1], W, RAD)]
    return len(segs) > 0, segs


def _cross_point(path):
    """first point where the path centerline crosses the wall mid-plane x=(xmin+xmax)/2 (for the figure marker)."""
    xc = 0.5 * (W[0] + W[1])
    for i in range(len(path) - 1):
        x0, x1 = path[i][0], path[i + 1][0]
        if (x0 - xc) * (x1 - xc) <= 0 and x0 != x1:
            t = (xc - x0) / (x1 - x0)
            y = path[i][1] + t * (path[i + 1][1] - path[i][1])
            if W[2] <= y <= W[3]:
                return [round(xc, 3), round(y, 3)]
    return None


def main():
    print("=" * 92)
    print("  S-B downstream vignette: deterministic A* on honest vs self-deceptive map")
    print(f"  W={W} d={D} xi={XI} (d<xi={D < XI})  robot_r={RAD}  grid_res={RES}  S={START} G={GOAL}")
    print("=" * 92)

    honest_path, L_h = astar([W])          # honest reported map: W present
    decep_path, L_d = astar([])            # self-deceptive reported map: W erased

    # ---- STOP-condition checks (geometry must actually be set up right) ----
    stops = []
    if honest_path is None:
        stops.append("honest A* found NO path (W must be passable around its ends)")
    if decep_path is None:
        stops.append("deceptive A* found NO path")
    if not stops:
        h_cross, _ = crosses_true_wall(honest_path)
        d_cross, d_segs = crosses_true_wall(decep_path)
        if h_cross:
            stops.append("honest path CROSSES true W (it must go AROUND) -> geometry/inflation wrong")
        if not d_cross:
            stops.append("deceptive short path does NOT cross true W (it must) -> move S/G so the shortcut goes through W")
        if not (L_d < L_h):
            stops.append(f"deceptive path NOT shorter (L_d={L_d:.3f} >= L_h={L_h:.3f}) -> the deception must offer a shortcut")
    if stops:
        print("\n  🔴 STOP — geometry not set up correctly:")
        for s in stops:
            print("     -", s)
        print("  (reporting actual numbers; not forcing the expected result)")
        if honest_path is not None:
            print(f"     honest: len={L_h:.3f} crosses={crosses_true_wall(honest_path)[0]}")
        if decep_path is not None:
            print(f"     deceptive: len={L_d:.3f} crosses={crosses_true_wall(decep_path)[0]}")
        sys.exit(2)

    h_cross, _ = crosses_true_wall(honest_path)
    d_cross, d_segs = crosses_true_wall(decep_path)
    h_verdict = "FLAGGED" if h_cross else "PASS"
    d_verdict = "FLAGGED" if d_cross else "PASS"
    xp = _cross_point(decep_path)

    print(f"\n  honest map    -> path goes AROUND W   len={L_h:.3f} m  crosses_true_wall={h_cross}  verdict={h_verdict}")
    print(f"  deceptive map -> path goes THROUGH W   len={L_d:.3f} m  crosses_true_wall={d_cross}  verdict={d_verdict}")
    print(f"  shortcut saved by the deception: {L_h - L_d:.3f} m ({100*(L_h-L_d)/L_h:.0f}% shorter); first wall-crossing at {xp}")
    print(f"  auditor ON  -> deceptive plan FLAGGED -> rejected -> replan on true map -> legal {L_h:.3f} m path")
    print(f"  auditor OFF -> deceptive plan executed -> path traverses impassable wall (collision/stuck)")

    out = {
        "geometry": {"wall_W_centerline": [[round(0.5*(W[0]+W[1]), 3), W[2]], [round(0.5*(W[0]+W[1]), 3), W[3]]],
                     "wall_W_aabb": list(W), "d": D, "xi": XI,
                     "start": list(START), "goal": list(GOAL), "robot_radius": RAD, "arena": list(ARENA)},
        "honest": {"path": _simplify(honest_path), "length": round(L_h, 4),
                   "crosses_true_wall": bool(h_cross), "verdict": h_verdict},
        "deceptive": {"path": _simplify(decep_path), "length": round(L_d, 4),
                      "crosses_true_wall": bool(d_cross), "verdict": d_verdict,
                      "first_wall_crossing": xp, "n_crossing_segments": len(d_segs)},
        "gated_outcome": {
            "auditor_on": f"deceptive plan FLAGGED -> rejected -> replan on true map -> legal {round(L_h,3)} m path (or safe abort)",
            "auditor_off": "deceptive plan executed -> path traverses impassable wall W (collision/stuck); the geometric fact 'plan crosses an impassable wall' is itself the unsafe consequence"},
        "meta": {"grid_res": RES, "astar": "deterministic 8-connected grid A*, euclidean cost+heuristic, "
                 "fixed neighbour order, fixed heap tie-break (insertion counter), diagonal corner-cut forbidden",
                 "footprint": f"circle radius {RAD}; planner inflates by RAD+{PLAN_MARGIN} (safety margin, removes grid chord-cut grazing), auditor checks at the true footprint radius {RAD}",
                 "auditor_predicate": "relational_oracle._seg_crosses_aabb(p0,p1,W,radius) on the plan vs TRUE geometry",
                 "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
                 "note": "deceptive map = true geometry minus W (the same thin barrier the twin self-deceives about "
                 "in Scenario B / tunneling). Planner is a fixed deterministic A*; the illegal path is a CONSEQUENCE of "
                 "consuming the corrupted map (propagation/inheritance), NOT learned exploitation. Overlap with RQ3 is "
                 "intentional: same fabrication, same crossing predicate; the new angle is the downstream consumer."}}
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "downstream_planning_vignette.json"), "w"), indent=2)
    print("\n  [OK] data:", os.path.join(here, "downstream_planning_vignette.json"))
    try:
        make_figure(honest_path, decep_path, xp, os.path.join(here, "downstream_planning_vignette.png"))
    except Exception as e:
        print("  [WARN] figure skipped:", e)


def make_figure(honest_path, decep_path, xp, path_png):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    # inflated wall (config space) faint, then true wall solid
    ax.add_patch(Rectangle((W[0]-RAD, W[2]-RAD), (W[1]-W[0])+2*RAD, (W[3]-W[2])+2*RAD,
                           facecolor="#cccccc", edgecolor="none", alpha=0.5, zorder=1))
    ax.add_patch(Rectangle((W[0], W[2]), W[1]-W[0], W[3]-W[2],
                           facecolor="#444444", edgecolor="k", zorder=2, label="impassable wall W (d=0.20)"))
    hp = np.array(honest_path); dp = np.array(decep_path)
    ax.plot(dp[:, 0], dp[:, 1], "-", color="#d33", lw=2.4, zorder=3,
            label="plan on self-deceptive map (W erased) — ILLEGAL, FLAGGED")
    ax.plot(hp[:, 0], hp[:, 1], "-", color="#2a7", lw=2.4, zorder=3,
            label="plan on honest map (W present) — legal, PASS")
    if xp:
        ax.plot([xp[0]], [xp[1]], "x", color="#a00", ms=14, mew=3, zorder=5)
        ax.annotate("plan crosses impassable wall\n(non-penetration oracle FLAGS)", xp,
                    textcoords="offset points", xytext=(8, 34), fontsize=8.5, color="#a00",
                    ha="left", arrowprops=dict(arrowstyle="->", color="#a00", lw=1))
    ax.plot(*START, "o", color="#063", ms=11, zorder=6); ax.annotate("  S", START, fontsize=11, color="#063", va="center")
    ax.plot(*GOAL, "o", color="#063", ms=11, zorder=6); ax.annotate("  G", GOAL, fontsize=11, color="#063", va="center")
    ax.set_xlim(ARENA[0], ARENA[1]); ax.set_ylim(ARENA[2], ARENA[3]); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(); fig.savefig(path_png, dpi=150, bbox_inches="tight")
    print("  [OK] figure:", path_png)


if __name__ == "__main__":
    main()
