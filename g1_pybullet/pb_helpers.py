# -*- coding: utf-8 -*-
"""
pb_helpers.py  ——  G1 · PyBullet real-engine generalization: scene / twin-report / invariants / audit-reuse layer
================================================================================
Port the 2D platform's audit **logic** to PyBullet (an independent third-party real engine), testing two things:
  1) can the physics layer (energy/momentum/non-penetration/EC5') catch **PyBullet's own native numerical pathologies** (non-circular · crown jewel).
  2) does the contract layer (C1-C3+C_I) / joint still work in a 3D real-engine state space (ported, using our report injectors).

🔴 Architecture red line: keep "the engine's internal high-fidelity state = truth; the derived odom/sensor report = report" (the backend is the single source of truth, the frontend is pure observation).
   Never port in frontend-physics-authority / local-dead-reckoning.

🔴 Non-circular point: the physics pathologies are produced by the **engine's real numerics** (set inducing conditions, no hand-set fault state); the audit logic is ours,
   but the fault under test comes from PyBullet.
"""

import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

# reuse the 2D platform's audit logic (pure functions, state stream/geometry) — the audit logic is ported, the 2D platform is not changed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit"))
from integrity_audit import check_truth_odom_fork, check_seq_integrity, check_feed_liveness  # noqa: E402
from joint_audit import traj_vs_map, _circle_aabb_pen                                          # noqa: E402

G = 9.8


# ====================================================================
# scenes
# ====================================================================
def connect(dt=1.0 / 240.0, gravity=True):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -G if gravity else 0)
    p.setTimeStep(dt)
    return dt


def add_wall(center, half_extents):
    """Add a thin wall (GEOM_BOX, static). Returns (uid, aabb_xy=(xmin,xmax,ymin,ymax))."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    uid = p.createMultiBody(0, col, -1, center, [0, 0, 0, 1])   # mass=0 → static
    aabb = (cx - hx, cx + hx, cy - hy, cy + hy)
    return uid, aabb


def add_ball(mass, pos, radius=0.1, damping=0.0):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    uid = p.createMultiBody(mass, col, -1, pos, [0, 0, 0, 1])
    p.changeDynamics(uid, -1, linearDamping=damping, angularDamping=damping)
    return uid


def add_box_body(mass, pos, half_extents=(0.1, 0.1, 0.1), damping=0.0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    uid = p.createMultiBody(mass, col, -1, pos, [0, 0, 0, 1])
    p.changeDynamics(uid, -1, linearDamping=damping, angularDamping=damping)
    return uid


# ====================================================================
# physical-truth readout (truth)
# ====================================================================
def base_state(uid):
    """Returns (pos(3,), quat(4,), v(3,), w(3,)) — PyBullet high-fidelity truth."""
    pos, quat = p.getBasePositionAndOrientation(uid)
    v, w = p.getBaseVelocity(uid)
    return np.array(pos), np.array(quat), np.array(v), np.array(w)


def kinetic_energy(uid, mass, inertia_diag=None):
    """½m‖v‖² + ½ωᵀIω (single rigid body; when inertia_diag is omitted, the approximate spherical inertia makes the rotational term negligible)."""
    _, _, v, w = base_state(uid)
    ke = 0.5 * mass * float(np.dot(v, v))
    if inertia_diag is not None:
        ke += 0.5 * float(np.dot(inertia_diag, w * w))
    return ke


def potential_energy(uid, mass):
    pos, _, _, _ = base_state(uid)
    return mass * G * float(pos[2])


def mechanical_energy(uid, mass, inertia_diag=None):
    return kinetic_energy(uid, mass, inertia_diag) + potential_energy(uid, mass)


def linear_momentum(uid, mass):
    _, _, v, _ = base_state(uid)
    return mass * v


# ====================================================================
# twin report layer (report = odom derived from truth, mirroring the 2D _integrate_odom)
# ====================================================================
class OdomReporter:
    """Backend single source of truth (truth) → derived odom report (report): eats the true-state velocity + slip-noise integration (dead reckoning).
    Isomorphic to the 2D platform's _integrate_odom: v_odom=v*(1+slip)+N(0,slip)*|v|. Never use local prediction to replace truth."""

    def __init__(self, init_xy, slip=0.05, seed=0):
        self.odom = np.array(init_xy, dtype=np.float64)
        self.slip = slip
        self.rng = np.random.default_rng(seed)

    def update(self, truth_v_xy, dt):
        v = np.array(truth_v_xy, dtype=np.float64)
        if self.slip > 0:
            v = v * (1 + self.slip) + self.rng.normal(0, self.slip, 2) * np.abs(v)
        self.odom = self.odom + v * dt
        return self.odom.copy()


# ====================================================================
# physics audit (porting the EC1 energy-budget / EC5' geometric-non-penetration idea to a 3D real engine)
# ====================================================================
def energy_audit_series(E_series, W_series=None, floor_abs=1e-2, floor_rel=1e-2, k_persist=3):
    """EC1(3D): mechanical-energy budget. With no external work ΔE should ≈0 (conservation); with actuator work W, ΔE≈W.
    The floor must be > the engine integrator's noise floor (measured ~8e-4 J/step). Over the bound for k consecutive steps → red.
    Returns (ok, detail, max_resid)."""
    E = np.asarray(E_series, dtype=np.float64)
    dE = np.diff(E)
    W = np.zeros_like(dE) if W_series is None else np.asarray(W_series, dtype=np.float64)[:len(dE)]
    resid = dE - W
    run = 0; first = None; worst = 0.0
    for i, r in enumerate(resid):
        fl = floor_abs + floor_rel * abs(E[i])
        if abs(r) > fl:
            if run == 0:
                first = i
            run += 1
            if abs(r) > abs(worst):
                worst = r
            if run >= k_persist:
                return False, f"energy-budget residual over the bound for {run} consecutive steps (from step{first}); max residual {worst:.3e}J", float(worst)
        else:
            run = 0
    return True, f"energy budget self-consistent (max single-step residual {np.max(np.abs(resid)):.3e}J within the noise floor)", float(np.max(np.abs(resid)))


def energy_conservation_check(E_series, work_in=0.0, floor_abs=1e-2, floor_rel=0.02):
    """EC1 (conservation form): for a conservative system (no external work, work_in≈0) the mechanical energy should be bounded by E0+work.
    max(E) significantly exceeding E0+work_in (over the noise floor) = energy injected by the engine (pathology). Catches elastic-bounce energy gain, etc."""
    E = np.asarray(E_series, dtype=np.float64)
    e0 = E[0]
    bound = e0 + work_in + floor_abs + floor_rel * abs(e0)
    excess = float(E.max() - bound)
    if excess > 0:
        i = int(np.argmax(E))
        return False, (f"mechanical energy over the conservation upper bound: max(E)={E.max():.3f}J > E0+work+floor {bound:.3f}J "
                       f"(over by {excess:.3f}J @step{i}) — energy injected by the engine"), excess
    return True, f"mechanical energy conserved (max(E)={E.max():.3f}J ≤ bound {bound:.3f}J, within the noise floor)", excess


def ec5prime_truth_vs_map(truth_xy_traj, walls_aabb, radius, pen_floor=1e-3):
    """EC5'(3D planar projection): truth (x,y) trajectory vs claimed wall geometry (does not trust the engine's contact report, geometric recompute).
    Catches the truth really crossing a wall that the engine's collision detection missed (e.g. high-speed tunneling). Reuses the 2D joint_audit geometry."""
    r = traj_vs_map(np.asarray(truth_xy_traj), walls_aabb, radius, "truth")
    r["check"] = "EC5P_TRUTH_MAP_3D"
    return r


def _segment_crosses_aabb(p0, p1, aabb, radius):
    """Whether segment p0→p1 (inflated by radius) intersects the wall AABB (slab method). For the swept detection of high-speed tunneling."""
    xmin, xmax, ymin, ymax = aabb
    xmin -= radius; xmax += radius; ymin -= radius; ymax += radius
    d = np.asarray(p1, float) - np.asarray(p0, float)
    tmin, tmax = 0.0, 1.0
    for lo, hi, o, dd in ((xmin, xmax, p0[0], d[0]), (ymin, ymax, p0[1], d[1])):
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


def ec5prime_swept(truth_xy_traj, walls_aabb, radius):
    """EC5' (swept form): check whether the **segment** between adjacent truth frames crosses any wall AABB.
    Catches "high-speed tunneling" — when the object's per-step displacement > wall thickness and it jumps over the wall between frames, the per-frame point check misses it, the swept segment check catches it."""
    T = np.asarray(truth_xy_traj)
    crossings = []
    for i in range(len(T) - 1):
        for w in walls_aabb:
            if _segment_crosses_aabb(T[i], T[i + 1], w, radius):
                crossings.append(i)
                break
    if crossings:
        return {"check": "EC5P_SWEPT_3D", "ok": False, "status": "RED",
                "detail": f"truth-trajectory segment crosses the claimed wall: {len(crossings)} illegal segments, first crossing @frame{crossings[0]}"
                          f" (high-speed tunneling: jumps over the wall in one step)",
                "locator": {"first_crossing_frame": crossings[0], "n_crossings": len(crossings),
                            "pos0": [round(float(T[crossings[0]][0]), 3), round(float(T[crossings[0]][1]), 3)]}}
    return {"check": "EC5P_SWEPT_3D", "ok": True, "status": "GREEN",
            "detail": "truth trajectory has no segment crossing a wall (swept non-penetration legal)", "locator": None}


def engine_reported_penetration(uid, wall_uids):
    """Engine-reported penetration: the most-negative distance from getClosestPoints (the engine may under-report → 0 during tunneling). For comparison with EC5'."""
    worst = 0.0
    for w in wall_uids:
        for cp in p.getClosestPoints(uid, w, distance=0.05):
            d = cp[8]   # contactDistance (negative when penetrating)
            if d < 0:
                worst = max(worst, -d)
    return worst
