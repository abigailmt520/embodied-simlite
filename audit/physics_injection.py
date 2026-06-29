# -*- coding: utf-8 -*-
"""
physics_injection.py  ——  Phase1a · physics self-deception injectors (hidden inside the integrator, purely for the energy audit to prove it catches fakes)
============================================================================================
The "physics-layer" counterpart of the contract layer's fault_injection.py. What is injected here is not an after-the-fact tamper of an already-recorded session,
but a **real corruption** of embodied_env's dynamics integration core (_integrate_dynamics) — it must take effect during simulation,
because a physics fault changes the trajectory itself (cannot be injected after the fact).

Each injector returns a `physics_fault` config dict; assigning it to env.physics_fault activates it; env=None means clean.
Key: an injector only changes the parameters used for "actual integration", while the energy ledger (W_act/D_damp/E_kin) still settles by the env's **declared constants**
(MASS/C_LIN/...) → the real kinetic-energy change ΔE and the declared energy budget (W_act−D_damp) develop a residual, caught by the energy audit.

Five physics self-deceptions (each breaking a different conservation law):
    P-1 negative damping/gain   —— actual damping made negative → energy from nothing (violates the 2nd law of thermodynamics).
    P-2 force-to-acceleration double count —— actual force ×2 while the ledger counts single → KE increment exceeds actuator work (violates Newtonian consistency).
    P-3 drop dissipation but claim conservation —— actual integration drops the damping term while the ledger still reports D_damp (violates energy-ledger self-consistency, the physics-layer Potemkin village).
    P-4 velocity over the actuator bound —— skip the inertial lag and overshoot → velocity over the physical limit + KE jump (violates the actuator power bound).
    P-5 misreport mass/inertia  —— actual integration uses half the mass while the ledger uses the declared mass → KE inconsistent with the budget (violates energy self-consistency).
"""


def clean(env=None):
    """Clean mode: clear any physics fault (env.physics_fault=None)."""
    if env is not None:
        env.physics_fault = None
    return None


def inject_p1_neg_damp(env):
    """P-1: make the actual damping coefficients negative (−C_LIN/−C_ANG) → damping becomes "gain", energy created without work each step."""
    fault = {"mode": "P-1_neg_damp",
             "c_lin_eff": -env.C_LIN, "c_ang_eff": -env.C_ANG}
    env.physics_fault = fault
    return fault


def inject_p2_force_double(env):
    """P-2: actual integration uses 2× force while the ledger settles by single force → KE increment exceeds actuator work."""
    fault = {"mode": "P-2_force_double", "force_mult": 2.0}
    env.physics_fault = fault
    return fault


def inject_p3_drop_dissipation(env):
    """P-3: actual integration drops the damping dissipation term (c_eff=0) while the ledger still reports D_damp by the declared C_LIN (falsely claims conservation)."""
    fault = {"mode": "P-3_drop_dissipation",
             "c_lin_eff": 0.0, "c_ang_eff": 0.0}
    env.physics_fault = fault
    return fault


def inject_p4_skip_lag(env):
    """P-4: skip the inertial lag and 1.5× overshoot → actual velocity over the actuator bound + KE jump from nothing."""
    fault = {"mode": "P-4_skip_lag", "skip_lag": True, "overshoot": 1.5}
    env.physics_fault = fault
    return fault


def inject_p5_mass_misreport(env):
    """P-5: actual integration uses half the mass (easier to accelerate) while the ledger/kinetic-energy use the declared mass → KE inconsistent with the budget."""
    fault = {"mode": "P-5_mass_misreport", "m_eff": 0.5 * env.MASS}
    env.physics_fault = fault
    return fault


INJECTORS = {
    "P-1_neg_damp": inject_p1_neg_damp,
    "P-2_force_double": inject_p2_force_double,
    "P-3_drop_dissipation": inject_p3_drop_dissipation,
    "P-4_skip_lag": inject_p4_skip_lag,
    "P-5_mass_misreport": inject_p5_mass_misreport,
}

DESCRIPTIONS = {
    "P-1_neg_damp": "actual damping made negative → energy created from nothing (violates the 2nd law of thermodynamics)",
    "P-2_force_double": "actual force ×2 while ledger counts single → KE increment exceeds work done (violates Newtonian consistency)",
    "P-3_drop_dissipation": "integration drops the damping term but ledger still reports dissipation → falsely claims conservation (energy-ledger self-consistency broken)",
    "P-4_skip_lag": "skip inertia + overshoot → velocity over the actuator bound + KE jump (actuator power bound broken)",
    "P-5_mass_misreport": "actual half mass while ledger claims full mass → KE inconsistent with the budget (energy self-consistency broken)",
}

# ====================================================================
# collision-fidelity self-deception injectors (Phase2, hidden inside _resolve_wall_collisions; only manifest with maze + a wall hit)
# ====================================================================
def inject_cf1_over_bounce(env):
    """CF-1: restitution e>1 (v is amplified after a wall hit) → collision creates energy from nothing (violates collision energy non-negativity E_contact≥0)."""
    fault = {"mode": "CF-1_over_bounce", "bounce_eff": 1.3}
    env.physics_fault = fault
    return fault


def inject_cf2_skip_pushout(env):
    """CF-2: skip the penetration pushout (robot stays inside the wall) but settle as usual → violates the non-penetration invariant (penetration>0)."""
    fault = {"mode": "CF-2_skip_pushout", "skip_pushout": True}
    env.physics_fault = fault
    return fault


def inject_cf3_phantom_contact(env):
    """CF-3: the ledger claims collision dissipation (E_contact_decl>0) while velocity is not decayed → violates energy-ledger self-consistency (the collision-flavored Potemkin village)."""
    fault = {"mode": "CF-3_phantom_contact", "phantom_contact": True}
    env.physics_fault = fault
    return fault


COLLISION_INJECTORS = {
    "CF-1_over_bounce": inject_cf1_over_bounce,
    "CF-2_skip_pushout": inject_cf2_skip_pushout,
    "CF-3_phantom_contact": inject_cf3_phantom_contact,
}

COLLISION_DESCRIPTIONS = {
    "CF-1_over_bounce": "restitution e>1 → collision creates energy (violates collision energy non-negativity)",
    "CF-2_skip_pushout": "skip penetration pushout but falsely claim it was resolved (violates the non-penetration invariant)",
    "CF-3_phantom_contact": "ledger claims collision dissipation while velocity is not decayed (collision Potemkin village, breaks energy-ledger self-consistency)",
}

COLLISION_EXPECTED_CHECK = {
    "CF-1_over_bounce": "EC4_COLLISION_NONNEG",   # signature: collision creates energy (E_contact_act<0)
    "CF-2_skip_pushout": "EC5_NON_PENETRATION",   # signature: residual penetration after resolution >0
    "CF-3_phantom_contact": "EC1_ENERGY_BUDGET",  # signature: claimed dissipation with no real loss → budget residual
}


# the energy-audit check each injector is "expected to be flagged RED" by (signature check; calibrated to the measured conservation-law violation signature)
# the EC1 energy-budget residual is the "universal net" — all 5 are caught by it (no escape); EC2/EC3 provide specific localization.
EXPECTED_CHECK = {
    "P-1_neg_damp": "EC2_NO_FREE_ENERGY",     # signature: energy from nothing (ΔE>W_act), 2nd law
    "P-2_force_double": "EC1_ENERGY_BUDGET",   # signature: force/work inconsistency → budget residual (Newtonian consistency)
    "P-3_drop_dissipation": "EC1_ENERGY_BUDGET",  # signature: phantom dissipation → budget residual (ledger self-consistency)
    "P-4_skip_lag": "EC3_ACTUATOR_BOUND",     # signature: velocity over the actuator bound
    "P-5_mass_misreport": "EC1_ENERGY_BUDGET",  # signature: KE inconsistent with the budget (observable only with a transient)
}
