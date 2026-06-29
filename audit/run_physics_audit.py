# -*- coding: utf-8 -*-
"""
run_physics_audit.py  ——  Phase1a · physics-layer energy audit "RED/GREEN" one-shot run + evidence output
====================================================================================
Gate P2 (pass clean · green): under clean dynamics the energy audit EC1/EC2/EC3 are all green, zero false positives (residual within the numeric floor).
Gate P1 (audit catches fakes · red): inject the five physics self-deceptions P-1..P-5 into the dynamics integration core; the energy audit must flag each RED and localize.

The whole thing runs the real embodied_env dynamics core (_integrate_dynamics real integration + a real energy ledger);
the injectors are hidden inside the integrator (really corrupting the trajectory), the audit red/green are real verdicts, the residuals are real numbers (no hardcode).

Drive method (open-loop "sinusoidal varying command", reproducible, policy-independent):
    the command varies sinusoidally with the step (v_tgt∈[0.15,0.95], w_tgt∈[-0.9,0.9]), spanning STEPS steps without reset.
    Why varying rather than fixed command: mass is observable only during "acceleration" (steady-state F=c·v is independent of mass) —
    under a fixed command the system quickly reaches steady state, and the P-5 mass misreport has zero residual at steady state and escapes (a measured lesson). A continuously varying command
    keeps the system always in transient → continuous acceleration → mass/force faults manifest every step, while the sinusoidal peaks keep v_act over the bound (P-4).

Run: python audit/run_physics_audit.py
Artifacts: audit/physics_sessions/*.json, audit/energy_redgreen_matrix.png
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodied_env import EmbodiedNavEnv                         # noqa: E402
from energy_audit import audit_session, format_report          # noqa: E402
import physics_injection as pi                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = os.path.join(HERE, "physics_sessions")
SEED = 7
STEPS = 200
# control mode selected by environment variable: 'A'=target velocity (Phase1a default) / 'B'=force control (Phase1b). The audit acts on the dynamics core, independent of mode.
CONTROL_MODE = os.environ.get("EMBODIED_AUDIT_MODE", "A")


def _bounds():
    """EC3 actuator speed limit (A/B modes have different physical top speeds)."""
    if CONTROL_MODE == "B":
        return EmbodiedNavEnv.V_PHYS_MAX_B, EmbodiedNavEnv.W_PHYS_MAX_B
    return EmbodiedNavEnv.V_PHYS_MAX, EmbodiedNavEnv.W_PHYS_MAX


def command(t):
    """Open-loop sinusoidal varying command: continuous transient to expose mass/force faults. Returns [v,w] or [f_l,f_r] by mode."""
    if CONTROL_MODE == "B":
        common = 0.65 + 0.30 * np.sin(2 * np.pi * t / 24.0)   # common-mode force (forward / accel-decel → transient)
        diff = 0.30 * np.sin(2 * np.pi * t / 17.0)            # differential-mode force (turning)
        return np.array([common - diff, common + diff], dtype=np.float32)  # [f_l, f_r]∈[-1,1]
    v = 0.55 + 0.40 * np.sin(2 * np.pi * t / 24.0)   # ∈[0.15,0.95], peaks keep exceeding the bound for P-4
    w = 0.90 * np.sin(2 * np.pi * t / 17.0)          # ∈[-0.9,0.9]
    return np.array([v, w], dtype=np.float32)


def record_session(configure_fault):
    """Run STEPS steps of the fixed command, recording the energy ledger per frame (from the energy field of the get_render_state contract).

    configure_fault(env): configure env.physics_fault after reset (no-op for the clean mode).
    Span without reset: a reset zeros the energy ledger, which cannot show continuous residual accumulation.
    """
    env = EmbodiedNavEnv(slip=0.0, control_mode=CONTROL_MODE)  # slip off: look only at dynamics, exclude odom noise
    env.reset(seed=SEED)
    configure_fault(env)
    rows = []
    for t in range(STEPS):
        env.step(command(t))
        st = env.get_render_state()       # take the energy ledger via the contract (same source as the platform's source of truth)
        e = st["energy"]
        rows.append({
            "step": st["step"], "seq": st["seq"],
            "E_kin": e["E_kin"], "dE": e["dE"],
            "W_act": e["W_act"], "D_damp": e["D_damp"],
            "v_act": st["v_act"], "w_act": st["w_act"],
        })
    return rows


def main():
    os.makedirs(SESS_DIR, exist_ok=True)
    print("=" * 76)
    print("  Phase1a · physics-layer energy audit — real RED/GREEN run (clean green + five injections red)")
    print("=" * 76)

    # ---- Gate P2: clean baseline → expect all green ----
    clean = record_session(pi.clean)
    json.dump(clean, open(os.path.join(SESS_DIR, "clean.json"), "w"))
    print("\n" + "─" * 76)
    print("[Gate P2 | audit passes clean dynamics (expected: all green, zero false positives)]")
    print("─" * 76)
    vmax, wmax = _bounds()
    res_clean = audit_session(clean, v_max=vmax, w_max=wmax)
    print(f"  [control mode: {CONTROL_MODE}]  EC3 speed bound v_max={vmax:.3f}, w_max={wmax:.3f}")
    print(format_report(res_clean))
    # report the clean residual magnitude (should be ~machine precision)
    max_r = max(abs(f["dE"] - (f["W_act"] - f["D_damp"])) for f in clean)
    print(f"    clean residual max|ΔE−(W_act−D_damp)| = {max_r:.3e} J")
    gate_p2 = res_clean["passed"]

    # ---- Gate P1: inject P-1..P-5 → expect each flagged red and located ----
    print("\n" + "─" * 76)
    print("[Gate P1 | audit catches fakes: inject P-1..P-5, expect each flagged RED and located]")
    print("─" * 76)
    gate1 = {}
    matrix = {"clean": res_clean}
    for name, inj in pi.INJECTORS.items():
        sess = record_session(inj)
        json.dump(sess, open(os.path.join(SESS_DIR, f"injected_{name}.json"), "w"))
        res = audit_session(sess, v_max=vmax, w_max=wmax)
        matrix[name] = res
        tgt = pi.EXPECTED_CHECK[name]
        tgt_red = any((not c["ok"]) and c["check"] == tgt for c in res["checks"])
        caught = (not res["passed"]) and tgt_red
        gate1[name] = caught
        print(f"\n  ▶ inject [{name}]: {pi.DESCRIPTIONS[name]}")
        print(f"    expected RED check: {tgt}")
        print(format_report(res))
        print(f"    → catch {'succeeded ✅' if caught else 'FAILED ❌ (needs analysis, do not hide INV-D)'}")
    gate1_ok = all(gate1.values())

    # ---- summary ----
    print("\n" + "=" * 76)
    print("  acceptance-gate summary")
    print("=" * 76)
    print(f"  Gate P1 (audit catches fakes · RED): {'✅ PASS (5/5 injections each flagged RED and located)' if gate1_ok else '❌ FAIL (see above)'}")
    print(f"  Gate P2 (passes clean · GREEN): {'✅ PASS (clean all green, zero false positives)' if gate_p2 else '❌ FAIL (false positive)'}")

    # ---- RED/GREEN matrix figure ----
    try:
        make_matrix_figure(matrix)
    except Exception as e:
        print(f"  [WARN] matrix figure skipped ({e})")

    return gate1_ok and gate_p2


def make_matrix_figure(matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    for cand in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if os.path.exists(cand):
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    GF, GE, GT = "#d7f3e3", "#2faa6a", "#11623b"
    RF, RE, RT = "#fbd9d9", "#d84141", "#8a1c1c"
    HF, HT = "#2b2f3a", "#ffffff"

    rows = [("EC1_ENERGY_BUDGET", "EC1 · energy budget\nΔE=W−D self-consistent"),
            ("EC2_NO_FREE_ENERGY", "EC2 · no free energy\nΔE≤W_act"),
            ("EC3_ACTUATOR_BOUND", "EC3 · actuator bound\n|v|≤v_max")]
    cols = [("clean", "clean dynamics\n(Gate P2)"),
            ("P-1_neg_damp", "P-1\nneg damping"),
            ("P-2_force_double", "P-2\nforce double-count"),
            ("P-3_drop_dissipation", "P-3\ndrop dissipation"),
            ("P-4_skip_lag", "P-4\nover bound"),
            ("P-5_mass_misreport", "P-5\nmass misreport")]

    by = {}
    for ck in matrix:
        by[ck] = {c["check"]: c for c in matrix[ck]["checks"]}

    nrow, ncol = len(rows), len(cols)
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    ax.set_xlim(0, ncol + 1.3); ax.set_ylim(0, nrow + 1.4); ax.axis("off")
    cw, rh = 1.0, 0.95
    x0, y0 = 1.3, 0.25

    def box(x, y, w, h, fc, ec):
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), w - 0.08, h - 0.08,
                     boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=1.6))

    for j, (_, label) in enumerate(cols):
        x = x0 + j * cw
        box(x, y0 + nrow * rh, cw, rh * 0.95, HF, HF)
        ax.text(x + cw / 2, y0 + nrow * rh + rh * 0.46, label, ha="center", va="center",
                color=HT, fontsize=10.5, fontweight="bold")
    for i, (_, label) in enumerate(rows):
        y = y0 + (nrow - 1 - i) * rh
        box(0.08, y, 1.2, rh, HF, HF)
        ax.text(0.08 + 1.2 / 2, y + rh / 2, label, ha="center", va="center",
                color=HT, fontsize=10, fontweight="bold")

    for i, (rk, _) in enumerate(rows):
        y = y0 + (nrow - 1 - i) * rh
        for j, (ck, _) in enumerate(cols):
            x = x0 + j * cw
            chk = by[ck][rk]
            if chk["ok"]:
                box(x, y, cw, rh, GF, GE)
                ax.text(x + cw / 2, y + rh * 0.62, "✓ PASS", ha="center", va="center",
                        color=GT, fontsize=12, fontweight="bold")
            else:
                box(x, y, cw, rh, RF, RE)
                loc = chk.get("locator") or {}
                val = (loc.get("max_residual_J") or loc.get("max_excess_J")
                       or loc.get("max_overshoot") or "")
                ax.text(x + cw / 2, y + rh * 0.66, "✗ FAIL", ha="center", va="center",
                        color=RT, fontsize=12, fontweight="bold")
                ax.text(x + cw / 2, y + rh * 0.30, f"@step{loc.get('first_step','?')}\n{val}",
                        ha="center", va="center", color=RT, fontsize=7.6)

    fig.suptitle("Embodied-SimLite · Physics-layer Energy Audit RED/GREEN",
                 fontsize=14.5, fontweight="bold", y=0.99)
    ax.text((ncol + 1.3) / 2, 0.03,
            "Gate P2: clean dynamics all-green with zero false positives (residual ≈ machine precision)  |  "
            "Gate P1: the five physics self-deceptions P-1..P-5 are each flagged RED and located by the energy audit "
            "— the physics audit is proven able to \"catch the fake\"",
            ha="center", va="center", fontsize=9, color="#333")
    suffix = f"_{CONTROL_MODE.lower()}mode"
    out = os.path.join(HERE, f"energy_redgreen_matrix{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] RED/GREEN matrix saved: {out}")


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 2)
