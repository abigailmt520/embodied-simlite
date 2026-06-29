# -*- coding: utf-8 -*-
"""
run_collision_audit.py  ——  Phase2 · collision-fidelity audit "RED/GREEN" real run
=====================================================================
Gate C2 (pass clean · green): under clean rectangular collision (penetration pushout + bounce e=0.5) EC1/EC4/EC5 are all green, zero false positives.
Gate C1 (audit catches fakes · red): inject the three collision self-deceptions CF-1/CF-2/CF-3; the energy/non-penetration audit must flag each RED and localize.

The collision faults are hidden inside embodied_env._resolve_wall_collisions (really corrupting the collision resolution); the audit consumes the collision ledger
(E_contact_decl/act, penetration) via the get_render_state contract.

Drive: place the robot in front of an outer wall in the maze and charge at full speed (A-mode [v=1,w=0]); after a wall hit it is pushed out + bounces and keeps re-hitting →
      consecutive collision frames for the audit to judge. Open-loop, reproducible, policy-independent.

Run: python audit/run_collision_audit.py
Artifacts: audit/collision_sessions/*.json, audit/collision_redgreen_matrix.png
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
SESS_DIR = os.path.join(HERE, "collision_sessions")
SEED = 7
STEPS = 40
START = np.array([37.0, 20.0])   # in front of the outer wall [39,40]
ACTION = np.array([1.0, 0.0], dtype=np.float32)   # charge at full speed into the +x outer wall


def record_session(configure_fault):
    env = EmbodiedNavEnv(slip=0.0, control_mode="A", map_type="maze")
    env.reset(seed=SEED)
    env.pos = START.copy(); env.theta = 0.0; env.v_act = 0.0; env.w_act = 0.0
    configure_fault(env)
    rows = []
    for _ in range(STEPS):
        env.step(ACTION)
        st = env.get_render_state(); e = st["energy"]
        rows.append({
            "step": st["step"], "seq": st["seq"], "E_kin": e["E_kin"], "dE": e["dE"],
            "W_act": e["W_act"], "D_damp": e["D_damp"],
            "E_contact_decl": e["E_contact_decl"], "E_contact_act": e["E_contact_act"],
            "penetration": e["penetration"], "v_act": st["v_act"], "w_act": st["w_act"],
        })
    return rows


def main():
    os.makedirs(SESS_DIR, exist_ok=True)
    vmax, wmax = EmbodiedNavEnv.V_PHYS_MAX, EmbodiedNavEnv.W_PHYS_MAX
    print("=" * 78)
    print("  Phase2 · collision-fidelity audit — RED/GREEN (clean green + CF-1/2/3 injections red)")
    print("=" * 78)

    clean = record_session(pi.clean)
    json.dump(clean, open(os.path.join(SESS_DIR, "clean.json"), "w"))
    print("\n[Gate C2 | pass clean rectangular collision (expected: all green)]")
    res_clean = audit_session(clean, v_max=vmax, w_max=wmax, with_collision=True)
    print(format_report(res_clean))
    ncontact = sum(1 for f in clean if f["E_contact_decl"] > 0)
    maxr = max(abs(f["dE"] - (f["W_act"] - f["D_damp"] - f["E_contact_decl"])) for f in clean)
    print(f"    clean: contact frames={ncontact}, energy residual max={maxr:.3e} J, residual penetration max="
          f"{max(f['penetration'] for f in clean):.2e} m")
    gate_c2 = res_clean["passed"]

    print("\n[Gate C1 | inject CF-1/CF-2/CF-3, expect each flagged RED and located]")
    gate1 = {}
    matrix = {"clean": res_clean}
    for name, inj in pi.COLLISION_INJECTORS.items():
        sess = record_session(inj)
        json.dump(sess, open(os.path.join(SESS_DIR, f"injected_{name}.json"), "w"))
        res = audit_session(sess, v_max=vmax, w_max=wmax, with_collision=True)
        matrix[name] = res
        tgt = pi.COLLISION_EXPECTED_CHECK[name]
        tgt_red = any((not c["ok"]) and c["check"] == tgt for c in res["checks"])
        caught = (not res["passed"]) and tgt_red
        gate1[name] = caught
        print(f"\n  ▶ inject [{name}]: {pi.COLLISION_DESCRIPTIONS[name]}")
        print(f"    expected RED check: {tgt}")
        print(format_report(res))
        print(f"    → catch {'succeeded ✅' if caught else 'FAILED ❌ (needs analysis, do not hide INV-E)'}")
    gate1_ok = all(gate1.values())

    print("\n" + "=" * 78)
    print(f"  Gate C1 (collision catches fakes · RED): {'✅ PASS (3/3 each flagged RED and located)' if gate1_ok else '❌ FAIL (see above)'}")
    print(f"  Gate C2 (passes clean · GREEN): {'✅ PASS (clean all green, zero false positives)' if gate_c2 else '❌ FAIL (false positive)'}")
    try:
        make_matrix(matrix)
    except Exception as e:
        print(f"  [WARN] matrix figure skipped ({e})")
    return gate1_ok and gate_c2


def make_matrix(matrix):
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
    rows = [("EC1_ENERGY_BUDGET", "EC1 · energy budget\nΔE=W−D−E_c"),
            ("EC4_COLLISION_NONNEG", "EC4 · collision non-neg\nE_contact≥0"),
            ("EC5_NON_PENETRATION", "EC5 · non-penetration\npenetration≈0")]
    cols = [("clean", "clean collision\n(Gate C2)"), ("CF-1_over_bounce", "CF-1\nover-bounce"),
            ("CF-2_skip_pushout", "CF-2\nskip pushout"), ("CF-3_phantom_contact", "CF-3\nphantom dissipation")]
    by = {ck: {c["check"]: c for c in matrix[ck]["checks"]} for ck in matrix}
    nrow, ncol = len(rows), len(cols)
    fig, ax = plt.subplots(figsize=(12, 5.4))
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
        ax.text(0.68, y + rh / 2, label, ha="center", va="center", color=HT,
                fontsize=9.5, fontweight="bold")
    for i, (rk, _) in enumerate(rows):
        y = y0 + (nrow - 1 - i) * rh
        for j, (ck, _) in enumerate(cols):
            x = x0 + j * cw
            chk = by[ck][rk]
            if chk["ok"]:
                box(x, y, cw, rh, GF, GE)
                ax.text(x + cw / 2, y + rh * 0.6, "✓ PASS", ha="center", va="center",
                        color=GT, fontsize=11.5, fontweight="bold")
            else:
                box(x, y, cw, rh, RF, RE)
                loc = chk.get("locator") or {}
                val = (loc.get("max_residual_J") or loc.get("min_E_contact_J")
                       or loc.get("max_penetration_m") or "")
                ax.text(x + cw / 2, y + rh * 0.64, "✗ FAIL", ha="center", va="center",
                        color=RT, fontsize=11.5, fontweight="bold")
                ax.text(x + cw / 2, y + rh * 0.3, f"@s{loc.get('first_step','?')}\n{val}",
                        ha="center", va="center", color=RT, fontsize=7.6)
    fig.suptitle("Embodied-SimLite · Collision-fidelity Audit RED/GREEN",
                 fontsize=13.5, fontweight="bold", y=0.99)
    ax.text((ncol + 1.3) / 2, 0.03,
            "Gate C2: clean rectangular collision all-green with zero false positives  |  "
            "Gate C1: CF-1/2/3 are each flagged RED and located by the energy / collision-non-neg / non-penetration audit",
            ha="center", va="center", fontsize=8.6, color="#333")
    out = os.path.join(HERE, "collision_redgreen_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] collision RED/GREEN matrix saved: {out}")


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 2)
