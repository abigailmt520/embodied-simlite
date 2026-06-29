# -*- coding: utf-8 -*-
"""
run_leakage_audit.py  ——  Phase3 · contract-layer C_I (mutual-information leakage) audit real run + comparison with C1
================================================================================
Gate C_I-2 (pass clean · green): under honest odom, I(Δodom;Δtruth) is within the noise-budget bound, the C_I audit is green, no false positives.
Gate C_I-1 (audit catches fakes · red): inject the L-1/L-2/L-3 leaks → I over the noise-budget bound → C_I flags RED + localizes.
Comparison with C1: C_I is a **principled quantitative generalization** of C1's "error too small is suspicious" — C1 catches only the full leak L-1,
        while C_I additionally catches what C1 misses: L-2 (partial leak, still drifts) / L-3 (large error but deterministic).

🔴 Two-slip measurement (INV-E honest reliability):
    slip=0.30 (estimable regime): all three leaks cleanly detected, the primary evidence.
    slip=0.05 (platform deployment, odom very accurate): the bound ~3.3 nats approaches the KSG reliable ceiling, L-1/L-2 still caught,
        L-3 missed — honestly reporting the limitation of MI estimation in the high-SNR regime (where C1's magnitude check is more practical).

Run: python audit/run_leakage_audit.py
Artifacts: audit/leakage_compare.png, audit/leakage_summary.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodied_env import EmbodiedNavEnv                         # noqa: E402
from integrity_audit import check_truth_odom_fork              # noqa: E402（C1）
import leakage_audit as la                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_INC = 1500          # increment sample count (required for KSG reliability)


def collect_increments(slip, n_target=N_INC):
    """Run the real env (random policy to lay out trajectories), collect in-episode per-step increments, concatenated to n_target."""
    env = EmbodiedNavEnv(slip=slip, control_mode="A")
    dT_all, dO_all = [], []
    s = 0
    while sum(len(x) for x in dT_all) < n_target:
        env.reset(seed=600 + s); s += 1
        T, O = [env.pos.copy()], [env.odom_pos.copy()]
        done = False
        while not done:
            _, _, te, tr, _ = env.step(env.action_space.sample())
            T.append(env.pos.copy()); O.append(env.odom_pos.copy())
            if te or tr:
                done = True
        T, O = np.array(T), np.array(O)
        dT_all.append(la.increments(T)); dO_all.append(la.increments(O))
    dT = np.vstack(dT_all)[:n_target]; dO = np.vstack(dO_all)[:n_target]
    return dT, dO


def pseudo_traj(d):
    """Increment array → pseudo position trajectory (cumsum), whose increments equal the original array (for ci_audit/C1 to consume)."""
    out = np.zeros((d.shape[0] + 1, d.shape[1]))
    out[1:] = np.cumsum(d, axis=0)
    return out


def run_c1(truth_traj, odom_traj):
    """Feed the pseudo trajectory to the contract layer C1 (check_truth_odom_fork), returning (ok, detail)."""
    sess = [{"recv_t": float(i), "seq": i, "step": i,
             "truth": {"x": float(truth_traj[i][0]), "y": float(truth_traj[i][1]), "theta": 0.0},
             "odom": {"x": float(odom_traj[i][0]), "y": float(odom_traj[i][1]), "theta": 0.0},
             "terminated": False, "truncated": False, "link_status": "online"}
            for i in range(len(truth_traj))]
    r = check_truth_odom_fork(sess)
    return r["ok"], r["detail"]


def audit_one(slip):
    dT, dO = collect_increments(slip)
    Ttraj = pseudo_traj(dT)
    budget, b_std, closed = la.noise_budget_bound(dT, slip)
    print("\n" + "=" * 80)
    print(f"  slip={slip}  (N={len(dT)} increments)  noise-budget bound={budget:.3f}±{b_std:.3f} nats "
          f"(closed-form ref {closed:.2f})  red threshold={budget + la.MARGIN_NATS:.3f}")
    print("=" * 80)

    cases = {"clean": dO}
    for name, inj in la.LEAK_INJECTORS.items():
        cases[name] = la.increments(inj(Ttraj, pseudo_traj(dO)))

    rows = {}
    for name, dOcase in cases.items():
        Otraj = pseudo_traj(dOcase)
        ci = la.ci_audit(Ttraj, Otraj, slip)
        c1_ok, c1_detail = run_c1(Ttraj, Otraj)
        rows[name] = {"mi": ci["locator"]["mi_full_nats"], "ci_red": (not ci["ok"]),
                      "c1_red": (not c1_ok)}
        tag = "clean" if name == "clean" else name
        print(f"\n  ▶ {tag}:  I={ci['locator']['mi_full_nats']:.3f} nats  vs bound {budget:.3f}")
        print(f"      C_I audit: {'🔴RED leak' if not ci['ok'] else '🟢GREEN legit'}  | "
              f"C1(error too small): {'🔴RED' if not c1_ok else '🟢GREEN'}")
    return {"slip": slip, "budget": round(budget, 3), "thresh": round(budget + la.MARGIN_NATS, 3),
            "rows": rows}


def main():
    print("Phase3 · contract-layer mutual-information leakage audit (C_I) real run + comparison with C1")
    res03 = audit_one(0.30)     # primary evidence: estimable regime
    res005 = audit_one(0.05)    # deployment regime: honestly report the limitation

    # —— verdict + C1 comparison table ——
    print("\n" + "=" * 80)
    print("  acceptance gates + comparison with C1 (slip=0.30 primary evidence)")
    print("=" * 80)
    r = res03["rows"]
    gate_ci2 = not r["clean"]["ci_red"]                       # clean: no false positive
    gate_ci1 = all(r[k]["ci_red"] for k in ("L-1_full_leak", "L-2_partial_leak", "L-3_privileged"))
    print(f"  Gate C_I-2 pass clean: {'✅ green (no false positive)' if gate_ci2 else '❌ false positive'}")
    print(f"  Gate C_I-1 catch three leaks: {'✅ 3/3 flagged RED' if gate_ci1 else '❌ see below'}")
    print("\n  comparison table (🔴=flagged RED/caught, 🟢=passed/missed):")
    print(f"  {'case':<18}{'C1(error too small)':<22}{'C_I(mutual info)':<16}")
    for k in ("clean", "L-1_full_leak", "L-2_partial_leak", "L-3_privileged"):
        print(f"  {k:<18}{'🔴' if r[k]['c1_red'] else '🟢':<22}{'🔴' if r[k]['ci_red'] else '🟢':<16}")
    print("  → C1 catches only the full leak L-1; C_I additionally catches what C1 misses: L-2 (partial) / L-3 (large error but deterministic) = a principled generalization of C1")

    # —— slip=0.05 limitation reported honestly ——
    r5 = res005["rows"]
    l3_missed = not r5["L-3_privileged"]["ci_red"]
    print("\n  [reliability · deployment slip=0.05] L-1 caught:" + ("✅" if r5["L-1_full_leak"]["ci_red"] else "❌")
          + " | L-2 caught:" + ("✅" if r5["L-2_partial_leak"]["ci_red"] else "❌")
          + " | L-3 caught:" + ("❌ missed (honest)" if l3_missed else "✅"))
    if l3_missed:
        print("    analysis: slip=0.05 odom is very accurate → the budget bound ~3.3 nats approaches the KSG reliable ceiling; the L-3 rotation-leak MI estimate "
              "falls below the bound and is missed. In this high-SNR regime C1's magnitude check is more practical; C_I's value is the quantitative coverage in the estimable regime (slip≈0.3) + catching what C1 misses.")

    json.dump({"slip_0.30": res03, "slip_0.05": res005,
               "gate_ci2_clean_green": gate_ci2, "gate_ci1_3leaks_red": gate_ci1,
               "slip005_L3_missed": bool(l3_missed)},
              open(os.path.join(HERE, "leakage_summary.json"), "w"), indent=2, ensure_ascii=False)
    try:
        make_figure(res03, res005)
    except Exception as e:
        print(f"  [WARN] figure skipped ({e})")
    return gate_ci1 and gate_ci2


def make_figure(res03, res005):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for cand in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if os.path.exists(cand):
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = ["clean", "L-1_full_leak", "L-2_partial_leak", "L-3_privileged"]
    short = ["clean", "L-1 full-leak", "L-2 partial", "L-3 privileged-rot"]
    for ax, res, title in ((axes[0], res03, "slip=0.30 (estimable regime): all three leaks detected"),
                           (axes[1], res005, "slip=0.05 (deployed, odom very accurate): L-3 missed (honest)")):
        mis = [res["rows"][k]["mi"] for k in order]
        cols = ["#2c7"] + ["#d33" if res["rows"][k]["ci_red"] else "#fa0" for k in order[1:]]
        ax.bar(range(4), mis, color=cols)
        ax.axhline(res["budget"], ls="--", color="#333", label=f"noise-budget bound {res['budget']:.2f}")
        ax.axhline(res["thresh"], ls=":", color="#a00", label=f"red threshold {res['thresh']:.2f}")
        ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=9)
        ax.set_ylabel("I(Δodom;Δtruth) nats"); ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=8)
        for i, m in enumerate(mis):
            ax.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=8.5)
    fig.suptitle("Embodied-SimLite · contract-layer mutual-information leakage audit (C_I) = a principled generalization of C1's \"error too small\"",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "leakage_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] comparison figure saved: {out}")


def replot_from_json():
    """Re-render leakage_compare.png from the committed leakage_summary.json
    (the live audit uses unseeded Monte-Carlo, so re-running would jitter the
    numbers; replotting keeps the figure consistent with the archived data)."""
    d = json.load(open(os.path.join(HERE, "leakage_summary.json")))
    make_figure(d["slip_0.30"], d["slip_0.05"])


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_from_json()
        sys.exit(0)
    ok = main()
    sys.exit(0 if ok else 1)
