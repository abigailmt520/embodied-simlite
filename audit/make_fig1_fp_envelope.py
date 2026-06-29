# -*- coding: utf-8 -*-
"""
make_fig1_fp_envelope.py  ——  paper Figure 1 · relational-oracle healthy false-positive (FP) envelope
================================================================================
Triptych (matching the paper caption):
  ① thin-wall geometry d<ξ (the wall is thinner than the noise budget) — the report can land in the free space on the other side, only the displacement segment crosses;
  ② persistence separation — honest noise crossing occupies <0.4 of the window (occasional), fabricated fills the whole window =1.0 (persistent);
  ③ healthy FP rate vs trajectory length — the naive point check rises sharply as the drift δ_max exceeds the local clearance, the gated relational stays flat at 0;
     mark the drift-budget gate where δ_max crosses clearance (~160 steps).

🔴 Data source (a single self-consistent experiment, reproducible from paper2-final): `run_scenB_irreducibility.py` → `scenB_irreducibility.json`'s
   soundness (part e: δ_max/clear/naive FP/gated FP vs length) + specificity (persistence) + params (d,ξ).
   panel ②'s per-seed crossing fraction is deterministically recomputed by the same script's healthy_near_wall (fixed seed 2000+, consistent with the JSON summary).
   Note: the naive/δ numbers the paper §7.4 text currently cites come from a different experiment g5_stats.json part_d (different seeds), differing slightly from this figure (the self-consistent source)
   at L=40/80/160 — see the report; this figure faithfully uses scenB part(e).

Run: python3 audit/make_fig1_fp_envelope.py   → audit/fig1_fp_envelope.png
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_scenB_irreducibility as RSB                      # noqa: E402  data-source script
from relational_oracle import relational_oracle            # noqa: E402

D = json.load(open(os.path.join(HERE, "scenB_irreducibility.json")))
PARAMS = D["params"]                                       # d, xi
SND = {int(k): v for k, v in D["soundness"]["rows"].items()}
SPEC = D["specificity"]
Ls = [20, 40, 80, 160, 320]


def _set_font():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for cand in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if os.path.exists(cand):
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def recompute_persistence_fracs():
    """panel ②: use the same script's healthy_near_wall to deterministically recompute the 30-seed crossing fraction (persist=0, count single frames)."""
    fracs = []
    for si in range(SPEC["n_seed"]):
        th, oh = RSB.healthy_near_wall(2000 + si)
        fracs.append(relational_oracle(th, oh, RSB.WALL_THIN, RSB.RADIUS, persist_frac=0.0)["frac"])
    return np.array(fracs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    _set_font()
    import matplotlib.pyplot as plt

    fracs = recompute_persistence_fracs()
    # self-consistency check: the recomputed persistence should match the JSON summary
    assert abs(fracs.mean() - SPEC["healthy_frac_mean"]) < 1e-6, (fracs.mean(), SPEC["healthy_frac_mean"])
    assert abs(fracs.max() - SPEC["healthy_frac_max"]) < 1e-6

    d, xi = PARAMS["d"], PARAMS["xi"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))

    # ── ① thin-wall geometry d<ξ ───────────────────────────────────────────────
    tv, ov = RSB.scenB_v2()                               # truth (free on side A) / report (free on side B)
    w = RSB.WALL_THIN[0]                                   # (xmin,xmax,ymin,ymax)
    ax1.add_patch(plt.Rectangle((w[0], tv[:, 1].min()), w[1] - w[0], np.ptp(tv[:, 1]),
                                color="#555", alpha=0.75, zorder=1))
    ax1.plot(tv[:, 0], tv[:, 1], "-", color="#2a7", lw=2.4, label="true state $x_t$ (free on side A)", zorder=3)
    ax1.plot(ov[:, 0], ov[:, 1], "-", color="#37a", lw=2.4, label="reported $o_t$ (free on side B, not in wall)", zorder=3)
    for i in range(0, len(tv), 9):
        ax1.plot([tv[i, 0], ov[i, 0]], [tv[i, 1], ov[i, 1]], "-", color="#d33", lw=1.1, alpha=0.8, zorder=2)
    ax1.plot([], [], "-", color="#d33", label="displacement $o_t - x_t$ (crosses wall)")
    # noise-budget ξ tolerance bar at mid-trajectory
    ymid = tv[len(tv) // 2, 1]
    ax1.annotate("", xy=(tv[len(tv)//2, 0] + xi, ymid - 0.25), xytext=(tv[len(tv)//2, 0], ymid - 0.25),
                 arrowprops=dict(arrowstyle="<->", color="#a60", lw=1.4))
    ax1.text(tv[len(tv)//2, 0] + xi / 2, ymid - 0.42, f"noise budget ξ={xi:.2f}", color="#a60",
             ha="center", fontsize=8)
    ax1.text((w[0]+w[1])/2, tv[:, 1].max() + 0.1, f"wall thickness d={d:.2f}", color="#222", ha="center", fontsize=8)
    ax1.set_xlim(29.45, 30.75); ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title(f"① Thin-wall geometry: d={d:.2f} < ξ={xi:.2f}\nendpoints free, only the displacement crosses")
    ax1.legend(fontsize=7.5, loc="upper right")

    # ── ② persistence separation ─────────────────────────────────────────
    ax2.hist(fracs, bins=12, range=(0, 1), color="#2a7", alpha=0.65,
             label=f"honest noise (healthy, N={SPEC['n_seed']})\nmax={fracs.max():.2f}")
    ax2.axvline(SPEC["scenB_v2_frac"], color="#d33", lw=2.6,
                label=f"fabrication (Scenario B) frac={SPEC['scenB_v2_frac']:.2f}")
    ax2.axvline(0.4, ls=":", color="#888", label="separation region (honest < 0.4)")
    ax2.set_xlabel("fraction of window with in-wall frames"); ax2.set_ylabel("seed count")
    ax2.set_title("② Persistence separation\ntransient noise (<0.4) vs persistent fabrication (=1.0)")
    ax2.legend(fontsize=8, loc="upper center")

    # ── ③ healthy FP rate vs trajectory length (naive sharp rise vs gated flat at 0) ──────────
    naive = np.array([SND[L]["naive_fp"] / SND[L]["n"] for L in Ls])
    gated = np.array([SND[L]["rel_fp"] / SND[L]["n"] for L in Ls])
    dmax = np.array([SND[L]["delta_max"] for L in Ls])
    cmed = np.array([SND[L]["clear_med"] for L in Ls])
    ax3.plot(Ls, naive, "o-", color="#d33", lw=2.2, label="naïve point check FP (rises with drift)")
    ax3.plot(Ls, gated, "s-", color="#2a7", lw=2.2, label="gated relational FP (flat at 0)")
    ax3.set_xscale("log"); ax3.set_xticks(Ls); ax3.set_xticklabels([str(L) for L in Ls])
    ax3.xaxis.set_minor_locator(plt.NullLocator())        # fix: disable the log minor ticks, removing the 320 / 3×10² label overlap
    ax3.set_ylim(-0.05, 1.08); ax3.set_xlabel("trajectory length L (steps, log)"); ax3.set_ylabel("healthy FP rate")
    # secondary axis: δ_max and clearance
    axb = ax3.twinx()
    axb.plot(Ls, dmax, "^--", color="#a60", lw=1.6, alpha=0.9, label="measured drift δ_max")
    axb.plot(Ls, cmed, "v:", color="#368", lw=1.6, alpha=0.9, label="local clearance (median)")
    axb.set_ylabel("distance (m)"); axb.set_yscale("symlog", linthresh=0.1)
    # drift-budget gate: where δ_max overtakes clearance (between 80 and 160; marked at ~160)
    gate = next((L for L in Ls if SND[L]["delta_max"] > SND[L]["clear_med"]), None)
    if gate:
        ax3.axvspan(gate, Ls[-1], color="#d33", alpha=0.07)
        ax3.axvline(gate, ls="--", color="#888", lw=1.4)
        ax3.text(gate, 0.5, f" drift-budget gate:\n δ_max > clearance @ L≈{gate}", color="#666",
                 fontsize=7.5, va="center")
    h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left")
    ax3.set_title("③ Healthy FP rate vs trajectory length\nnaïve rises steeply as δ_max overtakes clearance; gated flat at 0")

    fig.suptitle("Figure 1 · Healthy false-positive envelope of the relational oracle — "
                 "Thin wall (d<ξ): persistence separation + drift-budget gating for soundness",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_fp_envelope.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    # also report the actual numbers used in this figure (for audit)
    print("\nactual numbers in this figure (scenB_irreducibility.json part e, self-consistent source):")
    print(f"  L          : {Ls}")
    print(f"  δ_max (m)  : {[round(x,3) for x in dmax]}")
    print(f"  clear_med  : {[round(x,3) for x in cmed]}")
    naive_s = ["%d/%d" % (SND[L]["naive_fp"], SND[L]["n"]) for L in Ls]
    gated_s = ["%d/%d" % (SND[L]["rel_fp"], SND[L]["n"]) for L in Ls]
    print(f"  naïve FP   : {naive_s}")
    print(f"  gated FP   : {gated_s}")
    print(f"  persistence: healthy max={fracs.max():.3f} mean={fracs.mean():.3f}; fabricated={SPEC['scenB_v2_frac']}")
    print(f"  drift-budget gate @ L≈{gate}")


if __name__ == "__main__":
    main()
