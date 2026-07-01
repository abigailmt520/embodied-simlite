#!/usr/bin/env python3
"""Figure 3 (§7.3 / RQ3 cross-engine): PyBullet's contact solver injects mechanical
energy into an e=1 (conservative) elastic bounce, ratcheting it far above the EC1
conservation budget — which is exactly what the ported oracle catches RED.

Reads the real per-step series from energy_injection_series.json (produced by
run_energy_injection.py); plots nothing synthetic. Two panels (Δt = 1/30 s and
1/120 s) because the peaks differ by ~6×. No figure title is baked in (the LaTeX
\\caption supplies it); axis labels + legends only.

    conda run -n g1-pybullet python g1_pybullet/plot_energy_injection.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "energy_injection_series.json")
OUT = os.path.join(HERE, "cross_engine_energy_injection.png")

d = json.load(open(DATA))
runs = d["runs"]
keys = ["dt_1_30", "dt_1_120"]            # main, then control

E_COLOR = "#0072B2"      # mechanical-energy curve
BOUND_COLOR = "#CC0000"  # EC1 conservation ceiling

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, k in zip(axes, keys):
    r = runs[k]
    E = r["E_series_J"]
    dt = r["dt"]
    t = [i * dt for i in range(len(E))]
    bound, E0 = r["EC1_bound_J"], r["E0_J"]
    pk = E.index(r["E_peak_J"])
    fb = r["first_breach_step"]

    ax.plot(t, E, color=E_COLOR, lw=1.6, label="mechanical energy  E(t) = KE + PE")
    ax.axhline(bound, ls="--", lw=1.4, color=BOUND_COLOR,
               label=f"EC1 conservation budget ≈ {bound:.1f} J")
    ax.axhline(E0, ls=":", lw=1.2, color="#777777",
               label=f"E0 = {E0:.1f} J  (= m g h, the conserved value)")

    # first budget breach (first bounce) + global peak
    ax.scatter([t[fb]], [E[fb]], s=42, color="#222222", zorder=5)
    ax.annotate(f"first breach\nt = {t[fb]:.2f} s", (t[fb], E[fb]),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color="#222222")
    ax.scatter([t[pk]], [E[pk]], marker="*", s=150, color="#222222", zorder=5)
    ax.annotate(f"peak {E[pk]:.0f} J\n(+{r['pct_increase']:.0f}%)", (t[pk], E[pk]),
                textcoords="offset points", xytext=(-10, -28), fontsize=8.5,
                ha="right", color="#222222")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("mechanical energy (J)")
    ax.set_title(f"Δt = 1/{r['hz']} s  ({r['hz']} Hz)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved ->", OUT)
