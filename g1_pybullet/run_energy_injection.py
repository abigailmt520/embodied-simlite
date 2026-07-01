#!/usr/bin/env python3
"""Re-run the RQ3 / §7.3 cross-engine elastic-bounce energy-injection experiment and
persist the per-step mechanical-energy series to energy_injection_series.json.

Deterministic PyBullet simulation (no randomness) — this is a faithful re-run, not a
synthetic curve. The experiment logic lives unchanged in
`g1c_pathologies.pathology_energy_injection()`; this script only drives it at two
timesteps (Δt = 1/30 s main, Δt = 1/120 s control) and saves the results.

Run from a PyBullet environment, e.g.:
    conda run -n g1-pybullet python g1_pybullet/run_energy_injection.py
"""
import json
import os
import sys

import pybullet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1c_pathologies import pathology_energy_injection  # noqa: E402

try:
    from importlib.metadata import version as _pkg_version
    PB_VERSION = _pkg_version("pybullet")
except Exception:
    PB_VERSION = "unknown"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_injection_series.json")

record = {}
for dt in (1.0 / 30.0, 1.0 / 120.0):          # main, then control
    pathology_energy_injection(dt=dt, record=record)

payload = {
    "experiment": "RQ3 cross-engine elastic-bounce energy injection (e=1 conservative ball, restitution=1.0)",
    "quantity": "mechanical energy E = KE + PE (J) per step; E0 = m*g*h = potential energy at the drop height",
    "ec1_bound": "EC1 conservation upper bound = E0 + work_in + floor_abs + floor_rel*|E0| (work_in=0, floor_abs=1e-2, floor_rel=0.02)",
    "meta": {
        "pybullet_version": PB_VERSION,
        "pybullet_api_version": pybullet.getAPIVersion(),
        "python": sys.version.split()[0],
    },
    "runs": record,
}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nsaved -> {OUT}")
for k, r in record.items():
    print(f"  {k}: E0={r['E0_J']:.3f} J  peak={r['E_peak_J']:.3f} J  "
          f"bound={r['EC1_bound_J']:.3f} J  +{r['pct_increase']:.1f}%  "
          f"first_breach@step{r['first_breach_step']} (t={r['first_breach_time_s']:.3f}s)  "
          f"audit_red={r['audit_red']}")
