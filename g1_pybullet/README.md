# G1 · PyBullet real-engine generalization (separate environment)

Test whether the audit suite can catch **an independent third-party engine (PyBullet)'s own native numerical pathologies** (non-circular), and port the contract/joint layers to 3D. The audit logic reuses the 2D platform's `../audit/` (integrity_audit / joint_audit); the 2D platform itself is not changed.

## Environment (separate conda env, base untouched)
On Python 3.13 / recent macOS, PyBullet has no wheel and source compilation fails → use the conda-forge prebuilt binary in a separate env:
```bash
conda create -n g1-pybullet -c conda-forge python=3.13 pybullet numpy scipy matplotlib -y
```

## Running
Note: `cross_fidelity.py` additionally needs `gymnasium` in the env (`conda run -n g1-pybullet pip install gymnasium`, **do not pull in torch/sb3**).
```bash
conda run -n g1-pybullet python g1_pybullet/g1a_baseline.py     # healthy baseline + noise floor (checkpoint 1)
conda run -n g1-pybullet python g1_pybullet/g1b_tunneling.py    # native high-speed tunneling → physics audit (checkpoint 2 · crown jewel)
conda run -n g1-pybullet python g1_pybullet/g1c_pathologies.py  # energy injection + contract layer + joint (3D)
conda run -n g1-pybullet python g1_pybullet/cross_fidelity.py   # Cross-Fidelity energy contrast (twin passes self-check vs cross-fidelity oracle)
```

## Files
- `pb_helpers.py` — scenes / twin reporter (OdomReporter) / invariants / audit-reuse layer (incl. the swept EC5′, energy-conservation upper bound).
- `g1a_baseline.py` / `g1b_tunneling.py` / `g1c_pathologies.py` — the three stages.
- `cross_fidelity.py` — **Cross-Fidelity energy contrast**: PyBullet (reality) vs the 2D simplified twin (report) run in the same process, proving "internal consistency ≠ consistency with reality" (the twin passes EC1–EC5, while the cross-fidelity oracle catches the ledger-energy divergence at contact). See `../docs/CrossFidelity-Energy.md`.

## Key results (see ../docs/G1-PyBullet-Generalization.md)
- Engine energy noise floor ≈ 8.3e-4 J/step (the audit threshold must be > this).
- High-speed tunneling (200 m/s): the engine self-reports penetration=0 (completely missed), **only the swept EC5′ catches it** (non-circular · crown jewel).
- Energy injection (elastic bounce e=1): caught by the energy-conservation upper-bound audit.
- Contract C1-C3 ported to 3D: zero false positives on healthy + each injector (odom=truth / frozen seq) caught.
- warm-start ghost force: not cleanly measured (hard to isolate, honest negative/uncertain).
