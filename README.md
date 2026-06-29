# Beyond Single-Layer Oracles: Runtime Verification of Digital-Twin Self-Deception

> Reproducibility artifact for the paper **"Beyond Single-Layer Oracles: Runtime Verification of Digital-Twin Self-Deception."**
>
> We treat **digital-twin self-deception** — where a twin's *reported* state diverges from physical reality while passing each single-channel check — as a **runtime-verification** problem. The artifact provides (i) a lightweight 2-D embodied-navigation twin with a deliberately decoupled *backend-truth / frontend-observation* architecture, and (ii) a layered **audit suite** of oracles (contract, physics, relational, and cross-fidelity) that is itself first proven able to catch fabrication.

**Frozen snapshot:** the exact state behind the paper is tagged **`paper2-final`**. The previous paper's artifact is frozen at branch `master` / tag `paper1-final` and is **not** modified by this work.

---

## 1. What this is

A digital twin is a simplified model of reality. *Twin self-deception* = the twin's reported channel $o_t$ diverges from the ground-truth channel $x_t$ while the divergence stays hidden to single-channel checks. The platform realizes the dual-channel interface directly:

- **Backend = single source of truth.** It integrates the true physical state $x_t$ (force-based dynamics: Newton + viscous damping, semi-implicit sub-stepping; rigid obstacles/walls with circle–AABB collision; an *exact energy budget* $\Delta E = W_\text{act}-D_\text{damp}$ telescoping to machine precision).
- **Frontend = pure observation viewport.** It performs no physics. The reported channel $o_t$ is dead-reckoning odometry under a seeded slip model (zero slip ⇒ $o_t\equiv x_t$; nonzero slip ⇒ *honest*, generated drift).

The auditor is an **external monitor** over these two channels:

| Layer | Oracles | Catches |
|---|---|---|
| Contract (report self-consistency) | C1 true-fork / C2 sequence monotonicity / C3 disconnect-freezes / C_I mutual-information leakage | echoed odometry, frozen seq, "online" while disconnected, privileged-state leakage |
| Physics (engine self-consistency) | EC1 energy budget / EC2 no free energy / EC3 actuator bound / EC4 collision non-negative / EC5 non-penetration (ledger) / EC5′ truth-vs-map (geometric) | energy/penetration/ghost-obstacle violations |
| Relational (report × physics) | joint odom-vs-map (point) + **relational displacement-crossing** (through-crossing, persistence-gated) | dual-state coupling: report in a *different free region* from truth |
| Cross-fidelity (twin vs high-fidelity reality) | energy divergence of the twin's ledger vs PyBullet at contact | the twin passes all its *internal* ECs yet diverges from reality |

The headline result: **single-layer oracles (even map-equipped) are not enough** — the relational oracle catches a fault no single projection expresses (Table 1, Table 2), and a twin that passes all its internal physics oracles can still diverge from high-fidelity reality (Figure 3).

---

## 2. Directory structure

```
embodied-sim-lite/  (branch paper2-embodied-simlite; frozen tag paper2-final)
├── embodied_env.py            # the 2-D twin / platform (gymnasium env): dynamics core,
│                              #   true-fork odometry, analytic LiDAR, energy+collision ledger
├── train_agent.py             # PPO training (stable-baselines3); inference_server.py / ros_bridge.py = serving
├── ppo_embodied_agent*.pth    # byte-frozen trained weights (paper version + dynamics/maze/B-mode variants)
├── audit/                     # the audit suite + experiments + data + figures
│   ├── integrity_audit.py     #   contract oracles C1–C3
│   ├── leakage_audit.py, mi_estimator.py   #   contract C_I (KSG mutual information)
│   ├── energy_audit.py        #   physics oracles EC1–EC5
│   ├── joint_audit.py         #   EC5′ + joint odom-vs-map
│   ├── relational_oracle.py   #   relational displacement-crossing oracle (+ clearance)
│   ├── audit_suite.py         #   resident 3-layer suite (run_suite, coupling_label)
│   ├── fault_injection.py, physics_injection.py   #   injector catalog
│   ├── run_coverage_matrix.py #   RQ2 Table 1 (coverage matrix vs 4 baselines)
│   ├── run_scenB_irreducibility.py  #   RQ2 Table 2 (irreducibility) + FP/soundness data
│   ├── make_fig1_fp_envelope.py     #   Figure 1 (FP envelope) from scenB data
│   ├── run_g5_statistics.py   #   RQ1 detection (Wilson CI) + Figure 2 (CI sensitivity)
│   ├── run_coupling_test.py   #   resident-suite coupling separation + Scenario B2 regression
│   ├── run_action1.py         #   contract-layer red/green regression + base PPO eval
│   └── *.json, *.png          #   committed result data and figures
├── g1_pybullet/               # high-fidelity PyBullet (SEPARATE conda env)
│   ├── pb_helpers.py, g1a/g1b/g1c_*.py   #   engine-native pathology generalization (G1)
│   └── cross_fidelity.py      #   Figure 3 (cross-fidelity energy contrast)
├── docs/                      # per-result write-ups (RQ4, G5, ScenB, CrossFidelity, Phase*)
│   └── analysis/              #   early design/analysis notes (pre-implementation history)
├── PLATFORM_ITERATION_LOG.md  # development log (living document)
└── requirements.txt
```

---

## 3. Environment

**Core (Python 3.13; 3.10+ should work):**

```bash
pip install -r requirements.txt   # numpy, gymnasium, matplotlib, torch, stable-baselines3, ...
```

`scipy` is optional: the KSG mutual-information estimator (`audit/mi_estimator.py`) uses `scipy` if present and falls back to a numpy-only implementation otherwise (the committed `g5_stats.json` was produced via the fallback).

**Cross-fidelity / PyBullet (Figure 3 and the G1 generalization) — a separate conda environment** (PyBullet has no wheel on Python 3.13 / recent macOS; `torch` is intentionally *not* installed here):

```bash
conda create -n g1-pybullet -c conda-forge python=3.13 pybullet numpy scipy matplotlib -y
conda run -n g1-pybullet pip install gymnasium
```

---

## 4. Reproducing the main results

Each line below maps a committed script to the table/figure it produces. Detection/false-positive experiments are driven by **seeded, scripted action sequences over the frozen engine**; the trained PPO policy (also byte-frozen) provides the navigation baseline.

```bash
# RQ1 — detection recall (30/30, Wilson 95% CI) + Figure 2 (C_I sensitivity, joint envelope)
python audit/run_g5_statistics.py          # → audit/g5_stats.json, audit/g5_sensitivity.png

# RQ2 Table 1 — coverage matrix (10 instances × 5 methods; M4 naive-parallel misses Scenario B, only M5 catches)
python audit/run_coverage_matrix.py        # → audit/coverage_matrix.json, audit/coverage_matrix.png

# RQ2 Table 2 — relational irreducibility (d<ξ): map-equipped baselines miss, only the relational oracle catches
python audit/run_scenB_irreducibility.py   # → audit/scenB_irreducibility.json, audit/scenB_irreducibility.png

# Figure 1 — healthy false-positive envelope of the relational oracle (built from the scenB experiment's data)
python audit/make_fig1_fp_envelope.py      # → audit/fig1_fp_envelope.png

# Resident audit suite — coupling-label separation + Scenario B2 (displacement-crossing) integration regression
python audit/run_coupling_test.py          # → audit/coupling_summary.json

# Contract-layer red/green regression + base PPO evaluation
python audit/run_action1.py                # → audit/eval_metrics.png

# Figure 3 — cross-fidelity energy contrast (twin passes all internal ECs, diverges from PyBullet at contact)
conda run -n g1-pybullet python g1_pybullet/cross_fidelity.py   # → g1_pybullet/cross_fidelity_energy.{json,png}

# G1 — engine-native pathology generalization on an independent high-fidelity engine (non-circular)
conda run -n g1-pybullet python g1_pybullet/g1a_baseline.py     # noise floor (≈8.3e-4 J/step)
conda run -n g1-pybullet python g1_pybullet/g1b_tunneling.py    # 200 m/s tunneling: engine reports penetration=0, swept EC5′ catches
conda run -n g1-pybullet python g1_pybullet/g1c_pathologies.py  # elastic-collision energy injection caught by the conservation oracle
```

---

## 5. Frozen-artifact discipline

- `master` (tag `paper1-final`, commit `7b54625`) is the **previous** paper's artifact (zero-inertia kinematics + contract-only audit) and is left **byte-unchanged**.
- This branch (`paper2-embodied-simlite`, frozen tag **`paper2-final`**) is the Paper-2 artifact: the dynamics platform + the full layered audit suite.
- During the audit evaluation the **physics engine and the trained weights are not modified** — every reported detection and false positive is produced by running the audit layer over executions of this fixed artifact, so the evaluation is a test of the oracles, not of a moving platform.

## 6. License

See [LICENSE](LICENSE).
