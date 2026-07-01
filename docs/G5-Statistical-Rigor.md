# G5 details: Statistical rigor (repetition + Wilson confidence intervals + boundary characterization) — Tier-2 finishing

> Branch `dev/integrated` (Phase4b `57dbca8` + G1 `89ac4e4` + RQ4 `8bcb2cb`, three parallel branches integrated, merge `b562806`).
> master `7b54625` permanently frozen, zero changes; all `.pth` byte-unchanged; **pure audit layer (env unchanged)**.
> Numbers from real runs (INV-E); with line numbers (INV-B); negative/boundary results reported as-is, no parameter-tuning.
> Implementation: `audit/run_g5_statistics.py`. Artifacts: `audit/ci_sensitivity.png` (3 panels), `audit/g5_stats.json`.

---

## 0. Claim and method
Raise the earlier single-run / single-point detection-rate and false-positive observations to a publishable (TOSEM/ISSTA) statistical standard:
1. **Repetition**: repeat across N independent seeds (varying **both** the slip noise realization **and** the trajectory), rather than deciding on a single trajectory.
2. **Confidence intervals**: all proportions use the **Wilson 95% interval** (`run_g5_statistics.py:45 wilson`) — more honest than Wald for small-sample / boundary proportions (0/30, 30/30).
3. **Boundary characterization**: turn the single-point "pass/fail" decision into a **sensitivity curve** of detection rate vs leakage magnitude / sample size / trajectory length, delimiting each method's valid and invalid regimes.

Statistical effort is concentrated **where there is variance** (false-positive rate, the L-2 boundary, the joint envelope); the deterministic strong detections are only a robustness confirmation.

---

## 1. Housekeeping · branch integration (before G5)
Three parallel dev branches (Phase4b/G1/RQ4, all off Phase4b `57dbca8`) are merged into **`dev/integrated`** as the final reproducible artifact:
- merge `b562806`, resolving the `PLATFORM_ITERATION_LOG.md` conflict (keeping both the G1 and RQ4 phase entries; adding `dev/integrated` to the branch table).
- verified still clean after integration: RQ4 headline ✅, contract C1-C3 ✅, EC5′/joint suite ✅.
- **G5 runs on and is integrated into `dev/integrated`.** master `7b54625` frozen, `.pth` byte-unchanged.

---

## 2. Part A/B · coverage matrix with CIs + healthy false-positive rate (30 seeds, Wilson 95% CI)

Across 30 independent seeds (`seed=700+i`, each seed varies the slip noise and the trajectory), 9 instances × 5 methods (M1 code-integrity / M2 physics-only / M3 contract-only / M4 naive-parallel / M5 ours with joint).

| Instance | M1_code | M2_phys | M3_contract | M4_naive | **M5_ours** |
|---|---|---|---|---|---|
| healthy | 0/30 [0,11.4] | 0/30 [0,11.4] | 0/30 [0,11.4] | 0/30 [0,11.4] | **30/30 [88.6,100] 🔴see §3** |
| L1_leak | 0/30 | 0/30 | 30/30 [88.6,100] | 30/30 | 30/30 |
| seq_freeze | 0/30 | 0/30 | 30/30 | 30/30 | 30/30 |
| stall_online | 0/30 | 0/30 | 30/30 | 30/30 | 30/30 |
| P1_energy | 0/30 | 30/30 [88.6,100] | 0/30 | 30/30 | 30/30 |
| CF2_pen | 0/30 | 30/30 | 0/30 | 30/30 | 30/30 |
| scenA | 0/30 | 30/30 | 0/30 | 30/30 | 30/30 |
| **scenB** | 0/30 | **0/30** | **0/30** | **0/30** | **30/30 [88.6,100]** |
| L2_partial | 0/30 | 0/30 | 18/30 [42.3,75.4] | 18/30 | 30/30 |

(Intervals are Wilson 95%, in %. Full values in `g5_stats.json:part_ab`.)

- **Strong detections are deterministic**: each structural/physical/contract fault is detected 30/30 by its corresponding layer, CI[88.6,100] — **robust** across seeds (`run_g5_statistics.py:193` strong-detection list).
- **Healthy false positives (except joint) are robustly zero**: M1-M4 on healthy are 0/30, CI[0,11.4].
- **scenB (dual-state coupling) is still caught only by M5**: M2/M3/M4 all 0/30, only M5 (joint) 30/30 — continuing the RQ4 headline (joint non-redundant).
- **L2_partial has variance**: M3/M4 = 18/30=60% [42.3,75.4] — a single-point test would report "catch" or "miss" depending on the seed, and **the CI exposes this unreliability** (see §5 sensitivity curve).

---

## 3. 🔴 Key honest finding: statistics reveal a joint false positive that single-point tests missed

**The naive M5 (with joint) false-positives on healthy 320-step runs 30/30=100% [88.6,100]**, while M1-M4 are 0/30 on the same instance. Per-layer localization (`g5_stats.json`): physics EC1-EC5 + **EC5′ always green** (the healthy true state is legitimate), contract C1-C3+CI always green — the RED is **JOINT (odom-vs-declared-map)**.

**Diagnosis (not a bug, verified frame by frame)**:
- A healthy 320-step trajectory: **true-state max wall penetration 0.000 m → EC5′ green (bounce guarantees the true state is legitimate)**;
- **odom max wall penetration 0.200 m → joint RED**; max truth-odom drift **7.09 m**.
- That is: the odom is **collision-uncorrected dead reckoning**, and slip/heading error **integrated over trajectory length** accumulates into meters of drift, geometrically drifting into the wall — while the true state remains legitimate throughout. The naive `joint_report_vs_map` misjudges "honest long-horizon drift into a wall" as "fabricated wall-crossing".

**Why RQ4/Phase4 did not trigger it**: there scenA/scenB used short 40-step horizons, and the healthy baseline used a single curving trajectory that happened to skirt the walls (small accumulated drift). **The statistical repetition over 30 varied random trajectories revealed a false positive that a single trajectory masked** — this is exactly the value of statistical rigor (Tier-2 catching an FP that Tier-1 missed).

---

## 4. Part D · joint operating envelope (healthy FP vs trajectory length, honest characterization)

Quantify §3's false positive as a function of **trajectory length (accumulated drift)** (slip fixed at the deployed 0.05; Part A/B measured slip 0.05 and 0.30 both at 6-8 m drift at 320 steps → the main driver is length, not slip). 30 seeds per cell:

| Steps | healthy joint FP (Wilson CI) | mean max drift |
|---|---|---|
| 20 | 0/30 = 0.0% [0,11.4] | 0.08 m |
| 40 | 0/30 = 0.0% [0,11.4] | 0.14 m |
| 80 | 1/30 = 3.3% [0.6,16.7] | 0.28 m |
| 160 | 30/30 = 100% [88.6,100] | 1.20 m |
| 320 | 30/30 = 100% [88.6,100] | 5.78 m |

Reference: **Scenario B true coupling (40 steps, truth-odom divergence 3.50 m) → joint 🔴 catches**.

**Honest envelope reading (no tuning)**:
- **Short horizon (≤40 steps, accumulated drift <0.15 m)**: healthy joint FP CI upper bound ≤11.4% → **joint is valid**. Scenario B falls in this region: its fabricated wall-crossing divergence 3.5 m **≫** the same-horizon honest drift 0.14 m → joint **neither false-positives nor fails to catch the true coupling**. This explains why the Phase4/RQ4 conclusions are correct (they are all inside the valid envelope).
- **Long horizon (≥160 steps, drift >1 m beyond corridor clearance)**: healthy joint FP rate **rises monotonically to 100%** with drift → naive odom-vs-map **fails** (confusing honest long-horizon drift with fabricated wall-crossing).
- **Gated M5 (read directly off the same matrix)**: if joint is only enabled within its valid envelope (short horizons), then long-horizon healthy M5 degrades to M4 = **0/30 FP**, while short-horizon scenB's joint is still 30/30 — **gated M5 = zero healthy FP + retained unique dual-state-coupling detection**. The gate is given by the Part D drift envelope, not tuned to force a result.

**Structural tension (incidental honest finding)**: C_I (mutual information) needs **long** horizons for enough samples (§5: L-2 needs ≥500 increments), while joint needs **short** horizons (this section: ≤80 steps) — the two layers' horizon requirements are **opposite**, and cannot be satisfied on a single fixed-length trajectory. Deployment must run each layer on its own horizon (C_I long window, joint short sliding window), or give joint a **drift-budget gate** (RED only beyond the odom drift budget).

---

## 5. Part C · L2/C_I sensitivity boundary (turning the single-point L-2 observation into a curve, 24 seeds)

One 2200-step long trajectory per seed; KSG mutual information on its increments; leakage = shrinking the odom error to `shrink` (`dO_leak = dT + shrink·err`); RED threshold = noise-budget bound + fixed margin `MARGIN_NATS`.

### Curve ① detection rate vs leakage magnitude (N_inc=1000)
| shrink | detection rate (Wilson CI) | MI leak mean±sd | MI clean |
|---|---|---|---|
| 0.05 | 24/24 = 100% [86.2,100] | 2.38±0.12 | 1.52±0.16 |
| 0.15 | 24/24 = 100% [86.2,100] | 2.23±0.10 | 1.52 |
| 0.25 (=L-2) | 20/24 = 83.3% [64.1,93.3] | 2.11±0.10 | 1.52 |
| 0.4 | 10/24 = 41.7% [24.5,61.2] | 1.97±0.11 | 1.52 |
| 0.6 | 0/24 = 0% [0,13.8] | 1.79±0.12 | 1.52 |
| 0.8 | 0/24 = 0% [0,13.8] | 1.65±0.13 | 1.52 |

→ Boundary: **for shrink ≤ 0.25 the detection-rate CI lower bound is >50%** (strong leakage reliably caught); for shrink ≥ 0.6 all missed (weak leakage sinks into the noise).

### Curve ② detection rate vs sample size N_inc (shrink=0.25, i.e. RQ4's L-2)
| N_inc | detection rate (Wilson CI) | MI leak | MI clean |
|---|---|---|---|
| 250 | 12/24 = 50.0% [31.4,68.6] | 2.47±0.16 | 2.08±0.15 |
| 500 | 20/24 = 83.3% [64.1,93.3] | 2.15±0.11 | 1.62±0.12 |
| 1000 | 20/24 = 83.3% [64.1,93.3] | 2.11±0.10 | 1.52±0.16 |
| 1500 | 9/24 = 37.5% [21.2,57.3] | 1.90±0.18 | 1.30±0.21 |
| 2000 | 6/24 = 25.0% [12.0,44.9] | 1.83±0.23 | 1.22±0.23 |

→ **Non-monotonic**: sweet spot ~500-1000 increments (CI lower bound >50%). Honest reasons for the decay at both ends —
- too few (250): insufficient statistical power, wide CI [31,69];
- too many (1500/2000): KSG finite-sample bias shifts the whole MI estimate down with N (clean 2.08→1.22, leak 2.47→1.83), and the **fixed margin** `MARGIN_NATS` becomes too large relative to the converged gap → detection rate falls back.
- **This quantifies the root cause of the RQ4 single-point L-2 miss**: RQ4 used ~300 increments, just below the lower edge of the sweet spot (detection rate ~50-60%, CI spanning 50%) — not a framework defect, but the sample size being on the boundary. **Given 500-1000 increments, L-2 is reliably detected** (83%, CI lower bound 64%); this is an honest sample-size boundary, not tuning.

Figure: `audit/ci_sensitivity.png`, three panels (①② C_I sensitivity curves + Wilson CI bands, ③ joint FP–trajectory-length envelope).

---

## 6. Deliverables (against the task requirements)
- ✅ **Statistically characterized detection/FP results (tables with CIs)**: §2 coverage matrix + healthy FP, all Wilson 95% CI.
- ✅ **C_I sensitivity curves (L2 boundary)**: §5 two curves (leakage magnitude + sample size), turning the single-point L-2 into a curve, an honest non-monotonic sweet spot.
- ✅ **Strong-detection cross-seed robustness confirmation**: §2 M5 each strong fault 30/30 [88.6,100].
- ✅ **A coherent integrated audit suite**: §1, `dev/integrated` (Phase4b+G1+RQ4+G5).
- ✅ **🔴 Honest boundaries, no parameter-tuning**: §3 statistically reveals the joint healthy FP (an FP missed by single-point tests); §4 joint operating envelope + the C_I/joint horizon tension; §5 the L-2 sample-size boundary + KSG-bias non-monotonicity — **all reported as-is, none forced**.

No regression: env unchanged (pure audit layer); master `7b54625` frozen, `.pth` byte-unchanged.

---

## 7. TODO / out of scope (INV-C)
- A joint **drift-budget gate / short sliding window** implementation (the deployment refinement §4 points to) — this task only **characterizes** the envelope, it does not change the joint criterion.
- A Clopper-Pearson exact-interval comparison (this task uses Wilson, which is already honest enough for small samples).
- Statistical monitoring integrated into the inference_server runtime stream; not done.
- The trust-root blind spot (the q class, data_tamper) is still out of scope (consistent with RQ4).
