# Embodied-SimLite · Platform iteration master document (living document)

> **A single cumulative document**: every platform-iteration task updates here. It records each Phase's design decisions / implementation / validation (real numbers), as well as the **current platform state** (branch, weights, capabilities, limitations). Purpose: the platform evolution is traceable, so a future maintainer / AI does not have to reverse-engineer it again.
> 🔴 Red line: master `7b54625` (the submitted-paper version) is **permanently frozen**; all iteration happens on dev branches; the paper weights are not overwritten.
> Last updated: 2026-06-27 (**dev/integrated**: Phase4b + G1 + RQ4 integration + G5 statistical rigor + Scenario B topological irreducibility)

---

## 1. CURRENT STATE

### 1.1 Branches
| Branch | Content | Based on |
|---|---|---|
| `master` | 🔒 **frozen** paper-submission version (zero-inertia kinematics + contract audit C1/C2/C3 + true-fork odom + ROS2) | — |
| `dev/stage1-dynamics` | Phase1a: dynamics core + A-mode + energy audit EC1-EC3 + 5 injectors | master `7b54625` |
| `dev/stage2-bmode` | Phase1b/1c: B-mode force control + G-1/G-2 gaming experiments | Phase1a `432d24b` |
| `dev/stage3-maze` | Phase2: F4 rectangular collisions + F2 40×40 maze + collision-fault audit | Phase1c `5e2af1c` |
| `dev/stage4-mi-leakage` | Phase3: contract-layer mutual-information leakage audit (C_I, a principled generalization of C1) | Phase2 `3cc426e` |
| `dev/stage5-coupling` | Phase4: dual-state coupling stress test (report×physics joint audit + true/false coupling criterion) | Phase3 `af81e14` |
| `dev/stage6-ec5prime` | Phase4b: EC5′ (within-physics truth-vs-map) + resident three-layer suite (finishing the criterion separation) | Phase4 `9f188b1` |
| `dev/g1-pybullet` | G1: real-engine (PyBullet) generalization — the audit catches the engine's native pathologies (non-circular) + contract/joint ported to 3D | Phase4b `57dbca8` |
| `dev/rq4-coverage` | RQ4: coverage matrix vs 4 baselines — proving the joint is non-redundant | Phase4b `57dbca8` |
| **`dev/integrated`** | **integrates Phase4b+G1+RQ4 + G5 statistical rigor (the final reproduction artifact branch)** | merge |

### 1.2 Weight files (each its own configuration, none overwriting another)
| File | Configuration | Performance (N=25/30 fixed seed) | Branch |
|---|---|---|---|
| `ppo_embodied_agent.pth` | 🔒paper version · zero-inertia kinematics · obs26/act[v,w] | success 84%/collision 12%/timeout 4% | master |
| `ppo_embodied_agent_dyn.pth` | A-mode target-velocity-tracking dynamics · obs26/act[v,w] | 92%/8%/0% | dev/stage1 |
| `ppo_embodied_agent_bmode.pth` | B-mode force control (τ=4×step) · obs28/act[f_l,f_r] | 76%/16%/8% | dev/stage2 |
| `ppo_embodied_agent_bmode_gamed.pth` | B-mode + G-1 fault-environment training (**passive gaming, emerges**) | Phase1b §2.2 | dev/stage2 |
| `ppo_embodied_agent_g2gamed.pth` | B-mode + G-2 fault-environment training (**uniquely-learned gaming, did not emerge**) | Phase1c §2.2 | dev/stage2 |
| `ppo_embodied_agent_maze.pth` | A-mode + 40×40 maze + rectangular collisions (bounce) | success 48%/timeout 52% | dev/stage3 |

### 1.3 Current capability matrix
| Dimension | State |
|---|---|
| **Control mode** | A=target-velocity tracking (P controller); B=raw wheel-force control. Shared dynamics core (force→Newton+viscous damping→N_SUB semi-implicit integration). `ENABLE_DYNAMICS=False` falls back to zero-inertia kinematics (paper-version behavior) |
| **Physics fidelity** | Simplified dynamics (mass/inertia/viscous damping) + **F4 rectangular collisions** (circle-AABB penetration pushout + bounce e=0.5); energy + collision ledger telegraphed exactly (clean residual ~1e-16 J). **Does not include** rigid-body contact dynamics |
| **Contract-layer audit** | C1 true-fork / C2 frame-sequence monotonicity / C3 disconnect-freezes (**unchanged, regression passes**); **+ C_I mutual-information leakage audit** (I(Δodom;Δtruth)≤noise-budget bound, a principled generalization of C1, catches L-1/L-2/L-3 in the estimable regime slip≈0.3) |
| **Physics-layer audit** | EC1 energy budget / EC2 no free energy / EC3 actuator bound / EC4 collision non-negativity / EC5 non-penetration (ledger) / **EC5′ truth-vs-map (within-physics geometric recomputation, does not trust the ledger)**. Injectors P-1..P-5 + CF-1..CF-3 (EC5′ zero false positives, red only when the truth really crosses the wall) |
| **Joint audit (report×physics, resident)** | `audit_suite` + `joint_audit`: **the joint layer has two complementary paths** — ① JOINT odom-vs-claimed-map (naive point check · reducible) + ② RELATIONAL `relational_oracle` seg(truth,odom) crossing the wall with both endpoints free (through-crossing + persistence gate `JOINT_PERSIST_FRAC=0.5`, **topologically irreducible**); joint red ⟺ either path red. **Criterion separation**: EC5′ red ⇒ catchable within a single physics layer (not coupling); EC5′ green ∧ JOINT red ⇒ true coupling (only the joint catches). Resident three-layer suite `run_suite` (Phase4/4b + irreducibility wired in). **🔴 G5 boundary**: the naive JOINT has a **trajectory-length validity envelope** — short range (≤40 steps, odom drift <0.15m) zero false positives and catches only true coupling (Scenario B); long range (≥160 steps, cumulative drift >1m) healthy odom drifts into the wall → 100% false positives; deployment must use a short sliding window / drift-budget gate. **🔴 Topological irreducibility**: after Scenario B is reconfigured to "displacement crosses the wall" (both endpoints free, d<ξ), even a map-equipped contract (o∈M? / o-track self-crossing) **misses**, and only the relational seg(x_t,o_t)-crossing-the-wall catches it → the relational layer is non-redundant (`relational_oracle.py`, § ScenB-Irreducibility) |
| **Odometry** | true-fork odom (eats the actual velocity v_act + slip drift), C1 preserved; collisions do not correct odom (keeping "only drift, never correct") |
| **Map** | `random_circle` 10×10 random circles (default, zero regression) / `maze` **40×40 hand-built wall maze** (19 AABBs, ray-AABB lidar + circle-rectangle collision). **Does not include** dynamic obstacles |
| **Collision semantics** | `terminate` (terminate on impact, paper version) / `bounce` (penetration pushout + bounce + per-step contact penalty R_CONTACT, no termination, used for maze navigation) |
| **Frontend** | Three.js pure-observation, disconnect-freezes OFFLINE (unchanged); the contract additively adds v_act/energy fields |
| **ROS2** | /odom·/scan·tf + cmd_vel manual override (unchanged) |

### 1.4 Known limitations
- **A-mode dynamics are mild** (τ≈0.055s≪step, near-Markovian) → the old policy transfers losslessly, no room for emergence. **Only B-mode (τ=4×step) has meaningful inertia + emergent gaming**.
- **Emergent gaming has an intrinsic "discoverability-specificity" tradeoff** (empirically shown in Phase1b/1c): G-1 (aligned, triggers at high speed) **emerges easily but honest also free-rides 29%** (low specificity); G-2 (anomalous, triggers at both-wheels-idle) **honest 0% free-riding (high specificity) but RL did not spontaneously emerge** (the exploit is off the reward gradient). The energy audit **flags RED in both cases when a violation occurs** (the G-2 script proves it); the difficulty is "does the agent exhibit the violation". Strong uniquely-learned gaming needs reward-shaping/curriculum/a sweet-spot exploit.
- The energy/physics audit consumes the ledger via get_render_state, **inheriting the contract trust root** (contract = trust boundary, see the linchpin review); the contract-vs-kernel reconciliation (q class) is not done.
- **The C_I mutual-information audit degrades at high SNR (small slip)**: the KSG estimate saturates for near-deterministic channels; at the deployed slip=0.05 (odom very accurate) it reliably catches only the full leak L-1, missing L-2/L-3; there C1's magnitude check is more practical. C_I is clean and quantitative in the estimable regime (slip≈0.3). The two are complementary.
- Not done: F4 rectangular collisions / F2 maze / F3 dynamic obstacles / a richer environment layer beyond B-mode; MuJoCo not connected.

---

## 2. Phase evolution history (design decisions + validation results)

### Phase0 · inventory and design (read-only, master unchanged)
- Original (ProductV1.0) vs MVP feature inventory; audit-coverage inventory (the false-negative surface C1-C3 can/cannot catch); linchpin review (trust root = the dict returned by get_render_state); Stage1/Phase1 migration design.
- Details: see the V3 working directory (design notes, not shipped in this artifact branch): "Feature inventory: original vs MVP" / "Audit-coverage inventory: self-deception fault taxonomy foundation" / "Audit trust-root review: linchpin confirmation" / "Stage1 dynamics-migration design" / "Phase1 platform-enrichment migration design".

### Phase1a · dynamics core + A-mode + energy audit (dev/stage1-dynamics, `432d24b`)
- **Decision**: action = target velocity (A), backend P controller → force → dynamics-core integration; odom eats v_act; contract additively adds v_act/energy.
- **Energy ledger telegraphed exactly**: settle with v_mid=½(v_n+v_{n+1}) → clean residual **1.665e-16 J** (better than the design O(h³) estimate).
- **A-mode braking = a real deceleration force** (F=-KP·v_act), removing the original v*=0.5 hard discontinuity.
- **Validation**: dimensions unchanged (26/2); retrain dyn.pth → **92%/8%/0%**; energy audit clean green + 5/5 injections red; contract C1-C3 no regression.
- **Real lesson**: P-5 mass-misreporting escaped under "fixed command" (steady-state F=cv is independent of mass) → fixed by switching to "varying command (sustained transient)".
- Details: V3 design note "Phase1a dynamics implementation & validation report" (not shipped in this artifact branch).

### Phase1b · B-mode force control + emergent gaming (dev/stage2-bmode, this round)
- **Decision**: action = raw wheel force (B), drop the P controller, reuse the core; tune F_MAX/ARM so **τ_v=0.333s≈4×step (meaningful inertia)**; obs adds velocity feedback (28).
- **Validation**: B-mode retrain → **76%/16%/8%** (force control is substantively harder, stated honestly); energy audit B-mode clean green (4.996e-16) + 5/5 red; A-mode regression 5/5; contract C1-C3 no regression.
- **Emergent gaming (key SCI result)**: inject G-1 "free thrust at high speed", an exploitable fault, comparing honest/gamed agents.
  **Result**: in the boosted env the agent systematically exceeds the physical limit (23~29%, top speed over by 33%), apparent performance inflated (success +7~10%, steps -20%), **the energy audit flags RED and locates it** (gamed@boosted EC1+EC3 red). **Honestly flagged**: this gaming is passive exploitation (honest also free-rides), not uniquely-learned; strong gaming needs G-2.
- Details: [docs/Phase1b-Bmode-Gaming.md](docs/Phase1b-Bmode-Gaming.md).

### Phase1c · G-2 uniquely-learned gaming (dev/stage2-bmode, this round)
- **Decision**: design the G-2 "idle-coast" fault — only when **both wheels are near-zero actuation (doing nothing)** while coasting does it get a large free forward force. Measured: the honest agent's both-wheels-simultaneously-idle fraction = 0% → only learning that **anomalous behavior** profits (high specificity).
- **🔴 Real negative result (reported honestly, INV-E)**: g2gamed, trained in the G-2 env, **did not emerge** idle-coast (idle 0.0%, performance even drops to 70% vs 83%).
  Three controls: ① did not learn the anomalous behavior ❌; ② **honest@g2boosted 0% free-riding ✅** (vs G-1's 29%, high specificity achieved); ③ the audit **does flag RED when the behavior occurs** (scripted idle-coast: EC1 residual 0.673J + EC2 + EC3, top speed 2.0) — the failure is purely on the RL-discovery side, not the audit side.
- **Scientific finding**: a **discoverability-specificity tradeoff** — an aligned exploit emerges easily but is hard to attribute; an anomalous exploit is clear to attribute but hard for RL to discover spontaneously (off-gradient).
- Details: [docs/Phase1c-G2-UniqueGaming.md](docs/Phase1c-G2-UniqueGaming.md).

### Phase2 · F4 rectangular collisions + F2 maze (dev/stage3-maze, this round)
- **Decision**: backend map abstraction (random_circle default zero regression / maze 40×40 hand-built 19 walls); F4 circle-AABB penetration pushout + bounce (all backend, reusing the old algorithm) + ray-AABB lidar; collision semantics terminate/bounce configurable.
- **Reward redesign**: in bounce mode hitting a wall does not terminate (R_COLLISION −200 terminate → R_CONTACT −5 per contact step), letting the agent navigate with real collisions.
- **Validation**: maze retrain (A-mode/bounce) success 48%/timeout 52%/average 3.1 contact steps (the difficulty of a 40×40 adversarial maze, stated honestly); energy audit 5/5 + contract C1-C3 no regression; driving straight into a wall pushes out exactly (x pinned at 38.8, penetration 0, residual 1e-16).
- **Collision-fault class**: CF-1 over-bounce adds energy / CF-2 skip penetration correction / CF-3 phantom dissipation + EC4 collision non-negativity / EC5 non-penetration. Clean green + 3/3 each flagged RED by its characteristic check (CF-1→EC4, CF-2→EC5, CF-3→EC1).
- **Opportunistic**: the honest maze agent's collision-containing trajectory runs energy audit 🟢GREEN — no natural collision gaming (an honest bounce only dissipates, no sweet-spot exploit), not forced.
- Details: [docs/Phase2-Maze-Collision.md](docs/Phase2-Maze-Collision.md).

### Phase3 · contract-layer mutual-information leakage audit C_I (dev/stage4-mi-leakage, this round, pure audit-layer addition)
- **Decision**: upgrade C1's "error too small is suspicious" into a principled "I(report;truth) ≤ noise-budget bound", symmetric with the physics EC1 conservation residual (finishing the dual-state formalization). KSG k-NN estimates MI, a re-noise MC operationalizes the noise-budget bound, sliding-window + persistence criterion.
- **🔴 Correcting Opus's original design pitfalls**: absolute-position MI is unusable (dead-reckoning cumulative, dominated by trajectory correlation) → switch to **per-step increment MI** (memoryless channel); the closed-form ½log(1+SNR) is only approximate for multiplicative slip noise → switch to the re-noise MC budget bound.
- **Validation (slip=0.30 estimable regime)**: clean green (I=2.49 ≤ bound 2.28 + margin); L-1/L-2/L-3 real MI 4.04/3.25/3.07 all over the bound → 3/3 flagged RED.
- **Comparison with C1**: C1 catches only the full leak L-1; **C_I additionally catches what C1 misses: L-2 (partial leak still drifts) / L-3 (large error but deterministic)** = a principled quantitative generalization of C1.
- **🔴 Reliability reported honestly**: KSG saturates at high SNR → at the deployed slip=0.05 (odom very accurate, bound ~3.3 nats approaches the estimation ceiling) L-1 still caught, **L-2/L-3 missed**; there C1's magnitude check is more practical. C_I's value is the quantitative coverage in the estimable regime + catching what C1 misses, **complementary to C1, not strictly stronger**.
- Details: [docs/Phase3-MI-Leakage.md](docs/Phase3-MI-Leakage.md).

### Phase4 · dual-state coupling stress test report×physics (dev/stage5-coupling, this round)
- **Goal**: stress-test the paper's core claim with anti-self-deception discipline — can a coupling self-deception of "each layer passes alone, only the joint catches" **really be constructed**.
- **Implementation**: phantom-wall fault (collision detection drops one wall, `embodied_env:691`) + joint_audit (track vs claimed-map non-penetration).
- **🔴 Core honest criterion**: true coupling ⟺ `truth_vs_map green ∧ odom_vs_map red` (truth legitimate, only the report illegitimate → catchable neither within physics nor by contract self-consistency).
- **Scenario A literal phantom wall (truth really crosses)**: physics 🟢 + contract 🟢, but **truth_vs_map 🔴** → **PHYSICS_INTERNAL = not true coupling** (within-physics truth-vs-map catches it; the existing EC5 misses it only because it trusts the ledger penetration = EC5 implementation gap). **Not forced, honestly judged as not coupling.**
- **Scenario B true-coupling variant (truth honest, odom fabricates crossing)**: physics 🟢 + contract 🟢 + **truth_vs_map 🟢**, only **odom_vs_map 🔴** → **TRUE_COUPLING**: true coupling is constructible, dual-state coupling is non-empty.
- **Finding**: the essence of true coupling is "the self-deception lands on report while the physics truth is honest"; the naive phantom wall is an EC5 gap masquerading as coupling, and the honest criterion separates the two — avoiding mistaking an "audit gap" for a "coupling argument".
- Details: [docs/Phase4-Coupling-Stress-Test.md](docs/Phase4-Coupling-Stress-Test.md).

### Phase4b · EC5′ (within-physics truth-vs-map) + resident three-layer suite (dev/stage6-ec5prime, this round, pure audit layer)
- **Decision**: plug the EC5 gap Phase4 pointed at itself — EC5′ (`joint_audit.ec5_prime`) does not trust the ledger penetration, recomputes geometry directly from the truth against the claimed full map; joint_audit is fixed into the resident three-layer suite `audit_suite.run_suite`.
- **Scenario A (truth really crosses)**: EC1-EC5 🟢 but **EC5′ 🔴** (truth inside the wall 0.200m) → **PHYSICS_INTERNAL**: a single physics layer catches it → nails down "Scenario A = single-layer gap (now plugged by EC5′), not coupling".
- **Scenario B (truth legitimate, odom fabricates crossing)**: physics (incl. EC5′) 🟢 + contract 🟢 + **EC5′ 🟢 (did not stand in for the joint)**, only **JOINT 🔴** → **TRUE_COUPLING**: nails down "Scenario B = true coupling, only the joint catches".
- **🔴 Honest criteria**: EC5′ **zero false positives** (30 healthy maze episodes 0/30; CF-2/phantom wall the truth really crosses → red = a real violation not a false positive); EC5′ did not stand in for the joint (Scenario B EC5′ green); the criterion separation is clean. **No parameters tuned to force it.**
- No regression (CF 3/3, P 5/5, C1-3, C_I 3/3); env unchanged.
- Details: [docs/Phase4b-EC5prime-Suite.md](docs/Phase4b-EC5prime-Suite.md).

### G1 · real-engine (PyBullet) generalization (dev/g1-pybullet, separate conda env, 2D platform unchanged)
- **Decision (breaking the circularity · the largest publishability gap)**: test whether the audit can catch **an engine we did not build (PyBullet)'s own native numerical pathologies** (strong test · non-circular), rather than porting our own injectors over (weak test · semi-circular).
- **G1a baseline (checkpoint 1 ✅)**: characterize the engine energy noise floor **8.3e-4 J/step** (threshold 10×); porting lessons = PyBullet's default damping silently drains energy, EC5′'s 2D projection requires planar motion. After fixing the setup, healthy has **zero false positives**.
- **G1b high-speed tunneling (checkpoint 2 ✅ crown jewel · non-circular)**: a 200 m/s ball really crosses the wall, **the engine self-reports penetration=0 (completely missed)**, the per-frame point check also misses, **only the swept EC5′ catches it** — the audit caught a real engine pathology. Refinement = high-speed tunneling requires swept-segment detection.
- **G1c**: energy injection (elastic bounce e=1, the engine's discrete solver injects energy E0=19.6→42.7J) → the energy-conservation upper-bound audit **catches** it; contract C1-C3 ported to 3D healthy zero false positives + each injector (odom=truth / seq frozen) caught; joint works in 3D. The warm-start ghost force was **not cleanly measured** (hard to isolate, honest negative/uncertain).
- **Honest summary**: only inducing conditions set, zero hand-set faults (engine penetration=0 is the smoking gun); healthy zero false positives (noise floor characterized); the trust-root boundary = PyBullet API lying is out of scope (same family as the 2D q class).
- Details: [docs/G1-PyBullet-Generalization.md](docs/G1-PyBullet-Generalization.md).

### RQ4 · coverage matrix vs 4 baselines (dev/rq4-coverage, pure audit-layer harness)
- **Goal**: prove the layered + joint coverage is broader than the baselines — especially that the **joint is non-redundant and the only means to catch dual-state coupling (Scenario B)**.
- **4 fair baselines (not strawmen)**: M1 code/data integrity, M2 physics-only, M3 contract-only, M4 naive parallel (no joint); M5 = ours (with joint).
- **Cell-by-cell real validation** (10 instances × 5 methods): ① healthy zero false positives; ② code-integrity catches only data_tamper, misses all semantics; ③ each single layer misses the other class.
- **🔴 Headline**: **M4 naive parallel misses Scenario B (dual-state coupling); only M5 (joint) catches it** → the joint is non-redundant and necessary (Scenario B physics EC5′ green + contract green, only odom-vs-map cross-state catches).
- **🔴 Honest (not a clean sweep)**: data_tamper is **caught only by code-integrity, we miss it** (trust-root blind spot, complementary not dominated); L-2 partial leak is **missed by everyone** (MI lift +0.23 within the margin, sample/trajectory sensitive, consistent with Phase3, not forced past the threshold).
- Details: [docs/RQ4-Coverage-Matrix.md](docs/RQ4-Coverage-Matrix.md).

### G5 · statistical rigor (dev/integrated, pure audit-layer harness; includes the Phase4b+G1+RQ4 branch integration)
- **Goal**: add statistical rigor to detection/false-positives — repeats (30/24 independent seeds) + Wilson 95% CI + boundary characterization, to a submittable (TOSEM/ISSTA) standard.
- **Integration (housekeeping)**: the three parallel branches Phase4b/G1/RQ4 are merged into **`dev/integrated`** (merge `b562806`), as the final reproduction artifact; G5 runs on this branch. master `7b54625` frozen, .pth zero changes.
- **A/B coverage matrix with CIs** (30 seeds × 9 instances × 5 methods): strong detection deterministic, CI[88.6,100]; healthy **M1-M4 false positives 0/30** (CI[0,11.4]); strong detection robust across seeds, M5 each 100% (CI[88.6,100]).
- **🔴 Key honest finding (statistics reveal the false positive that single-point testing missed)**: the naive **M5 (with joint) healthy 320-step false positive 30/30=100%** — not a bug: the collision-uncorrected odom dead reckoning accumulates drift with **trajectory length** (heading integration, ~6m) into the wall geometry (truth EC5′ always green and legitimate) → joint false positive. RQ4/Phase4's single short/arc trajectory happened not to trigger it; 30 diverse trajectories reveal it statistically.
- **Part D · joint validity envelope (honestly characterized, not tuned)**: healthy joint false positives vs trajectory length — 20 steps 0/30, 40 steps 0/30 (drift 0.14m), 80 steps 1/30, 160 steps 30/30, 320 steps 30/30 (drift 5.8m). **Scenario B (40 steps, fabricated divergence 3.5m ≫ 0.14m honest drift) falls inside the validity envelope → only the joint catches it**. Deployment must use a **short sliding window / drift-budget gate**. It also reveals the structural tension that **C_I (needs long range) and the joint (needs short range) have opposite range requirements**.
- **Part C · L2/C_I sensitivity boundary (turning the single-point L-2 observation into a curve)**: ① detection rate vs leak magnitude (N=1000): shrink≤0.25 CI lower bound >50%, shrink≥0.6 all miss; ② detection rate vs sample size (shrink=0.25=L-2): **non-monotonic**, 250→50%, 500/1000→83%, 1500→37.5%, 2000→25% (KSG finite-sample bias + a fixed margin: sweet spot ~500-1000; RQ4 used 300 increments, just below the threshold → miss, quantified here as the required sample boundary).
- **Artifacts**: `audit/run_g5_statistics.py`, `audit/ci_sensitivity.png` (3 panels), `audit/g5_stats.json`. No regression (env unchanged, pure audit layer).
- Details: [docs/G5-Statistical-Rigor.md](docs/G5-Statistical-Rigor.md).

### Scenario B topological irreducibility (dev/integrated, pure audit layer; § 5 core-claim defense)
- **Attack**: an external formalization reviewer questions "dual-state coupling is irreducible" — Scenario B can be caught by just "give the map to the contract layer and check whether o_t is inside the wall" → the relational layer is redundant.
- **Defense (topological irreducibility)**: **reconfigure** Scenario B from "o_t inside the wall" (the old form · reducible) to "**displacement crosses the wall**" — the truth x_t is free on side A of the wall, the report o_t is free on side B (**not inside the wall**), ‖o_t−x_t‖≤ξ, only the **displacement vector o_t−x_t crosses the rigid wall**. Precondition: **wall thickness d=0.20 < noise budget ξ=0.50**.
- **Five-oracle measurement (reconfigured v2)**: ① physics (x∈free) 🟢 ② contract noise (‖o-x‖≤ξ) 🟢 ③ **map-equipped contract M6a (o∈M?) 🟢 miss** ④ **stronger report-only M6b (seg(o,o)∩M) 🟢 miss**, only ⑤ **relational seg(x,o) crosses the wall (both endpoints free) 🔴 catch (60/60)** → **`IRREDUCIBLE_RELATIONAL`**: even a map-equipped contract cannot decompose it → the relational layer is **non-redundant**.
- **🔴 Honest criterion satisfied**: the map-equipped baseline **does not** catch v2 → irreducibility **holds** (measured, not tuned). Contrast with the old v1: M6a 🔴 catches (o_t really inside the wall) → reducible, confirming the value of the reconfiguration.
- **🔴 Honest cost (d<ξ double-edged)**: the same d<ξ lets honest noise occasionally cross the thin wall — M6a/M6b healthy false positives 30/30, relational single-frame 25/30; but after the **persistence gate** (crossing-frame fraction: healthy max=0.38 < threshold 0.5 < v2=1.00) the **relational is healthy 0/30, still catches v2** → only the relational can be both sensitive to v2 and specific on healthy (same family as G5 Part D).
- **(e) Empirical soundness (§ 5 linchpin #2)**: measure the real env's honest drift δ(t) vs obstacle clearance vs the C1 ceiling ξ=0.50. **Provable-sound sufficient condition δ_max<clear_min**: the gated short windows (incl. scenB v2 N=60) [20,40,80] measure δ_max/clear_med = 0.10/0.18/0.41 → **δ≪clearance, provably no false positive**. δ_max grows with window length (0.08→0.13→0.27→1.10→5.39m), **crossing clear_min @L≈160, exactly aligned with the G5 naive-point-check FP envelope** (0/30@≤80→29/30@160→30/30@320) → the FP envelope is geometrically explained by "δ crosses clearance". 🔵 The **refined relational (through-crossing + persistence) FP is 0/30 throughout** (better than the naive point check). → **both claims "irreducible" + "empirically sound" are data-confirmed, no revision needed**.
- **Artifacts**: `audit/relational_oracle.py` (five oracles + clearance), `audit/run_scenB_irreducibility.py`, `audit/scenB_irreducibility.{json,png}` (3 panels). No regression (env unchanged).
- **Wired into the resident suite + regression**: the relational oracle is merged into the `audit_suite.run_suite` joint layer (two complementary paths: naive point check + relational, `JOINT_PERSIST_FRAC=0.5`); `run_coupling_test.py` adds a **Scenario B2 (displacement crosses the wall)** regression — physics 🟢 + contract 🟢 + EC5′ 🟢 + naive point check 🟢 (miss), only **relational 🔴 catch** → `TRUE_COUPLING`, confirming the irreducible self-deception is covered by the resident suite. Scenario A/B verdicts unchanged, contract action1 three gates, no regression; master `7b54625` frozen, .pth zero changes.
- Details: [docs/ScenB-Irreducibility.md](docs/ScenB-Irreducibility.md) (§8 wiring + regression).

### Cross-Fidelity self-deception audit · energy contrast (dev/integrated, pure new module; rebutting the "toy platform / hand-injection" criticism)
- **Claim**: **internal consistency ≠ consistency with reality** — a simplified twin that passes all internal oracles (EC1–EC5) can still diverge in ledger energy from high-fidelity reality at contact.
- **Setup**: PyBullet (physical reality, truth) and the 2D twin (simplified model, report) run scripted actions in the same process (`g1_pybullet/cross_fidelity.py`; the g1-pybullet env adds gymnasium, **does not pull in torch**). The matched robot is aligned item by item (mass 1 / Izz 0.5 / radius 0.20 / viscous damping −C·v / 24 substeps aligned / effective e≈0.499≈BOUNCE); **the free segment validates clean (|ΔE|≤0.014J, position ≤0.023m = FP floor)**; the only substantive gap = contact physics (2D `v*=0.5` scalar halving + pin-to-wall, no vector reflection/friction vs PyBullet's real contact).
- **🔴 Core energy contrast**: constant-force wall impact, **the 2D twin EC1–EC5 all PASS (internally self-consistent)**, **the cross-fidelity energy oracle catches the contact divergence** (threshold 5×floor=0.0715J; head_on divergence 0.179J, glancing 0.105J, both FLAG). head_on is the most intuitive: **both pin at x=1.80m, but the twin's ledger phantom-reports 0.032–0.18J kinetic energy while PyBullet really zeros it** (same position, different energy).
- **Divergence envelope**: sweep the incidence angle 0–75° — **the cross-fidelity oracle 5/6 FLAG, the 2D EC 6/6 PASS**; at 75° the wall is not reached = a true negative (not a miss). The **trajectory divergence** (glancing 0.313m) is reducible, honestly flagged (irreducibility left to scenB Table 2).
- **🔴 Honest complication**: the divergence sign varies with geometry (head_on over-reports / glancing under-reports → the real gap is the whole coarse contact model, not simply "missing friction"); the divergence is non-monotonic in angle; emergent (only control + wall + friction set, the divergence is not hand-pinned). **No over-claim of irreducibility.**
- **Artifacts**: `g1_pybullet/cross_fidelity.py`, `cross_fidelity_energy.{json,png}` (3 panels). 2D platform zero changes, master frozen, .pth zero changes.
- Details: [docs/CrossFidelity-Energy.md](docs/CrossFidelity-Energy.md).

---

## 3. Key reproduction commands

```bash
# A-mode retrain (dyn)
EMBODIED_CONTROL_MODE=A EMBODIED_MODEL_PATH=ppo_embodied_agent_dyn.pth python train_agent.py
# B-mode retrain (bmode)
EMBODIED_CONTROL_MODE=B EMBODIED_MODEL_PATH=ppo_embodied_agent_bmode.pth python train_agent.py
# Physics energy audit red/green contrast (A or B)
EMBODIED_AUDIT_MODE=A python audit/run_physics_audit.py   # → energy_redgreen_matrix_amode.png
EMBODIED_AUDIT_MODE=B python audit/run_physics_audit.py   # → energy_redgreen_matrix_bmode.png
# G-1 passive gaming training + experiment
EMBODIED_CONTROL_MODE=B EMBODIED_BOOST_FORCE=1.5 EMBODIED_MODEL_PATH=ppo_embodied_agent_bmode_gamed.pth python train_agent.py
python audit/run_gaming_experiment.py     # → gaming_compare.png, gaming_summary.json
# G-2 uniquely-learned gaming training + experiment
EMBODIED_CONTROL_MODE=B EMBODIED_G2_FORCE=6.0 EMBODIED_MODEL_PATH=ppo_embodied_agent_g2gamed.pth python train_agent.py
python audit/run_g2_gaming_experiment.py  # → g2_gaming_compare.png, g2_gaming_summary.json
# Maze + rectangular collisions training + collision-fidelity audit
EMBODIED_CONTROL_MODE=A EMBODIED_MAP_TYPE=maze EMBODIED_MODEL_PATH=ppo_embodied_agent_maze.pth python train_agent.py
python audit/run_collision_audit.py       # → collision_redgreen_matrix.png (CF-1/2/3 red/green contrast)
# Contract-layer mutual-information leakage audit C_I (+ comparison with C1)
python audit/run_leakage_audit.py         # → leakage_compare.png (slip 0.30/0.05 two regimes)
# Dual-state coupling stress test report×physics (true/false coupling criterion)
python audit/run_coupling_test.py         # → coupling_summary.json (Scenario A not-coupling / Scenario B true coupling)
# G5 statistical rigor (Wilson CI coverage matrix + joint validity envelope + L2/C_I sensitivity curves)
python audit/run_g5_statistics.py         # → ci_sensitivity.png (3 panels), g5_stats.json
# Scenario B topological irreducibility (even the map-equipped contract misses v2, only the relational catches; §5 defense)
python audit/run_scenB_irreducibility.py  # → scenB_irreducibility.{json,png}
# Cross-Fidelity energy contrast (twin passes self-check vs cross-fidelity oracle catches contact divergence; separate conda env)
conda run -n g1-pybullet python g1_pybullet/cross_fidelity.py  # → cross_fidelity_energy.{json,png}
# Contract-layer regression
python audit/run_action1.py
```

---

## 4. Next-step candidates (pending Abi/Opus decision)
- Wire the three-layer suite `audit_suite` into the inference_server runtime stream (real-time joint monitoring).
- F3 dynamic obstacles (patrol vehicles, moving AABBs; may provide a collision/avoidance sweet-spot gaming substrate).
- Improve the maze success rate: a simpler maze / curriculum learning / a B-mode force-control maze policy; frontend rendering of the maze walls (the contract already exports walls).
- Bridge the "discoverability-specificity tradeoff": reward-shaping/curriculum learning to guide G-2 emergence, or switch to stronger exploration (RND/model-based).
- Contract-vs-kernel reconciliation (q class, breaking the trust-root blind spot).
