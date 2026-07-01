# Cross-Fidelity self-deception audit · energy contrast (lead) + trajectory divergence / divergence envelope

> Thesis: **internal consistency ≠ consistency with reality**. A simplified digital twin that **passes all of its internal physics oracles (EC1–EC5)** can still, at every contact, have an **energy ledger that diverges from high-fidelity physical reality** — so one must audit against reality, not merely check internal self-consistency.
>
> Implementation: `g1_pybullet/cross_fidelity.py` (separate conda env `g1-pybullet`, add `gymnasium`, **no torch/sb3**).
> Artifacts: `g1_pybullet/cross_fidelity_energy.json`, `cross_fidelity_energy.png`.
> 🔴 Architecture red line: PyBullet = physical reality (truth); the 2D twin (`EmbodiedNavEnv`) = the simplified model under audit (its E_kin ledger = the report).
> The 2D platform is **unchanged**; master `7b54625` frozen; no `.pth` changes. **Pure additive module.**

---

## 0. Setup (Steps 0–1)

- **Step 0 env unification**: add `gymnasium` to `g1-pybullet` (no torch/sb3), so the 2D twin and PyBullet **co-run in one process under scripted actions**.
- **Step 1 matching robot** (`cross_fidelity.py:make_robot`): a PyBullet cylinder unicycle, **item-by-item aligned** to the 2D twin — radius 0.20, mass 1.0, **rotational inertia about z Izz=0.5** (explicitly overriding the physical disk's 0.02 = the 2D `INERTIA_COEF·MASS`), viscous damping `−C_LIN·v / −C_ANG·w` (applied explicitly per substep with the body's own damping set to zero, exactly matching the 2D ODE). Time alignment: one 2D control step (`DT=0.1`) = **24 PyBullet substeps** (`PB_DT=1/240`).
- **Restitution calibration** (`measure_restitution`): PyBullet composes restitution as a product of the two bodies → set each body to `REST_BODY=√0.5`, measured **effective e_eff≈0.499 ≈ 2D BOUNCE=0.5** (clean attribution: the divergence comes from the contact **model**, not from a restitution mismatch I introduced).
- **Planar projection** (`_planar_project`): each substep zero `vz=wx=wy` (→ z/roll/pitch do not drift). Both are planar motions, so this projection is **matching**, not a gap. 🔴 Measured lesson: never use `resetBasePositionAndOrientation` (it zeros the velocity and kills the motion).

### Matching fidelity + residual differences (honest)
- **Step 1 free-flight validation (no contact)**: straight-line accelerate + coast, 2D vs PyBullet —
  **energy |ΔE| max=0.0143 J (~1.3% of the 1.10 J peak KE)**; **position diff max=0.023 m**. → ✅ clean match, **used as the FP floor**.
- **Residual differences (flagged)**: ① different integration schemes (2D semi-implicit N_SUB=5/h=0.02 vs PyBullet 1/240) → ~0.01 J / 0.02 m free-flight floor (characterized); ② the 2D twin is a **nonholonomic unicycle** (velocity always along the heading), whereas a free PyBullet body **can develop lateral velocity at contact** — this is **itself part of the contact-fidelity gap** (intended, not contamination); ③ control is **straight-line / constant forward force**, keeping both sides' velocity along the heading in free flight → avoids the unicycle-vs-free-body drift difference during turning (narrowing the gap to contact).

**The only substantive gap = contact physics**: the 2D wall model (`embodied_env.py:706` `v_act*=BOUNCE` scalar speed ×0.5 + pushout, **no vector reflection / friction / contact manifold**) vs PyBullet's true contact (vector restitution + Coulomb friction + contact manifold).

---

## 1. Energy contrast (PRIMARY) — the core comparison

The same constant-force control sequence (B-mode `[0.8,0.8]`, steady-state v_ss=1.2 < `V_PHYS_MAX_B`=1.65 → EC3 does not false-flag) drives both sides; the robot accelerates and hits a wall. **Force is kept on at contact → head-on = pressing the wall, glancing = sustained sliding along the wall.**
Cross-fidelity energy-oracle threshold = **5×free-flight floor = 5×0.0143 = 0.0715 J**.

| Scenario | first contact step | pre-contact KE | **2D twin self-check EC1–EC5** | free-flight FP floor | **contact energy divergence** | cross-fidelity oracle | end E_2d / E_pb |
|---|---|---|---|---|---|---|---|
| head_on (0°) | 19 | 0.715 J | **🟢 all PASS** | 9.16e-3 J | **0.179 J** | **🔴 FLAG** | 0.032 / **0.000** J |
| glancing_45 (45°) | 25 | 0.719 J | **🟢 all PASS** | 9.16e-3 J | **0.105 J** | **🔴 FLAG** | 0.032 / 0.081 J |

**🔴 Core comparison established**: in both scenarios the **2D twin passes all of its internal physics oracles EC1–EC5** (the ledger `ΔE=W_act−D_damp−E_contact_decl` is self-consistent, `energy_audit.py:57`), **yet both are caught by the cross-fidelity energy oracle as diverging from PyBullet reality**. → **a twin that passes every self-check is still diverging from reality**.

### head_on step by step (most striking: same position, energy diverges)
In free flight (steps 15–18) the two energies agree (|ΔE|~1e-3) and positions are in sync (x_2d≈x_pb). At contact (step 19, **both at x=1.800 m**):
- **PyBullet's true E_pb drops to zero immediately** (hits the wall, the constant force is balanced by the normal contact force → at rest);
- **the 2D twin's ledger E_2d=0.179 J → decays to a residual 0.032 J**: its `v*=0.5` model + the constant force **re-injects** the halved speed each step → the twin "thinks" it is still **jittering against the wall carrying kinetic energy**. **At the same position (x=1.80 m), the twin hallucinates 0.032–0.18 J of KE while reality is at rest (0 J).**

---

## 2. FP characterization and threshold (linchpin)

- **Free flight (no contact) = FP floor**: measured **9.16e-3 J** (the pre-contact segment within the scenario) / 0.0143 J (independent validation) — both sides match cleanly.
- **Contact divergence**: 0.10–0.30 J, **10–30× the floor**.
- **Threshold 5×floor = 0.0715 J separates cleanly**: contact divergence (≥0.105) ≫ threshold ≫ free-flight floor (≤0.014), **>7× margin**, no false positives.
- **True-negative confirmation**: at 75° incidence the robot **does not reach the wall** within 50 steps (glancing, slow toward the wall) → **no contact → divergence=0 → correctly not FLAGged** (not a miss).

---

## 3. Trajectory divergence (SECONDARY, reducible) + divergence envelope (SECONDARY)

- **Trajectory divergence**: post-contact trajectory diff — **glancing_45 = 0.313 m** (PyBullet slides along the wall, the 2D scalar halving, emergent divergence); **head_on = 0.000 m** (both pin at the wall face x=1.80 m → **positions reconverge, only energy diverges** — head-on is a pure-energy self-deception of "same position, different energy"). This is a **reducible** divergence (large-magnitude, catchable by the contract `‖x_2d−x_pb‖` over budget); **flagged honestly as not irreducible** (irreducibility evidence stays in scenB Table 2).
- **Divergence envelope (sweep incidence angle 0–75°)**:

| Incidence angle | 0° | 15° | 30° | 45° | 60° | 75° |
|---|---|---|---|---|---|---|
| contact energy divergence (J) | 0.179 | 0.176 | 0.150 | 0.105 | **0.295** | 0.000 (no wall contact) |
| 2D twin EC1–EC5 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| cross-fidelity oracle | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🟢 (true negative) |

→ **cross-fidelity oracle FLAGs at 5/6 incidence angles** (the 6th is a true non-contact); **2D twin EC1–EC5 all PASS at 6/6**.

---

## 4. emergent vs constructed (flagged)

- **emergent**: the energy/trajectory **divergence** arises **naturally** from PyBullet's true contact physics vs the 2D simplified model — I only set the control sequence + wall geometry + friction coefficient, and **hand-pinned no divergent state/fault**. The contact itself emerges from "driving toward a wall".
- **constructed**: the **scenario** (driving toward a wall) is a constructed setup, but the **divergence within it is emergent** (not injected). Wall placement / control are scene setup, **not the kind of fault injection in §7**.

---

## 5. 🔴 Honest assessment (does the energy contrast work cleanly + complications)

**The energy contrast works cleanly**: the core comparison holds, the FP floor is clean and separable, the divergence is emergent. But there are the following complications, reported honestly:

1. **The "sign" of the divergence varies with geometry** (not simply "the twin lacks friction → the twin has lower energy"):
   - head_on: the twin **over-reports** (hallucinated 0.032 vs reality 0); glancing: the twin **under-reports** (0.032 vs reality 0.081). The real gap is **the entire crude contact model** (scalar speed halving + pinning, no vector reflection/friction); "lacks friction" is only one part of it. The cross-fidelity oracle catches **|divergence|**, sign-independent — the thesis doesn't depend on the sign, but the specific phrasing "lacks friction" **should not be used alone**.
2. **The divergence is non-monotonic in incidence angle** (60° largest at 0.295, 45° smallest at 0.105): the contact dynamics vary non-trivially with geometry. Presented as-is, not smoothed.
3. **75° has no contact** (too shallow a glance, the wall isn't reached in 50 steps): a true negative, not a miss. Extending the step count would make it contact; this experiment keeps it to show "no contact → no false alarm".
4. **The two trajectories diverge after contact**, so the contact-segment energy comparison conflates "ledger error + trajectory difference" — both stem from the same contact-fidelity gap; the free-flight floor proves the pre-contact match.
5. **Matching residuals** (integration scheme, unicycle-vs-free-body) leave a ~0.01 J / 0.02 m free-flight floor — explicitly characterized and used as the threshold baseline, not hidden.

**Overall**: the energy contrast holds and is clean — **a simplified twin passes all of its internal EC1–EC5, yet its energy ledger emergently diverges from PyBullet reality at contact, caught by the cross-fidelity oracle.** This directly answers the "2D toy platform / hand-injected fault" criticism: the fault (divergence) emerges from the **real fidelity gap** between an **independent third-party high-fidelity engine** and the simplified model, not from our design. **No over-claim of irreducibility** (the trajectory divergence is reducible, flagged honestly; irreducibility evidence stays in scenB Table 2).

---

## 6. Out of scope / TODO (INV-C)
- A **per-quantity decomposition of true friction dissipation** during sustained sliding (via `getContactPoints` friction impulses) not broken out; this experiment uses total mechanical-energy divergence (sufficient).
- Integrating the cross-fidelity oracle into the resident `audit_suite` (this experiment is a standalone harness).
- A trained policy replacing scripted actions (D5, bonus): `train_agent.py` + the existing `ppo_embodied_agent_maze.pth` are available, not on the critical path, not done.
- An opportunistic relational demo (Step 5): the trajectory divergence / divergence envelope did not naturally produce a "small-metric, topological" case → **no scene engineering done** (per discipline).
