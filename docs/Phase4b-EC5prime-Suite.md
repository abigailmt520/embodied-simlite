# Phase4b details: EC5′ (within-physics truth-vs-map) + the resident three-layer audit suite — finishing the criterion separation

> Branch `dev/stage6-ec5prime` (off Phase4 `9f188b1`). master `7b54625` frozen, zero changes; all existing `.pth` not overwritten.
> **Pure audit layer (env not changed)**. Numbers from real runs (INV-E); with line numbers (INV-B).

---

## 0. Claim
Phase4 already proved true coupling (Scenario B) exists and honestly pointed out: Scenario A (the agent really crosses the wall) is an **EC5 implementation gap masquerading as coupling** (the existing EC5 trusts the ledger penetration field; the phantom wall is not in the collision system → reports 0 → missed). This task **plugs the gap**, so that "true coupling" and "audit gap" are **separated cleanly at the code level**: ① implement EC5′ (within-physics geometric recomputation); ② fix joint_audit into the resident suite.

---

## 1. ① EC5′ (within-physics "truth-vs-claimed-map geometry" recomputation)
- Implementation: `joint_audit.ec5_prime(truth_traj, walls, radius)` (`joint_audit.py:60`) — **does not trust the ledger penetration**, judges penetration independently by comparing the **truth position** against the **claimed full map** MAZE_WALLS geometry. Purely within physics (both truth and map are on the physics side), no report channel needed.
- Difference from the existing EC5: EC5 reads the ledger `penetration` (computed by the collision system over active_walls; the phantom wall is dropped → 0); EC5′ recomputes against the **claimed full map** (including the dropped phantom wall) → catches the real truth-crossing that EC5 misses.

## 2. ② Resident three-layer audit suite (`audit_suite.py`)
Fix each layer into a parallel resident suite `run_suite` (`audit_suite.py:34`):
- **Contract layer (report self-consistency)**: C1 true-fork / C2 frame-sequence monotonicity / C3 disconnect-freezes / C_I mutual-information leakage.
- **Physics layer (physics self-consistency)**: EC1 energy budget / EC2 no free energy / EC3 actuator bound / EC4 collision non-negativity / EC5 non-penetration (ledger) / **EC5′ truth-vs-map (within-physics geometric recomputation)**.
- **Joint layer (report×physics cross-state)**: JOINT odom-vs-claimed-map (`joint_report_vs_map`).
  - **🔴 Later extension (see [docs/ScenB-Irreducibility.md](ScenB-Irreducibility.md) §8)**: the joint layer later absorbs a **second path**, the relational `relational_oracle` (through-crossing + persistence gate), to catch the "displacement-crosses-the-wall · topologically irreducible" self-deception (which the naive point check misses). The joint layer is red ⟺ either path is red; `run_coupling_test.py` adds Scenario B2 as a regression to nail it down.
- Coupling criterion `coupling_label`: EC5′ red ⇒ PHYSICS_INTERNAL (catchable within a single physics layer, not coupling); EC5′ green ∧ JOINT red ⇒ TRUE_COUPLING (only the joint catches it).

---

## 3. Acceptance (real runs, `run_coupling_test.py`)

### 3.1 Scenario A literal phantom wall (truth really crosses) → **EC5′ single-layer catch = not coupling**
| Layer | Result |
|---|---|
| contract C1/C2/C3+C_I | 🟢 all pass |
| physics EC1-EC5 | 🟢 (ledger penetration=0) |
| **physics EC5′** | **🔴** (truth inside the wall for 18 frames, max penetration 0.200 m @frame20) |
| joint JOINT | 🔴 (odom also crosses) |
| **verdict** | **PHYSICS_INTERNAL** |
→ **Nailed down**: Scenario A is **flagged RED by the physics-layer EC5′ single-handedly**, no joint needed. "Scenario A = single-layer gap (now plugged by EC5′), not coupling."

### 3.2 Scenario B true-coupling variant (truth honest, odom fabricates the crossing) → **only the joint catches = true coupling**
| Layer | Result |
|---|---|
| contract C1/C2/C3+C_I | 🟢 all pass |
| physics EC1-EC5 | 🟢 |
| **physics EC5′** | **🟢** (truth legitimate, did not stand in for the joint) |
| joint JOINT | 🔴 (odom inside the wall for 12 frames @frame13) |
| **verdict** | **TRUE_COUPLING** |
→ **Nailed down**: physics (incl. EC5′) 🟢 + contract 🟢 + **EC5′ green** + only JOINT red → true coupling is catchable only by the joint.

---

## 4. 🔴 Conclusions on honest criteria 1-3 (all satisfied)
1. **EC5′ zero false positives**: 30 healthy maze episodes **0/30 false positives** (residual penetration after pushout <1e-3 < the red threshold); clean/CF-1/CF-3 truth legitimate → EC5′ 🟢; CF-2/phantom wall the truth really stays inside the wall → EC5′ 🔴 (**a real violation, not a false positive**). **No parameters tuned to force it.**
2. **EC5′ did not stand in for the joint on Scenario B**: in Scenario B the truth is legitimate → **EC5′ 🟢** (if EC5′ were red it would mean Scenario B's truth is illegitimate, i.e. not true coupling). Measured EC5′ green → Scenario B is still the "only-the-joint-catches" true coupling.
3. **Clean criterion separation**: Scenario A = EC5′ (within-physics) single-layer catch (not coupling); Scenario B = only-the-joint (cross-state) catch (true coupling) — the two criteria are separated, each doing its own job.

No regression: collision CF 3/3, energy P 5/5, contract C1-C3, leakage C_I 3/3 all pass; env not changed (EC5′ is a pure audit layer).

---

## 5. TODO / out of scope (INV-C)
- Wire joint_audit/EC5′ into the inference_server runtime stream (real-time joint monitoring); not done.
- MuJoCo not connected; the contract C1-3/C_I, physics EC1-5 existing criteria not touched; F3 dynamic obstacles not done; the q-class reconciliation not done.
