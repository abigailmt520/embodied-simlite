# Phase4 details: dual-state coupling stress test (report×physics) — stress-testing the core claim itself with anti-self-deception discipline

> Branch `dev/stage5-coupling` (off Phase3 `af81e14`). master `7b54625` frozen, zero changes; all existing `.pth` not overwritten.
> Numbers from real runs (INV-E); with line numbers (INV-B). 🔴 The supreme discipline of this phase: **honestly judge whether the coupling truly exists, never tune parameters to force it.**

---

## 0. Claim and the most important honest criterion
**Goal**: construct a self-deception such that (a) the contract layer run alone = pass, (b) the physics layer run alone = pass, but (c) the report×physics joint = catch. Only this proves "dual-state coupling", as opposed to two side-by-side audits.

**🔴 Distinguishing true coupling from false coupling (the core contribution of this phase)**: the simulation has two position signals — `truth` (physical ground truth) and `odom` (the contract report). Whether a wall-crossing self-deception is **truly coupled** is judged by `truth_vs_map`:
- **`truth_vs_map` RED** (the truth really crosses the wall) → catchable **within physics alone** (truth vs map, both on the physics side) → **not true coupling** (merely an EC5 implementation gap).
- **`truth_vs_map` GREEN and `odom_vs_map` RED** (truth is legitimate, only the report is illegitimate) → catchable neither within physics nor by contract self-consistency, **only by the report×physics joint** → **true coupling**.

Formalized criterion (`joint_audit.coupling_verdict` `:60`): `truth_vs_map.ok ∧ ¬odom_vs_map.ok ⟺ TRUE_COUPLING`.

---

## 1. Implementation
- **Phantom-wall physics fault** (`embodied_env._resolve_wall_collisions:691-693`): `physics_fault["phantom_walls"]=[idx]` makes the collision-detection layer **drop the specified wall** → the robot passes through "legitimately" (no collision, momentum conserved, energy self-consistent, ledger penetration still 0).
- **Joint audit** (`joint_audit.py`): `traj_vs_map` (`:37`) checks per-frame whether the robot circle falls inside a claimed wall AABB; `coupling_verdict` (`:60`) returns `truth_vs_map / odom_vs_map / verdict∈{TRUE_COUPLING, PHYSICS_INTERNAL, NO_VIOLATION}`.
- **Stress test** (`run_coupling_test.py`): each scenario runs physics EC1-EC5 + contract C1/C2/C3+C_I + joint truth/odom_vs_map.

---

## 2. Scenario A · literal "phantom wall" (the truth really crosses the wall) — 🔴 judged honestly: **not true coupling**
`scenario_a` (`:88`): in the maze, drop the death-corridor left wall (idx4); the robot drives straight through at [v=1]. **Real run result**:
| Layer | Result |
|---|---|
| physics EC1-EC5 | 🟢 all pass (no collision → no energy anomaly, ledger penetration=0) |
| contract C1/C2/C3+C_I | 🟢 all pass (odom is the truth's honest dead reckoning) |
| joint truth_vs_map | **🔴 truth crosses the wall** |
| joint odom_vs_map | 🔴 report crosses the wall |
| **verdict** | **PHYSICS_INTERNAL** |

**🔴 Honest conclusion**: a literal phantom wall is **not true coupling**. `truth_vs_map` RED = the truth really crosses → catchable **within physics alone** (truth vs claimed map), no report channel needed. The current EC1-EC5 miss it **only because EC5 trusts the ledger's penetration field** (the phantom wall is not in the collision system → reports 0); this is an **EC5 implementation gap** (same family as the linchpin trust-root blind spot), not "must rely on coupling of two layers". A complete EC5′ (independently recompute penetration from truth vs the claimed full map) would catch it within physics. **No parameters were tuned to manufacture coupling — judged honestly as PHYSICS_INTERNAL.**

---

## 3. Scenario B · the true-coupling variant (truth honest, odom fabricates the crossing) — ✅ **true coupling**
`scenario_b` (`:104`): the truth runs honestly (blocked by wall idx4, legitimately bounces), but odom is fabricated as a constant-velocity straight wall-crossing track (internally self-consistent dead reckoning). **Real run result**:
| Layer | Result |
|---|---|
| physics EC1-EC5 | 🟢 all pass (honest truth dynamics) |
| contract C1/C2/C3+C_I | 🟢 all pass (the fabricated odom is constant-velocity self-consistent: C1 large error → not suspected of leakage, C_I low MI, C2/C3 normal) |
| joint truth_vs_map | **🟢 truth legitimate** |
| joint odom_vs_map | **🔴 report crosses the wall** |
| **verdict** | **TRUE_COUPLING** |

**Conclusion**: true coupling **is constructible**. Truth legitimate (within-physics truth_vs_map green) + physics passes + contract passes, **only `odom_vs_map` (report×physics) RED** → catchable neither within physics (truth legitimate) nor by contract self-consistency (odom self-consistent), **only by the joint check of the reported track against the claimed physical map**. This is the existence proof of dual-state coupling.

---

## 4. Core findings (for the paper)
1. **A true-coupling self-deception of the "each layer alone passes, only the joint catches" kind really exists** (Scenario B) — dual-state coupling is non-empty, the claim holds.
2. **But the essence of the coupling is "the self-deception lands on report, while the physics truth is honest"**: the criterion = `truth_vs_map green ∧ odom_vs_map red`. If the truth itself violates physics (truth_vs_map red), it is catchable within physics and does not constitute true coupling (Scenario A).
3. **The value of the counter-example**: the naive "phantom wall" (truth crosses) **looks** like coupling (existing EC/contract both pass, only the joint is red), but that is an **EC5 implementation gap** masquerading as coupling. The honest criterion `truth_vs_map` separates the two, avoiding mistaking an "audit gap" for a "coupling argument" — this is exactly stress-testing the claim itself with anti-self-deception discipline: **the argument for true coupling must exclude pseudo-coupling that "a single layer could have caught, just wasn't implemented".**
4. **Platform implication**: the existing physics-layer EC5 trusts the ledger penetration (does not independently recompute truth vs map) → recommend adding EC5′ (within-physics truth-vs-claimed-map) to plug Scenario-A-class gaps; whereas Scenario-B-class true coupling **must** rely on the report×physics joint audit layer (the joint_audit added in this phase).

---

## 5. TODO / out of scope (INV-C)
- Wire joint_audit into the runtime audit pipeline (real-time joint monitoring); add the EC5′ within-physics truth-vs-map recomputation (plug the Scenario A gap).
- MuJoCo not connected; the contract C1-3 / physics EC1-5 existing criteria not touched; F3 dynamic obstacles not done; the q-class reconciliation not done (an independent contract-vs-kernel channel is still out of scope).
- No regression: collision CF 3/3, energy P 5/5, contract C1-3, maze env_checker all pass (env only gains a phantom_walls test hook).
