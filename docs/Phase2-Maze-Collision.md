# Phase2 details: F4 rectangular collisions + F2 maze (richer environment layer, design + validation)

> Branch `dev/stage3-maze` (off Phase1c `5e2af1c`). master `7b54625` frozen, zero changes; the paper/dyn/bmode/bmode_gamed/g2gamed.pth all not overwritten.
> Numbers from real runs (INV-E); with line numbers (INV-B). 🔴 Physics/collision/lidar are all backend; the frontend stays pure-observation, unchanged (INV-D).

---

## 1. Milestone 1 · F4 rectangular collisions + F2 maze + reward redesign

### 1.1 Backend map abstraction (embodied_env.py)
- `map_type` (`:149-163`): `random_circle` (default, 10×10 random circles, **Phase0-1c behavior byte-preserved, zero regression**) / `maze` (40×40 hand-built wall maze).
- `MAZE_WALLS` (`:109-129`): **19 AABB segments** (4 outer walls + a death corridor / chaos maze / U-shaped deadlock valley / extreme narrow slit, porting old:616-640's named scenes into [0,40]).
- Instantiates arena size / wall list / collision semantics; `random_circle` has no inner walls, `maze` has 19 walls.

### 1.2 F4 rectangular collisions (all backend, reusing the old algorithm)
- **Ray-AABB lidar** (`_ray_aabbs` `:607`, the slab method vectorized over rays): used by maze (including outer walls); random_circle still uses `_ray_walls` (arena boundary). 🔴 All analytic backend, not the original frontend raycasting.
- **Circle-AABB collision + penetration pushout** (`_circle_aabb_overlap` `:641` + `_resolve_wall_collisions` `:668`): compute penetration via the AABB nearest point, push out along the normal (when the center is inside the box, along the minimum-penetration axis, old:649-655); **2 iterations to resolve corners** (old:645).
- **Bounce** (`BOUNCE=0.5` `:106`): after hitting a wall `v_act,w_act ← e·(·)`, KE drops to e², **E_contact=(1−e²)·KE_before ≥ 0 dissipative** (replicating the old `v*=0.5` decay, not a true 2D reflection — consistent with the unicycle scalar model).
- **odom does no collision correction** (preserving the audit philosophy, not reviving the implicit correction at old:744).
- Measured (driving straight into an outer wall): the robot is pinned at x=38.8 (=39−radius, exact pushout), v_act ×0.5 after impact, **residual penetration=0**, E_contact_act==E_contact_decl=0.24 (≥0), energy residual ~1e-16 J.

### 1.3 🔴 Collision semantics + reward redesign
- `collision_mode`: `terminate` (paper version, terminate on impact + R_COLLISION=−200, random_circle default) / `bounce` (maze default).
- **Bounce redesign** (`_compute_reward` `:523`): hitting a wall **does not terminate** (already pushed out + bounced), instead **R_CONTACT=−5 per contact step** (`:107`); only reaching the goal terminates, timeout truncates.
- **Effect on R_COLLISION/training**: from a large −200 terminal penalty → a small −5 per-step contact penalty; episodes are **longer** (no sudden death, average 402 steps, 52% timeout), and the agent learns to **navigate with real collisions** (average 3.1 contact steps — bounce and continue after a wall hit rather than dying), which is exactly the goal of the richer environment layer.

### 1.4 Retrain the maze policy (ppo_embodied_agent_maze.pth)
- A-mode (target velocity is easier to learn for maze navigation), maze, bounce, 1.5M steps (~2900 fps, slightly slower with 19-wall lidar/collision).
- **N=25 fixed-seed evaluation**: success **48%** (12/25) / timeout **52%** / average 402 steps / average 3.1 contact steps.
- **Honest statement**: 48% reflects the difficulty of the 40×40 **adversarial maze** (the original was designed "to defeat SLAM/Nav2": death corridor / U-shaped deadlock / narrow slit). This is an honest number for true collision physics + complex topology, not a regression; a simpler maze / more steps / B-mode could improve it, but the F4/F2 implementation itself is this phase's deliverable.

### 1.5 No-regression confirmation
- **Energy audit** (random_circle, EMBODIED_AUDIT_MODE=B): clean green (residual 4.996e-16) + **P-1..P-5 5/5 flagged RED**.
- **Contract C1/C2/C3**: Gate 1 (3/3 injections RED) + Gate 2 (healthy all green) + Gate 3 (84%) — all pass.
- `audit_session` gains a `with_collision` switch (default False) → zero impact on old calls; EC1's E_contact_decl term degenerates when there is no collision (E_contact=0) → the old P-audit is unchanged.

---

## 2. Milestone 2 · collision-fidelity fault class + audit

### 2.1 Three collision self-deception injectors (physics_injection.py, hidden inside `_resolve_wall_collisions`)
| Injection | What it changes (line) | Conservation law broken |
|---|---|---|
| **CF-1 over_bounce** | restitution e=1.3>1 (`_resolve_wall_collisions` reads `bounce_eff` `:685`) | collision energy non-negativity (collision creates energy from nothing) |
| **CF-2 skip_pushout** | skip the penetration pushout (robot stays inside the wall, `skip_pushout` `:686`) | non-penetration invariant |
| **CF-3 phantom_contact** | the ledger claims collision dissipation but the velocity is not actually decayed (`phantom_contact` `:687`) | energy-ledger self-consistency (a collision-flavored Potemkin village) |

### 2.2 New audit checks (energy_audit.py)
- **EC1 extended**: budget residual `r = ΔE − (W_act − D_damp − E_contact_decl)` (including the declared collision dissipation; degenerates to the original form without collision).
- **EC4 COLLISION_NONNEG**: `E_contact_act ≥ 0` (a collision only dissipates, never creates energy; any single-frame creation is flagged).
- **EC5 NON_PENETRATION**: after resolution `penetration ≤ 1e-4 m` (no overlap with the wall after pushout).

### 2.3 Validation (real numbers, `audit/run_collision_audit.py`, evidence figure `collision_redgreen_matrix.png`)
**Gate C2 clean**: EC1-EC5 all green, 18 collision frames, energy residual **1.943e-16 J**, residual penetration **0 m**.
**Gate C1 three injections 3/3 flagged RED and located** (each caught by its characteristic check):
| Injection | EC1 | EC4 | EC5 | catch | locator |
|---|:---:|:---:|:---:|:---:|---|
| clean | 🟢 | 🟢 | 🟢 | — | residual 1.9e-16 |
| CF-1 over-bounce | 🔴 | 🔴 | 🟢 | ✅ | E_contact_act=−0.221 J @step23 |
| CF-2 no pushout | 🟢 | 🟢 | 🔴 | ✅ | penetration=0.698 m @step34 |
| CF-3 phantom dissipation | 🔴 | 🟢 | 🟢 | ✅ | residual 0.240 J @step23 |

- Each injection has a **specific signature**: CF-1→EC4 (collision creates energy), CF-2→EC5 (residual penetration), CF-3→EC1 (claimed dissipation with no real loss). No misses, no false positives.

### 2.4 Opportunistic observation: natural gaming in the maze+collision env (not forced)
- Running the honest maze agent's collision-containing trajectory through the energy+collision audit → **🟢GREEN**: no collision-energy gaming naturally appeared.
- Analysis: an honest bounce (e≤1) **only dissipates**, with no exploitable collision-energy loophole — without injection (CF-1 etc.) there is no sweet-spot exploit. Consistent with G-1/G-2: the audit flags RED **when a violation occurs**, and honest (no violation) is green. Continuing Phase1c's "discoverability-specificity tradeoff": the collision scenario did not naturally provide an aligned sweet-spot exploit.

---

## 3. TODO / out of scope (INV-C)
- Not done: F3 dynamic obstacles (next task); MuJoCo not connected (self-built lightweight enrichment); the contract-layer C1-3 code not touched; the q-class reconciliation not done; no forced collision-gaming bridge.
- Frontend rendering of the maze walls (get_render_state already additively exports a `walls` field that the frontend can consume; the frontend is still pure-observation, the frontend code was not changed in this phase).
- Improving the maze success rate: a simpler maze / curriculum learning / a B-mode force-control maze policy — deferred.
