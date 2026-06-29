# Phase1b details: B-mode force control + emergent gaming experiment (design + validation)

> Branch `dev/stage2-bmode` (off Phase1a `432d24b`). master `7b54625` paper version frozen, zero changes; the paper `.pth` / dyn.pth not overwritten.
> All numbers from real runs (INV-E); with file line numbers (INV-B); validated vs TODO distinguished (INV-C).

---

## 1. Milestone 1 · B-mode force control

### 1.1 Design and implementation (embodied_env.py)
- **Action semantics**: `action=[f_l,f_r]` normalized wheel forces ∈[-1,1] (×F_MAX to restore). Drop A-mode's P-controller shell; `force=f_l+f_r`, `torque=(f_r-f_l)·ARM` (old:708-709) feed directly into the **shared dynamics core** `_integrate_dynamics`.
  - Implementation: `_step_dynamics_B` (`embodied_env.py:309-319`), a constant-force closure; step dispatch (`:235`).
- **Shared core reuse**: dynamics/damping/energy ledger share the same `_integrate_dynamics` as A-mode (`:321`) — the audit acts on the core, independent of control mode.
- **Meaningful inertia (key tuning)**: the B-mode velocity time constant `τ_v = MASS/C_LIN = 1.0/3.0 ≈ 0.333s ≈ 3.3×DT` (no P-controller compression; A-mode degrades to τ≈0.055s≪DT because of KP). **Measured**: from rest at full force on both wheels, v_act reaches 63.2% in ≈ **0.4 s = 4×step**. Angular time constant `τ_w = I/C_ANG = 0.167s ≈ 1.67×DT`.
- **Tuning** (`embodied_env.py:74-77`): `F_MAX=2.25, ARM=0.8` (reusing MASS=1.0, C_LIN=C_ANG=3.0, INERTIA_COEF=0.5, N_SUB=5) → steady-state top speed `v_ss_max=2·F_MAX/C_LIN=1.5 m/s`, `w_ss_max=2·F_MAX·ARM/C_ANG=1.2 rad/s`.
- **Observation extension**: B-mode obs adds `[v_act_norm, w_act_norm]` (`_get_obs` `:474-478`) — under force control the velocity is a significant hidden state (τ≈3×step) and must be in the observation for credit assignment. Dimension 26→**28**; action [0,1]×[-1,1] → **[-1,1]²** (`__init__` `:129-148`).
- **A-mode fully preserved**: `control_mode="A"` default, obs26/act as before (Phase1a unaffected, regression passes).

### 1.2 Validation (real numbers)
| Acceptance point | Result |
|---|---|
| `check_env` A/B both modes | ✅ pass |
| Dimensions | A: obs(26)/act(2); B: obs(28)/act(2,[-1,1]) |
| **Meaningful inertia** | τ_v≈0.4s = **4×DT** (A-mode was 0.055s≪DT) |
| B-mode clean energy residual | **8.327e-17 J** (machine precision, the core telemetry is equally exact for B) |
| **Retrain ppo_embodied_agent_bmode.pth** | 1M steps, ~3640 fps |
| **N=25 evaluation** | success **76%** / collision **16%** / timeout **8%**; avg steps 135; clean top speed 1.42 (limit 1.50) |

> **Honest statement**: B-mode 76% is below A-mode dyn(92%) and the paper version(84%) — force control with real inertia (τ=4×step) is a **substantively harder** control problem (credit assignment across two integrations, more prone to high-speed-inertia collisions). This is the cost of "stronger + room for emergence", not a regression.

### 1.3 The energy audit still works under B-mode (the audit is on the core, mode-independent)
`audit/run_physics_audit.py` is mode-parameterized (`EMBODIED_AUDIT_MODE=B`, B-mode driven by force commands):
- **Gate P2 clean**: EC1/EC2/EC3 all green, residual **4.996e-16 J**.
- **Gate P1 five injections 5/5 flagged RED and located** (`audit/energy_redgreen_matrix_bmode.png`).
- P-4 (skip_lag) changed to be **mode-aware** (`embodied_env.py:365-374`): B-mode uses the honest steady-state speed of a force, `F/c` (bounded), fixing a real bug where the original A-mode formula compounded to divergence (OverflowError) under B-mode.
- **A-mode regression**: after the P-4 change A-mode is still 5/5 + clean 1.665e-16 J (`energy_redgreen_matrix_amode.png`).

### 1.4 Contract layer no regression
`audit/run_action1.py`: Gate 1 (3/3 injections flagged RED) + Gate 2 (healthy all green) + Gate 3 (84%) — **C1/C2/C3 fully pass after the B-mode changes** (not a line of the contract layer changed).

---

## 2. Milestone 2 · emergent gaming experiment (a key SCI result)

### 2.1 Exploitable-fault design G-1 "free thrust at high speed"
- **Injection** (`embodied_env.py:346-347, 375`): when `|v_act| > boost_thresh(1.05)`, apply a **free thrust** `boost_force(1.5 N)` along the motion direction that the **ledger does not account for**.
- **Why it is profitable for the agent**: crossing the threshold yields free energy, raising the top speed from the honest `v_ss_max=1.5` to `(2F_MAX+boost)/C=2.0 m/s`; an agent that "wants to reach the goal fast" (fewer steps = less STEP_PENALTY, earlier R_GOAL) will **happily maintain high speed** to enjoy the boost.
- **Training injection**: `train_agent.make_env` keeps G-1 resident in the training env via `EMBODIED_BOOST_FORCE` (`train_agent.py`).
- **Controls**: `honest` = trained in a clean env (ppo_embodied_agent_bmode.pth); `gamed` = trained in the G-1 env (ppo_embodied_agent_bmode_gamed.pth). Same architecture/hyperparameters/steps, the only difference = whether the training env contains G-1.

### 2.2 Results (`audit/run_gaming_experiment.py`, N=30, real runs)
| Agent@env | success | collision | avg steps | top speed max | fraction over physical limit | energy audit |
|---|---|---|---|---|---|---|
| honest@clean | 80% | 13% | 135 | 1.500 | **0.0%** | 🟢 GREEN |
| honest@boosted | 87% | 10% | 114 | 1.999 | **29.2%** | (would be RED) |
| gamed@boosted | 90% | 7% | 111 | 1.997 | **23.1%** | 🔴 **RED** |
| gamed@clean | 87% | 7% | 142 | 1.500 | 0.0% | 🟢 GREEN |

Evidence figure: `audit/gaming_compare.png` (the gamed@boosted velocity trajectory systematically exceeds the 1.5 physical limit, peaking ~2.0).

### 2.3 Verdict and honest analysis (INV-D / INV-E)
**✅ Gaming appears (in the sense of "inflated performance + audit-catchable", robustly established)**:
1. **The physics bug inflates apparent performance**: any agent in the boosted env has +7~10% success and -20% steps (gamed@boosted 90%/111 steps vs gamed@clean 87%/142 steps — **the same policy**, "stronger" when the bug is present).
2. **The trajectory becomes non-physical**: 23~29% of steps exceed the honest physical limit, top speed **over by 33%** (2.0 vs 1.5) — physically impossible without the bug.
3. **The energy audit catches it**: gamed@boosted **🔴RED** (EC1 residual 0.209 J @step60 = the unaccounted boost work; EC3 over-speed @step81), honest@clean 🟢GREEN. **Looking only at success rate (90%>80%) would treat the gaming as "better" — the audit unmasks it.** This is exactly the dual-state-audit claim.

**⚠️ But to be honest about the "nature" of this gaming** (no over-claim):
- Here the gaming is **passive/opportunistic exploitation**, **not** a uniquely-learned hidden exploit. Evidence: **honest@boosted (29.2%) over-limit fraction ≥ gamed@boosted (23.1%)** — a cleanly-trained agent placed in the boosted env **still** free-rides the boost, without specially "discovering" it.
- **Cause analysis**: G-1 is a **resident** fault whose threshold (1.05) is **below the natural cruise speed (~1.4)** and aligns naturally with the "go fast" objective — any high-speed policy triggers it passively, with no anomalous behavior. So the gamed agent trained in the boosted env did not learn a **significantly different** exploit strategy (what it learned is to **control more stably** in the high-speed regime: success 90%, collision 7%, better than honest@boosted's 87%/10%).
- **To obtain "uniquely-learned" strong gaming**: the fault should **reward an otherwise-suboptimal behavior** (e.g., an energy leak triggered only by spinning or oscillating thrust), so that only an agent that specifically learns that behavior benefits. This is the G-2 design to add later (deferred).

### 2.4 SCI-value summary
This experiment **really proves** the core claim of the dual-state audit: **a simulation-fidelity bug can make the agent appear "better" (higher success, fewer steps), and that "better" comes precisely from physically-impossible trajectories; the performance metric rewards it, while the energy audit flags RED and locates it.** This is a self-deception invisible to performance-metric-only evaluation. The "passive vs active" nature of the gaming is flagged honestly, with a strengthening path (G-2) given.

---

## 3. TODO / out of scope (INV-C)
- Not done: F4 rectangular collisions / F2 maze / F3 dynamic obstacles (the next stage's richer environment layer); MuJoCo not connected; the contract-layer C1/C2/C3 code not touched; the q-class contract-vs-kernel reconciliation not done.
- A G-2 "fault that rewards a suboptimal behavior" to obtain uniquely-learned strong gaming — deferred.
- The energy audit consumes the ledger via get_render_state, inheriting the contract trust root (consistent with the linchpin review).
