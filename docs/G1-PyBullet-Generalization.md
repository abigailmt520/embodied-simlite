# G1 details: Real-engine (PyBullet) generalization — a cross-engine honest report

> Branch `dev/g1-pybullet` (off Phase4b `57dbca8`). The 2D platform master `7b54625` is frozen, zero changes; the paper and all `.pth` byte-unchanged.
> **The audit logic is ported/reused; the 2D platform itself is unchanged**; the PyBullet code is entirely in `g1_pybullet/`, in a separate conda env `g1-pybullet`.
> Numbers from real runs (INV-E); with file line numbers (INV-B). 🔴 Throughout, only **inducing conditions** are set; a faulty state is never hand-set.

---

## 0. Claim and the non-circularity core
The deepest objection: is the audit suite only effective on "our own toy + our own hand-set faults", or can it catch **the engine's own real pathologies inside an engine we did not build**?
- ✅ **Strong test (non-circular, crown jewel)**: trigger PyBullet's own native numerical pathologies (high-speed tunneling / energy injection) and test whether the **physics audit** catches them.
- ✅ **Support (porting)**: port the contract layer C1-C3 / joint to a 3D real-engine state space, test with our report injectors, and show they still work.

---

## 1. Feasibility (honest record)
- On this machine (Python 3.13 / macOS-26 arm64) PyBullet has **no prebuilt wheel and source compilation fails** (clang hits the new SDK header `_stdio.h:322`).
- conda-forge has a prebuilt `pybullet-3.25 py313 osx-arm64`; installing it in the base env conflicts with `requests` → **created a separate env `g1-pybullet`** (leaving base untouched).
- Run: `conda run -n g1-pybullet python g1_pybullet/g1a_baseline.py` (b/c likewise).

---

## 2. G1a healthy baseline + noise floor (checkpoint 1 ✅ clean)
- **Engine energy noise floor**: a frictionless free projectile (a conservative system) has a per-step |ΔE| ≈ **8.3e-4 J/step** (systematic, std~1e-15); cumulative drift 0.48%/355 steps. **The audit threshold must be > this floor** (take 10× = 1e-2 J/step) — analogous to the Phase-3 KSG noise-budget bound.
- **🔴 Porting lesson**: PyBullet's **default linear damping of 0.04 silently drains energy** (with it on, a free projectile loses -24.6% energy) → the energy budget must explicitly subtract the engine damping, or disable damping when measuring conservation.
- **🔴 Debugging lesson (honest)**: the first non-penetration baseline **false-positived** — I forgot to add a ground plane → the ball fell through the floor (z→-8), and EC5′ uses a 2D (x,y) projection that ignores z → misjudged "inside a wall". Fix: add a ground plane to keep **planar motion** (z∈[0.099,0.104] constant), so EC5′'s 2D projection is valid (the platform's v_z≈0 invariant). **This was my own setup error, not an engine pathology, recorded honestly.**
- **Result**: after the fix, the healthy baseline has **zero false positives** — energy audit 🟢 (floor 1e-2), non-penetration 🟢 (the ball is stopped by the wall at x=1.85, engine penetration ≈0).

---

## 3. G1b high-speed tunneling (checkpoint 2 ✅ non-circular, crown jewel)
Inducing conditions: a thin wall (half-thickness 0.02 m) + high speed + default no CCD (no hand-set fault). `g1b_tunneling.py`, dt=1/120 s:
| Speed | per-step displacement | passes through? | **engine-reported penetration** | EC5′ point check | **EC5′ swept** |
|---|---|---|---|---|---|
| 50 m/s | 0.42 m | yes | 0.042 (partially detected) | 🔴 | 🔴 |
| **200 m/s** | 1.67 m | yes | **0.00 (completely missed)** | **🟢 miss** | **🔴 catch** |
| 400 m/s | 3.33 m | yes | **0.00** | 🟢 miss | 🔴 catch |

**🔴 Core conclusion (non-circular)**: at 200 m/s the ball **genuinely passes through the wall** and **PyBullet reports penetration=0** (the engine's discrete collision **completely misses it** — this is the engine's real numerical pathology, not hand-set by us). **A per-frame point check also misses** (the ball jumps over the whole wall in one step, no frame is inside the wall, same cause as the engine miss = discrete sampling); **only the swept EC5′ (segment-vs-wall-core) flags it RED**. That is: **our audit logic caught a real pathology of an engine we did not build.**
- **Porting finding**: the 2D platform's EC5′ is a per-frame point check (adequate at the 2D low speeds); a real engine's high-speed tunneling **requires swept-segment detection** (`pb_helpers.ec5prime_swept:188`). This is a genuine refinement discovered during generalization.

---

## 4. G1c other pathologies + contract layer + joint
`g1c_pathologies.py`:
- **Pathology · energy injection (non-circular)**: an elastic bounce (e=1, a conservative system), and PyBullet's discrete contact solver **injects energy** — E0=19.6→max=42.7 J (dt=1/120), worsening with the step (dt=1/30 gives +1178%, max=250.5 J, mechanical energy = KE+PE). **The energy audit (conservation upper bound, `energy_conservation_check:146`) catches it RED** (over budget by 22.7 J). Non-circular: e=1 should conserve, and the injection is the engine's.
- **Pathology · warm-start ghost force (honest negative/uncertain)**: removing a support and testing for a lateral anomaly at the disconnect transient → **no significant anomaly observed** (|v_xy|=1.2e-5≈0). **🔴 Reported as-is**: a warm-start ghost force aliases with normal gravity fall and is **hard to isolate cleanly**; this test is not a complete diagnosis. **Not counted as a catch; honestly scoped as "not cleanly measured".**
- **Contract layer in 3D (ported, support)**: a husky differential drive, odom = true-state velocity + slip dead reckoning → C1/C2/C3 **all green on healthy (zero false positives)**; injecting odom=truth → **C1 🔴 catches**; injecting a frozen seq → **C2 🔴 catches**. **The contract audit still works in a 3D real-engine state space.**
- **joint in 3D (support)**: the true-state husky vs the declared wall, EC5′ 🟢 legitimate; a fabricated odom crossing the wall → joint 🔴 catches.

---

## 5. G1d cross-engine generalization honest report (summary)

### 5.1 Pathology catch/miss list (real results)
| Pathology (PyBullet native) | inducing condition | audit verdict | note |
|---|---|---|---|
| high-speed tunneling | thin wall + high speed + no CCD | ✅ **catch** (swept EC5′) | engine-reported penetration=0 (completely missed), point check also misses, only swept catches |
| energy injection (elastic bounce) | e=1 conservative system | ✅ **catch** (energy conservation upper bound) | over budget by 22.7 J; worsens with step |
| warm-start ghost force | remove support | ⚠️ **not cleanly measured** | aliases with gravity, hard to isolate; honest negative/uncertain |

### 5.2 Three honest criteria (INV-E)
1. **Hand-set fault / tuned to force it? No.** Tunneling/energy-injection both set only **inducing conditions** (thin wall + high speed / e=1); the fault is produced by the engine's numerics — the **engine-reported penetration=0** at tunneling is the smoking gun (the engine misses it itself, we did not fabricate a violating state).
2. **Does detection depend on a hand-tuned scenario? Tunneling at 200 m/s / e=1 are standard inducing parameters**, not finely tuned to a critical edge to trigger; the sweep is recorded honestly (50 stopped, 200+ tunnels, threshold clear).
3. **Does healthy false-positive? No (after the setup fix).** The noise floor is characterized (8.3e-4 J/step), threshold 10×; non-penetration requires planar motion (ground plane added). The first false positive was my setup error, recorded honestly and fixed.

### 5.3 Trust-root boundary (honest scope-out)
- The audit reads the true state via the **PyBullet API** (getBasePositionAndOrientation/getBaseVelocity) — **if PyBullet were to lie via its API** (return a wrong true state), the audit could not catch it. This is the **same family** as the 2D platform's `get_render_state` trust-root blind spot (the linchpin q class), and is likewise out of scope.

### 5.4 Paper significance
- **The strong-generalization claim holds (in part)**: the framework can detect self-deception on the **native pathologies of an independently-built real engine (PyBullet)** (high-speed tunneling, energy injection), **non-circularly** — not just on our toy.
- **Honestly scoped**: the warm-start ghost force was not cleanly measured (hard to isolate); the trust-root boundary (engine API lying) is out of scope.
- **Genuine refinements**: EC5′ must be extended to **swept** to catch high-speed tunneling; the energy budget must explicitly handle the **engine default damping + integration noise floor**.

---

## 6. TODO / out of scope (INV-C)
- A clean diagnosis of the warm-start ghost force (needs a more dedicated contact-impulse probe); a full-geometry 3D EC5′ (dropping the planar assumption); folding the engine damping model into the energy audit.
- The 2D platform / master were not modified; the existing contract C1-3/CI and physics EC1-5 criteria were not touched (only ported/reused + a 3D refinement).
