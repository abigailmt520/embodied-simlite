# Phase3 details: contract-layer mutual-information leakage audit (C_I) — finishing the dual-state formalization

> Branch `dev/stage4-mi-leakage` (off Phase2 `3cc426e`). master `7b54625` frozen, zero changes; all existing `.pth` not overwritten.
> **Phase3 is a pure audit-layer addition (not a line of env / contract C1-3 changed)**. Numbers from real runs (INV-E); with line numbers (INV-B).

---

## 0. Claim and symmetry
Upgrade the validated C1 heuristic ("Truth–Odom error too small is suspicious") into a **principled theory**, **symmetric** with the physics layer's energy conservation:
- Physics layer: the ΔE conservation residual (EC1).
- Contract layer: **I(report;truth) ≤ noise-budget bound** (C_I) — the information about truth carried by a legitimate report channel is upper-bounded by its noise budget; over the bound = leakage.

---

## 1. 🔴 Engineering corrections to Opus's original design (design pitfalls, flagged honestly)

**Pitfall 1: absolute-position MI is unusable.** The original design "I(odom_pos;truth_pos) ≤ ½log(1+Var/σ²)" does not hold for **absolute position**: odom is a **cumulative integral** of dead reckoning (at episode start odom≡truth, then drifts), so the absolute-position MI is dominated by the **shared-trajectory correlation** (not the noise channel), near-infinite after reset → it would false-positive on a healthy system.
**Correction**: estimate MI on the **per-step displacement increments Δpos** (a memoryless channel; `leakage_audit.py` uses `increments` throughout).

**Pitfall 2: the closed-form ½log(1+SNR) is only approximate for multiplicative slip noise.** The odom noise is `slip·|v|·Z` (multiplicative, velocity-dependent), not standard AWGN; the closed-form bound is systematically biased (measured at slip=0.05: closed-form 5.15 vs estimable truth ~3.3).
**Correction**: the noise-budget bound uses **the MI of the truth increments re-noised by the declared noise model** (Monte-Carlo operationalization, `noise_budget_bound` `:38`); the closed form is printed only for reference.

**Pitfall 3 (most important, reliability): KSG saturates at high SNR / near-determinism.** See §4.

---

## 2. Milestone 1 · C_I audit + noise-budget bound + estimator

### 2.1 KSG MI estimator (`mi_estimator.py:42`)
Kraskov estimator 1 (k-NN, Chebyshev norm, per-dimension standardization, digamma). Self-check (a known Gaussian channel): σ=1.0 estimates 0.31 vs theory 0.35; σ=0.3 estimates 0.99 vs 1.25 (underestimate at high MI, consistent with KSG's known bias).

### 2.2 Noise-budget bound (`noise_budget_bound` `:38`)
Re-noise the truth increments under the declared model `Δodom_legit=(1+slip)·Δtruth + slip·|Δtruth|·N(0,I)`, estimate its MI as the bound (5 MC repeats, mean±std).

### 2.3 C_I audit criterion (`ci_audit` `:66`)
Within a sliding window (window=500/stride=250), estimate I(Δodom;Δtruth); over `bound + margin(MARGIN_NATS=0.4` `:28`)` and persisting ≥ K_PERSIST(2) → flag RED + locate (mirroring the C1/EC gating).

### 2.4 Clean baseline (real MI vs bound)
| slip | N | noise-budget bound | clean I | threshold | C_I audit |
|---|---|---|---|---|---|
| **0.30** | 1500 | 2.281±0.021 | **2.494** | 2.681 | 🟢 GREEN (no false positive) |
| 0.05 | 1500 | 3.319±0.019 | 2.819 | 3.719 | 🟢 GREEN |

Clean I is within bound+margin → green, zero false positives ✅.

---

## 3. Milestone 2 · leakage injectors + validation + comparison with C1

### 3.1 Three leakage injectors (contract layer, `leakage_audit.py`)
| Injection | What it changes (line) | C1 (error too small) | C_I (mutual information) |
|---|---|:---:|:---:|
| **L-1 full leak** | odom secretly = truth (`:108`) | error≈0 → 🔴catch | I=∞ → 🔴catch |
| **L-2 partial leak** | odom = truth + noise shrunk to 1/4 (still drifts, `:113`) | still large error → 🟢**miss** | I over bound → 🔴catch |
| **L-3 privileged transform** | odom increment = truth increment rotated 35° (deterministic, large error, `:125`) | large error → 🟢**miss** | I=∞ → 🔴catch |

### 3.2 Validation (slip=0.30 primary evidence, real MI)
- Gate C_I-2 clean green ✅; Gate C_I-1 three leaks **3/3 flagged RED** ✅.
- Real MI: L-1 **4.039** / L-2 **3.251** / L-3 **3.068** nats, all > threshold 2.681 (bound 2.281). Evidence figure `leakage_compare.png`.

### 3.3 Comparison with C1 (C_I = a principled quantitative generalization of C1)
| Case | C1 | C_I |
|---|:---:|:---:|
| clean | 🟢 | 🟢 |
| L-1 full leak | 🔴 | 🔴 |
| L-2 partial leak | 🟢 miss | 🔴 catch |
| L-3 privileged transform | 🟢 miss | 🔴 catch |
- **C1 catches only the full leak L-1** (error≈0); **C_I additionally catches what C1 misses: L-2 (partial leak still drifts) / L-3 (large error but deterministic)** — C_I generalizes C1's "error magnitude" heuristic into a quantitative "information exceeds the noise budget" criterion. This finishes the contract layer of the dual-state formalization.

---

## 4. 🔴 Reliability reported honestly (INV-E)

### 4.1 MI estimation sample size / bias
- N=1500 increments, 2D-2D (4D joint), k=4. KSG underestimates high MI (already seen in the self-check); for **deterministic relations** (L-1/L-3) it returns a **finite saturated value** (~4 nats, not literal ∞). So detection is via "over the bound", not the absolute value.

### 4.2 Limitation at the deployed slip=0.05 (**key honest point**)
The platform deploys slip=0.05 → odom is **very accurate** → the noise-budget bound ~3.3 nats **approaches the KSG reliable ceiling** → healthy and leak both crowd into the saturation region, with poor separation:
| slip=0.05 | clean | L-1 | L-2 | L-3 |
|---|---|---|---|---|
| I (nats) | 2.82🟢 | 4.02🔴catch | 3.48🟢**miss** | 3.02🟢**miss** |
- **L-1 still caught; L-2/L-3 missed** (below the high bound, and L-2 jitters near the threshold across repeats → unreliable).
- **Conclusion**: the C_I audit is clean, quantitative, and catches what C1 misses in the **estimable regime (slip≈0.3)**; in the **high-SNR deployment regime (slip=0.05, odom too accurate)** the MI estimate saturates and detection power degrades — there **C1's magnitude check is actually more practical**. The two are **complementary**: C1 is simple and robust for the full leak; C_I is principled/quantitative and catches partial/privileged leaks in the estimable regime.

### 4.3 No over-claim
- C_I is not "strictly stronger" — it is a **principled generalization** of C1 that broadens coverage in the estimable regime (L-2/L-3), but is limited by KSG in the deployed small-slip high-SNR regime. The paper should present it **dual-state-symmetrically** (physics EC1 conservation residual ↔ contract C_I information bound) and honestly flag the estimable interval.

---

## 5. TODO / out of scope (INV-C)
- Improving small-slip detection: a better MI estimator (e.g. MINE/ensemble), larger samples, or a log-transformed channel; deferred.
- MuJoCo not connected; the physics EC / contract C1-3 code not touched; F3 dynamic obstacles not done; the q-class reconciliation not done (an independent contract-vs-kernel channel is still out of scope — C_I still consumes truth/odom via get_render_state, inheriting the trust root).
