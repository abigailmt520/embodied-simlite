# RQ4 details: Coverage matrix — the layered + relational framework vs. 4 fair baselines

> Branch `dev/rq4-coverage` (off Phase4b `57dbca8`). master `7b54625` frozen, zero changes; the paper and all `.pth` byte-unchanged.
> **Pure audit-layer addition** (`audit/run_coverage_matrix.py`, reusing the existing suite). Numbers from real runs, **verified cell by cell** (INV-E, nothing assumed).

---

## 0. Claim
Show that the layered + relational audit covers more than existing baselines — in particular that the **joint layer is non-redundant and is the only means of catching the dual-state coupling (Scenario B)**.
🔴 Discipline: give each baseline the **best implementation a competent engineer would write** (no strawman); if any baseline catches something we miss, **report it honestly** (goal = honest comparison, not winning everything).

## 1. The 5 methods (fair, best implementations)
- **M1 code/data integrity** (hash/checksum) — the existing RV/attestation line: verifies stream checksums; post-hoc tampering → caught; semantic self-deception (a deceptive state produced by legitimate code) → self-consistent data → missed.
- **M2 physics-only** (EC1–EC5 + EC5′, no contract/joint).
- **M3 contract-only** (C1–C3 + C_I, no physics/joint).
- **M4 naive-parallel** (physics + contract side by side, **no joint cross-check**) — the key control.
- **M5 ours (with joint)**.

## 2. Coverage matrix (real runs, slip=0.30 C_I-estimable regime; 🔴 = caught / · = missed)
| Method ＼ instance | healthy | L1 full-leak | L2 partial-leak | seq-freeze | disconnect-as-live | P1 energy | CF2 penetration | Scenario A ghost | **Scenario B coupling** | data-tamper |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| M1 code-integrity | · | · | · | · | · | · | · | · | · | **🔴** |
| M2 physics-only | · | · | · | · | · | 🔴 | 🔴 | 🔴 | · | · |
| M3 contract-only | · | 🔴 | · | 🔴 | 🔴 | · | · | · | · | · |
| M4 naive-parallel (no joint) | · | 🔴 | · | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | **·** | · |
| **M5 ours (with joint)** | · | 🔴 | · | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | **🔴** | · |

Evidence figure `audit/coverage_matrix.png`; per-cell data `audit/coverage_matrix.json`.

## 3. Per-item conclusions (all verified cell by cell from real runs)
- **① Zero false positives on healthy**: all 5 methods green on the healthy instance. ✅
- **② Code integrity**: catches only `data_tamper` (its designed threat), **misses all semantic self-deception** (semantic deception produced by legitimate code, checksums self-consistent). ✅ Confirms "attestation verifies provenance/integrity, not semantic correctness".
- **③ Single layers each miss the other class**: physics-only misses contract self-deception (L1 etc.); contract-only misses physical self-deception (P1/CF2/Scenario A). ✅
- **④ 🔴 Headline (the RQ4 core)**: **M4 naive-parallel misses Scenario B (dual-state coupling); only M5 (with joint) catches it**.
  Scenario B has a legitimate true state (physics EC5′ green) + self-consistent odometry (contract green) → both layers green in parallel, and only the joint odom-vs-map cross-check catches it.
  **→ the joint layer is non-redundant and is the only means of catching the dual-state coupling.** ✅
- **⑤ 🔴 Honest (not winning everything)**: `data_tamper` is caught **only by code-integrity; we (M5) miss it** — the semantic audit **trusts the data stream** (a trust-root blind spot, of the same family as the 2-D linchpin q).
  **Code-integrity and our framework are complementary, not dominated**; written honestly into the paper's limitations. ✅

## 4. 🔴 Honest additional finding (not hidden)
- **L-2 partial leakage: missed by every method (including ours).** Measured C_I: MI(L-2)=2.435, **above clean 2.206** (leakage visible) but **below threshold 2.558** (bound 2.158 + margin 0.4) —
  at this scale (maze, 300 increments, turning trajectory) the +0.23 MI rise is within the detection margin. In Phase 3 (1500 increments, random straight lines) the L-2 rise was ~0.97 and was caught.
  **Conclusion: L-2 detection is sensitive to sample size / trajectory distribution / margin (consistent with the Phase-3 C_I reliability limit), not a defect unique to this framework; reported honestly, with no parameter-tuning to force it past the threshold.**
- That is, the coverage matrix is **not "we win everything"** — L-2 is a blind spot for all, and data_tamper is code-integrity's specialty. Our unique value lands precisely on the **joint-catches-dual-state-coupling (Scenario B)** cell.

## 5. Paper significance (RQ4)
- **Headline**: naive "physics + contract in parallel" is insufficient to cover **dual-state coupling** — a **report × physics joint cross-check (joint)** is required. This is the irreplaceable coverage the layered + relational framework provides over all baselines.
- **Honest positioning**: the framework does not dominate code-integrity (different threat models, complementary); weak L-2-type leakage is a recognized hard case (sample/margin sensitive). Broader coverage ≠ winning everything.

---

## 6. TODO / out of scope (INV-C)
- Better detection of weak L-2 leakage (larger samples / better MI estimation / adaptive margin); a joint + code-integrity combination covering both the provenance and semantic axes.
- The 2-D platform / master were not modified; existing criteria untouched (pure reuse + new harness). No regression.
