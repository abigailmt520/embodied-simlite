# -*- coding: utf-8 -*-
"""
make_audit_figure.py  ——  Action1 · RED/GREEN evidence figure (paper §4.3 figure)
====================================================================
Load the 4 archived sessions (healthy + injections 1-A/1-B/1-C), **re-run the audit for real**,
and render the verdict matrix as a "RED/GREEN" figure:
    rows = the three checks C1 / C2 / C3
    columns = healthy system (gate 2) | injection 1-A | injection 1-B | injection 1-C  (gate 1)
    each cell = green ✓PASS / red ✗FAIL + real localization info (frame/seq/error/recv_t)

All red/green and localization text in the figure comes from the real return of `audit_session()`, not hardcoded (INV-2).

Run: python audit/make_audit_figure.py   (first run run_action1.py to generate sessions/*.json)
Artifact: audit/audit_redgreen_matrix.png
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrity_audit import audit_session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.join(HERE, "sessions")

# ---- register a CJK font (matplotlib's default font has no Chinese) ----
for cand in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/Library/Fonts/Arial Unicode.ttf",
             "/System/Library/Fonts/Hiragino Sans GB.ttc"):
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

GREEN_FILL, GREEN_EDGE, GREEN_TXT = "#d7f3e3", "#2faa6a", "#11623b"
RED_FILL, RED_EDGE, RED_TXT = "#fbd9d9", "#d84141", "#8a1c1c"
HEAD_FILL, HEAD_TXT = "#2b2f3a", "#ffffff"

ROWS = [("C1", "C1 · TRUTH_ODOM_FORK\nTruth↔Odom true fork"),
        ("C2", "C2 · SEQ_INTEGRITY\nframe-seq monotonic"),
        ("C3", "C3 · FEED_LIVENESS\ndisconnect freezes/OFFLINE")]

COLS = [("healthy", "Healthy system (Gate 2)\nHealthy"),
        ("1-A_truth_copy", "Injection 1-A\nodom ← truth"),
        ("1-B_seq_freeze", "Injection 1-B\nseq frozen"),
        ("1-C_stall_running", "Injection 1-C\nstall→online")]

CHECK_ID = {"C1": "C1_TRUTH_ODOM_FORK", "C2": "C2_SEQ_INTEGRITY",
            "C3": "C3_FEED_LIVENESS"}


def load_audit(col_key):
    if col_key == "healthy":
        path = os.path.join(SESS, "healthy.json")
    else:
        path = os.path.join(SESS, f"injected_{col_key}.json")
    with open(path) as fh:
        sess = json.load(fh)
    res = audit_session(sess)
    return {c["check"]: c for c in res["checks"]}


def cell_text(row_key, chk):
    """Compress the real audit result into short readable cell text."""
    loc = chk.get("locator") or {}
    if chk["ok"]:
        head = "✓ PASS"
        if row_key == "C1":
            body = f"max err={loc.get('max_err_m','?')} m\nL={loc.get('truth_path_len_m','?')} m (true fork)"
        elif row_key == "C2":
            body = "seq increases monotonically with data"
        else:
            body = "stale frame marked OFFLINE"
    else:
        head = "✗ FAIL"
        if row_key == "C1":
            body = (f"max err={loc.get('max_err_m','?')} m ≈ 0\n"
                    f"L={loc.get('truth_path_len_m','?')} m → no fork / copies truth")
        elif row_key == "C2":
            if "frozen_seq" in loc:
                body = f"seq frozen@{loc['frozen_seq']}\nframe {loc.get('frame_index')}, t={loc.get('recv_t')}s"
            else:
                body = f"seq regresses {loc.get('prev_seq')}→{loc.get('seq')}\nframe {loc.get('frame_index')}"
        else:
            span = loc.get("recv_t_span")
            body = (f"stale@seq{loc.get('frozen_seq')} but link=online\n"
                    f"t={span}s → disconnected yet running")
    return head, body


def main():
    audits = {ck: load_audit(ck) for ck, _ in COLS}

    nrow, ncol = len(ROWS), len(COLS)
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, ncol + 1.15)
    ax.set_ylim(0, nrow + 1.4)
    ax.axis("off")

    cw, ch = 1.0, 1.0
    x0, y0 = 1.15, 0.25
    rh = 0.95

    def box(x, y, w, h, fill, edge, rad=0.06):
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), w - 0.08, h - 0.08,
                     boxstyle=f"round,pad=0.0,rounding_size={rad}",
                     fc=fill, ec=edge, lw=1.6))

    # column headers
    for j, (_, label) in enumerate(COLS):
        x = x0 + j * cw
        box(x, y0 + nrow * rh, cw, rh * 0.95, HEAD_FILL, HEAD_FILL)
        ax.text(x + cw / 2, y0 + nrow * rh + rh * 0.48, label, ha="center", va="center",
                color=HEAD_TXT, fontsize=11, fontweight="bold")
    # row headers
    for i, (_, label) in enumerate(ROWS):
        y = y0 + (nrow - 1 - i) * rh
        box(0.06, y, 1.05, rh, HEAD_FILL, HEAD_FILL)
        ax.text(0.06 + 1.05 / 2, y + rh / 2, label, ha="center", va="center",
                color=HEAD_TXT, fontsize=10.5, fontweight="bold")

    # cells
    for i, (rk, _) in enumerate(ROWS):
        y = y0 + (nrow - 1 - i) * rh
        for j, (ck, _) in enumerate(COLS):
            x = x0 + j * cw
            chk = audits[ck][CHECK_ID[rk]]
            if chk["ok"]:
                box(x, y, cw, rh, GREEN_FILL, GREEN_EDGE)
                hc, bc = GREEN_TXT, GREEN_TXT
            else:
                box(x, y, cw, rh, RED_FILL, RED_EDGE)
                hc, bc = RED_TXT, RED_TXT
            head, body = cell_text(rk, chk)
            ax.text(x + cw / 2, y + rh * 0.74, head, ha="center", va="center",
                    color=hc, fontsize=12.5, fontweight="bold")
            ax.text(x + cw / 2, y + rh * 0.34, body, ha="center", va="center",
                    color=bc, fontsize=8.2)

    fig.suptitle("Embodied-SimLite · Integrity Audit RED/GREEN evidence",
                 fontsize=15, fontweight="bold", y=0.985)
    ax.text((ncol + 1.15) / 2, 0.04,
            "Gate 2: the healthy system is all-green with zero false positives   |   "
            "Gate 1: injections 1-A/1-B/1-C are each flagged RED and located by the corresponding check   "
            "— the audit is proven able to \"catch the fake\", not a forever-green Potemkin village",
            ha="center", va="center", fontsize=9.5, color="#333")

    out = os.path.join(HERE, "audit_redgreen_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] RED/GREEN matrix saved: {out}")
    # also verify: the first column should be all green, the latter three columns red on the diagonal
    ok = (all(audits["healthy"][CHECK_ID[r]]["ok"] for r, _ in ROWS)
          and not audits["1-A_truth_copy"]["C1_TRUTH_ODOM_FORK"]["ok"]
          and not audits["1-B_seq_freeze"]["C2_SEQ_INTEGRITY"]["ok"]
          and not audits["1-C_stall_running"]["C3_FEED_LIVENESS"]["ok"])
    print(f"[CHECK] healthy all-green + three injections flagged RED on the diagonal: {'PASS' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
