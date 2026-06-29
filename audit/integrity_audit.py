# -*- coding: utf-8 -*-
"""
integrity_audit.py  ——  Action1 · Anti-Self-Deception Integrity Audit
=========================================================================================
This module is the "auditor" itself: input a runtime state stream (session, per-frame contract + the monitoring client's sampling time and
link-status claim), output the RED/GREEN + localization of a set of checks.

Design principle (the soul of paper §4.3/4.4, see PRD):
    the audit itself must first be proven able to "catch fakes". An audit that is forever green is a Potemkin village.
    So each check targets a class of **real self-deception signal**, and **really turns red and localizes** when the corresponding fake instrument is deliberately injected.

Three core checks (corresponding to acceptance gate 1's 1-A / 1-B / 1-C):
    C1  TRUTH_ODOM_FORK   —— do Truth and Odom truly fork. Catches the "splice Odom back onto truth" fake instrument (ox=robot.x).
    C2  SEQ_INTEGRITY     —— is the frame seq self-consistent with "data is updating". Catches "data moves but the frame seq is frozen/regressing".
    C3  FEED_LIVENESS     —— does a disconnect freeze immediately and mark OFFLINE. Catches "feed interrupted yet still claims online/running".

Session frame-record contract (list[dict]):
    {
      "recv_t": float,                      # the wall-clock time the monitoring client sampled this frame (s)
      "seq": int,                           # contract frame seq (embodied_env globally monotonic)
      "truth": {"x","y","theta"},           # truth pose (= contract robot)
      "odom":  {"x","y","theta"},           # odometry pose (= contract odom)
      "step": int, "terminated": bool, "truncated": bool,
      "link_status": "online" | "offline",  # the system/frontend's claim of link liveness (a healthy system sets offline on disconnect)
    }

Pure stdlib + numpy, no env/torch dependency; can run independently on any session (incl. offline json).
"""

import math

# ---- decision thresholds (conservative, physically interpretable; not tuned to fit results, see INV-2) ----
EPS_FORK = 1e-4        # the error upper bound for Truth–Odom to count as "no fork" (m): a healthy system far exceeds this after slip
MIN_MOTION = 0.5       # the minimum truth cumulative path for C1 to apply (m): insufficient path → N/A rather than a false positive
EPS_MOVE = 1e-9        # the minimum Euclidean increment to judge "the pose moved" (m)
STALE_TOL = 2          # the consecutive stale-frame count C3 tolerates: beyond it the feed is deemed stopped (tolerates 1 frame of jitter)


def _xy(p):
    return (p["x"], p["y"])


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _moved(prev, cur):
    """Whether the truth pose moved between two frames (position or heading)."""
    return (_dist(prev["truth"], cur["truth"]) > EPS_MOVE
            or abs(prev["truth"]["theta"] - cur["truth"]["theta"]) > EPS_MOVE)


def _result(name, desc, ok, detail, locator=None):
    return {
        "check": name,
        "desc": desc,
        "status": "GREEN" if ok else "RED",
        "ok": bool(ok),
        "detail": detail,
        "locator": locator,
    }


# ====================================================================
# C1 · Truth–Odom true fork
# ====================================================================
def check_truth_odom_fork(session):
    """Catch the "fake instrument": Odom is spliced back onto truth (the error link is dead, Truth≡Odom).

    Evaluated only on online frames. Track the truth cumulative path L and the max Truth–Odom error E_max:
        - insufficient path (L<MIN_MOTION): N/A (no false positive);
        - sufficient path but E_max≈0: flag RED "no fork / suspected truth copy", and localize where it occurs.
    """
    online = [f for f in session if f.get("link_status", "online") == "online"]
    if len(online) < 2:
        return _result("C1_TRUTH_ODOM_FORK", "Truth and Odom truly fork",
                       True, "not enough online frames, N/A")

    L = 0.0
    e_max, e_max_seq = 0.0, None
    for i in range(1, len(online)):
        L += _dist(online[i - 1]["truth"], online[i]["truth"])
    for f in online:
        e = _dist(f["truth"], f["odom"])
        if e > e_max:
            e_max, e_max_seq = e, f["seq"]

    if L < MIN_MOTION:
        return _result("C1_TRUTH_ODOM_FORK", "Truth and Odom truly fork",
                       True, f"truth cumulative path L={L:.3f}m < {MIN_MOTION}m, insufficient motion, N/A")

    if e_max < EPS_FORK:
        return _result(
            "C1_TRUTH_ODOM_FORK", "Truth and Odom truly fork", False,
            f"within path L={L:.2f}m the max Truth–Odom error is only {e_max:.2e}m (<{EPS_FORK:.0e}) — "
            f"odometry has no drift / suspected copying of truth (fake instrument)",
            locator={"max_err_m": e_max, "truth_path_len_m": round(L, 3),
                     "n_online_frames": len(online)})
    return _result(
        "C1_TRUTH_ODOM_FORK", "Truth and Odom truly fork", True,
        f"within path L={L:.2f}m, max Truth–Odom error {e_max:.3f}m @seq={e_max_seq} (true fork)",
        locator={"max_err_m": round(e_max, 4), "truth_path_len_m": round(L, 3)})


# ====================================================================
# C2 · frame-seq self-consistency (catches "data moves but the frame seq is frozen/regressing")
# ====================================================================
def check_seq_integrity(session):
    """The frame seq must be self-consistent with "data is updating":

    For adjacent online frames:
        - seq regresses (seq[i] < seq[i-1]) → flag RED;
        - truth is moving but seq does not increase (data is live yet the frame seq lies) → flag RED.
    (truth and seq both frozen is "stale/disconnected", left to C3 to judge, not misjudged here.)
    """
    prev = None
    for i, f in enumerate(session):
        if f.get("link_status", "online") != "online":
            prev = None  # do not compare across an offline segment
            continue
        if prev is not None:
            ps, cs = prev["seq"], f["seq"]
            if cs < ps:
                return _result("C2_SEQ_INTEGRITY", "frame seq monotonic and self-consistent", False,
                               f"frame seq regresses: seq {ps} → {cs}",
                               locator={"frame_index": i, "prev_seq": ps, "seq": cs,
                                        "recv_t": f["recv_t"]})
            if _moved(prev, f) and cs == ps:
                return _result("C2_SEQ_INTEGRITY", "frame seq monotonic and self-consistent", False,
                               f"data updating (truth moved {_dist(prev['truth'], f['truth']):.3f}m) "
                               f"but frame seq frozen at seq={cs}",
                               locator={"frame_index": i, "frozen_seq": cs,
                                        "recv_t": f["recv_t"]})
        prev = f
    return _result("C2_SEQ_INTEGRITY", "frame seq monotonic and self-consistent", True,
                   "all online frames' frame seq increases monotonically with data updates")


# ====================================================================
# C3 · disconnect freezes (catches "feed interrupted yet still claims online/running")
# ====================================================================
def check_feed_liveness(session):
    """Disconnect freezes immediately and marks OFFLINE:

    A "stale frame" = truth and seq both unchanged but recv_t advancing (wall clock running, data frozen).
    If the link still claims online while stale frames persist ≥ STALE_TOL → flag RED (disconnected yet shown running).
    A healthy system sets link_status to offline during a disconnect, so stale frames are exempted → green.
    """
    prev = None
    stale_run = 0
    run_start = None
    for i, f in enumerate(session):
        online = f.get("link_status", "online") == "online"
        if prev is not None and online:
            stale = (not _moved(prev, f)) and (f["seq"] == prev["seq"]) \
                    and (f["recv_t"] > prev["recv_t"])
            if stale:
                if stale_run == 0:
                    run_start = i
                stale_run += 1
                if stale_run >= STALE_TOL:
                    return _result(
                        "C3_FEED_LIVENESS", "disconnect freezes immediately and marks OFFLINE", False,
                        f"feed stopped updating (from seq={f['seq']}, {stale_run} consecutive frames frozen, "
                        f"wall clock still advancing) but link still claims online — disconnected yet shown running",
                        locator={"stale_from_frame": run_start, "frozen_seq": f["seq"],
                                 "recv_t_span": [session[run_start]["recv_t"], f["recv_t"]],
                                 "link_status": "online"})
            else:
                stale_run = 0
        else:
            stale_run = 0
        prev = f
    return _result("C3_FEED_LIVENESS", "disconnect freezes immediately and marks OFFLINE", True,
                   "no 'disconnected yet shown running': stale frames are all correctly marked OFFLINE frozen")


CHECKS = [check_truth_odom_fork, check_seq_integrity, check_feed_liveness]


def audit_session(session):
    """Run all checks on a session, returning the aggregated result."""
    results = [c(session) for c in CHECKS]
    passed = all(r["ok"] for r in results)
    return {"passed": passed, "verdict": "GREEN" if passed else "RED", "checks": results}


def format_report(audit, title=""):
    """Render as a paste-back-able text report."""
    lines = []
    if title:
        lines.append(title)
    overall = "🟢 ALL GREEN (GREEN)" if audit["passed"] else "🔴 self-deception detected (RED)"
    lines.append(f"  audit verdict: {overall}")
    for r in audit["checks"]:
        mark = "🟢 GREEN" if r["ok"] else "🔴 RED  "
        lines.append(f"    [{mark}] {r['check']:<22} {r['desc']}")
        lines.append(f"             └─ {r['detail']}")
        if not r["ok"] and r["locator"]:
            lines.append(f"             └─ locator: {r['locator']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage: python integrity_audit.py <session.json>")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        sess = json.load(fh)
    res = audit_session(sess)
    print(format_report(res, title=f"== audit {sys.argv[1]} =="))
    sys.exit(0 if res["passed"] else 2)
