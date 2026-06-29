# -*- coding: utf-8 -*-
"""
fault_injection.py  ——  Action1 · fake-instrument injectors (purely for the audit to prove it catches fakes, never enters the production path)
================================================================================
Inject three real "self-deceptions / fake instruments" into a *healthy* session, to verify the audit really catches fakes (acceptance gate 1).

Injection is a **real corruption** of the state stream (replicating defects that really occurred historically), not writing "demo-only red output":
    1-A truth_copy     —— splice Odom back onto truth frame by frame (replicating the ox=robot.x Potemkin village of the V3 refactor era).
    1-B seq_freeze     —— data keeps updating while the frame seq is frozen (the frame seq lies).
    1-C stall_running  —— flip the link claim of a "stalled-frozen segment" from offline back to online (disconnected yet shown running).

Each injector takes a deep copy of the healthy session and returns the corrupted session; the healthy original is not modified (one-click restore =
do not inject / reuse the healthy session).
"""

import copy


def inject_truth_copy(session):
    """1-A: Odom ← Truth (copy truth frame by frame). Simulates the "error link is dead, odometry = truth" fake instrument."""
    s = copy.deepcopy(session)
    for f in s:
        f["odom"] = {"x": f["truth"]["x"], "y": f["truth"]["y"],
                     "theta": f["truth"]["theta"]}
    return s


def inject_seq_freeze(session, start_frac=0.3, span=12):
    """1-B: freeze the frame seq within an online window where data is still updating (data moves but seq does not)."""
    s = copy.deepcopy(session)
    online_idx = [i for i, f in enumerate(s) if f.get("link_status") == "online"]
    if not online_idx:
        return s
    a = online_idx[int(len(online_idx) * start_frac)]
    frozen = s[a]["seq"]
    for i in range(a, min(a + span, len(s))):
        if s[i].get("link_status") == "online":
            s[i]["seq"] = frozen     # pin the frame seq; truth/odometry advance as usual (data is live)
    return s


def inject_stall_running(session):
    """1-C: flip the healthy session's "stalled-frozen segment" (offline) back to online, creating "disconnected yet shown running"."""
    s = copy.deepcopy(session)
    flipped = 0
    for f in s:
        if f.get("link_status") == "offline":
            f["link_status"] = "online"   # data still frozen, yet falsely claims online/running
            flipped += 1
    if flipped == 0:
        # when the healthy session has no stall segment, fallback: create a stretch of frozen-yet-online stale frames at the tail
        last = copy.deepcopy(s[-1])
        t = last["recv_t"]
        for k in range(1, 11):
            fr = copy.deepcopy(last)
            fr["recv_t"] = round(t + k * 0.1, 3)   # wall clock advances
            fr["link_status"] = "online"           # data frozen yet claims online
            s.append(fr)
    return s


INJECTORS = {
    "1-A_truth_copy": inject_truth_copy,
    "1-B_seq_freeze": inject_seq_freeze,
    "1-C_stall_running": inject_stall_running,
}

DESCRIPTIONS = {
    "1-A_truth_copy": "splice Odom back onto the truth frame by frame (odom←truth), replicating the ox=robot.x Potemkin village",
    "1-B_seq_freeze": "freeze the frame sequence number while data keeps updating (the frame seq lies)",
    "1-C_stall_running": "flip the link claim of a stalled-frozen segment from offline back to online (disconnected yet shown as running)",
}
