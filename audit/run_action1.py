# -*- coding: utf-8 -*-
"""
run_action1.py  ——  Action1 · one-shot run of the three gates + evidence output
=====================================================
Gate 1 (audit catches fakes · red): inject the three fake instruments 1-A/1-B/1-C into a healthy V3; the audit must flag each RED and localize.
Gate 2 (pass healthy · green): the healthy V3 system audits all-green with no false positives.
Gate 3 (basic evaluation): run the trained PPO for N≥20 episodes, computing success/collision/average-reach-steps + a figure.

The whole thing runs real physics (embodied_env + _integrate_odom true fork) and the real PPO policy;
the audit red/green are real verdicts, the metrics come from real episodes; nothing hardcoded/faked (INV-2).

Run: python audit/run_action1.py
Artifacts: audit/sessions/*.json, audit/eval_metrics.png, audit/eval_episodes.csv
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodied_env import EmbodiedNavEnv                       # noqa: E402
from integrity_audit import audit_session, format_report     # noqa: E402
import fault_injection as fi                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = os.path.join(HERE, "sessions")
MODEL_PATH = os.path.join(os.path.dirname(HERE), "ppo_embodied_agent.pth")
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

SEED_SESSION = 11
SEED_EVAL = 100
N_LIVE = 240            # online live frame count of the healthy session
N_DISCONNECT = 15       # simulated disconnect-frozen frame count (healthy system: link_status=offline)
N_EVAL_EPISODES = 25    # Gate 3 episode count (≥20)


def load_ppo(env):
    """Rebuild a PPO isomorphic to the training-time one and load the weights (reusing inference_server's load logic)."""
    m = PPO("MlpPolicy", env, policy_kwargs=POLICY_KWARGS, device="cpu")
    m.policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.policy.eval()
    return m


def _record(frame, recv_t, link_status):
    r, o = frame["robot"], frame["odom"]
    return {
        "recv_t": round(recv_t, 3),
        "seq": frame["seq"],
        "truth": {"x": r["x"], "y": r["y"], "theta": r["theta"]},
        "odom": {"x": o["x"], "y": o["y"], "theta": o["theta"]},
        "step": frame["step"],
        "terminated": frame["terminated"],
        "truncated": frame["truncated"],
        "link_status": link_status,
    }


def build_healthy_session():
    """Really drive env+PPO to produce a healthy session: a live online segment + a disconnected offline frozen segment."""
    env = EmbodiedNavEnv(slip=EmbodiedNavEnv.SLIP_FACTOR)
    model = load_ppo(env)
    obs, info = env.reset(seed=SEED_SESSION)
    dt = env.DT
    t = 0.0
    session = []

    # —— live online segment: real PPO inference, auto-reset at episode end (seq does not reset across episodes, stays monotonic) ——
    for _ in range(N_LIVE):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.get_render_state(reward=reward, terminated=terminated,
                                     truncated=truncated, info=info)
        t += dt
        session.append(_record(frame, t, "online"))
        if terminated or truncated:
            obs, info = env.reset()

    # —— disconnect segment: feed stops updating, data frozen at the last frame; the healthy system explicitly marks OFFLINE (wall clock still advancing) ——
    last = json.loads(json.dumps(session[-1]))   # deep copy
    for _ in range(N_DISCONNECT):
        t += dt
        fr = json.loads(json.dumps(last))
        fr["recv_t"] = round(t, 3)
        fr["link_status"] = "offline"            # disconnect freezes and marks OFFLINE (the correct behavior fixed in Phase0)
        session.append(fr)

    return session


def run_eval():
    """Gate 3: N-episode PPO evaluation, returning per-episode records and aggregate metrics."""
    env = EmbodiedNavEnv(slip=EmbodiedNavEnv.SLIP_FACTOR)
    model = load_ppo(env)
    rows = []
    for i in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=SEED_EVAL + i)
        done = False
        success = collided = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                done = True
                success = bool(info.get("is_success", False))
                collided = bool(info.get("collided", False))
        outcome = "success" if success else ("collision" if collided else "timeout")
        rows.append({"episode": i, "seed": SEED_EVAL + i, "steps": steps,
                     "success": int(success), "collision": int(collided),
                     "outcome": outcome})
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    summary = {
        "n_episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "timeout_rate": sum(1 for r in rows if r["outcome"] == "timeout") / n,
        "avg_steps_success": (sum(r["steps"] for r in succ) / len(succ)) if succ else None,
        "avg_steps_all": sum(r["steps"] for r in rows) / n,
    }
    return rows, summary


def plot_eval(summary, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rates = [summary["success_rate"], summary["collision_rate"], summary["timeout_rate"]]
    bars = ax1.bar(["success", "collision", "timeout"], rates,
                   color=["#2c7", "#d33", "#fa0"])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("rate")
    ax1.set_title(f"PPO over N={summary['n_episodes']} episodes (fixed seeds)")
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}", ha="center")

    steps = [r["steps"] for r in rows]
    colors = {"success": "#2c7", "collision": "#d33", "timeout": "#fa0"}
    ax2.bar(range(len(rows)), steps, color=[colors[r["outcome"]] for r in rows])
    ax2.set_xlabel("episode")
    ax2.set_ylabel("steps")
    ax2.set_title("per-episode steps (color = outcome)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    os.makedirs(SESS_DIR, exist_ok=True)
    print("=" * 74)
    print("  Action 1 · Anti-self-deception audit + base evaluation  ——  three-gate live run")
    print("=" * 74)

    # ---- build the healthy session ----
    healthy = build_healthy_session()
    json.dump(healthy, open(os.path.join(SESS_DIR, "healthy.json"), "w"))
    print(f"\n[healthy session] {len(healthy)} frames "
          f"(online {sum(1 for f in healthy if f['link_status']=='online')} + "
          f"offline {sum(1 for f in healthy if f['link_status']=='offline')}), "
          f"seq {healthy[0]['seq']}→{healthy[-1]['seq']}")

    # ============ Gate 2: healthy system → all green ============
    print("\n" + "─" * 74)
    print("[Gate 2 | Audit passes the healthy system (expected: all green)]")
    print("─" * 74)
    res_healthy = audit_session(healthy)
    print(format_report(res_healthy))
    gate2_ok = res_healthy["passed"]

    # ============ Gate 1: inject the three fake instruments → flag each RED ============
    print("\n" + "─" * 74)
    print("[Gate 1 | Audit catches fakes: inject 1-A/1-B/1-C, expect each flagged RED and located]")
    print("─" * 74)
    gate1 = {}
    expected_check = {"1-A_truth_copy": "C1_TRUTH_ODOM_FORK",
                      "1-B_seq_freeze": "C2_SEQ_INTEGRITY",
                      "1-C_stall_running": "C3_FEED_LIVENESS"}
    for name, inj in fi.INJECTORS.items():
        injected = inj(healthy)
        json.dump(injected, open(os.path.join(SESS_DIR, f"injected_{name}.json"), "w"))
        res = audit_session(injected)
        # whether this injection is flagged RED by the "corresponding check"
        tgt = expected_check[name]
        tgt_red = any((not c["ok"]) and c["check"] == tgt for c in res["checks"])
        gate1[name] = {"caught": (not res["passed"]) and tgt_red, "result": res}
        print(f"\n  ▶ Inject [{name}]: {fi.DESCRIPTIONS[name]}")
        print(f"    expected RED check: {tgt}")
        print(format_report(res))
        print(f"    → catch fake {'OK ✅' if gate1[name]['caught'] else 'FAILED ❌'}")

    gate1_ok = all(v["caught"] for v in gate1.values())

    # ============ Gate 3: basic evaluation ============
    print("\n" + "─" * 74)
    print(f"[Gate 3 | Base evaluation: PPO × N={N_EVAL_EPISODES} episodes (fixed seeds)]")
    print("─" * 74)
    rows, summary = run_eval()
    with open(os.path.join(HERE, "eval_episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_eval(summary, rows, os.path.join(HERE, "eval_metrics.png"))
    print(f"  success_rate   : {summary['success_rate']:.1%}")
    print(f"  collision_rate : {summary['collision_rate']:.1%}")
    print(f"  timeout_rate   : {summary['timeout_rate']:.1%}")
    avg_s = summary["avg_steps_success"]
    print(f"  avg steps to goal (successful episodes): {avg_s:.1f}" if avg_s else "  avg steps to goal (successful episodes): no successful episode")
    print(f"  avg steps (all episodes)               : {summary['avg_steps_all']:.1f}")
    print(f"  [OK] figure: audit/eval_metrics.png   per-episode: audit/eval_episodes.csv")

    # ============ summary ============
    print("\n" + "=" * 74)
    print("  Acceptance-gate summary")
    print("=" * 74)
    print(f"  Gate 1 (audit catches fakes · RED): {'✅ PASS (3/3 injections flagged RED and located)' if gate1_ok else '❌ FAIL'}")
    print(f"  Gate 2 (passes healthy · GREEN)   : {'✅ PASS (healthy system all green)' if gate2_ok else '❌ FAIL (false positive)'}")
    print(f"  Gate 3 (base evaluation)          : ✅ produced real metrics + figure (success rate {summary['success_rate']:.0%})")
    json.dump(summary, open(os.path.join(HERE, "eval_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
