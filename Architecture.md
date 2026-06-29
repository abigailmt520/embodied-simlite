# Embodied-SimLite Architecture Upgrade Document V2

> Milestone: **refactor and unify the RL physics kernel and the sim-to-real control gateway**
> First principle: **fully decouple physics rollout, AI decision-making, and twin rendering**
> Status: the server side (inference gateway + twin frontend + data contract) passes local testing (see "Validation results" at the end); the ROS 2 bridge part needs validation in an Ubuntu + ROS 2 environment.
> Note: this document was written during the V2 refactor; since then the odometry has been upgraded to a **true fork** (injecting real slip drift, not `odom≡truth`), and an **anti-self-deception integrity audit** plus a global frame sequence number `seq` were added — the content below has been updated accordingly, see the repo README for details.

---

## 1. Upgrade background and historical debt

The old `ProductV1.0.py` welded three things that should be orthogonal into one process:

- **Physics rollout**: `physics_loop()` ran rigid-body / odometry / dynamic obstacles in an async loop at 60Hz inside FastAPI;
- **Communication and rendering**: the same file hosted the Three.js frontend and exchanged data bidirectionally over `/ws/simulation`;
- **Global mutable state**: one giant `state` dict, read and written by the physics loop, the WebSocket send/receive, and collision resolution.

This coupling caused two fatal problems:

1. **No high-speed training**: the physics logic was locked to the `asyncio.sleep(1/60)` clock and WebSocket serialization; the "saturate a single core, repeatedly `step()` at the hardware limit" that reinforcement learning needs was simply impossible.
2. **A messy data contract**: the frontend, backend, and ROS 2 bridge each read and wrote the flat fields `x/y/yaw/ox/oy/oyaw/lidar/...`, so renaming any field anywhere triggered a three-way cascade failure.

The goal of V2 is to cut these three layers fully apart and rebuild the whole system around a **pure-compute physics kernel** as the common denominator.

---

## 2. Core design philosophy: the Compute-Swap Architecture

### 2.1 One-sentence definition

> **One pure-compute physics kernel (`embodied_env.py`), wrapped in two different "scheduling shells": the training state strips away every clock and all communication to let a single core run flat-out; the inference state re-wraps it in a 60Hz async heartbeat to broadcast the twin state outward. The two shells do not pollute each other.**

"Swap" means exactly this: training and inference consume the same compute budget, only **swapping it from "waiting on the clock / waiting on the network" to "pure compute"** — during training the rendering, communication, and throttling overhead is all refunded to the CPU for `step()` iteration.

### 2.2 Three-layer decoupling

| Layer | Carrier | Responsibility | Explicitly does not do |
|----|------|------|--------------|
| **Physics rollout** | `embodied_env.py` (`EmbodiedNavEnv`) | A pure-numpy synchronous functional state machine: kinematic integration, O(1) circle-circle collision, analytic ray-cast LiDAR | No network / rendering / async-clock logic |
| **AI decision** | `inference_server.py` (PPO policy network) | 60Hz per-tick inference action; can be preempted by a manual command | No physics-integration logic |
| **Twin rendering** | the Three.js frontend embedded in `inference_server.py` | Only consumes the broadcast state for visualization | Does not take part in physics computation (V2 removed the frontend raycasting) |

### 2.3 Why the physics kernel can be "one kernel, dual use"

`EmbodiedNavEnv` is a standard `gymnasium.Env`; its key property is **no side effects, no I/O, no sleep**:

- Kinematics: differential-drive (unicycle) semi-implicit integration;
- Collision: circular chassis vs circular obstacle, center-distance test, O(1) per obstacle;
- LiDAR: analytic "ray-circle" / "ray-wall" intersection, vectorized O(N_RAYS), no per-pixel scan needed.

Precisely because it is just a pure-functional `step()` of "input action → return observation/reward", we get:

```
                    ┌─────────────────────────────┐
                    │   embodied_env.py (pure kernel) │
                    │   reset() / step() / no I/O   │
                    └──────────────┬──────────────┘
            training shell ↙                          ↘ inference shell
 ┌────────────────────────┐         ┌──────────────────────────────┐
 │  train_agent.py         │         │  inference_server.py          │
 │  strip the clock,       │         │  asyncio 60Hz heartbeat + WS  │
 │  full-speed single-core │         │  broadcast                     │
 │  → PPO weights .pth/.zip │         │  → twin obs domain / ROS 2 bridge │
 └────────────────────────┘         └──────────────────────────────┘
```

### 2.4 Inference-state pipeline (the data flow per tick)

```
read Observation
   → decision-authority arbitration (Override? manual command : PPO inference)
   → env.step(action)  advance physics by one frame
   → env.get_render_state()  serialize the twin state
   → manager.broadcast(json)  WebSocket broadcast to all frontends / ROS 2 bridges
   → if the episode terminated/truncated then env.reset(), the twin demo keeps rolling
   → drift-compensating heartbeat: sleep the remaining time rather than a fixed 1/60, suppressing cumulative clock drift
```

Drift compensation is the key to 60Hz precision: accumulate against `next_tick += TICK_DT`, `sleep(next_tick - now)`, and if a single tick overruns, reset the baseline, avoiding "catch-up bursts".

### 2.5 Unified data contract (`get_render_state()` new nested format)

V2 replaces the old flat fields with one **nested contract**; the rendering side and the ROS 2 side recognize only this one structure:

```jsonc
{
  "robot":    { "x": 2.35, "y": 6.66, "theta": 2.40, "radius": 0.20 },  // Truth (noise-free baseline)
  "odom":     { "x": 2.41, "y": 6.49, "theta": 2.31 },  // odometry (with real slip drift, independent of truth)
  "goal":     { "x": 0.58, "y": 7.31, "radius": 0.40 },
  "obstacles":[ { "x": .., "y": .., "r": .. }, ... ],   // regenerated on each episode reset
  "lidar":    [ 5.0, 3.2, ... ],                         // 24 real range readings (m)
  "lidar_range": 5.0,
  "arena":    { "w": 10.0, "h": 10.0 },                  // origin at the corner, 0..w / 0..h
  "seq": 1234,                  // global monotonic frame sequence number (does not reset across episodes, for the integrity audit to verify)
  "step": 81, "reward": -0.5,
  "terminated": false, "truncated": false,
  "distance": 1.82,
  "control_mode": "rl"          // added by the server: rl | override (does not pollute the env contract)
}
```

> Design: the truth `robot` is the noise-free baseline; the odometry `odom` is integrated independently by `embodied_env._integrate_odom()` injecting **real slip drift**, deviating monotonically from truth over the trajectory (**true fork**, see README "true-fork odometry"). The frontend "odometry ghost" `ghostGroup` renders `odom`, drifting away from the truth body over time; ROS 2 `/odom` publishes `odom` (the drifting odometry) rather than the truth. `seq` is the global monotonic frame sequence number, for the anti-self-deception audit to verify frame-sequence monotonicity.

### 2.6 Unified service: a single entry point

`inference_server.py` now wears three hats and is the **system's single entry point**:

- `@app.get("/")` → serves the whole Three.js twin observation domain directly (HTML migrated from the old version, `HTMLResponse`);
- `@app.get("/health")` → health-check JSON (the health info that used to occupy `/` retreats here);
- `@app.websocket("/ws")` → downstream broadcast of the twin state + upstream receipt of manual-override commands.

The old `physics_loop()` and the global `state` dict are **fully deprecated and not migrated** — the physics rollout is taken over entirely by `embodied_env.py`.

---

## 3. The 2s sim-to-real override mechanism (Override Control)

### 3.1 Goal

Currently the PPO decides automatically inside the inference gateway and dispatches control. But in a real deployment, the operator or Nav2 needs to **intervene manually** at any time via ROS 2 `/cmd_vel`. The mechanism's goal is:

> Once a manual `/cmd_vel` arrives, **seamlessly suspend and preempt** the PPO's control authority, and for the next **2 seconds** execute only the manual command; when the window expires, **automatically hand back** to the RL, achieving a seamless switch of sim-to-real control authority.

### 3.2 Full pipeline

```
ROS 2 node (teleop / Nav2)
   │  publishes geometry_msgs/Twist to /cmd_vel (or TwistStamped on /cmd_vel_nav)
   ▼
ros_bridge.py  EmbodiedRos2Bridge.process_cmd(v_x, w_z)
   │  passes the real physical quantities through, no scaling:
   │  ws.send({"cmd_vel": {"linear": v_x, "angular": w_z}})
   ▼
inference_server.py  /ws upstream handler
   │  parse cmd_vel → override.submit(linear, angular, now)
   ▼
OverrideController (2s-window arbiter)
   │  normalized conversion + record the expiry instant expiry = now + 2.0
   ▼
simulation_loop()  per-tick decision-authority arbitration
       manual = override.get(now)
       action = manual if manual is not None else model.predict(obs)
```

### 3.3 Key engineering details

**(a) Normalization is done on the gateway side; the bridge stays generic**
ROS 2's `cmd_vel` is a real physical quantity (`linear.x` m/s, `angular.z` rad/s). The bridge **passes it through verbatim**, and the gateway's `OverrideController` converts it uniformly back into the env's normalized action space, so the physics kernel is completely agnostic to whether control comes from "manual / AI":

```python
v = clip(linear  / MAX_LIN_VEL, 0.0, 1.0)   # MAX_LIN_VEL = 1.0 m/s
w = clip(angular / MAX_ANG_VEL, -1.0, 1.0)  # MAX_ANG_VEL = 1.5 rad/s
```

This way the bridge need not know any env-internal constant; switching robot model later changes only one place on the gateway.

**(b) Time-window arbiter**
`OverrideController` caches "the most recent action + its expiry instant"; the arbitration logic is minimal:

```python
class OverrideController:
    def submit(self, linear, angular, now):
        self._action = np.array([v, w], dtype=np.float32)
        self._expiry = now + self.window_s          # window_s = 2.0

    def get(self, now):
        return self._action if (self._action is not None and now < self._expiry) else None
```

- **Preempt**: within the window `get()` returns the manual action, which `simulation_loop` adopts directly, skipping `model.predict`;
- **Renew**: continuously publishing `/cmd_vel` keeps refreshing `expiry`, so manual control can continue indefinitely;
- **Auto hand-back**: 2 seconds after publishing stops, `get()` returns `None`, and decision authority returns seamlessly to the PPO.

**(c) Visualization feedback that does not pollute the physics contract**
Before broadcasting, the gateway **adds** a `control_mode` field (`rl` / `override`) to the state dict, and the frontend telemetry panel highlights "control authority" in red as "ROS 2 manual override". This field is added by the server and is **not written into** `embodied_env.get_render_state()`, keeping the physics-kernel contract pure.

**(d) Clock consistency**
Both `submit` and `get` use the same `asyncio` event-loop clock `loop.time()` (a monotonic clock), sharing a source with the 60Hz heartbeat, ensuring the 2s window is precise and unaffected by wall-clock jumps.

---

## 4. File responsibilities at a glance (V2)

| File | Role | V2 change |
|------|------|---------|
| `embodied_env.py` | Pure-compute physics kernel + the new data contract `get_render_state()` | As the contract baseline, **unchanged** this time |
| `train_agent.py` | Training shell (strip the clock, full-speed single-core) | — |
| `inference_server.py` | **Single entry point**: 60Hz inference heartbeat + `/` serves the frontend + `/ws` broadcast/override | HTML migrated in, new `/` route, `OverrideController`, decision arbitration; deprecated `physics_loop`/`state` |
| `ros_bridge.py` | ROS 2 protocol translation layer | `on_message` reads `data["odom"]` (drifting odometry) / `data["lidar"]`; `/cmd_vel` passed through as `{"cmd_vel":{...}}`; connects `/ws` (needs a ROS 2 environment to run) |
| `Architecture_V2.md` | This document | New |

---

## 5. Validation results (server-side local testing)

> The following are **server-side** (inference gateway + frontend + data contract + override mechanism) local test results.
> The end-to-end closed loop involving real ROS 2 topics / rviz2 / Nav2 **needs validation in an Ubuntu + ROS 2 environment** and is out of this machine's scope.

Launching `inference_server.py`, measured:

| Validation item | Result |
|--------|------|
| Model load + 60Hz heartbeat | ✅ `Model loaded: ppo_embodied_agent.pth` / `Uvicorn running on :8000` |
| `GET /` serves the frontend | ✅ `content-type: text/html`, returns the full Three.js page (`/health` retreats to JSON) |
| New nested-contract broadcast | ✅ keys = `robot/goal/obstacles/lidar/lidar_range/arena/step/reward/...`; 24-ray lidar, `lidar_range=5.0`, 6 obstacles, 10×10 arena |
| RL actually drives | ✅ `robot.x` changes continuously across frames |
| `/cmd_vel` preemption | ✅ after publishing, `control_mode` immediately flips to `override` |
| 2s override window release | ✅ after publishing stops, auto hand-back to `rl` in **~2.01s** (measured 2.01s, expected 2.0s) |

---

## 6. Integration test steps (reproduction guide)

```bash
# 1) Launch the inference gateway (the system's single entry point)
cd embodied-simlite
python inference_server.py            # listens on 0.0.0.0:8000

# 2) Open the twin observation domain in a browser, verify the pure-RL closed loop
#    http://localhost:8000/

# 3) In another terminal, launch the ROS 2 bridge node (needs a ROS 2 environment)
python ros_bridge.py                  # connects ws://127.0.0.1:8000/ws by default
# cross-machine: SIM_GATEWAY_WS=ws://<gateway-IP>:8000/ws python ros_bridge.py

# 4) Verify the seamless switch of sim-to-real control authority
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.8}}'
#    → gateway telemetry switches to "ROS 2 manual override"; after publishing stops for 2s, auto hand-back to "RL automatic"
```
