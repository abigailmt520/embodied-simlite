# -*- coding: utf-8 -*-
"""
inference_server.py
===================
The inference state of the "compute-swap architecture": 60Hz async inference + twin-state broadcast gateway + frontend observation-domain host.

This file is the [system-level single entry point]. It wears three hats:
    1. Hosts the PPO policy network, driving the pure-compute physics kernel embodied_env.py with a tight 60Hz heartbeat;
    2. Via `@app.get("/")` serves the whole Three.js twin-observation-domain frontend directly (migrated from the old ProductV1.0);
    3. Via `/ws` broadcasts the new nested data contract of `env.get_render_state()`, and receives the `/cmd_vel`
       manual-override command from the ROS 2 bridge, implementing the "seamless sim-to-real control switch" (Override Control).

First principle (full decoupling):
    - Physics rollout: entirely delegated to embodied_env.py (a pure-numpy synchronous state machine); this file contains no physics logic;
      the old ProductV1.0's `physics_loop()` and global `state` dict are fully deprecated and not migrated.
    - AI decision: the PPO policy network infers here at 60Hz; it can be preempted by a ROS 2 manual command within a 2s window.
    - Twin rendering: only the frontend Three.js consumes the broadcast state; the server does not render.

Data flow per tick:
    read Observation → (Override? manual command : PPO inference) Action → env.step() advances physics →
    serialize the twin state → WebSocket broadcast to all frontend observation windows / ROS 2 bridges.

Dependencies: fastapi, uvicorn, websockets, stable-baselines3, torch, numpy.
Run: python inference_server.py   (or uvicorn inference_server:app --host 0.0.0.0 --port 8000)
"""

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from stable_baselines3 import PPO

from embodied_env import EmbodiedNavEnv

# —— must match train_agent.py exactly, otherwise load_state_dict shapes mismatch ——
MODEL_PATH = "ppo_embodied_agent.pth"
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
TICK_HZ = 60.0                     # twin heartbeat frequency
TICK_DT = 1.0 / TICK_HZ

# —— sim-to-real control switch: after receiving /cmd_vel, block RL auto inference within this window, prioritize the manual override ——
OVERRIDE_WINDOW_S = 2.0


# ====================================================================
# manual override controller (Override Control): ROS 2 /cmd_vel → preempt RL inference
# ====================================================================
class OverrideController:
    """Caches the most recent ROS 2 manual command and preempts RL auto inference within OVERRIDE_WINDOW_S.

    cmd_vel is a real physical quantity (linear.x m/s, angular.z rad/s); here it is converted uniformly back to the env's
    normalized action space [v∈[0,1], w∈[-1,1]], so the physics kernel is fully agnostic to the "manual/AI" two control paths.
    """

    def __init__(self, window_s: float = OVERRIDE_WINDOW_S):
        self.window_s = window_s
        self._action: np.ndarray | None = None
        self._expiry = 0.0

    def submit(self, linear: float, angular: float, now: float):
        v = float(np.clip(linear / EmbodiedNavEnv.MAX_LIN_VEL, 0.0, 1.0))
        w = float(np.clip(angular / EmbodiedNavEnv.MAX_ANG_VEL, -1.0, 1.0))
        self._action = np.array([v, w], dtype=np.float32)
        self._expiry = now + self.window_s

    def active(self, now: float) -> bool:
        return self._action is not None and now < self._expiry

    def get(self, now: float) -> np.ndarray | None:
        """Within the window return the manual action; when the window expires return None (hand back to RL)."""
        return self._action if self.active(now) else None


override = OverrideController()


# ====================================================================
# WebSocket connection manager: maintains all frontend observation windows / ROS 2 bridges, broadcasts uniformly
# ====================================================================
class ConnectionManager:
    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: str):
        # copy to avoid concurrent modification of the set during broadcast; clean up dead connections along the way
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# global singletons: the inference env kernel + policy model (initialized in lifespan)
env = EmbodiedNavEnv(render_mode=None)
model: PPO | None = None


def load_model() -> PPO:
    """Rebuild a PPO isomorphic to the training-time one and load the .pth weights (policy state_dict).

    Note: directly constructing PPO does not actually train; it only reproduces the network structure to receive the state_dict.
    If you kept the SB3 native zip archive, you can also use the more robust:
        return PPO.load("ppo_embodied_agent", device="cpu")
    """
    m = PPO("MlpPolicy", env, policy_kwargs=POLICY_KWARGS, device="cpu")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    m.policy.load_state_dict(state_dict)
    m.policy.eval()   # inference state: turn off training behaviors like dropout/batchnorm
    return m


# ====================================================================
# background async coroutine: tight 60Hz heartbeat inference loop (incl. manual-override preemption)
# ====================================================================
async def simulation_loop():
    """Maintain a precise 60Hz with drift compensation, deciding per tick and broadcasting the twin state.

    Decision arbitration: if a ROS 2 /cmd_vel was received within the 2s override window, execute the manual action; otherwise PPO auto inference.
    """
    loop = asyncio.get_event_loop()
    obs, info = env.reset()
    next_tick = loop.time()

    while True:
        now = loop.time()

        # —— 1) decision arbitration: manual override takes priority, otherwise PPO inference ——
        manual = override.get(now)
        if manual is not None:
            action = manual          # sim-to-real switch: ROS 2 manual command preempts
        else:
            # a small MLP's predict is a sub-millisecond synchronous op, fine to run directly in the event loop;
            # if a heavy network is used, switch to await loop.run_in_executor(...) to avoid blocking the heartbeat.
            action, _ = model.predict(obs, deterministic=True)

        # —— 2) physics step (the physics kernel is fully agnostic to the manual/AI two control paths) ——
        obs, reward, terminated, truncated, info = env.step(action)

        # —— 3) serialize and broadcast the twin state to all frontend observation windows / ROS 2 bridges ——
        state = env.get_render_state(reward=reward, terminated=terminated,
                                     truncated=truncated, info=info)
        # the server adds a control-authority marker (does not pollute the env contract), for the frontend telemetry to display
        state["control_mode"] = "override" if manual is not None else "rl"
        await manager.broadcast(json.dumps(state, separators=(",", ":")))

        # —— 4) auto-reset at episode end, keeping the twin demo rolling ——
        if terminated or truncated:
            obs, info = env.reset()

        # —— 5) drift-compensating heartbeat: sleep the remaining time rather than a fixed 1/60, suppressing cumulative clock drift ——
        next_tick += TICK_DT
        sleep_time = next_tick - loop.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # a single tick overran (compute too slow): reset the baseline, avoiding catch-up bursts
            next_tick = loop.time()


# ====================================================================
# FastAPI lifecycle: load the model and start the background heartbeat on startup, cancel gracefully on shutdown
# ====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    print(f">>> Model loaded: {MODEL_PATH}, starting {TICK_HZ:.0f}Hz twin inference heartbeat...")
    task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print(">>> Inference heartbeat stopped.")


app = FastAPI(title="Embodied-SimLite Inference Gateway", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Unified WS endpoint: both the frontend Three.js observation window and the ROS 2 bridge connect here.

    - Downstream: the server broadcasts env.get_render_state()'s twin state at 60Hz.
    - Upstream: only recognizes the manual-override control sent by the ROS 2 bridge
            {"cmd_vel": {"linear": <m/s>, "angular": <rad/s>}}; other messages are ignored.
    """
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            cv = data.get("cmd_vel") if isinstance(data, dict) else None
            if isinstance(cv, dict):
                override.submit(
                    linear=float(cv.get("linear", 0.0)),
                    angular=float(cv.get("angular", 0.0)),
                    now=asyncio.get_event_loop().time(),
                )
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


@app.get("/health")
async def health():
    """Health check / connection info."""
    return {
        "service": "Embodied-SimLite Inference Gateway",
        "tick_hz": TICK_HZ,
        "clients": len(manager.active),
        "ws_endpoint": "/ws",
    }


# ====================================================================
# Task 1: unified frontend-backend service — the Three.js twin observation domain migrated from the old ProductV1.0
# Task 2: ws.onmessage rewritten, strictly adapted to the nested contract of embodied_env.get_render_state()
# ====================================================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Embodied-SimLite | RL Inference Twin Observatory</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/litegraph.js/css/litegraph.css">
    <style>
        body, html { margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background-color: #1a1a1a; font-family: sans-serif; color: white; }
        #main-container { display: flex; width: 100vw; height: 100vh; flex-direction: row; }
        #left-panel { flex: 0 0 36%; height: 100%; border-right: 1px solid #333; background-color: #222; position: relative; min-width: 200px; }
        #resizer { width: 6px; cursor: ew-resize; background-color: #333; transition: background 0.2s; z-index: 100; }
        #resizer:hover { background-color: #00ffcc; }
        #right-panel { flex: 1; height: 100%; background-color: #111; position: relative; overflow: hidden; min-width: 200px; }
        canvas { display: block; outline: none; }

        .panel-title {
            position: absolute; top: 10px; left: 20px;
            color: #00ffcc; font-family: monospace;
            z-index: 1000; pointer-events: none;
            text-shadow: 1px 1px 2px black;
            background: rgba(0,0,0,0.6);
            padding: 8px 12px; border-radius: 6px;
        }
        .app-brand { display: block; font-size: 1.2em; font-weight: bold; color: #fff; margin-bottom: 4px; }

        #telemetry { position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; font-family: monospace; border: 1px solid #444; z-index: 1000; pointer-events: none; min-width: 240px;}
        .tel-row { display: flex; justify-content: space-between; margin: 5px 0; font-size: 13px; }
        .truth { color: #00ffcc; }
        .odom { color: #ff4444; }
        #canvas-container { width: 100%; height: 100%; display: block; }

        #view-hint {
            position: absolute; bottom: 20px; right: 20px; color: rgba(255,255,255,0.5);
            font-family: monospace; font-size: 12px; pointer-events: none; z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="main-container">
        <div id="left-panel">
            <h2 class="panel-title">
                <span class="app-brand">Embodied-SimLite | RL Inference Twin Observatory</span>
                🧩 Module 1: Perception & Control Blueprint
            </h2>
            <canvas id="node-canvas"></canvas>
        </div>
        <div id="resizer"></div>
        <div id="right-panel">
            <h2 class="panel-title">🌐 Module 2: Cyber-Physical Twin Observatory</h2>
            <div id="telemetry">
                <h3 style="margin-top:0; color:#fff; border-bottom:1px solid #555; padding-bottom:5px;">📡 Telemetry</h3>
                <div class="tel-row truth"><span>True X:</span><span id="true_x">0.00</span></div>
                <div class="tel-row truth"><span>True Y:</span><span id="true_y">0.00</span></div>
                <div class="tel-row truth"><span>True Yaw:</span><span id="true_yaw">0.00</span></div>
                <hr style="border: 0.5px solid #333;">
                <div class="tel-row odom"><span>Odom X:</span><span id="odom_x">0.00</span></div>
                <div class="tel-row odom"><span>Odom Y:</span><span id="odom_y">0.00</span></div>
                <div class="tel-row odom"><span>Odom Yaw:</span><span id="odom_yaw">0.00</span></div>
                <hr style="border: 0.5px solid #333;">
                <div class="tel-row" style="color:#ffcc00;"><span>To goal:</span><span id="dist_val">0.00 m</span></div>
                <div class="tel-row" style="color:#ffcc00;"><span>Step / reward:</span><span id="step_reward">0 / 0.0</span></div>
                <div class="tel-row" style="color:#ffaa00; font-weight:bold;"><span>🕹️ Control:</span><span id="ctrl_mode">RL auto</span></div>
                <div class="tel-row" style="color:#ff4444; font-weight:bold;"><span>Episode:</span><span id="epi_status">Running</span></div>
                <p>Status: <span id="wsStatus" style="color: yellow;">Connecting...</span></p>
            </div>
            <div id="view-hint">LMB: rotate | RMB: pan | wheel: zoom | R: reset view</div>
            <!-- INV-1 / D-017: explicit OFFLINE overlay when the WS disconnects, view frozen, no local dead-reckoning continuation -->
            <div id="offline-overlay" style="display:none; position:absolute; inset:0; z-index:2000;
                 background:rgba(20,0,0,0.55); backdrop-filter:grayscale(0.8) blur(1px);
                 display:none; align-items:center; justify-content:center; pointer-events:none;">
                <div style="font-family:monospace; text-align:center; color:#ff6666;
                     border:2px solid #ff4444; border-radius:12px; padding:22px 30px;
                     background:rgba(0,0,0,0.8);">
                    <div style="font-size:1.6em; font-weight:bold;">⚠ OFFLINE</div>
                    <div style="margin-top:8px; color:#ffaaaa;">Source-of-truth link lost · view frozen</div>
                    <div style="margin-top:4px; font-size:0.8em; color:#cc8888;">Frontend performs no local dead-reckoning; waiting for backend to reconnect…</div>
                </div>
            </div>
            <div id="canvas-container"></div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/litegraph.js/build/litegraph.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        // flat mirror of the old blueprint nodes: x/y/yaw from the truth d.robot; ox/oy/oyaw from the odometry d.odom (with slip drift)
        window.simData = { x: 0, y: 0, yaw: 0, ox: 0, oy: 0, oyaw: 0, v: 0 };

        window.onload = function() {
            // connect uniformly to the inference gateway's /ws (replaces the old /ws/simulation)
            const wsUrl = `ws://${window.location.host}/ws`;
            const ws = new WebSocket(wsUrl);

            // ---- INV-1 / D-017: disconnect freezes and explicitly marks OFFLINE (the frontend never dead-reckons the pose locally) ----
            // Note: this frontend's pose is only assigned in ws.onmessage, animate() does no integration,
            // so after a disconnect the view naturally freezes (not dead reckoning). The functions below only make the "frozen" fact
            // explicit, eliminating the twin-authority contradiction of "connection lost yet still shown running".
            function setOffline() {
                document.getElementById('wsStatus').innerHTML = "<span style='color: #ff6666;'>❌ Disconnected (OFFLINE · view frozen)</span>";
                document.getElementById('offline-overlay').style.display = 'flex';
                const st = document.getElementById('epi_status');
                st.innerText = '⚠ OFFLINE'; st.style.color = '#ff6666';
            }
            function clearOffline() {
                document.getElementById('offline-overlay').style.display = 'none';
            }

            ws.onopen = () => { clearOffline(); document.getElementById('wsStatus').innerHTML = "<span style='color: lime;'>✅ Inference gateway online</span>"; };
            ws.onclose = () => setOffline();
            ws.onerror = () => setOffline();

            var graph = new LGraph();
            var canvas = new LGraphCanvas("#node-canvas", graph);

            graph.onBeforeStep = function() {
                for (let i = 0; i < graph._nodes.length; ++i) {
                    if (graph._nodes[i].pos[1] < 170) graph._nodes[i].pos[1] = 170;
                }
            };

            function syncCanvasSize() {
                const lp = document.getElementById('left-panel');
                const cv = document.getElementById('node-canvas');
                cv.width = lp.clientWidth; cv.height = lp.clientHeight;
                if (canvas) { canvas.resize(); canvas.draw(true, true); }
            }

            // ---------- blueprint nodes (educational/visualization purpose; control authority is handed to RL/ROS, the nodes no longer write back to ws) ----------
            function WatchNode() {
                this.addInput("X", "number"); this.addInput("Y", "number"); this.addInput("Yaw", "number");
                this.properties = { x: 0, y: 0, yaw: 0 };
                this.color = "#143"; this.bgcolor = "#264"; this.size = [160, 70]; this.title_text_color = "#00ffcc";
            }
            WatchNode.title = "👁️ Pose Monitor";
            WatchNode.prototype.onExecute = function() {
                this.properties.x = this.getInputData(0) || 0; this.properties.y = this.getInputData(1) || 0; this.properties.yaw = this.getInputData(2) || 0;
            };
            WatchNode.prototype.onDrawBackground = function(ctx) {
                if (this.flags.collapsed) return;
                ctx.fillStyle = "#00ffcc"; ctx.font = "bold 14px monospace";
                let x_val = (this.properties.x || 0).toFixed(2);
                let y_val = (this.properties.y || 0).toFixed(2);
                let yaw_deg = ((this.properties.yaw || 0) * 180 / Math.PI) % 360;
                if (yaw_deg > 180) yaw_deg -= 360; else if (yaw_deg < -180) yaw_deg += 360;
                ctx.fillText(`${x_val}`, 50, 25);
                ctx.fillText(`${y_val}`, 50, 45);
                ctx.fillText(`${yaw_deg.toFixed(1)}°`, 50, 65);
            };
            LiteGraph.registerNodeType("Analysis/Pose Monitor", WatchNode);

            function OdomNode() {
                this.addOutput("Est. X", "number"); this.addOutput("Est. Y", "number"); this.addOutput("Est. Yaw", "number");
                this.color = "#622"; this.bgcolor = "#833"; this.size = [180, 80]; this.title_text_color = "#00ffcc";
            }
            OdomNode.title = "⚙️ Wheel Odometry (Odom)";
            OdomNode.prototype.onExecute = function() { this.setOutputData(0, window.simData.ox); this.setOutputData(1, window.simData.oy); this.setOutputData(2, window.simData.oyaw); };
            LiteGraph.registerNodeType("Sensor/Wheel Odometry", OdomNode);

            function TruthNode() {
                this.addOutput("True X", "number"); this.addOutput("True Y", "number"); this.addOutput("True Yaw", "number");
                this.color = "#266"; this.bgcolor = "#388"; this.size = [180, 80]; this.title_text_color = "#00ffcc";
            }
            TruthNode.title = "🛰️ Ground Truth (Truth)";
            TruthNode.prototype.onExecute = function() { this.setOutputData(0, window.simData.x); this.setOutputData(1, window.simData.y); this.setOutputData(2, window.simData.yaw); };
            LiteGraph.registerNodeType("Sensor/Ground Truth", TruthNode);

            var nTruth = LiteGraph.createNode("Sensor/Ground Truth"); nTruth.pos=[30, 200]; graph.add(nTruth);
            var nOdom = LiteGraph.createNode("Sensor/Wheel Odometry"); nOdom.pos=[30, 320]; graph.add(nOdom);
            var nWatchTruth = LiteGraph.createNode("Analysis/Pose Monitor"); nWatchTruth.pos=[300, 200]; graph.add(nWatchTruth);
            var nWatchOdom = LiteGraph.createNode("Analysis/Pose Monitor"); nWatchOdom.pos=[300, 320]; graph.add(nWatchOdom);
            nTruth.connect(0, nWatchTruth, 0); nTruth.connect(1, nWatchTruth, 1); nTruth.connect(2, nWatchTruth, 2);
            nOdom.connect(0, nWatchOdom, 0); nOdom.connect(1, nWatchOdom, 1); nOdom.connect(2, nWatchOdom, 2);
            graph.start();

            // ==========================================
            // 3D rendering and controls
            // ==========================================
            const rp = document.getElementById('right-panel');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, rp.clientWidth/rp.clientHeight, 0.1, 1000);

            // initial view is a placeholder; once the arena size arrives, focus on the arena center
            let initialCameraPos = new THREE.Vector3(5, -8, 12);
            let initialCameraTarget = new THREE.Vector3(5, 5, 0);
            camera.position.copy(initialCameraPos);
            camera.lookAt(initialCameraTarget);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(rp.clientWidth, rp.clientHeight);
            document.getElementById('canvas-container').appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;
            controls.minDistance = 2;
            controls.maxDistance = 80;
            controls.maxPolarAngle = Math.PI / 2 - 0.05;
            controls.target.copy(initialCameraTarget);

            scene.add(new THREE.AmbientLight(0xffffff, 0.7));
            const light = new THREE.DirectionalLight(0xffffff, 0.8); light.position.set(5, 10, 10); scene.add(light);

            const grid = new THREE.GridHelper(40, 40, 0x444444, 0x222222);
            grid.rotation.x = Math.PI/2;
            scene.add(grid);

            // ---------- dynamic scene: arena boundary / circular obstacles / goal (all driven by the new contract) ----------
            const wallMat = new THREE.MeshLambertMaterial({color: 0x555577});
            let arenaBuilt = false;

            function buildArena(w, h) {
                // floor
                const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
                    new THREE.MeshLambertMaterial({color: 0x182838}));
                floor.position.set(w/2, h/2, -0.01);
                scene.add(floor);
                // four boundary walls (aligned with env's 0..w / 0..h axis-aligned arena)
                const t = 0.1, hh = 0.4;
                const segs = [
                    [w, t, w/2, 0], [w, t, w/2, h],   // bottom, top
                    [t, h, 0, h/2], [t, h, w, h/2],   // left, right
                ];
                segs.forEach(([sw, sh, cx, cy]) => {
                    const m = new THREE.Mesh(new THREE.BoxGeometry(sw, sh, hh), wallMat);
                    m.position.set(cx, cy, hh/2);
                    scene.add(m);
                });
                // focus on the arena center
                initialCameraTarget = new THREE.Vector3(w/2, h/2, 0);
                initialCameraPos = new THREE.Vector3(w/2, h/2 - 0.85*h, 1.3*h);
                camera.position.copy(initialCameraPos);
                controls.target.copy(initialCameraTarget);
                controls.update();
                arenaBuilt = true;
            }

            // circular obstacles: the obstacle set changes on each episode reset; rebuild wholesale when a change is detected
            const obstacleMat = new THREE.MeshLambertMaterial({color: 0xff6600});
            let obstacleMeshes = [];
            let obstacleSig = "";
            function syncObstacles(obstacles) {
                const sig = obstacles.map(o => `${o.x.toFixed(2)},${o.y.toFixed(2)},${o.r.toFixed(2)}`).join('|');
                if (sig === obstacleSig) return;
                obstacleSig = sig;
                obstacleMeshes.forEach(m => scene.remove(m));
                obstacleMeshes = [];
                obstacles.forEach(o => {
                    const geo = new THREE.CylinderGeometry(o.r, o.r, 0.6, 24);
                    const m = new THREE.Mesh(geo, obstacleMat);
                    m.rotation.x = Math.PI / 2;       // cylinder axis toward +Z
                    m.position.set(o.x, o.y, 0.3);
                    scene.add(m);
                    obstacleMeshes.push(m);
                });
            }

            // goal: changes with each episode reset
            let goalMesh = null;
            function syncGoal(goal) {
                if (!goalMesh) {
                    goalMesh = new THREE.Mesh(
                        new THREE.CylinderGeometry(goal.radius, goal.radius, 0.04, 32),
                        new THREE.MeshBasicMaterial({color: 0x00ff66, transparent: true, opacity: 0.55}));
                    goalMesh.rotation.x = Math.PI / 2;
                    scene.add(goalMesh);
                }
                goalMesh.position.set(goal.x, goal.y, 0.02);
            }

            // ---------- the vehicle body (robotGroup) and the odometry ghost (ghostGroup) ----------
            const chassisGeo = new THREE.BoxGeometry(0.4, 0.3, 0.18);
            const robotGroup = new THREE.Group();
            const chassis = new THREE.Mesh(chassisGeo, new THREE.MeshLambertMaterial({color: 0x00cc88}));
            chassis.position.set(0, 0, 0.12);
            robotGroup.add(chassis);

            const arrowShape = new THREE.Shape();
            arrowShape.moveTo(0.22, 0); arrowShape.lineTo(-0.12, 0.15);
            arrowShape.lineTo(-0.04, 0); arrowShape.lineTo(-0.12, -0.15);
            arrowShape.lineTo(0.22, 0);
            const arrowGeo = new THREE.ShapeGeometry(arrowShape);
            const arrow = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({color: 0xffffff}));
            arrow.position.set(0.05, 0, 0.22);
            robotGroup.add(arrow);
            scene.add(robotGroup);

            // odometry ghost: renders the backend's d.odom (with slip drift). Over time it visibly deviates from the green body — this is the "true fork"
            const ghostGroup = new THREE.Group();
            const gChassis = new THREE.Mesh(chassisGeo, new THREE.MeshLambertMaterial({color: 0xff3333, transparent: true, opacity: 0.45}));
            gChassis.position.set(0, 0, 0.12);
            ghostGroup.add(gChassis);
            const gArrow = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({color: 0xffaaaa, transparent: true, opacity: 0.55}));
            gArrow.position.set(0.05, 0, 0.22);
            ghostGroup.add(gArrow);
            scene.add(ghostGroup);

            // ---------- LiDAR rays: consume the server's analytic ranging directly, no frontend raycasting ----------
            const lidarMat = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.6 });
            const lidarGeo = new THREE.BufferGeometry();
            const lidarLines = new THREE.LineSegments(lidarGeo, lidarMat);
            scene.add(lidarLines);

            function updateLidar(robot, lidar) {
                // env ray offset: linspace(-π, π, N, endpoint=False) → off_i = -π + i*(2π/N)
                const N = lidar.length;
                if (N === 0) return;
                const positions = new Float32Array(N * 6);
                const z = 0.22;
                for (let i = 0; i < N; i++) {
                    const off = -Math.PI + i * (2 * Math.PI / N);
                    const ang = robot.theta + off;
                    const dist = lidar[i];
                    const b = i * 6;
                    positions[b]   = robot.x;            positions[b+1] = robot.y;            positions[b+2] = z;
                    positions[b+3] = robot.x + Math.cos(ang) * dist;
                    positions[b+4] = robot.y + Math.sin(ang) * dist;
                    positions[b+5] = z;
                }
                lidarGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                lidarGeo.attributes.position.needsUpdate = true;
                lidarGeo.computeBoundingSphere();
            }

            // ==========================================
            // 🎯 Task 2: ws.onmessage strictly adapts to the new nested contract of get_render_state()
            //   contract: { robot:{x,y,theta,radius}, goal:{x,y,radius},
            //          obstacles:[{x,y,r}], lidar:[m], lidar_range, arena:{w,h},
            //          step, reward, terminated, truncated, distance, control_mode }
            // ==========================================
            ws.onmessage = (e) => {
                clearOffline();   // a downstream message means online: remove the OFFLINE freeze overlay
                const d = JSON.parse(e.data);
                const robot = d.robot;
                const odom = d.odom || d.robot;   // source-of-truth odom; backward compatible: fall back to truth when the old contract has no odom

                // flat mirror: truth from d.robot, odometry from d.odom (no longer fabricating odom from truth)
                window.simData.x = robot.x; window.simData.y = robot.y; window.simData.yaw = robot.theta;
                window.simData.ox = odom.x; window.simData.oy = odom.y; window.simData.oyaw = odom.theta;

                if (!arenaBuilt && d.arena) buildArena(d.arena.w, d.arena.h);
                if (d.obstacles) syncObstacles(d.obstacles);
                if (d.goal) syncGoal(d.goal);

                // vehicle body pose (Truth)
                robotGroup.position.set(robot.x, robot.y, 0);
                robotGroup.rotation.z = robot.theta;
                // odometry ghost (Odom, with slip drift; deviates from the body over time)
                ghostGroup.position.set(odom.x, odom.y, 0);
                ghostGroup.rotation.z = odom.theta;

                // lidar rays
                if (d.lidar) updateLidar(robot, d.lidar);

                // ---------- telemetry panel ----------
                document.getElementById('true_x').innerText = robot.x.toFixed(2) + ' m';
                document.getElementById('true_y').innerText = robot.y.toFixed(2) + ' m';
                let ty = (robot.theta * 180 / Math.PI) % 360; if (ty > 180) ty -= 360; else if (ty < -180) ty += 360;
                document.getElementById('true_yaw').innerText = ty.toFixed(1) + '°';

                document.getElementById('odom_x').innerText = odom.x.toFixed(2) + ' m';
                document.getElementById('odom_y').innerText = odom.y.toFixed(2) + ' m';
                let oy_deg = (odom.theta * 180 / Math.PI) % 360; if (oy_deg > 180) oy_deg -= 360; else if (oy_deg < -180) oy_deg += 360;
                document.getElementById('odom_yaw').innerText = oy_deg.toFixed(1) + '°';

                document.getElementById('dist_val').innerText = (d.distance != null ? d.distance.toFixed(2) : '—') + ' m';
                document.getElementById('step_reward').innerText = `${d.step} / ${(d.reward != null ? d.reward.toFixed(2) : '0.00')}`;

                const isOverride = d.control_mode === 'override';
                const cm = document.getElementById('ctrl_mode');
                cm.innerText = isOverride ? 'ROS 2 manual override' : 'RL auto';
                cm.style.color = isOverride ? '#ff4444' : '#00ffcc';

                const st = document.getElementById('epi_status');
                if (d.terminated && d.distance != null && d.distance < (d.goal ? d.goal.radius : 0.4)) {
                    st.innerText = '🎯 Goal reached'; st.style.color = '#00ff66';
                } else if (d.terminated) {
                    st.innerText = '💥 Collision'; st.style.color = '#ff4444';
                } else if (d.truncated) {
                    st.innerText = '⏱ Timeout'; st.style.color = '#ffaa00';
                } else {
                    st.innerText = 'Running'; st.style.color = '#cccccc';
                }
            };

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();

            window.addEventListener('keydown', (e) => {
                if(e.target.tagName.toLowerCase() === 'input') return;
                if (e.key.toLowerCase() === 'r') {
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialCameraTarget);
                    controls.update();
                }
            });

            const resizer = document.getElementById('resizer'); const leftPanel = document.getElementById('left-panel'); let isResizing = false;
            resizer.addEventListener('mousedown', () => { isResizing = true; document.body.style.cursor = 'ew-resize'; });
            window.addEventListener('mousemove', (e) => {
                if (!isResizing) return; let percent = (e.clientX / window.innerWidth) * 100;
                if (percent > 15 && percent < 85) { leftPanel.style.flexBasis = percent + '%'; syncCanvasSize(); camera.aspect = rp.clientWidth / rp.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(rp.clientWidth, rp.clientHeight); }
            });
            window.addEventListener('mouseup', () => { isResizing = false; document.body.style.cursor = 'default'; });
            window.addEventListener('resize', () => { syncCanvasSize(); camera.aspect = rp.clientWidth / rp.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(rp.clientWidth, rp.clientHeight); });
            syncCanvasSize();
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def index():
    """Serve the Three.js twin-observation-domain frontend directly (migrated from the old ProductV1.0's html_content)."""
    return HTMLResponse(HTML_CONTENT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
