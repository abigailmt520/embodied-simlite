# -*- coding: utf-8 -*-
"""
embodied_env.py
================
Embodied-SimLite lightweight embodied-intelligence twin environment (standard gymnasium.Env wrapper).

Design philosophy:
    This env is the physics kernel of the "compute-swap architecture". It itself **contains no**
    network / rendering / async-clock logic; it is a pure-numpy synchronous functional state machine.
    This guarantees:
        - Training: train_agent.py calls step() repeatedly at the hardware limit, one core saturated;
        - Inference: inference_server.py calls it throttled by a 60Hz async heartbeat, broadcasting the twin state.
    One physics kernel, two scheduling shells, not polluting each other.

Physics model (deliberately minimal, zero rigid-body-dynamics dependency):
    - Kinematic integration (differential-drive / unicycle model):
          theta_{t+1} = theta_t + w * dt
          x_{t+1}     = x_t + v * cos(theta) * dt
          y_{t+1}     = y_t + v * sin(theta) * dt
    - Anti-tunneling: circular obstacles + circular chassis, single-obstacle detection is O(1) (center distance < sum of radii = collision).
    - LiDAR: analytic "ray-circle" and "ray-wall" intersection, no per-pixel scan, single ray-single obstacle is also O(1).
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EmbodiedNavEnv(gym.Env):
    """Lightweight autonomous-navigation obstacle-avoidance environment.

    Observation space (continuous):
        [ lidar_0 ... lidar_{N-1},  dist_norm,  yaw_err_norm ]   dim = N + 2
        - lidar_i      : the i-th laser ray's normalized range, ∈ [0, 1] (1 means no obstacle, range limit reached)
        - dist_norm    : chassis-to-goal relative distance / arena diagonal, ∈ [0, 1]
        - yaw_err_norm : yaw-angle error toward the goal / π, ∈ [-1, 1]

    Action space (continuous):
        [ v, w ]
        - v : linear-velocity command, ∈ [0, 1]    (internally multiplied by MAX_LIN_VEL to restore real m/s, no reversing)
        - w : angular-velocity command, ∈ [-1, 1]   (internally multiplied by MAX_ANG_VEL to restore real rad/s)
    """

    metadata = {"render_modes": [None]}

    # ====================== physics / scene constants ======================
    ARENA_W = 10.0          # arena width (m)
    ARENA_H = 10.0          # arena height (m)
    ROBOT_RADIUS = 0.20     # chassis radius (m)
    GOAL_RADIUS = 0.40      # goal-reached radius (m)
    DT = 0.10               # kinematic integration step (s), corresponds to 10Hz decision frequency

    MAX_LIN_VEL = 1.0       # linear-velocity limit (m/s), the target speed at v=1.0 (A-mode command limit)
    MAX_ANG_VEL = 1.5       # angular-velocity limit (rad/s), the target angular rate at |w|=1.0

    # ====================== dynamics core (Phase1a · F1 simplified-dynamics backend) ======================
    # Design: the action [v,w] is interpreted as "target velocity"; the backend uses mass/inertia/viscous-damping
    #       to drive the "actual velocity" first-order toward the target, then integrates the pose from the actual velocity.
    #       Zero rigid-body dependency, pure numpy, analytically auditable energy.
    # Layering: _integrate_dynamics() is the shared core (takes force/torque functions); A-mode wraps a P controller around it
    #       (target velocity → force). Later B-mode feeds raw wheel forces [f_l,f_r] directly, reusing the same core.
    ENABLE_DYNAMICS = True   # True = force-control dynamics (with inertia/damping); False = fall back to the original zero-inertia kinematics (regression control)
    MASS         = 1.0       # vehicle mass m (kg) (keeping the original default old:710)
    INERTIA_COEF = 0.5       # moment of inertia I = INERTIA_COEF * MASS (keeping the original κ=0.5, old:711)
    C_LIN        = 3.0       # linear viscous-damping coefficient (kg/s): rewrites the original post-multiply *0.95 as the standard force term -c·v (auditable)
    C_ANG        = 3.0       # angular viscous-damping coefficient
    KP_V         = 12.0      # A-mode velocity-tracking P gain (target velocity → actuator force)
    KP_W         = 12.0      # A-mode angular-velocity-tracking P gain
    N_SUB        = 5         # physics substeps per env.step (h = DT / N_SUB)

    # ---------------------- B-mode force control (Phase1b) ----------------------
    # action = raw wheel forces [f_l, f_r] (normalized ∈[-1,1], internally ×F_MAX). Drop A-mode's P-tracking layer,
    # directly force=f_l+f_r, torque=(f_r-f_l)·ARM (old:708-709), feed the same dynamics core.
    # "meaningful inertia": the B-mode velocity time constant τ_v = MASS/C_LIN (no P-controller acceleration),
    #   = 1.0/3.0 ≈ 0.333s ≈ 3.3×DT — comparable to / larger than the step (A-mode degrades to τ≈0.055s≪DT because of KP).
    F_MAX        = 2.25      # single-wheel force limit (N): both wheels at full force force=2·F_MAX → steady-state v_ss_max=2·F_MAX/C_LIN=1.5 m/s
    ARM          = 0.8       # differential moment arm; torque=(f_r-f_l)·ARM → steady-state w_ss_max=2·F_MAX·ARM/C_ANG=1.2 rad/s
    # B-mode physical speed limit (steady-state top speed + 10% margin for EC3 over-bound detection; beyond it is non-physical)
    V_PHYS_MAX_B = (2.0 * F_MAX / C_LIN) * 1.10
    W_PHYS_MAX_B = (2.0 * F_MAX * ARM / C_ANG) * 1.10

    # actuator physical speed limit (A-mode: steady-state v_ss=KP/(KP+C)·MAX, leaving 25% margin for the over-bound audit EC3)
    V_PHYS_MAX   = MAX_LIN_VEL * 1.25
    W_PHYS_MAX   = MAX_ANG_VEL * 1.25

    # ====================== odometry drift (D-018: Odom real slip) ======================
    # the wheel-odometry "slip factor" relative to truth: the odom integration applies a multiplicative bias + same-magnitude proportional noise,
    # making odometry deviate monotonically from truth over the trajectory (true fork). When =0, odom coincides with truth point by point,
    # degenerating to the original ideal kinematic kernel — serving as the "before" control and the "no-slip" regression baseline.
    SLIP_FACTOR = 0.05

    N_RAYS = 24             # LiDAR core ray count (after downsampling)
    LIDAR_RANGE = 5.0       # LiDAR range limit (m)
    LIDAR_FOV = 2.0 * np.pi # field of view, 2π means 360° all-around

    N_OBSTACLES = 6         # number of circular obstacles
    OBS_R_MIN = 0.4         # obstacle radius lower bound
    OBS_R_MAX = 0.9         # obstacle radius upper bound

    MAX_STEPS = 500         # max steps per episode (used for truncated)

    # ====================== F4 rectangular collisions + F2 maze (Phase2 richer environment layer) ======================
    # Map backend: 'random_circle' = 10×10 random circles (Phase0-1c behavior, default, zero regression);
    #           'maze' = 40×40 hand-built wall maze (AABB walls + circle-rectangle collision + ray-AABB lidar).
    # Collision semantics: 'terminate' = terminate on impact (paper version, default); 'bounce' = penetration pushout + velocity decay + per-step contact penalty, no termination.
    BOUNCE        = 0.5     # collision restitution e≤1: after a wall hit v_act←e·v_act (KE drops to e², E_contact≥0 dissipative)
    R_CONTACT     = -5.0    # bounce-mode per-contact-step penalty (replaces terminate-mode's R_COLLISION termination)
    # 40×40 hand-built maze walls (AABB: xmin,xmax,ymin,ymax): 4 outer walls + death corridor / chaos maze / U-shaped deadlock valley / extreme narrow slit
    MAZE_WALLS = [
        (0.0, 40.0, 0.0, 1.0), (0.0, 40.0, 39.0, 40.0),          # bottom, top outer walls
        (0.0, 1.0, 0.0, 40.0), (39.0, 40.0, 0.0, 40.0),          # left, right outer walls
        (28.0, 29.0, 5.0, 33.0), (33.0, 34.0, 5.0, 33.0), (28.0, 34.0, 33.0, 34.0),   # death corridor
        (5.0, 12.0, 28.0, 29.0), (5.0, 6.0, 22.0, 29.0), (5.0, 12.0, 22.0, 23.0),
        (11.0, 12.0, 23.0, 27.0), (11.0, 16.0, 27.0, 28.0),       # chaos maze
        (5.0, 12.0, 10.0, 11.0), (5.0, 6.0, 5.0, 11.0), (5.0, 12.0, 5.0, 6.0),         # U-shaped deadlock valley
        (18.0, 23.0, 18.0, 19.0), (15.0, 20.0, 15.0, 16.0), (18.0, 23.0, 12.0, 13.0),
        (23.0, 24.0, 13.0, 18.0),                                  # extreme narrow slit
    ]
    MAZE_W = 40.0
    MAZE_H = 40.0

    # ====================== reward-function weights ======================
    # [core tuning area] the weights below directly determine the agent's "personality"; the comments give the engineering meaning and tuning direction.
    K_PROGRESS   = 30.0     # ↑ dense progress reward: +30 for each 1m closer to the goal. This is the main learning signal,
                            #    and must be significantly larger than the other terms, or the agent falls into a stay-put local optimum.
    R_GOAL       = 200.0    # ↑ large staged positive reward for reaching the goal (triggers terminated).
    R_COLLISION  = -200.0   # ↓ large negative reward for collision (triggers terminated). Symmetric in magnitude with R_GOAL,
                            #    to avoid the agent learning the gambler's strategy of "rush the goal even at the cost of hitting a wall".
    STEP_PENALTY = -0.5     # ↓ fixed per-step penalty: forces the agent to take the shortest path and avoid redundant circling.
    K_SAFETY     = 2.0      # ↓ near-distance soft-penalty coefficient: penalty grows linearly after entering the safety buffer,
                            #    shaping avoidance behavior before a "hard collision", making the learning curve smoother.
    SAFE_DIST    = 0.6      # safety-buffer distance threshold (m); soft penalty begins when min(lidar) is below it.
    K_SMOOTH     = 0.3      # ↓ angular-velocity smoothness-penalty coefficient: suppresses high-frequency in-place jitter for smoother trajectories.

    def __init__(self, render_mode=None, seed=None, slip=None, control_mode="A",
                 map_type="random_circle", collision_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # odometry slip factor (configurable): None takes the class constant SLIP_FACTOR; passing 0.0 explicitly disables drift
        self.slip_factor = float(self.SLIP_FACTOR if slip is None else slip)

        # control mode: 'A' = target-velocity tracking (Phase1a, obs26/act[v,w]); 'B' = raw wheel force (Phase1b, obs28/act[f_l,f_r])
        assert control_mode in ("A", "B"), "control_mode must be 'A' or 'B'"
        self.control_mode = control_mode

        # map backend + collision semantics (Phase2): random_circle keeps the Phase0-1c behavior by default (zero regression)
        assert map_type in ("random_circle", "maze")
        self.map_type = map_type
        if map_type == "maze":
            self.arena_w, self.arena_h = self.MAZE_W, self.MAZE_H
            self.walls = list(self.MAZE_WALLS)       # AABB walls (maze)
            self.collision_mode = collision_mode or "bounce"   # maze defaults to bounce (navigation with real collisions)
        else:
            self.arena_w, self.arena_h = self.ARENA_W, self.ARENA_H
            self.walls = []                          # random-circle map has no inner walls (boundary handled by _ray_walls)
            self.collision_mode = collision_mode or "terminate"  # paper version defaults to terminate
        self._walls_arr = np.array(self.walls, dtype=np.float64).reshape(-1, 4)

        # observation space: N lidar rays (0~1) + dist (0~1) + yaw_err (-1~1); B-mode additionally adds v_act/w_act feedback
        #   (under force control the velocity is a significant hidden state, τ≈3×step; the agent must observe the actual velocity to do force→motion credit assignment)
        base_low = [np.zeros(self.N_RAYS, dtype=np.float32),
                    np.array([0.0, -1.0], dtype=np.float32)]
        base_high = [np.ones(self.N_RAYS, dtype=np.float32),
                     np.array([1.0, 1.0], dtype=np.float32)]
        if control_mode == "B":
            base_low.append(np.array([-1.0, -1.0], dtype=np.float32))   # v_act_norm, w_act_norm
            base_high.append(np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.concatenate(base_low), high=np.concatenate(base_high), dtype=np.float32)

        # action space: A=[v∈[0,1], w∈[-1,1]]; B=[f_l∈[-1,1], f_r∈[-1,1]] (normalized wheel forces, internally ×F_MAX)
        if control_mode == "B":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        # arena diagonal, for distance normalization (by the map's actual size)
        self._max_dist = float(np.hypot(self.arena_w, self.arena_h))

        # precompute each LiDAR ray's angular offset relative to chassis heading (uniformly distributed over 360°)
        self._ray_offsets = np.linspace(
            -self.LIDAR_FOV / 2.0, self.LIDAR_FOV / 2.0,
            self.N_RAYS, endpoint=False, dtype=np.float64,
        )

        # runtime state (initialized in reset)
        self.pos = None          # chassis position np.array([x, y]) (Truth, noise-free baseline)
        self.theta = None        # chassis heading (rad) (Truth)
        self.v_act = 0.0         # actual linear velocity (m/s) (dynamics integration state; in the zero-inertia mode always equals the target speed)
        self.w_act = 0.0         # actual angular velocity (rad/s)
        self.odom_pos = None     # odometry position np.array([x, y]) (with slip drift, integrated independently)
        self.odom_theta = None   # odometry heading (rad) (with slip drift)
        self.goal = None         # goal point np.array([x, y])
        self.obstacles = None    # obstacles np.array([[cx, cy, r], ...])
        self.prev_dist = None    # distance to goal at the previous step (for the progress-reward difference)
        self.step_count = 0
        # global monotonic frame sequence number (does not reset across episodes), for the integrity audit to verify "frame-seq monotonicity" — distinct from step_count which does reset
        self.frame_seq = 0
        self.last_lidar = None   # caches the most recent lidar, reused for rendering/broadcast

        # —— energy-audit ledger (this step's real numbers, consumed by energy_audit) ——
        #    ΔE_kin should ≈ W_act − D_damp (exact telemetry, clean residual to machine precision, see _integrate_dynamics)
        self.E_kin = 0.0         # current kinetic energy ½m·v_act² + ½I·w_act²
        self.last_dE = 0.0       # this step's kinetic-energy change
        self.last_W_act = 0.0    # this step's actuator net work (by the "declared" force, for audit reconciliation)
        self.last_D_damp = 0.0   # this step's damping dissipation (by the "declared" damping coefficients C_LIN/C_ANG)
        # —— collision ledger (Phase2, consumed by energy_audit's EC1/EC4/EC5) ——
        self.last_E_contact_decl = 0.0  # this step's "declared" collision dissipation = (1−BOUNCE²)·KE_before (by the declared restitution)
        self.last_E_contact_act = 0.0   # this step's "actual" collision kinetic-energy change = KE_before−KE_after (an injector can make it <0 = energy creation)
        self.last_penetration = 0.0     # this step's max residual penetration depth after resolution (should ≈0; >0 if CF-2 skips correction)
        # physics-fault injection hook (None = clean; set by audit/physics_injection.py; test only, never enters production)
        # of the form {"mode": "P-1_neg_damp", ...}; hidden inside _integrate_dynamics / _resolve_wall_collisions.
        self.physics_fault = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------
    # gymnasium standard interface: reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # initialize self.np_random (seeded random number generator)

        # 1) obstacles: random_circle generates random circles; maze has no circular obstacles (walls are static AABBs, not changing per episode)
        if self.map_type == "maze":
            self.obstacles = np.zeros((0, 3), dtype=np.float64)   # maze has no circular obstacles
        else:
            obstacles = []
            margin = self.OBS_R_MAX + 0.2
            for _ in range(self.N_OBSTACLES):
                r = self.np_random.uniform(self.OBS_R_MIN, self.OBS_R_MAX)
                cx = self.np_random.uniform(margin, self.arena_w - margin)
                cy = self.np_random.uniform(margin, self.arena_h - margin)
                obstacles.append([cx, cy, r])
            self.obstacles = np.array(obstacles, dtype=np.float64)

        # 2) random start and goal (rejection sampling: not overlapping any obstacle/wall, and the two keep enough separation)
        self.pos = self._sample_free_point(clearance=self.ROBOT_RADIUS + 0.1)
        min_sep = (0.4 if self.map_type != "maze" else 0.25) * self._max_dist
        for _ in range(200):
            self.goal = self._sample_free_point(clearance=self.GOAL_RADIUS + 0.1)
            if np.linalg.norm(self.goal - self.pos) > min_sep:
                break  # ensures the navigation task is long enough, avoiding degenerate one-step samples

        # 3) random initial heading
        self.theta = float(self.np_random.uniform(-np.pi, np.pi))

        # 4) odometry is calibrated to truth at episode start (cumulative error starts from 0.000)
        self.odom_pos = self.pos.copy()
        self.odom_theta = self.theta

        # 5) dynamics state reset: actual velocity zeroed, the episode starts from rest; energy + collision ledger cleared
        self.v_act = 0.0
        self.w_act = 0.0
        self.E_kin = 0.0
        self.last_dE = 0.0
        self.last_W_act = 0.0
        self.last_D_damp = 0.0
        self.last_E_contact_decl = 0.0
        self.last_E_contact_act = 0.0
        self.last_penetration = 0.0

        self.step_count = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))

        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info

    # ------------------------------------------------------------------
    # gymnasium standard interface: step (kinematic update + O(1) collision detection + reward)
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        a0, a1 = float(action[0]), float(action[1])

        # —— 1+2) the dynamics core advances the truth pose and settles the energy ledger (dispatched by control mode) ——
        if self.control_mode == "B":
            # B-mode: action = normalized raw wheel forces [f_l, f_r] (×F_MAX to restore), drop the P controller, feed the core directly.
            self._step_dynamics_B(a0 * self.F_MAX, a1 * self.F_MAX)
            # smoothness regularization uses the "actual angular velocity" (no w_cmd concept), suppressing pointless spinning.
            w_cmd = self.w_act / self.MAX_ANG_VEL
        else:
            # A-mode: action = target velocity [v, w]; ENABLE_DYNAMICS=False falls back to zero-inertia kinematics.
            self._step_dynamics(a0 * self.MAX_LIN_VEL, a1 * self.MAX_ANG_VEL)
            w_cmd = a1

        # —— 2b) wall collision resolution (Phase2/maze): circle-AABB penetration pushout + velocity bounce + collision ledger;
        #         random_circle has no walls → no-op (clean path, zero regression). Modifies self.pos/self.v_act.
        wall_contact = self._resolve_wall_collisions()

        # —— 2c) odometry integration (D-018 real slip): eats the "actual velocity" v_act/w_act (wheel encoders sense the actual, not the command),
        #         applies slip error → real fork (C1 unchanged). Position integration (incl. collision pushout) is done; here only odom.
        self._integrate_odom(self.v_act, self.w_act)

        self.step_count += 1
        self.frame_seq += 1        # global frame seq increases monotonically (does not reset across episodes)

        # —— 3) compute geometric quantities (after the collision pushout) ——
        dist = float(np.linalg.norm(self.goal - self.pos))
        lidar = self._cast_lidar()           # real ranging (meters)
        self.last_lidar = lidar
        min_lidar = float(lidar.min())

        # —— 4) termination + contact decision (by collision semantics) ——
        reached = dist < self.GOAL_RADIUS
        if self.collision_mode == "bounce":
            # bounce: a wall hit does not terminate (already pushed out + bounced), only a per-contact-step penalty; only reaching terminates.
            contact = wall_contact
            collided = wall_contact
            terminated = bool(reached)
        else:
            # terminate: paper version — out of bounds / hitting a circle terminates.
            collided = self._check_collision(min_lidar)
            contact = collided
            terminated = bool(collided or reached)
        truncated = bool(self.step_count >= self.MAX_STEPS)

        # —— 5) reward synthesis ——
        reward = self._compute_reward(
            dist=dist, w_cmd=w_cmd, min_lidar=min_lidar,
            collided=collided, reached=reached, contact=contact,
        )
        self.prev_dist = dist

        obs = self._get_obs(lidar=lidar)
        info = {
            "is_success": reached,
            "collided": collided,
            "contact": contact,
            "distance": dist,
            "min_lidar": min_lidar,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # dynamics core (Phase1a · F1): force → Newton + viscous damping → semi-implicit substep integration → truth pose
    #   layering: _integrate_dynamics is the shared core (takes force/torque closures, evaluated at the current actual velocity);
    #         _step_dynamics is the A-mode shell (target velocity → P-controller force). B-mode later feeds raw forces directly.
    # ------------------------------------------------------------------
    def _step_dynamics(self, v_tgt, w_tgt):
        """A-mode: convert the target velocity via a P controller into actuator forces, call the shared core to advance the truth pose.

        ENABLE_DYNAMICS=False: falls back to zero-inertia kinematics (v_act≡v_tgt, w_act≡w_tgt integrated directly),
        as a "no dynamics" regression control (the energy ledger has no physical meaning in this mode, zeroed).
        """
        if not self.ENABLE_DYNAMICS:
            self.v_act, self.w_act = v_tgt, w_tgt
            self.theta = self._wrap_angle(self.theta + self.w_act * self.DT)
            self.pos = self.pos + np.array(
                [self.v_act * np.cos(self.theta), self.v_act * np.sin(self.theta)]
            ) * self.DT
            self.E_kin = 0.5 * self.MASS * self.v_act ** 2 \
                + 0.5 * self.INERTIA_COEF * self.MASS * self.w_act ** 2
            self.last_dE = self.last_W_act = self.last_D_damp = 0.0
            return

        # A-mode P controller: force/torque = gain × (target velocity − current actual velocity), evaluated per substep at the live v_act.
        # When v_tgt=0, F=KP·(0−v_act)=−KP·v_act is a "real deceleration force" → natural braking,
        # replacing the original v*=0.5 hard discontinuity (old:716), and its negative work enters the energy ledger faithfully.
        lin_force_of = lambda v: self.KP_V * (v_tgt - v)
        ang_force_of = lambda w: self.KP_W * (w_tgt - w)
        self._integrate_dynamics(lin_force_of, ang_force_of)

    def _step_dynamics_B(self, f_l, f_r):
        """B-mode: raw wheel forces → net force / differential torque → shared core (drop A-mode's P-tracking layer).

        force=f_l+f_r, torque=(f_r−f_l)·ARM (old:708-709). The force is "constant" over a substep (does not vary with v),
        so lin_force_of/ang_force_of are constant closures — dynamics/damping/energy-ledger share the same core as A-mode,
        and the audit (EC1/EC2/EC3) acts on the core, independent of control mode.
        Meaningful inertia: v's time constant τ_v=MASS/C_LIN≈0.333s≈3.3×DT (no P-controller compression).
        """
        force = f_l + f_r
        torque = (f_r - f_l) * self.ARM
        self._integrate_dynamics(lambda v: force, lambda w: torque)

    def _integrate_dynamics(self, lin_force_of, ang_force_of):
        """Shared dynamics core: N_SUB semi-implicit substeps, Newton + viscous damping, settling the exact energy ledger.

        Conservation-law telemetry (key, the foundation of the energy audit):
            the discrete semi-implicit update v_{n+1}=v_n+(h/m)(F−c·v_n) telegraphs ½m·v² exactly:
                ΔKE = h·F·v_mid − h·c·v_n·v_mid,  v_mid=½(v_n+v_{n+1})
            so after defining W_act=Σ F·v_mid·h, D_damp=Σ c·v_n·v_mid·h (using the "declared" constants),
            the clean-run residual r=ΔE−(W_act−D_damp) should be at machine precision (≈0).
            An injector (physics_fault) changes only the "actual integration" while the ledger still settles by the declared constants → the residual becomes nonzero (caught by the audit).
        """
        h = self.DT / self.N_SUB
        m_decl = self.MASS                       # declared mass (for ledger/kinetic energy)
        I_decl = self.INERTIA_COEF * self.MASS   # declared moment of inertia
        c_lin_decl, c_ang_decl = self.C_LIN, self.C_ANG

        # —— unpack fault-injection parameters (clean mode takes all declared values; injection mode secretly changes the "actual" integration parameters) ——
        f = self.physics_fault or {}
        mode = f.get("mode")
        c_lin_eff = f.get("c_lin_eff", c_lin_decl)   # damping used in actual integration (P-1 negative / P-3 zeroed)
        c_ang_eff = f.get("c_ang_eff", c_ang_decl)
        force_mult = f.get("force_mult", 1.0)        # actual force multiplier (P-2 double-count=2.0)
        m_eff = f.get("m_eff", m_decl)               # actual integration mass (≠declared → P-5 misreport)
        skip_lag = f.get("skip_lag", False)          # P-4: skip inertial lag, actual velocity reaches/overshoots the target instantly
        overshoot = f.get("overshoot", 1.0)          # P-4: >1 overshoots the actuator bound (for EC3 detection)
        # G-1 emergent-gaming fault (passive): when |v0|>boost_thresh, apply a "free" thrust boost_force along the motion direction,
        #   the ledger does not account for it → at high speed it gets free energy (steady-state top speed over the physical limit), aligned with the "go fast" goal → any high-speed policy triggers it passively.
        boost_force = f.get("boost_force", 0.0)
        boost_thresh = f.get("boost_thresh", 0.0)
        # G-2 emergent-gaming fault (uniquely-learned): when **both wheels are near-zero actuation** (|net force|<g2_thresh AND |torque|<g2_thresh_tau,
        #   i.e. neither driving nor turning = "doing nothing, pure coasting") AND |v0|>g2_vmin, apply a large free forward force g2_force (ledger does not account for it).
        #   The honest agent **is always actuating** (either driving or turning); measured both-wheels-near-zero-simultaneously fraction = 0% → never triggers, no free-riding;
        #   only learning the **anomalous behavior** "release both wheels, coast on the free force, actuate briefly only when a turn is needed" is profitable (uniquely learned).
        g2_force = f.get("g2_force", 0.0)
        g2_thresh = f.get("g2_thresh", 0.0)          # net-force near-zero threshold (≈ not driving)
        g2_thresh_tau = f.get("g2_thresh_tau", 0.0)  # torque near-zero threshold (≈ not turning)
        g2_vmin = f.get("g2_vmin", 0.05)

        E0 = 0.5 * m_decl * self.v_act ** 2 + 0.5 * I_decl * self.w_act ** 2
        W_act = 0.0
        D_damp = 0.0

        for _ in range(self.N_SUB):
            v0, w0 = self.v_act, self.w_act
            F = lin_force_of(v0)
            tau = ang_force_of(w0)

            if skip_lag:
                # P-4: ignore inertia/damping, lock the actual velocity directly to overshoot × "the steady-state velocity this actuation state should have".
                #      overshoot>1 → persistently over the actuator bound + KE jumps from nothing (breaks the actuator power bound).
                # The steady-state velocity is computed by mode (both bounded, avoiding substep compounding divergence):
                #   A-mode: F=KP(v_tgt−v0) ⇒ steady state = v_tgt = v0 + F/KP;
                #   B-mode: F is a raw force ⇒ steady state = F/c_decl (angular tau/c_ang_decl).
                if self.control_mode == "B":
                    v_tgt_eq = (force_mult * F) / (c_lin_decl if c_lin_decl else 1.0)
                    w_tgt_eq = (force_mult * tau) / (c_ang_decl if c_ang_decl else 1.0)
                else:
                    v_tgt_eq = v0 + (F / self.KP_V if self.KP_V else 0.0)
                    w_tgt_eq = w0 + (tau / self.KP_W if self.KP_W else 0.0)
                v_new = overshoot * v_tgt_eq
                w_new = overshoot * w_tgt_eq
            else:
                # G-1: at high speed get a free thrust (ledger does not account for it) — energy injected from nothing, top speed over the physical limit.
                boost = boost_force * np.sign(v0) if (boost_force and abs(v0) > boost_thresh) else 0.0
                # G-2: when both wheels are near-zero actuation (not driving AND not turning = "doing nothing") while coasting, get a large free forward force (ledger does not account for it).
                g2 = (g2_force * np.sign(v0) if (g2_force
                      and abs(force_mult * F) < g2_thresh
                      and abs(force_mult * tau) < g2_thresh_tau
                      and abs(v0) > g2_vmin) else 0.0)
                a = (force_mult * F + boost + g2 - c_lin_eff * v0) / m_eff
                alpha = (force_mult * tau - c_ang_eff * w0) / (self.INERTIA_COEF * m_eff)
                v_new = v0 + a * h
                w_new = w0 + alpha * h

            v_mid = 0.5 * (v0 + v_new)
            w_mid = 0.5 * (w0 + w_new)
            # the ledger settles by the "declared" force and "declared" damping (an injector changes the actual, the ledger stays declared → the residual exposes the fault)
            W_act += (F * v_mid + tau * w_mid) * h
            D_damp += (c_lin_decl * v0 * v_mid + c_ang_decl * w0 * w_mid) * h

            self.v_act, self.w_act = v_new, w_new
            self.theta = self._wrap_angle(self.theta + self.w_act * h)
            self.pos = self.pos + np.array(
                [self.v_act * np.cos(self.theta), self.v_act * np.sin(self.theta)]
            ) * h

        # kinetic energy uses the "declared" mass (when P-5 misreports, E_kin is inconsistent with the real integration → nonzero residual)
        self.E_kin = 0.5 * m_decl * self.v_act ** 2 + 0.5 * I_decl * self.w_act ** 2
        self.last_dE = self.E_kin - E0
        self.last_W_act = W_act
        self.last_D_damp = D_damp

    # ------------------------------------------------------------------
    # odometry integration (D-018: inject real slip drift, making Odom genuinely deviate from Truth over time)
    # ------------------------------------------------------------------
    def _integrate_odom(self, v, w):
        """Integrate the odometry pose independently using the "slip-perturbed perceived velocity".

        Physical intuition (textbook model of differential-wheel odometry drift):
            wheel-speed encoders perceive the real motion with a multiplicative bias + random noise, dead reckoning accumulates step by step:
                v_odom = v*(1+slip) + N(0, slip)*|v|     linear velocity: systematic bias + proportional noise
                w_odom = w*(1+slip) + N(0, slip)*|w|     angular velocity: systematic bias + proportional noise
        Key points:
            - noise is proportional to speed → odometry does not drift when the robot is stationary (physically correct);
            - the systematic-bias term → error grows monotonically with cumulative path length (the "true fork", not random-walk cancellation);
            - noise comes from self.np_random (seeded) → fully reproducible, never a hardcoded fake curve (INV-2);
            - when slip=0, v_odom≡v, w_odom≡w and no random number is drawn → coincides with truth point by point (degenerates to the original behavior).
        Constraint: this function only reads (v,w) and its own odom state, never writes back self.pos/self.theta (truth is not polluted).
        """
        slip = self.slip_factor
        if slip <= 0.0:
            # degenerate mode (before control / slip=0 regression): odometry integrates identically to truth, odom≡truth, error always 0
            v_odom, w_odom = v, w
        else:
            v_odom = v * (1.0 + slip) + self.np_random.normal(0.0, slip) * abs(v)
            w_odom = w * (1.0 + slip) + self.np_random.normal(0.0, slip) * abs(w)

        # semi-implicit unicycle integration isomorphic to truth, but acting on the odometry's own state and "perceived velocity"
        self.odom_theta = self._wrap_angle(self.odom_theta + w_odom * self.DT)
        self.odom_pos = self.odom_pos + np.array(
            [v_odom * np.cos(self.odom_theta), v_odom * np.sin(self.odom_theta)]
        ) * self.DT

    # ------------------------------------------------------------------
    # reward function (dense shaping: progress-dominated + safety soft constraint + step/smoothness regularization)
    # ------------------------------------------------------------------
    def _compute_reward(self, dist, w_cmd, min_lidar, collided, reached, contact=False):
        # (a) dense progress reward: proportional to the "distance reduction", providing a continuous gradient signal
        reward = self.K_PROGRESS * (self.prev_dist - dist)

        # (b) fixed per-step penalty: encourages short paths, avoids useless circling
        reward += self.STEP_PENALTY

        # (c) angular-velocity smoothness penalty: suppresses high-frequency in-place jitter, smoother trajectory
        reward -= self.K_SMOOTH * abs(w_cmd)

        # (d) near-distance safety soft penalty: linear penalty upon entering the safety buffer (avoid before colliding)
        if min_lidar < self.SAFE_DIST:
            reward -= self.K_SAFETY * (self.SAFE_DIST - min_lidar) / self.SAFE_DIST

        # (e) termination/contact reward (redesigned by collision semantics)
        if reached:
            reward += self.R_GOAL
        elif self.collision_mode == "bounce":
            # bounce: a wall hit does not terminate, a small per-contact-step penalty (replaces R_COLLISION termination), forces learning avoidance but allows recovering after grazing a wall
            if contact:
                reward += self.R_CONTACT
        else:
            # terminate: paper version — a large negative reward terminating on impact
            if collided:
                reward += self.R_COLLISION

        return float(reward)

    # ------------------------------------------------------------------
    # observation construction (normalized)
    # ------------------------------------------------------------------
    def _get_obs(self, lidar=None):
        if lidar is None:
            lidar = self._cast_lidar()
            self.last_lidar = lidar

        lidar_norm = (lidar / self.LIDAR_RANGE).astype(np.float32)         # ∈ [0,1]

        dist = float(np.linalg.norm(self.goal - self.pos))
        dist_norm = np.float32(min(dist / self._max_dist, 1.0))           # ∈ [0,1]

        goal_angle = np.arctan2(self.goal[1] - self.pos[1],
                                self.goal[0] - self.pos[0])
        yaw_err = self._wrap_angle(goal_angle - self.theta)
        yaw_err_norm = np.float32(yaw_err / np.pi)                         # ∈ [-1,1]

        parts = [lidar_norm, np.array([dist_norm, yaw_err_norm], dtype=np.float32)]
        if self.control_mode == "B":
            # B-mode velocity feedback: under force control the velocity is a significant hidden state (τ≈3×step), needs to be in the observation for credit assignment.
            v_norm = np.float32(np.clip(self.v_act / self.V_PHYS_MAX_B, -1.0, 1.0))
            w_norm = np.float32(np.clip(self.w_act / self.W_PHYS_MAX_B, -1.0, 1.0))
            parts.append(np.array([v_norm, w_norm], dtype=np.float32))
        obs = np.concatenate(parts)
        # defensive clipping: rule out out-of-bounds from floating-point error, ensuring it passes env_checker and SB3 validation
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # LiDAR: analytic ray casting (vectorized, O(N_RAYS) numpy ops)
    # ------------------------------------------------------------------
    def _cast_lidar(self):
        """Return each ray's real range (m), truncated at LIDAR_RANGE."""
        angles = self.theta + self._ray_offsets                  # (N,) absolute ray angles
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N,2) unit directions
        dists = np.full(self.N_RAYS, self.LIDAR_RANGE, dtype=np.float64)
        P = self.pos

        # —— (1) ray vs circular obstacle (analytic intersection) ——
        for cx, cy, r in self.obstacles:
            L = np.array([cx, cy]) - P                # from ray origin toward the circle center
            tca = dirs @ L                            # (N,) projection length of the center onto the ray
            d2 = (L @ L) - tca ** 2                   # (N,) squared perpendicular distance from center to ray
            hit = (d2 <= r * r) & (tca > 0)           # the ray really passes through the circle and the circle is ahead
            thc = np.sqrt(np.clip(r * r - d2, 0.0, None))
            t0 = tca - thc                            # near-intersection parameter (i.e. the range)
            valid = hit & (t0 > 0)
            dists = np.where(valid, np.minimum(dists, t0), dists)

        # —— (2) ray vs walls: maze uses ray-AABB (incl. outer walls); random_circle uses the arena's four-sided boundary ——
        if self._walls_arr.shape[0] > 0:
            dists = self._ray_aabbs(P, dirs, dists)
        else:
            dists = self._ray_walls(P, dirs, dists)

        return np.clip(dists, 0.0, self.LIDAR_RANGE)

    def _ray_walls(self, P, dirs, dists):
        """Intersect rays with the axis-aligned rectangular arena boundary, updating the min range vectorized per wall (used by random_circle)."""
        dx, dy = dirs[:, 0], dirs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            for wx in (0.0, self.arena_w):
                t = (wx - P[0]) / dx
                y_hit = P[1] + t * dy
                valid = (dx != 0) & (t > 0) & (y_hit >= 0) & (y_hit <= self.arena_h)
                dists = np.where(valid, np.minimum(dists, t), dists)
            for wy in (0.0, self.arena_h):
                t = (wy - P[1]) / dy
                x_hit = P[0] + t * dx
                valid = (dy != 0) & (t > 0) & (x_hit >= 0) & (x_hit <= self.arena_w)
                dists = np.where(valid, np.minimum(dists, t), dists)
        return dists

    def _ray_aabbs(self, P, dirs, dists):
        """Ray-AABB (slab method, vectorized over rays), updating the min range per wall (used by maze, incl. outer walls)."""
        dx, dy = dirs[:, 0], dirs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            for wx1, wx2, wy1, wy2 in self.walls:
                tx1, tx2 = (wx1 - P[0]) / dx, (wx2 - P[0]) / dx
                ty1, ty2 = (wy1 - P[1]) / dy, (wy2 - P[1]) / dy
                tenter = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
                texit = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
                valid = (tenter <= texit) & (texit > 0) & (tenter > 0)
                t = np.where(valid, tenter, np.inf)
                dists = np.minimum(dists, np.where(np.isfinite(t), t, dists))
        return dists

    # ------------------------------------------------------------------
    # O(1) collision detection: circle-circle + out of bounds
    # ------------------------------------------------------------------
    def _check_collision(self, min_lidar):
        # (a) out of bounds: the chassis circle exceeds the arena
        x, y = self.pos
        if (x - self.ROBOT_RADIUS < 0 or x + self.ROBOT_RADIUS > self.arena_w or
                y - self.ROBOT_RADIUS < 0 or y + self.ROBOT_RADIUS > self.arena_h):
            return True
        # (b) tunneling into an obstacle: center distance < sum of radii (O(1) per obstacle)
        if self.obstacles.shape[0] > 0:
            diff = self.obstacles[:, :2] - self.pos                 # (K,2)
            center_dist = np.linalg.norm(diff, axis=1)              # (K,)
            if np.any(center_dist < self.obstacles[:, 2] + self.ROBOT_RADIUS):
                return True
        return False

    # ------------------------------------------------------------------
    # F4 rectangular collision (Phase2): circle-AABB penetration analysis + pushout + velocity bounce + collision ledger
    # ------------------------------------------------------------------
    def _circle_aabb_overlap(self, pos, walls=None):
        """Return (max_overlap, push_vector): the max penetration and pushout displacement of circle center pos against the wall AABBs (single-wall nearest-point method).
        walls=None takes self.walls; a subset can be passed (the collision-layer "phantom wall" fault passes a subset with the phantom wall dropped)."""
        r = self.ROBOT_RADIUS
        max_pen = 0.0
        push = np.zeros(2)
        for wx1, wx2, wy1, wy2 in (self.walls if walls is None else walls):
            cx = min(max(pos[0], wx1), wx2)      # nearest point on the AABB to the circle center
            cy = min(max(pos[1], wy1), wy2)
            dx, dy = pos[0] - cx, pos[1] - cy
            d = math.hypot(dx, dy)
            if d < r:                            # penetration
                pen = r - d
                if d > 1e-9:
                    nx, ny = dx / d, dy / d
                else:
                    # circle center falls inside the AABB: push out along the minimum-penetration axis (old:649-655)
                    dl, dr_, db, dt = pos[0] - wx1, wx2 - pos[0], pos[1] - wy1, wy2 - pos[1]
                    m = min(dl, dr_, db, dt)
                    if m == dl: nx, ny, pen = -1.0, 0.0, dl + r
                    elif m == dr_: nx, ny, pen = 1.0, 0.0, dr_ + r
                    elif m == db: nx, ny, pen = 0.0, -1.0, db + r
                    else: nx, ny, pen = 0.0, 1.0, dt + r
                if pen > max_pen:
                    max_pen = pen
                push += np.array([nx * pen, ny * pen])
        return max_pen, push

    def _resolve_wall_collisions(self):
        """maze/bounce: circle-AABB penetration pushout (2 iterations to resolve corners) + velocity bounce + collision ledger. No-op if there are no walls.

        Collision-fault injection (physics_fault, hidden here):
            CF-1 over_bounce(bounce_eff>1)  —— bounce adds energy (violates collision energy non-negativity).
            CF-2 skip_pushout               —— skip the penetration correction but settle as usual (violates the non-penetration invariant).
            CF-3 phantom_contact            —— ledger claims collision dissipation while velocity is not decayed (violates energy-ledger self-consistency).
        Ledger: last_E_contact_decl=(1−BOUNCE²)·KE_before (declared); last_E_contact_act=KE_before−KE_after (actual);
              last_penetration=residual penetration after resolution (should ≈0). EC1/EC4/EC5 judge from these.
        """
        self.last_E_contact_decl = 0.0
        self.last_E_contact_act = 0.0
        self.last_penetration = 0.0
        if self._walls_arr.shape[0] == 0:
            return False

        f = self.physics_fault or {}
        e_eff = f.get("bounce_eff", self.BOUNCE)     # CF-1: >1 adds energy
        skip_pushout = f.get("skip_pushout", False)  # CF-2
        phantom = f.get("phantom_contact", False)    # CF-3
        # WP "phantom wall" (Phase4 coupling stress test): the collision layer drops the specified wall index → the robot passes through that wall "legitimately"
        #   (no collision, momentum conserved, energy self-consistent, ledger penetration still 0). The truth really crosses the wall, but the physics ledger cannot see it.
        phantom_walls = f.get("phantom_walls", None)
        active_walls = (self.walls if not phantom_walls
                        else [w for i, w in enumerate(self.walls) if i not in set(phantom_walls)])

        I_decl = self.INERTIA_COEF * self.MASS
        ke_before = 0.5 * self.MASS * self.v_act ** 2 + 0.5 * I_decl * self.w_act ** 2

        contact = False
        for _ in range(2):                            # 2 iterations to resolve corners / narrow slits (old:645)
            pen, push = self._circle_aabb_overlap(self.pos, active_walls)   # the phantom wall does not take part in detection
            if pen > 0.0:
                contact = True
                if not skip_pushout:
                    self.pos = self.pos + push

        if contact:
            # velocity bounce: v/w decay by e (honest e≤1 dissipates; CF-1 e>1 adds energy). phantom does not decay (falsely claims dissipation).
            if not phantom:
                self.v_act *= e_eff
                self.w_act *= e_eff
            ke_after = 0.5 * self.MASS * self.v_act ** 2 + 0.5 * I_decl * self.w_act ** 2
            self.E_kin = ke_after
            self.last_dE += (ke_after - ke_before)          # fold the collision KE change into this step's ΔE
            self.last_E_contact_act = ke_before - ke_after  # actual (CF-1 makes it <0)
            self.last_E_contact_decl = (1.0 - self.BOUNCE ** 2) * ke_before  # declared (by class BOUNCE)
            self.last_penetration = self._circle_aabb_overlap(self.pos, active_walls)[0]  # residual penetration
        return contact

    # ------------------------------------------------------------------
    # utilities: sample a free point / angle normalization
    # ------------------------------------------------------------------
    def _sample_free_point(self, clearance):
        """Rejection-sample a point in the arena that does not overlap any obstacle/wall."""
        for _ in range(300):
            p = np.array([
                self.np_random.uniform(clearance, self.arena_w - clearance),
                self.np_random.uniform(clearance, self.arena_h - clearance),
            ])
            if self.obstacles.shape[0] > 0:
                d = np.linalg.norm(self.obstacles[:, :2] - p, axis=1)
                if not np.all(d > self.obstacles[:, 2] + clearance):
                    continue
            if self._walls_arr.shape[0] > 0 and self._circle_aabb_overlap_clear(p, clearance):
                continue
            return p
        return p  # fallback: return the last sample under extreme crowding

    def _circle_aabb_overlap_clear(self, pos, clearance):
        """True means pos is < clearance from some wall (not open enough, used for spawn/goal rejection sampling)."""
        for wx1, wx2, wy1, wy2 in self.walls:
            cx = min(max(pos[0], wx1), wx2)
            cy = min(max(pos[1], wy1), wy2)
            if math.hypot(pos[0] - cx, pos[1] - cy) < clearance:
                return True
        return False

    @staticmethod
    def _wrap_angle(a):
        """Normalize an arbitrary angle to [-π, π]."""
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # twin render-state export (for inference_server.py to serialize and broadcast to Three.js)
    # ------------------------------------------------------------------
    def get_render_state(self, reward=0.0, terminated=False,
                         truncated=False, info=None):
        """Return a JSON-serializable twin-observation-domain state dict (the frontend data contract)."""
        info = info or {}
        lidar = self.last_lidar if self.last_lidar is not None else self._cast_lidar()
        # odometry pose (with slip drift, independent of truth); before reset, falls back to truth when None to avoid serialization errors
        odom_pos = self.odom_pos if self.odom_pos is not None else self.pos
        odom_theta = self.odom_theta if self.odom_theta is not None else self.theta
        return {
            "robot": {
                "x": float(self.pos[0]),
                "y": float(self.pos[1]),
                "theta": float(self.theta),
                "radius": self.ROBOT_RADIUS,
            },
            # odometry (Odom): the frontend renders this field directly, must not fabricate it from truth (eliminating the "fake instrument")
            "odom": {
                "x": float(odom_pos[0]),
                "y": float(odom_pos[1]),
                "theta": float(odom_theta),
            },
            "goal": {"x": float(self.goal[0]), "y": float(self.goal[1]),
                     "radius": self.GOAL_RADIUS},
            "obstacles": [
                {"x": float(o[0]), "y": float(o[1]), "r": float(o[2])}
                for o in self.obstacles
            ],
            "lidar": [float(d) for d in lidar],       # real ranging (m), the frontend can draw rays
            "lidar_range": self.LIDAR_RANGE,
            "arena": {"w": self.arena_w, "h": self.arena_h},
            # F2 maze walls (AABB): the frontend can render them; random_circle is an empty list (backward compatible)
            "walls": [{"x1": float(w[0]), "x2": float(w[1]), "y1": float(w[2]), "y2": float(w[3])}
                      for w in self.walls],
            "seq": int(self.frame_seq),    # global monotonic frame seq (for the integrity audit to verify frame-seq monotonicity)
            "step": int(self.step_count),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "distance": float(info.get("distance", np.linalg.norm(self.goal - self.pos))),
            # —— Phase1a additive fields (backward compatible: old frontend / contract audit ignore unknown keys, zero impact) ——
            "v_act": float(self.v_act),     # actual linear velocity (with inertial lag; the frontend can show "command vs actual")
            "w_act": float(self.w_act),     # actual angular velocity
            "energy": {                      # energy ledger (consumed by energy_audit; this step's real numbers)
                "E_kin": float(self.E_kin),       # current kinetic energy ½m·v² + ½I·w²
                "dE": float(self.last_dE),        # this step's kinetic-energy change (incl. collision)
                "W_act": float(self.last_W_act),  # actuator net work (by the declared force)
                "D_damp": float(self.last_D_damp),  # damping dissipation (by the declared damping coefficients)
                "E_contact_decl": float(self.last_E_contact_decl),  # declared collision dissipation (by the declared restitution)
                "E_contact_act": float(self.last_E_contact_act),    # actual collision kinetic-energy change (CF-1 makes it <0)
                "penetration": float(self.last_penetration),        # residual penetration after resolution (CF-2 >0)
            },
        }


# register to gymnasium (optional, convenient for gym.make("EmbodiedNav-v0"))
try:
    gym.register(id="EmbodiedNav-v0", entry_point=EmbodiedNavEnv, max_episode_steps=EmbodiedNavEnv.MAX_STEPS)
except Exception:
    pass  # silently skip on duplicate registration


if __name__ == "__main__":
    # self-check: API compliance + random-policy smoke test
    from gymnasium.utils.env_checker import check_env

    env = EmbodiedNavEnv()
    check_env(env)  # passing means fully compliant with the gymnasium standard interface
    print("[OK] check_env passed.")

    obs, info = env.reset(seed=0)
    print(f"[OK] obs shape = {obs.shape}, dtype = {obs.dtype}")
    total = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            obs, info = env.reset()
    print(f"[OK] 50-step random rollout done, accumulated reward = {total:.2f}")
