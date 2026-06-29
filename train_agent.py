# -*- coding: utf-8 -*-
"""
train_agent.py
==============
The training state of the "compute-swap architecture": a hyper-real-time PPO training script.

Core principle — fully strip away the scheduling shell:
    this script only interacts with embodied_env.py's pure-compute kernel, **does not import** fastapi / websockets at all,
    and has no asyncio.sleep / 60Hz heartbeat. The env is driven repeatedly by PPO's rollout collector via synchronous
    env.step() function calls at the CPU's limit speed — this is exactly the engineering landing point of swapping
    "real-time rendering/communication compute" for "pure training compute": one core saturated, zero clock throttling.

Monitoring: integrated with TensorBoard, automatically logging metrics like rollout/ep_rew_mean (the cumulative-reward convergence curve).
Artifact: the trained weights are saved as ppo_embodied_agent.pth (the policy's state_dict).
"""

import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from embodied_env import EmbodiedNavEnv

# ====================== training hyperparameters ======================
# overridable by environment variables (dev branch: defaults to producing *_dyn weights, never overwrites the paper .pth / dyn.pth)
TOTAL_TIMESTEPS = int(os.environ.get("EMBODIED_TIMESTEPS", 1_000_000))  # total training steps
CONTROL_MODE = os.environ.get("EMBODIED_CONTROL_MODE", "A")             # 'A'=target velocity / 'B'=force control
MAP_TYPE = os.environ.get("EMBODIED_MAP_TYPE", "random_circle")         # 'random_circle' / 'maze'
MODEL_PATH = os.environ.get("EMBODIED_MODEL_PATH", "ppo_embodied_agent_dyn.pth")  # weight output
SB3_NATIVE_PATH = os.environ.get("EMBODIED_SB3_PATH", "ppo_embodied_agent_dyn")   # SB3 native zip
CKPT_PREFIX = os.environ.get("EMBODIED_CKPT_PREFIX", "ppo_dyn_ckpt")    # checkpoint prefix
TB_LOG_DIR = "./tb_embodied/"      # TensorBoard log directory
CHECKPOINT_DIR = "./checkpoints/"  # periodic-checkpoint directory

# policy network structure: deliberately kept lightweight (a two-layer 64 MLP), matching the "lightweight base" principle.
# Note: the inference script inference_server.py must use the **exact same** policy_kwargs,
#       otherwise load_state_dict fails due to layer-shape mismatch.
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))


class RewardLogCallback(BaseCallback):
    """Lightweight callback: additionally logs each episode's cumulative reward to TensorBoard's custom/ namespace,
    for finer-grained convergence observation beyond ep_rew_mean."""

    def _on_step(self) -> bool:
        # SB3 carries Monitor's episode stats in info["episode"] (r=cumulative reward, l=steps)
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.logger.record("custom/episode_reward", ep["r"])
                self.logger.record("custom/episode_length", ep["l"])
                if info.get("is_success"):
                    self.logger.record("custom/success", 1.0)
        return True


# —— emergent gaming experiment: an exploitable physics fault can be injected during training (hidden inside the integrator) ——
#    G-1 (passive): EMBODIED_BOOST_FORCE>0, free thrust at high speed, aligned with "go fast" → any high-speed policy triggers it passively.
#    G-2 (uniquely-learned): EMBODIED_G2_FORCE>0, near-zero-thrust coasting gets a large free forward force; only learning "cut thrust and coast" is profitable.
BOOST_FORCE = float(os.environ.get("EMBODIED_BOOST_FORCE", 0.0))
BOOST_THRESH = float(os.environ.get("EMBODIED_BOOST_THRESH", 1.05))
G2_FORCE = float(os.environ.get("EMBODIED_G2_FORCE", 0.0))
G2_THRESH = float(os.environ.get("EMBODIED_G2_THRESH", 0.35))      # net-force near-zero threshold
G2_THRESH_TAU = float(os.environ.get("EMBODIED_G2_THRESH_TAU", 0.3))  # torque near-zero threshold (both wheels idle)


def make_env():
    """Construct a single env (wrapped in Monitor to collect episode stats).
    A single DummyVecEnv instance suffices for the one-core-saturated scenario; to use multiple cores, change this to
    SubprocVecEnv + multiple make_env for a linear speedup."""
    env = EmbodiedNavEnv(render_mode=None, control_mode=CONTROL_MODE, map_type=MAP_TYPE)
    if BOOST_FORCE > 0.0:
        env.physics_fault = {"mode": "G-1_speed_boost",
                             "boost_force": BOOST_FORCE, "boost_thresh": BOOST_THRESH}
    elif G2_FORCE > 0.0:
        env.physics_fault = {"mode": "G-2_lazy_coast", "g2_force": G2_FORCE,
                             "g2_thresh": G2_THRESH, "g2_thresh_tau": G2_THRESH_TAU}
    env = Monitor(env)
    return env


def main():
    # lock torch thread count to 1: matching the compute-swap setting of "one CPU core saturated",
    # and avoiding the small network being slowed by scheduling overhead under multithreading.
    torch.set_num_threads(1)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,          # sampling steps per rollout
        batch_size=256,
        n_epochs=10,
        gamma=0.99,            # discount factor: navigation rewards are fairly long-horizon, use 0.99
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=TB_LOG_DIR,
        device="cpu",          # a lightweight MLP on CPU is actually faster than GPU scheduling
        verbose=1,
    )

    callbacks = [
        RewardLogCallback(),
        CheckpointCallback(save_freq=50_000, save_path=CHECKPOINT_DIR,
                           name_prefix=CKPT_PREFIX),
    ]

    print(">>> entering the super-real-time training loop (no network / no async clock, single core saturated)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
                progress_bar=True)

    # —— save artifacts ——
    # (1) save the policy weights as .pth as required (the inference script loads via load_state_dict)
    torch.save(model.policy.state_dict(), MODEL_PATH)
    print(f">>> weights saved: {MODEL_PATH}")

    # (2) also save the SB3 native zip archive (contains the full hyperparameters/optimizer, most robust, strongly recommended to keep)
    model.save(SB3_NATIVE_PATH)
    print(f">>> SB3 native archive saved: {SB3_NATIVE_PATH}.zip")
    print(">>> training complete. Run `tensorboard --logdir ./tb_embodied/` to view the convergence curves.")


if __name__ == "__main__":
    main()
