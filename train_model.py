import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import src.SB_rsk.tasks
import mujoco
import mujoco.viewer
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import draw_target_marker

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2_000_000,
    "env_id": "rsk_pos",
    "algo": "PPO",
    "forward_reward_weight": 1,
    "n_envs": 8,
}

ENV_KWARGS = dict(
    render_mode="rgb_array",
    exclude_current_positions_from_observation=True,
    include_cfrc_ext_in_observation=False,
    forward_reward_weight=config["forward_reward_weight"],
)

def make_env():
    def _init():
        env = gym.make(config["env_id"], **ENV_KWARGS)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    run = wandb.init(
        project="StableBaseline-rsk",
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    train_env = SubprocVecEnv([make_env() for _ in range(config["n_envs"])])

    model = PPO(
        config["policy_type"],
        train_env,
        verbose=1,
        device="cpu",
        tensorboard_log=f"runs/{run.id}"
    )
    print(model.device)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
    # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=10_000,
            verbose=2,
        ),
    )
    model.save("models/ppo_rsk_v1")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
    # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    train_env.close()
    run.finish()

    # Visualisation of the trained model

    viz_env = gym.make(config["env_id"], **ENV_KWARGS)
    obs, _ = viz_env.reset()

    viewer = mujoco.viewer.launch_passive(
        viz_env.unwrapped.model,
        viz_env.unwrapped.data,
        show_left_ui=False,
        show_right_ui=False
    )

    try:
        for i in range(10_000):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = viz_env.step(action)
            if terminated or truncated:
                obs, _ = viz_env.reset()

            draw_target_marker(
                viewer,
                viz_env.unwrapped._target_x,
                viz_env.unwrapped._target_y,
            )

            viewer.sync()
            time.sleep(1/60)
    except KeyboardInterrupt:
        pass

    viz_env.close()
    viewer.close()