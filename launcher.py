import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
import src.SB_rsk.tasks
import mujoco
import mujoco.viewer
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="StableBaseline-rsk",
    config=wandb.config,
    sync_tensorboard=True,
    monitor_gym=False, # don't log videos"
    save_code=True,
)

env = gym.make(
    "RSK",
    # render_mode="human",
    render_mode="rgb_array",
    exclude_current_positions_from_observation=False, 
    include_cfrc_ext_in_observation=False,
    forward_reward_weight = 1
)
env = Monitor(env)

print(f"env.action_space : {env.action_space}")

# model = A2C("MlpPolicy", env, verbose=1) # maybe try PPO insted but heavier to train
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
print(model.device)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
model.learn(
    total_timesteps=100_000,
    callback=WandbCallback(
        gradient_save_freq=100,       # log les gradients tous les N steps (optionnel, un peu lourd)
        model_save_path=f"models/{run.id}",
        model_save_freq=10_000,
        verbose=2,
    ),
)
model.save("models/ppo_rsk_v1")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

vec_env = model.get_env()
obs = vec_env.reset()

env.close()
viewer = mujoco.viewer.launch_passive(
    env.unwrapped.model,
    env.unwrapped.data,
    show_left_ui=False,
    show_right_ui=False
)

try:
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)

        viewer.sync()
        time.sleep(1/60)
except KeyboardInterrupt:
    pass