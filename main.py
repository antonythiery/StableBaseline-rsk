import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
import src.SB_rsk.tasks
import mujoco
import mujoco.viewer
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("RSK", render_mode="rgb_array", include_cfrc_ext_in_observation=False)

print(f"env.action_space : {env.action_space}")

model = A2C("MlpPolicy", env, verbose=1) # maybe try PPO insted but heavier to train
print(model.device)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
model.learn(total_timesteps=10_000)
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

vec_env = model.get_env()
obs = vec_env.reset()

viewer = mujoco.viewer.launch_passive(
    env.unwrapped.model,
    env.unwrapped.data,
    show_left_ui=False,
    show_right_ui=False
)

try:
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        viewer.sync()
except KeyboardInterrupt:
    pass