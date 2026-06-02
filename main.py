import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
import src.SB_rsk.tasks
import mujoco
import mujoco.viewer

env = gym.make("Antony", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)

vec_env = model.get_env()
obs = vec_env.reset()

viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

try:
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        zero_action = np.array([[0.0, 0.0, 0.0, 0.0]])
        obs, reward, done, info = vec_env.step(action)

        viewer.sync()
except KeyboardInterrupt:
    pass