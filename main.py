import numpy as np
import gymnasium as gym

from stable_baselines3 import A2C

import src.SB_rsk.tasks

env = gym.make("Antony", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    # action, _state = model.predict(obs, deterministic=True)
    zero_action = np.array([[0.0, 0.0, 0.0, 0.0]])
    obs, reward, done, info = vec_env.step(zero_action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()