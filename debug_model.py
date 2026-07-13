import time
import argparse

import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import src.SB_rsk.tasks  # nécessaire pour enregistrer l'env "RSK"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le .zip du modèle")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        "RSK",
        render_mode=None,  # pas de MujocoRenderer interne, on gère notre propre viewer
        exclude_current_positions_from_observation=False,
        include_cfrc_ext_in_observation=False,
        forward_reward_weight=1,
    )

    model = PPO.load(args.model, env=env)

    viewer = mujoco.viewer.launch_passive(
        env.unwrapped.model,
        env.unwrapped.data,
        show_left_ui=False,
        show_right_ui=False,
    )

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0

            while not done and viewer.is_running():
                action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1

                viewer.sync()
                time.sleep(1 / 60)

            print(f"Episode {ep}: reward={ep_reward:.3f}, steps={step_count}, "
                  f"terminated={terminated}, truncated={truncated}")
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        env.close()

if __name__ == "__main__":
    main()