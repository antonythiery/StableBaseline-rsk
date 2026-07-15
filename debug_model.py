import time
import argparse
import threading

import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import src.SB_rsk.tasks  # nécessaire pour enregistrer l'env "RSK"
from utils import draw_target_marker

class TargetInputHandler:
    def __init__(self):
        self.target_x = 0.0
        self.target_y = 0.0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._input_loop, daemon=True)
        self.thread.start()
    
    def _input_loop(self):
        while True:
            try:
                user_input = input("Entrez target (x y) ou 'reset': ").strip()
                if user_input.lower() == 'reset':
                    with self.lock:
                        self.target_x = 0.0
                        self.target_y = 0.0
                    print("Target réinitialisé à (0.0, 0.0)")
                else:
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            with self.lock:
                                self.target_x = x
                                self.target_y = y
                            print(f"Target défini à ({x}, {y})")
                        except ValueError:
                            print("Erreur: entrez deux nombres séparés par un espace")
                    else:
                        print("Erreur: format invalide. Utilisez 'x y' ou 'reset'")
            except EOFError:
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le .zip du modèle")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        "rsk_pos",
        render_mode=None,
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

    input_handler = TargetInputHandler()

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()

            with input_handler.lock:
                env.unwrapped._target_x = input_handler.target_x
                env.unwrapped._target_y = input_handler.target_y

            done = False
            ep_reward = 0.0
            step_count = 0

            while not done and viewer.is_running():

                with input_handler.lock:
                    env.unwrapped._target_x = input_handler.target_x
                    env.unwrapped._target_y = input_handler.target_y

                action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1

                draw_target_marker(
                    viewer,
                    env.unwrapped._target_x,
                    env.unwrapped._target_y,
                )

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