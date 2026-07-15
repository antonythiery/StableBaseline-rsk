## Fichiers

launcher.py permet de lancer l'entrainement. Cet entrainement est définit dans src/SB_rsk/tasks
mujoco_env.py est un fichier générique de RL mujoco. Ce fichier est appelé par rsk_vel.py
rsk_vel.py permet de définir l'entrainement du robot. Il contient les rewards, les observations, le reset etc ...



## Commands

- Run training : 
uv run train_model.py

- Run model testing : 
uv run debug_model.py --model models/ppo_rsk_v1.zip
