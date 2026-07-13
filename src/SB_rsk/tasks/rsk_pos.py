__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
import mujoco
# from gymnasium.envs.mujoco import MujocoEnv
from src.SB_rsk.tasks.mujoco_env import MujocoEnv 
from gymnasium.spaces import Box

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "robot" / "rsk" / "scene.xml"
model_path = str(ROBOT_XML)

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.0,
}


class RSKEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = model_path,
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: int | str = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0, 0.5),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = False,
        include_cfrc_ext_in_observation: bool = True,
        lateral_cost_weight: float = 1.0,
        angular_cost_weight: float = 0.1,
        target_radius: float = 0.1,
        progress_reward_weight: float = 1.0,
        distance_cost_weight: float = 1.0,
        stopping_reward_weight: float = 1.0,
        stopping_scale: float = 0.1,
        arrival_bonus: float = 10.0,
        target_range_min: float = 0.5,
        target_range_max: float = 1.5,
        **kwargs,
    ):  
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            target_radius,
            progress_reward_weight,
            distance_cost_weight,
            stopping_reward_weight,
            stopping_scale,
            arrival_bonus,
            target_range_min,
            target_range_max,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self._lateral_cost_weight = lateral_cost_weight
        self._angular_cost_weight = angular_cost_weight

        self._target_range_min = target_range_min
        self._target_range_max = target_range_max
        self._target_x = 1.0
        self._target_y = 0.0
        self._target_radius = target_radius
        self._progress_reward_weight = progress_reward_weight
        self._distance_cost_weight = distance_cost_weight    
        self._stopping_reward_weight = stopping_reward_weight
        self._stopping_scale = stopping_scale
        self._arrival_bonus = arrival_bonus 

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        base_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
        self._base_dofadr = int(self.model.jnt_dofadr[base_jid])

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        obs_size += 2 # for target_relative (x, y)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "target_relative": 2,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        x_position = float(self.data.qpos[self._base_dofadr + 0])
        y_position = float(self.data.qpos[self._base_dofadr + 1])
        yaw_position = float(self.data.qpos[self._base_dofadr + 5])
        x_velocity = float(self.data.qvel[self._base_dofadr + 0])
        y_velocity = float(self.data.qvel[self._base_dofadr + 1])
        yaw_velocity = float(self.data.qvel[self._base_dofadr + 5])

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_position, y_position, x_velocity, y_velocity, yaw_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position,
            "y_position": y_position,
            "yaw_position": yaw_position,
            "distance_from_origin": np.linalg.norm([x_position, y_position], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "yaw_velocity": yaw_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_xy_position(self):
        x_position = float(self.data.qpos[self._base_dofadr + 0])
        y_position = float(self.data.qpos[self._base_dofadr + 1])
        return x_position, y_position

    def _get_rew(self, x_position, y_position, x_velocity, y_velocity, yaw_velocity, action):
        dx = self._target_x - x_position
        dy = self._target_y - y_position
        distance_to_target = np.sqrt(dx**2 + dy**2)

        progress_reward = (self._prev_distance_to_target - distance_to_target) * self._progress_reward_weight
        self._prev_distance_to_target = distance_to_target

        distance_cost = self._distance_cost_weight * distance_to_target

        speed = np.sqrt(x_velocity**2 + y_velocity**2)
        at_target = distance_to_target < self._target_radius
        if at_target:
            stopping_reward = self._stopping_reward_weight * np.exp(-speed / self._stopping_scale)
            arrival_bonus = self._arrival_bonus
        else:
            stopping_reward = 0.0
            arrival_bonus = 0.0

        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        angular_cost = self._angular_cost_weight * (yaw_velocity**2)

        reward = (
            progress_reward
            + healthy_reward
            + stopping_reward
            + arrival_bonus
            - ctrl_cost 
            - contact_cost
            - distance_cost
            - angular_cost
        )

        reward_info = {
            "reward_progress": progress_reward,
            "reward_stopping": stopping_reward,
            "reward_arrival_bonus": arrival_bonus,
            "reward_distance_cost": -distance_cost,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "reward_angular": -angular_cost,
            "distance_to_target": distance_to_target,
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        x_position, y_position = self._get_xy_position()
        target_relative = np.array([
            self._target_x - x_position,
            self._target_y - y_position,
        ])

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force, target_relative))
        else:
            return np.concatenate((position, velocity, target_relative))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        angle = self.np_random.uniform(0, 2 * np.pi)
        radius = self.np_random.uniform(self._target_range_min, self._target_range_max)
        # self._target_x = radius * np.cos(angle)
        # self._target_y = radius * np.sin(angle)

        x_position, y_position = self._get_xy_position()
        self._prev_distance_to_target = np.sqrt(
            (self._target_x - x_position) ** 2 + (self._target_y - y_position) ** 2
        )

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
