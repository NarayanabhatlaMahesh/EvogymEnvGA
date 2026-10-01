import gymnasium as gym
import numpy as np
from evogym.envs.base import EvoGymBase
from evogym import EvoViewer


class JsonWorldEnv(EvoGymBase):
    metadata = {"render_modes": ["human"]}

    def render(self):
        """
        Render full 30x30 voxel world with fixed camera.
        """
        if not hasattr(self, "_viewer"):
            self._viewer = EvoViewer(
                self.sim,
                pos=(20, 5),          # center of 30x30 world
                view_size=(45, 20),    # FULL ENVIRONMENT
                resolution=(3600, 1800),
                target_rps=None        # fastest
            )
        return self._viewer.render(mode="img")

    def __init__(
        self,
        world,
        robot_name,
        total_timesteps,
        render_mode=None,
    ):
        super().__init__(world=world, render_mode=render_mode)

        self.robot_name = robot_name

        # =======================
        # TIMING
        # =======================
        self.sim_steps = 8
        self.max_steps = 1000
        self.step_count = 0

        self.total_timesteps = total_timesteps
        self.global_step = 0

        # =======================
        # SPACES
        # =======================
        self.n_act = len(self.get_actuator_indices(robot_name))

        # We define space based on the CURRENT robot
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_act + 1,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        # =======================
        # STATE
        # =======================
        self.prev_x = self.prev_y = 0.0
        self.prev_theta = 0.0
        self.ground_y = None

        self.filtered_action = None
        self.noise_state = None

        self.ground_steps = 0
        self.stagnant_steps = 0

    def _get_obs(self):
        vel = self.get_vel_com_obs(self.robot_name)
        pos = self.get_pos_com_obs(self.robot_name)

        theta = float(self.get_ort_obs(self.robot_name)[0])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ang_vel = theta - self.prev_theta

        return np.array(
            [
                vel[0],
                vel[1],
                pos[1] - self.ground_y,
                theta,
                ang_vel
            ],
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Re-detect actuator count in case the world/robot changed
        self.n_act = len(self.get_actuator_indices(self.robot_name))

        pos = self.get_pos_com_obs(self.robot_name)
        self.prev_x, self.prev_y = pos
        self.ground_y = pos[1]

        theta = float(self.get_ort_obs(self.robot_name)[0])
        self.prev_theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # CLEAR memory so shapes re-initialize in step()
        self.filtered_action = None
        self.noise_state = None

        self.step_count = 0
        self.ground_steps = 0
        self.stagnant_steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.global_step += 1

        # =======================
        # DYNAMIC SHAPE CORRECTION
        # =======================
        # Ensure 'action' matches the robot's physical actuator count + 1
        expected_size = self.n_act + 1
        if action.shape[0] != expected_size:
            # Slice if too large, pad with zeros if too small
            if action.shape[0] > expected_size:
                action = action[:expected_size]
            else:
                action = np.pad(action, (0, expected_size - action.shape[0]))

        # =======================
        # GLOBAL PROGRESS
        # =======================
        p = min(1.0, self.global_step / self.total_timesteps)
        p2 = p * p

        # =======================
        # ACTION PROCESSING
        # =======================
        raw = np.clip(action[:-1], -1.0, 1.0)
        speed = np.clip(action[-1], -1.0, 1.0)

        speed_bias = 0.6 * (1.0 - p)
        speed = np.clip(speed + speed_bias, -1.0, 1.0)

        amp = 1.0 - 0.25 * p2
        target = 1.0 + amp * raw

        # Initialize noise_state to match CURRENT robot muscles
        if self.noise_state is None:
            self.noise_state = np.random.randn(self.n_act)

        # Physics/Noise math (now guaranteed to match shapes)
        self.noise_state = 0.93 * self.noise_state + 0.03 * np.random.randn(self.n_act)
        target += (0.9 * (1.0 - p2) + 0.2) * self.noise_state

        target = np.clip(
            target,
            0.2 + p * 0.5,
            1.8 - p * 0.6
        )

        alpha = (0.8 * (1 - p) + 0.5 * p)
        alpha *= 0.3 + 0.7 * (speed + 1.0) / 2.0

        self.filtered_action = (
            target if self.filtered_action is None
            else (1 - alpha) * self.filtered_action + alpha * target
        )

        # =======================
        # PHYSICS
        # =======================
        for _ in range(self.sim_steps):
            super().step({self.robot_name: self.filtered_action})

        # =======================
        # STATE UPDATE
        # =======================
        pos = self.get_pos_com_obs(self.robot_name)
        x, y = pos

        dx = x - self.prev_x
        dy = y - self.prev_y

        self.prev_x, self.prev_y = x, y
        self.ground_y = min(self.ground_y, y)

        theta = float(self.get_ort_obs(self.robot_name)[0])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        ang_vel = theta - self.prev_theta
        self.prev_theta = theta

        airborne = y > self.ground_y + 0.25
        self.ground_steps = 0 if airborne else self.ground_steps + 1
        self.stagnant_steps = self.stagnant_steps + 1 if abs(dx) < 0.03 else 0

        # =======================
        # REWARD
        # =======================
        reward = -0.2

        if airborne and dy > 0.08 and dx > 0.05:
            reward += (4.0 + 16.0 * p2) * dy

        reward += 3.0 * dx
        reward -= 8.0 * max(-dx, 0.0)

        reward -= 0.8 * (theta ** 2)
        reward -= 0.3 * (ang_vel ** 2)

        if self.stagnant_steps > 15:
            reward -= 5.0

        if not airborne and p > 0.5:
            reward -= 0.5

        # =======================
        # TERMINATION
        # =======================
        # Corrected: ensure theta is float before comparison
        if float(theta) < -1.2:
            reward -= 40.0
            reward = np.clip(reward, -20.0, 20.0)
            return self._get_obs(), float(reward), True, False, {}

        reward = np.clip(reward, -20.0, 20.0)
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), float(reward), False, truncated, {}