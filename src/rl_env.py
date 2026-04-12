import gym
from gym import spaces
import numpy as np
import random

class EdgeSchedulerEnv(gym.Env):

    def __init__(self):
        super(EdgeSchedulerEnv, self).__init__()

        # -----------------------------
        # STATE SPACE (Telemetry)
        # -----------------------------
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([100, 100, 20, 8000]),
            dtype=np.float32
        )

        # -----------------------------
        # ACTION SPACE
        # 0 = RUN_BOTH
        # 1 = SKIP_YOLO
        # 2 = DELAY_BERT
        # -----------------------------
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.step_count = 0

    def reset(self):
        self.state = np.array([
            random.uniform(50, 70),   # temp
            random.uniform(40, 80),   # gpu
            random.uniform(5, 15),    # power
            random.uniform(1000, 4000) # ram
        ])
        self.step_count = 0
        return self.state

    def step(self, action):

        temp, gpu, power, ram = self.state

        # -----------------------------
        # SIMULATE EFFECT OF ACTION
        # -----------------------------
        if action == 0:  # RUN_BOTH
            temp += random.uniform(1, 3)
            gpu += random.uniform(5, 10)
            power += random.uniform(1, 3)

        elif action == 1:  # SKIP_YOLO
            temp -= random.uniform(1, 2)
            gpu -= random.uniform(3, 6)
            power -= random.uniform(1, 2)

        elif action == 2:  # DELAY_BERT
            temp -= random.uniform(0.5, 1.5)
            gpu -= random.uniform(2, 5)
            power -= random.uniform(0.5, 1.5)

        # Clamp values
        temp = np.clip(temp, 30, 90)
        gpu = np.clip(gpu, 0, 100)
        power = np.clip(power, 0, 20)
        ram = np.clip(ram + random.uniform(-100, 100), 500, 8000)

        self.state = np.array([temp, gpu, power, ram])

        # -----------------------------
        # REWARD FUNCTION
        # -----------------------------
        reward = (
            -0.3 * temp
            -0.2 * power
            -0.2 * gpu
            + 1.0 * (action == 0)   # encourage running both
        )

        # Penalty for overheating
        if temp > 80:
            reward -= 20

        self.step_count += 1
        done = self.step_count >= 100

        return self.state, reward, done, {}
