"""
rl_env.py — SchedulerEnv for Jetson Orin NX 8 GB (precision-switching)

Action space (4 discrete):
    0 = YOLO FP16  + BERT FP16   (max accuracy, highest load)
    1 = YOLO INT8  + BERT FP16   (reduce vision load)
    2 = YOLO FP16  + BERT INT8   (reduce NLP load)
    3 = YOLO INT8  + BERT INT8   (max efficiency, lowest load)

Both models ALWAYS run — the scheduler only changes their precision.
This matches the thesis proposal and produces richer comparisons than
skipping one model entirely.

Observation space (5 features):
    [gpu_temp_c, gpu_util_pct, power_w, ram_used_mb, temp_trend_c]
"""

import gym
from gym import spaces
import numpy as np
import random


# ─────────────────────────────────────────────
# Precision profiles (realistic Orin NX numbers)
# Each entry: (gpu_delta, temp_delta, power_delta, accuracy_score)
#
# FP16 → higher accuracy, more compute
# INT8 → lower accuracy, less compute (~40% faster, ~45% less power)
# ─────────────────────────────────────────────
PROFILES = {
    #           YOLO gpu  temp  power   BERT gpu  temp  power  joint_accuracy
    0: dict(yolo_gpu=22.0, yolo_temp=2.5, yolo_power=2.5,   # YOLO FP16
            bert_gpu=10.0, bert_temp=1.0, bert_power=1.0,   # BERT FP16
            accuracy=1.00),

    1: dict(yolo_gpu=13.0, yolo_temp=1.5, yolo_power=1.5,   # YOLO INT8
            bert_gpu=10.0, bert_temp=1.0, bert_power=1.0,   # BERT FP16
            accuracy=0.88),

    2: dict(yolo_gpu=22.0, yolo_temp=2.5, yolo_power=2.5,   # YOLO FP16
            bert_gpu= 6.0, bert_temp=0.6, bert_power=0.6,   # BERT INT8
            accuracy=0.90),

    3: dict(yolo_gpu=13.0, yolo_temp=1.5, yolo_power=1.5,   # YOLO INT8
            bert_gpu= 6.0, bert_temp=0.6, bert_power=0.6,   # BERT INT8
            accuracy=0.80),
}

ACTION_NAMES = {
    0: "YOLO_FP16+BERT_FP16",
    1: "YOLO_INT8+BERT_FP16",
    2: "YOLO_FP16+BERT_INT8",
    3: "YOLO_INT8+BERT_INT8",
}


class SchedulerEnv(gym.Env):

    # Jetson Orin NX 8 GB hardware limits
    TEMP_MAX  = 95.0
    TEMP_WARN = 70.0
    TEMP_SAFE = 65.0
    GPU_WARN  = 90.0
    POWER_MAX = 15.0
    RAM_MAX   = 7000.0
    RAM_MIN   = 500.0

    def __init__(self):
        super(SchedulerEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array( [20.0,   0.0, 0.0, self.RAM_MIN, -15.0], dtype=np.float32),
            high=np.array([self.TEMP_MAX, 100.0, self.POWER_MAX, self.RAM_MAX, 15.0],
                          dtype=np.float32)
        )

        self.action_space = spaces.Discrete(4)

        self.state      = None
        self.prev_temp  = None
        self.step_count = 0

    def reset(self):
        temp  = random.uniform(40.0, 58.0)
        gpu   = random.uniform(15.0, 45.0)
        power = random.uniform(3.0,  7.0)
        ram   = random.uniform(1500.0, 3500.0)

        self.prev_temp  = temp
        self.step_count = 0
        self.state = np.array([temp, gpu, power, ram, 0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        temp, gpu, power, ram, _ = self.state
        profile = PROFILES[int(action)]

        # 1. Natural cooldown
        gpu   = max(0.0,  gpu   - random.uniform(3.0, 8.0))
        temp  = max(20.0, temp  - random.uniform(0.4, 1.2))
        power = max(0.0,  power - random.uniform(0.3, 0.7))

        # 2. Stochastic background burst
        if random.random() < 0.4:
            gpu   += random.uniform(10.0, 25.0)
            temp  += random.uniform(1.5,  4.0)
            power += random.uniform(1.0,  2.5)

        # 3. Action effects — both models always run
        gpu   += profile["yolo_gpu"]   + profile["bert_gpu"]
        temp  += profile["yolo_temp"]  + profile["bert_temp"]
        power += profile["yolo_power"] + profile["bert_power"]
        ram   += random.uniform(80.0, 350.0)
        accuracy = profile["accuracy"]

        # 4. RAM drift
        ram += random.uniform(-200.0, 100.0)

        # 5. Clip
        gpu   = float(np.clip(gpu,   0.0, 100.0))
        temp  = float(np.clip(temp,  20.0, self.TEMP_MAX))
        power = float(np.clip(power, 0.0, self.POWER_MAX))
        ram   = float(np.clip(ram,   self.RAM_MIN, self.RAM_MAX))

        # 6. Temperature trend
        temp_trend = float(np.clip(temp - self.prev_temp, -15.0, 15.0))
        avg_temp   = (temp + self.prev_temp) / 2.0

        # 7. Reward — normalized so all terms are roughly ±5
        reward = (
            + 5.0  * accuracy
            - 0.05 * avg_temp
            - 0.03 * gpu
            - 0.15 * power
        )

        if temp > self.TEMP_WARN:
            reward -= 12.0
        if gpu > self.GPU_WARN:
            reward -= 8.0
        if 48.0 < temp < self.TEMP_SAFE:
            reward += 3.0
        # Encourage FP16 when system is cool (max accuracy when headroom allows)
        if action == 0 and temp < 55.0 and gpu < 60.0:
            reward += 2.0
        # Encourage full INT8 under genuine thermal stress (proactive saving)
        if action == 3 and temp > 62.0:
            reward += 1.5

        # 8. Update state
        self.prev_temp = temp
        self.state = np.array([temp, gpu, power, ram, temp_trend], dtype=np.float32)
        self.step_count += 1
        done = self.step_count >= 200

        info = {
            "accuracy":    accuracy,
            "avg_temp":    avg_temp,
            "action_name": ACTION_NAMES[int(action)],
        }

        return self.state, reward, done, info

    def render(self, mode="human"):
        temp, gpu, power, ram, trend = self.state
        print(
            f"Step {self.step_count:3d} | "
            f"Temp={temp:.1f}C  GPU={gpu:.1f}%  "
            f"Power={power:.2f}W  RAM={ram:.0f}MB  Trend={trend:+.2f}"
        )
