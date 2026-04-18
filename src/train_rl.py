"""
train_rl.py — PPO training for the Jetson Orin NX multimodal scheduler

Key fixes vs original:
  - 300,000 timesteps (up from 60,000) for meaningful convergence
  - EvalCallback saves the best model, not just the final snapshot
  - Action-distribution check printed after training to verify the agent
    is not collapsing to a single action (a sign of reward imbalance)
  - Reproducible seed set for NumPy and PyTorch
  - Hyperparameters tuned for the 5-dim observation / 3-action space
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl_env import SchedulerEnv


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────
BEST_MODEL_DIR  = "./best_model/"
CHECKPOINT_DIR  = "./checkpoints/"

os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────
# Wrap in Monitor so SB3 can track episode rewards/lengths
train_env = Monitor(SchedulerEnv())
eval_env  = Monitor(SchedulerEnv())


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    eval_freq=5_000,          # evaluate every 5 k steps
    n_eval_episodes=20,       # 20 episodes per evaluation
    deterministic=True,
    render=False,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_scheduler",
    verbose=1
)


# ─────────────────────────────────────────────
# PPO Model
#
# Key hyperparameter notes:
#   n_steps=2048  — rollout buffer length per update
#   batch_size=64 — mini-batch size for gradient updates
#   n_epochs=10   — how many times each rollout is reused
#   gamma=0.99    — discount; suits 200-step episodes
#   ent_coef=0.01 — entropy bonus to prevent premature convergence
#                   to a single action (critical given reward issues)
# ─────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    seed=SEED,
    # --- core hyperparameters ---
    gamma=0.99,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    # --- exploration ---
    ent_coef=0.01,          # entropy regularisation (was 0 in original)
    # --- value function ---
    vf_coef=0.5,
    max_grad_norm=0.5,
)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 300_000   # 5× original — needed for convergence

print(f"\n{'='*55}")
print(f"  Training PPO scheduler for {TOTAL_TIMESTEPS:,} timesteps")
print(f"  Best model → {BEST_MODEL_DIR}")
print(f"  Checkpoints → {CHECKPOINT_DIR}")
print(f"{'='*55}\n")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
)

# Save final model as well (best_model/ already has the best checkpoint)
model.save("ppo_scheduler_final")
print("\nFinal model saved as: ppo_scheduler_final.zip")


# ─────────────────────────────────────────────
# Post-training sanity check: action distribution
#
# If the agent is healthy it should choose all three
# actions in roughly: RUN_BOTH when cool, YOLO_ONLY
# when warm, BERT_ONLY only under severe thermal stress.
# A distribution of 100 % one action = reward collapse.
# ─────────────────────────────────────────────
print("\n--- Post-training action distribution (100 eval episodes) ---")
action_counts = {0: 0, 1: 0, 2: 0}
action_names  = {0: "RUN_BOTH", 1: "YOLO_ONLY", 2: "BERT_ONLY"}

check_env = SchedulerEnv()
for _ in range(100):
    obs   = check_env.reset()
    done  = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = check_env.step(int(action))
        action_counts[int(action)] += 1

total_actions = sum(action_counts.values())
for a, count in action_counts.items():
    pct = 100 * count / total_actions
    print(f"  {action_names[a]:12s}: {count:5d}  ({pct:.1f}%)")

print("\nTraining complete!")
print(f"Load best model with:  model = PPO.load('{BEST_MODEL_DIR}best_model')")
