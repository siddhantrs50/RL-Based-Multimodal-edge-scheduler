from stable_baselines3 import PPO
from rl_env import EdgeSchedulerEnv

env = EdgeSchedulerEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=20000)

model.save("ppo_scheduler")

print("Training complete!")
