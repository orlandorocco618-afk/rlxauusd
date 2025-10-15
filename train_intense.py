from stable_baselines3 import DQN
from trade_env import TradeEnv
import os

os.makedirs("models", exist_ok=True)

env = TradeEnv()

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.995,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.3,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="tb_intense",
)

model.learn(total_timesteps=100_000)
model.save("models/dqn_xauusd_intense")
print("Intense model saved.")