from stable_baselines3 import DQN
from trade_env import TradeEnv
import os

# 1. crea la cartella dove salveremo il modello
os.makedirs("models", exist_ok=True)

# 2. istanza dell'ambiente
env = TradeEnv()

# 3. crea l'agente DQN con iper-parametri mini-mini
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,
    verbose=1,
    tensorboard_log="tb",
)

# 4. addestramento (numero di passi ambientali)
model.learn(total_timesteps=20_000)

# 5. salva il modello
model.save("models/dqn_xauusd_first")
print("Modello salvato in models/dqn_xauusd_first.zip")