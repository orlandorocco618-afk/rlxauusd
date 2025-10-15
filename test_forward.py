from stable_baselines3 import DQN
from trade_env import TradeEnv
import pandas as pd
import numpy as np

MODEL_PATH = "models/dqn_xauusd_first"
TEST_CSV   = "xauusd_15m.csv"          # stesso file, filtriamo le date

# --- carichiamo l'agente addestrato ---
model = DQN.load(MODEL_PATH)

# --- filtriamo il periodo mai visto (ultimi 3 mesi) ---
df = pd.read_csv(TEST_CSV, parse_dates=True, index_col=0)
test_df = df.loc["2024-07-01":"2024-09-30"]

# --- creiamo un ambiente *solo* con queste candele ---
test_df.to_csv("test_slice.csv")       # file temporaneo
env = TradeEnv(csv_file="test_slice.csv", initial_cash=10_000)

# --- testiamo ---
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

# --- risultati ---
final_value = env._portfolio_value(env.df["Close"].iloc[env.current_step - 1])
buy_hold    = 10_000 * (env.df["Close"].iloc[-1] / env.df["Close"].iloc[0])
print(f"Agente DQN: {final_value:,.2f} $")
print(f"Buy & Hold: {buy_hold:,.2f} $")