import gymnasium as gym
import numpy as np
import pandas as pd

class TradeEnv(gym.Env):
    def __init__(self, csv_file="xauusd_15m.csv", initial_cash=10_000):
        super().__init__()
        # CSV: prima colonna contiene le date
        self.df = pd.read_csv(csv_file, sep=';', parse_dates=True, dayfirst=False)
        self.df.set_index(self.df.columns[0], inplace=True)
        self.df.index.name = 'Date'

        self.n_steps = len(self.df)
        self.initial_cash = initial_cash

        # spazi
        self.action_space = gym.spaces.Discrete(3)  # 0=flat, 1=long, 2=short
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # buffer
        self.current_step = 20
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 20
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0
        return self._get_obs(), {}

    def _get_obs(self):
        prices = self.df["Close"].iloc[self.current_step - 20:self.current_step + 1].values
        norm = (prices / prices[-1] - 1.0)[:20]  # prendiamo esattamente 20 valori
        return np.append(norm, self.position).astype(np.float32)

    def _portfolio_value(self, price):
        if self.position == 0:
            return self.cash
        pnl = (price - self.entry_price) * (1 if self.position == 1 else -1)
        return self.cash + pnl

    def step(self, action):
        price = self.df["Close"].iloc[self.current_step]
        prev_val = self._portfolio_value(price)

        # --- esecuzione ordine con costi reali ---
        cost_per_trade = 0.15   # 15 cent (spread + slip + comm)
        if action != self.position:  # cambio di posizione
            if action == 0:  # chiudi
                self.cash = self._portfolio_value(price)
                self.position = 0
            else:  # entra o flip
                if self.position != 0:  # prima chiudi
                    self.cash = self._portfolio_value(price)
                # applica costo e slittamento sul prezzo d’ingresso
                slip_cost = cost_per_trade * (1 if action == 1 else -1)
                self.entry_price = price + slip_cost
                self.cash -= cost_per_trade  # paghi il costo
                self.position = action

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        new_val = self._portfolio_value(price)
        reward = new_val - prev_val
        info = {}
        return self._get_obs(), reward, done, False, info

    def render(self):
        print(f"step={self.current_step}  cash={self.cash:.2f}  pos={self.position}")