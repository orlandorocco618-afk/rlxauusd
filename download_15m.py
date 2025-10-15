import yfinance as yf, pandas as pd

ticker = "EURUSD=X"
df = yf.download(ticker, start="2023-01-01", interval="1h", progress=False)
df = df[['Open','High','Low','Close','Volume']].dropna()
df.to_csv("xauusd_15m.csv")   # teniamo nome fisso per comodità
print("EURUSD 1h salvati:", len(df))