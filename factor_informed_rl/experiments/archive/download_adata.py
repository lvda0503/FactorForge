"""下载A股真实数据 via baostock"""
import baostock as bs
import pandas as pd
import numpy as np
import os

bs.login()

# 贵州茅台
ticker = "sh.600519"
rs = bs.query_history_k_data_plus(ticker,
    "date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn",
    start_date="2015-01-01", end_date="2025-12-31",
    frequency="d", adjustflag="2")

data = []
while (rs.error_code == "0") and rs.next():
    data.append(rs.get_row_data())
bs.logout()

df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount","pe","pb","turn"])
for c in ["open","high","low","close","volume","pe","pb","turn"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# 处理缺失
df["pe"] = df["pe"].replace(0, np.nan).ffill().bfill()
df["pb"] = df["pb"].replace(0, np.nan).ffill().bfill()

print(f"Ticker: {ticker}")
print(f"Days: {len(df)} ({df.index[0].date()} -> {df.index[-1].date()})")
print(f"Close: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
print(f"Total Return: {df['close'].iloc[-1]/df['close'].iloc[0]-1:.1%}")
print(f"PE: {df['pe'].min():.0f} ~ {df['pe'].max():.0f}")
print(f"PB: {df['pb'].min():.0f} ~ {df['pb'].max():.0f}")
print(f"NaN PE: {df['pe'].isna().sum()}, NaN PB: {df['pb'].isna().sum()}")

os.makedirs("d:/JoinQuant/quant_env/data_cache", exist_ok=True)
path = "d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl"
df.to_pickle(path)
print(f"\nSaved: {path}")
