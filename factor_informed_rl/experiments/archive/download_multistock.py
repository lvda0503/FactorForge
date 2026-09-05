"""下载多只A股数据"""
import baostock as bs
import pandas as pd
import numpy as np
import os

TICKERS = {
    "600519": "贵州茅台",
    "000001": "平安银行",
    "000858": "五粮液",
    "600036": "招商银行",
    "000333": "美的集团",
    "600276": "恒瑞医药",
    "000725": "京东方A",
    "601318": "中国平安",
}

bs.login()
cache_dir = "d:/JoinQuant/quant_env/data_cache"
os.makedirs(cache_dir, exist_ok=True)

for code, name in TICKERS.items():
    ticker = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
    path = f"{cache_dir}/baostock_{code}.pkl"

    if os.path.exists(path):
        print(f"[SKIP] {code} {name} (already cached)")
        continue

    rs = bs.query_history_k_data_plus(ticker,
        "date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn",
        start_date="2015-01-01", end_date="2025-12-31",
        frequency="d", adjustflag="2")

    data = []
    while (rs.error_code == "0") and rs.next():
        data.append(rs.get_row_data())

    if not data:
        print(f"[FAIL] {code} {name}: no data")
        continue

    df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount","pe","pb","turn"])
    for c in ["open","high","low","close","volume","pe","pb","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Clean
    df["pe"] = df["pe"].replace(0, np.nan).ffill().bfill()
    df["pb"] = df["pb"].replace(0, np.nan).ffill().bfill()
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].ffill().bfill()

    df.to_pickle(path)

    train_ret = df[df.index.year <= 2020]["close"].iloc[-1] / df[df.index.year <= 2020]["close"].iloc[0] - 1
    test_ret = df[df.index.year >= 2022]["close"].iloc[-1] / df[df.index.year >= 2022]["close"].iloc[0] - 1

    print(f"[OK] {code} {name}: {len(df)}d | Train: {train_ret:+.0%} | Test: {test_ret:+.0%} | NaN: {df.isna().sum().sum()}")

bs.logout()
print("\nAll stocks downloaded!")
