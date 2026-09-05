"""Download 8 more liquid A-shares"""
import baostock as bs, pandas as pd, numpy as np, os

TICKERS = {
    "600887": "伊利股份", "002415": "海康威视", "600030": "中信证券",
    "000651": "格力电器", "601166": "兴业银行", "600585": "海螺水泥",
    "000002": "万科A",    "601888": "中国中免",
}

bs.login()
cache = "d:/JoinQuant/quant_env/data_cache"

for code, name in TICKERS.items():
    ticker = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
    path = f"{cache}/baostock_{code}.pkl"
    if os.path.exists(path):
        print(f"[SKIP] {code} {name}")
        continue

    rs = bs.query_history_k_data_plus(ticker,
        "date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn",
        start_date="2015-01-01", end_date="2025-12-31",
        frequency="d", adjustflag="2")
    data = []
    while (rs.error_code == "0") and rs.next():
        data.append(rs.get_row_data())
    if not data:
        print(f"[FAIL] {code} {name}")
        continue
    df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount","pe","pb","turn"])
    for c in ["open","high","low","close","volume","pe","pb","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open","high","low","close","volume","pe","pb","turn"]:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    df.to_pickle(path)
    ret = df['close'].iloc[-1]/df['close'].iloc[0]-1
    print(f"[OK] {code} {name}: {len(df)}d, total ret={ret:+.0%}")
bs.logout()
print("Done!")
