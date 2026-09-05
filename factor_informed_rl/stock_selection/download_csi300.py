"""Download CSI300 daily data for all 300 constituents"""
import baostock as bs, pandas as pd, numpy as np, os, time

bs.login()

# Get constituent list
print("Getting CSI300 constituents...")
rs = bs.query_hs300_stocks()
constituents = []
while (rs.error_code == '0') and rs.next():
    row = rs.get_row_data()
    code = row[1]  # e.g. "sh.600000"
    constituents.append(code)
bs.logout()
print(f"Total: {len(constituents)} stocks")

cache_dir = "d:/JoinQuant/quant_env/data_cache/csi300"
os.makedirs(cache_dir, exist_ok=True)

downloaded = 0; failed = 0; skipped = 0
for i, ticker in enumerate(constituents):
    code = ticker.replace('sh.','').replace('sz.','')
    path = f"{cache_dir}/{code}.pkl"
    if os.path.exists(path):
        skipped += 1
        continue

    bs.login()
    time.sleep(0.3)  # Rate limit
    rs = bs.query_history_k_data_plus(ticker,
        "date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn",
        start_date="2015-01-01", end_date="2025-12-31",
        frequency="d", adjustflag="2")
    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    bs.logout()

    if not data:
        failed += 1
        if failed <= 5:
            print(f"  [{failed}] {ticker}: no data")
        continue

    df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount","pe","pb","turn"])
    for c in ["open","high","low","close","volume","pe","pb","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open","high","low","close","volume","pe","pb","turn"]:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    df.to_pickle(path)
    downloaded += 1

    if (i+1) % 30 == 0:
        print(f"  Progress: {i+1}/{len(constituents)} (ok={downloaded} skip={skipped} fail={failed})", flush=True)

print(f"\nDone! Downloaded={downloaded} Skipped={skipped} Failed={failed}")
