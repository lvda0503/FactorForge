"""选股管线 — 基于今日数据的全流程"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, os, time
t_start = time.time()
import baostock as bs

from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter, quality_filter
from factor_informed_rl.preprocessing.factor_engine import FactorEngine

CACHE_DIR = "d:/JoinQuant/quant_env/data_cache/csi300"
IND_PATH  = "d:/JoinQuant/quant_env/data_cache/csi300_industry.pkl"

# ── Step 1: Get industry mapping (cached) ──
print("[1/4] Industry data...", end=" ", flush=True)
if not os.path.exists(IND_PATH):
    bs.login()
    rs = bs.query_stock_industry()
    ind_map = {}
    while (rs.error_code == '0') and rs.next():
        row = rs.get_row_data()
        code = row[1].replace('sh.','').replace('sz.','')
        ind  = row[3] if row[3] else row[4]  # 证监会行业大类 or 门类
        ind_map[code] = ind
    bs.logout()
    pd.Series(ind_map).to_pickle(IND_PATH)
else:
    ind_map = pd.read_pickle(IND_PATH).to_dict()
print(f"{len(ind_map)} stocks mapped")

# ── Step 2: Load CSI300 stocks + find latest date ──
print("[2/4] Loading data...", end=" ", flush=True)
all_dates = set()
code_list = []
for f in os.listdir(CACHE_DIR):
    if not f.endswith('.pkl'): continue
    code = f.replace('.pkl','')
    code_list.append(code)

# Find latest common date
for f in os.listdir(CACHE_DIR)[:10]:
    if f.endswith('.pkl'):
        df = pd.read_pickle(f"{CACHE_DIR}/{f}")
        all_dates.update(df.index)
        break
for f in os.listdir(CACHE_DIR):
    if f.endswith('.pkl'):
        df = pd.read_pickle(f"{CACHE_DIR}/{f}")
        all_dates &= set(df.index)

latest = max(all_dates)
print(f"{len(code_list)} stocks, latest date: {latest.date()}")

# ── Step 3: Compute factors for each stock at latest date ──
print("[3/4] Computing factors...", flush=True)
VALUE_FACTORS   = ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"]
QUALITY_FACTORS = ["roc_60","beta_20","rsqr_20","vma_20","std_20"]
WINDOW = 60

def compute_snapshot(code, factors_list):
    path = f"{CACHE_DIR}/{code}.pkl"
    if not os.path.exists(path): return None
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    # PIT-safe PE percentile
    df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)

    if latest not in df.index: return None
    i = df.index.get_loc(latest)
    if i < WINDOW: return None

    engine = FactorEngine(factors_list)
    ohlcv = df.iloc[i-WINDOW:i+1][['open','high','low','close','volume']].values.astype(np.float64)
    row = df.iloc[i]
    factors = engine.compute_factors(ohlcv, ohlcv[:,4],
        pb_value=row.get('pb'), pe_percentile=row.get('pe_percentile'))
    mc = row.get('close', 100) * row.get('volume', 1e6)

    # 硬过滤所需的额外字段
    close_series = df['close']
    stock_data = {
        'pb': row.get('pb', 0),
        'pe': row.get('pe', 20),
        'close': row.get('close', 0),
        'amount': row.get('close', 0) * row.get('volume', 0),
        'roc_250': float(close_series.iloc[i] / close_series.iloc[max(0,i-250)] - 1) if i >= 250 else 0.0,
        'ma_200': float(close_series.iloc[max(0,i-200):i+1].mean()) if i >= 200 else 999999.0,
    }
    return {k: float(np.nan_to_num(np.asarray(v), nan=0.0)) for k,v in factors.items()}, mc, stock_data

# Compute for all stocks (Value + Quality)
val_fv = {}; qual_fv = {}; mkt_caps = {}
stock_info = {}
valid = 0
for code in code_list:
    r = compute_snapshot(code, VALUE_FACTORS)
    if r:
        fv, mc, sd = r
        stock_info[code] = sd
        for k, v in fv.items():
            val_fv.setdefault(k, {})[code] = v
        r2 = compute_snapshot(code, QUALITY_FACTORS)
        if r2:
            for k, v in r2[0].items():
                qual_fv.setdefault(k, {})[code] = v
            mkt_caps[code] = mc
            valid += 1
    if valid % 50 == 0:
        print(f"  {valid}/{len(code_list)} stocks computed", flush=True)
print(f"  Done: {valid} valid stocks")

# ── Step 4: Selection with Barra + Scoring ──
print(f"\n{'='*65}")
print(f"  CSI300 STOCK SELECTION — {latest.date()}")
print(f"  Valid stocks: {valid}")
print(f"{'='*65}")

neutralizer = BarraNeutralizer(ind_map)
scorer = FactorScorer()

for label, fv_dict, filter_fn in [
    ("Value-Defensive", val_fv, value_filter),
    ("Quality-Offensive", qual_fv, quality_filter),
]:
    # ── 硬过滤 ──
    valid_codes = filter_fn(stock_info)
    n_filtered = len(stock_info) - len(valid_codes)
    print(f"\n  [{label}] Hard filter: {len(valid_codes)}/{len(stock_info)} passed ({n_filtered} removed)")
    # Filter factors to valid codes only
    fv_filtered = {}
    for fn, vals in fv_dict.items():
        fv_filtered[fn] = {c: v for c, v in vals.items() if c in valid_codes}

    # Barra neutralize
    fv_neutral = {}
    for fn, vals in fv_filtered.items():
        fv_neutral[fn] = neutralizer.neutralize(vals, mkt_caps)

    # Score
    scores = scorer.score(fv_neutral)
    top12 = scorer.select_top_k(scores, k=12)

    print(f"\n  [{label}] Top-12 picks:")
    print(f"  {'Rank':<6} {'Code':<9} {'Name':<10} {'Industry':<12} {'Score':>7}")
    print(f"  {'-'*50}")
    for rank, (code, score) in enumerate(top12, 1):
        name = code  # 暂用code
        ind  = ind_map.get(code, 'Other')[:10]
        print(f"  {rank:<6} {code:<9} {name:<10} {ind:<12} {score:>7.4f}")

print(f"\n{'='*65}")
print(f"  Done! {time.time()-t_start:.0f}s elapsed")
