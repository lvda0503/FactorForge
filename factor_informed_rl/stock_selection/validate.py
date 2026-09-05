"""选股系统验证 — 12只A股排序回测"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, os
from scipy import stats

from factor_informed_rl.stock_selection import (
    UniverseFilter, BarraNeutralizer, FactorTransformer, FactorScorer, StockSelector
)
from factor_informed_rl.preprocessing.factor_engine import FactorEngine

STOCKS = {
    "600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui",
    "600887":"Yili","002415":"Hikvision","600030":"CITIC Sec","000651":"Gree",
    "601166":"Ind Bank","600585":"Conch","000002":"Vanke","601888":"CDFG",
}
# 申万一级行业 (手动映射)
INDUSTRY_MAP = {
    "600519":"食品饮料","000858":"食品饮料","000333":"家用电器","600276":"医药生物",
    "600887":"食品饮料","002415":"计算机","600030":"非银金融","000651":"家用电器",
    "601166":"银行","600585":"建筑材料","000002":"房地产","601888":"休闲服务",
}
cache = "d:/JoinQuant/quant_env/data_cache"

# 加载12只股票
all_data = {}
for code in STOCKS:
    path = f"{cache}/baostock_{code}.pkl"
    if os.path.exists(path):
        df = pd.read_pickle(path)
        for c in ['open','high','low','close','volume','pe','pb','turn']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        all_data[code] = df

print(f"Loaded {len(all_data)} stocks")

# 统计: test period returns
print("\nTest period (2022-2025) returns:")
for code, name in STOCKS.items():
    if code in all_data:
        test = all_data[code][all_data[code].index.year >= 2022]
        ret = test['close'].iloc[-1] / test['close'].iloc[0] - 1
        print(f"  {code} {name}: {ret:+.1%}")

# ============================================================
# 选股验证: 月度调仓, 对比 Top-K vs Bottom-K vs Random
# ============================================================
FACTORS = ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"]  # Value策略因子
TOP_K = 5

# 准备数据: 为每只股票计算因子值 (日频)
print(f"\nComputing factors for {len(all_data)} stocks...")
factor_data = {}  # {code: {date: {factor_name: value}}}
for code in STOCKS:
    if code not in all_data: continue
    df = all_data[code].copy()
    df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
    f_data = {}
    engine = FactorEngine(FACTORS, ic_window=120)
    for i in range(60, len(df)):
        row = df.iloc[i]
        date = df.index[i]
        ohlcv = df.iloc[i-60:i+1][['open','high','low','close','volume']].values.astype(np.float64)
        factors = engine.compute_factors(
            ohlcv, ohlcv[:,4], pb_value=row.get('pb'),
            pe_percentile=row.get('pe_percentile', 0.5))
        f_data[date] = {}
        for k, v in factors.items():
            try: f_data[date][k] = float(v)
            except: f_data[date][k] = 0.0
    factor_data[code] = f_data
print(f"Factor computation complete for {len(factor_data)} stocks")

# 选股回测: 每月第一天调仓
TEST_DATES = sorted(set().union(*[set(fd.keys()) for fd in factor_data.values()]))
TEST_DATES = [d for d in TEST_DATES if d.year >= 2022]
test_series = pd.Series(index=pd.DatetimeIndex(TEST_DATES), data=0)
rebalance_dates = test_series.resample('ME').first().index.tolist()
print(f"Rebalance dates: {len(rebalance_dates)} (monthly, 2022-2025)")

# 配置
neutralizer = BarraNeutralizer(INDUSTRY_MAP)
scorer = FactorScorer()

results = []
for i, date in enumerate(rebalance_dates):
    if i == 0: continue  # 需要前一期收益

    prev_date = rebalance_dates[i-1]

    # 收集当天所有股票的因子值
    fv = {fn: {} for fn in FACTORS}
    mc = {}
    for code in STOCKS:
        if code not in factor_data: continue
        if date not in factor_data[code]: continue
        fvals = factor_data[code][date]
        for fn in FACTORS:
            fv[fn][code] = fvals.get(fn, 0.0)
        # 市值 (用收盘价 * volume 做代理)
        if code in all_data:
            row = all_data[code].loc[date] if date in all_data[code].index else None
            if row is not None:
                mc[code] = row.get('close', 100) * row.get('volume', 1e6)

    if len(fv[FACTORS[0]]) < 5: continue

    # === 方案 A: 原始因子 + 打分 (无中性化, 无非线性) ===
    scores_raw = scorer.score(fv)
    top_raw = scorer.select_top_k(scores_raw, k=TOP_K)

    # === 方案 B: Barra中性化 + 打分 ===
    fv_neutral = {}
    for fn in FACTORS:
        fv_neutral[fn] = neutralizer.neutralize(fv[fn], mc)
    scores_neutral = scorer.score(fv_neutral)
    top_neutral = scorer.select_top_k(scores_neutral, k=TOP_K)

    # === 计算实际收益 (下月收益) ===
    for label, top_list in [("Raw", top_raw), ("Barra", top_neutral)]:
        for code, score in top_list:
            if code not in all_data: continue
            # 找到下个调仓日
            if i+1 < len(rebalance_dates):
                next_date = rebalance_dates[i+1]
                if code in all_data and date in all_data[code].index and next_date in all_data[code].index:
                    entry_price = all_data[code].loc[date]['close']
                    exit_price = all_data[code].loc[next_date]['close']
                    ret = exit_price / entry_price - 1
                    results.append({
                        'date': date, 'code': code, 'stock': STOCKS.get(code,''),
                        'method': label, 'score': score,
                        'return': ret, 'rank': top_list.index((code, score)) + 1,
                    })

    if i % 6 == 0:
        print(f"  [{date.date()}] Raw Top: {[c for c,_ in top_raw[:3]]} | "
              f"Barra Top: {[c for c,_ in top_neutral[:3]]}", flush=True)

# 汇总
df_r = pd.DataFrame(results)
if len(df_r) == 0:
    print("No results! Check data alignment.")
    exit()

print(f"\n{'='*65}")
print("  STOCK SELECTION VALIDATION (Monthly Rebalance, Top-5)")
print("=" * 65)

for method in ['Raw', 'Barra']:
    sub = df_r[df_r['method'] == method]
    if len(sub) == 0: continue

    # 平均收益
    avg_ret = sub.groupby('date')['return'].mean()
    ann_ret = float((1 + avg_ret.mean())**12 - 1)
    sharpe = float(avg_ret.mean() / (avg_ret.std() + 1e-10) * np.sqrt(12))
    max_dd = float(np.min(np.cumprod(1 + avg_ret) / np.maximum.accumulate(np.cumprod(1 + avg_ret)) - 1))

    # 等权全12股基准
    all_codes = list(set(df_r['code']))
    all_ret = df_r[df_r['method'] == 'Raw'].groupby('date').apply(
        lambda g: g[g['code'].isin(all_codes)]['return'].mean() if len(g) >= 3 else np.nan
    ).dropna()
    bh_ann = float((1 + all_ret.mean())**12 - 1)

    # 命中率: Top-5中有多少在下期跑赢等权
    hit_rate = []
    for date, g in sub.groupby('date'):
        all_mean = all_ret.get(date, 0)
        if all_mean is not None:
            hit_rate.append((g['return'] > all_mean).mean())
    hit = np.mean(hit_rate) * 100 if hit_rate else 0

    print(f"\n  [{method}]")
    print(f"    Annual Return:   {ann_ret:+.1%}")
    print(f"    Sharpe:          {sharpe:.2f}")
    print(f"    Max DD:          {max_dd:.1%}")
    print(f"    Hit Rate vs EW:  {hit:.0f}%")
    print(f"    Benchmark (EW12):{bh_ann:+.1%}")

# Head-to-head
raw_ret = df_r[df_r['method']=='Raw'].groupby('date')['return'].mean()
barra_ret = df_r[df_r['method']=='Barra'].groupby('date')['return'].mean()
common = raw_ret.index.intersection(barra_ret.index)
wins = (barra_ret[common] > raw_ret[common]).sum()
print(f"\n  Barra vs Raw: {wins}/{len(common)} months ({wins/len(common)*100:.0f}%)")
print("=" * 65)
