"""
PortfolioManager + V7 FI-PPO 全链路回测
=======================================
每日滚动选股 → PortfolioManager → FI-PPO交易 → vs CSI300
"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, os, time, pickle

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter, quality_filter
from factor_informed_rl.portfolio.manager import PortfolioManager

CACHE = "d:/JoinQuant/quant_env/data_cache/csi300"
IND_PATH = "d:/JoinQuant/quant_env/data_cache/csi300_industry.pkl"
MODEL_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
FACTOR_CACHE = "d:/JoinQuant/quant_env/data_cache/csi300_factors.pkl"
START, END = "2021-01-01", "2026-06-30"
WINDOW = 60

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "filter_fn": value_filter,
        "model": "Value-Defensive_600519_fi.pt",
    },
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "filter_fn": quality_filter,
        "model": "Quality-Offensive_600276_fi.pt",
    },
}

market_ctx = MarketContext()
ind_map = pd.read_pickle(IND_PATH).to_dict()
neutralizer = BarraNeutralizer(ind_map)
scorer = FactorScorer()

# ── Step 0: Pre-compute factors for all stocks (cached) ──
t0 = time.time()
print("[0/3] Factor computation...", flush=True)

if os.path.exists(FACTOR_CACHE):
    with open(FACTOR_CACHE, 'rb') as f:
        factor_cache = pickle.load(f)
    print(f"  Loaded cached: {len(factor_cache)} days", flush=True)
else:
    factor_cache = {}
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith('.pkl'): continue
        code = f.replace('.pkl','')
        df = pd.read_pickle(f"{CACHE}/{f}")
        for c in ['open','high','low','close','volume','pe','pb']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)

        for sn, cfg in STRATEGIES.items():
            F = cfg["factors"]
            engine = FactorEngine(F)
            for i in range(WINDOW, len(df)):
                date = df.index[i]; row = df.iloc[i]
                ohlcv = df.iloc[i-WINDOW:i+1][['open','high','low','close','volume']].values.astype(np.float64)
                fv = engine.compute_factors(ohlcv, ohlcv[:,4], pb_value=row.get('pb'),
                                            pe_percentile=row.get('pe_percentile',0.5))
                key = (date, sn, code)
                factor_cache[key] = {k: float(np.nan_to_num(np.asarray(v), nan=0.0))
                                    for k, v in fv.items() if k in F}
        if len(factor_cache) % 50000 == 0:
            print(f"  ...{len(factor_cache)} factor snapshots computed", flush=True)

    with open(FACTOR_CACHE, 'wb') as f:
        pickle.dump(factor_cache, f)
    print(f"  Saved {len(factor_cache)} factor snapshots ({time.time()-t0:.0f}s)", flush=True)

# ── Step 1: Get trading dates ──
dates = sorted(set(d for (d, _, _) in factor_cache.keys()
                   if pd.Timestamp(START) <= d <= pd.Timestamp(END)))
print(f"[1/3] Trading days: {len(dates)} ({dates[0].date()} → {dates[-1].date()})")

# ── Step 2: Load FI-PPO models ──
print("[2/3] Loading V7 models...", flush=True)
agents = {}
for sn, cfg in STRATEGIES.items():
    ckpt = torch.load(f"{MODEL_DIR}/{cfg['model']}")
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state']); model.eval()
    agents[sn] = {'model': model, 'state_dim': sb.state_dim, 'factors': cfg['factors']}

# ── Step 3: Run PortfolioManager for each strategy ──
print("[3/3] Portfolio backtest", flush=True)
results = {}

for sn in ["Value-Defensive", "Quality-Offensive"]:
    cfg = STRATEGIES[sn]
    pm = PortfolioManager(max_stocks=6, total_capital=1_000_000,
                          observe_days=5, build_days=8, exit_days=10, top_n=300)

    for day_idx, date in enumerate(dates):
        if day_idx % 100 == 0:
            print(f"  [{sn}] {date.date()} ({day_idx}/{len(dates)})", flush=True)

        # Collect factor values + stock data for this date
        stock_data = {}
        factor_vals = {fn: {} for fn in cfg["factors"]}
        prices = {}; returns_day = {}

        for code in [c for c in os.listdir(CACHE) if c.endswith('.pkl')]:
            code = code.replace('.pkl','')
            key = (date, sn, code)
            if key not in factor_cache: continue
            fv = factor_cache[key]
            for fn, val in fv.items():
                factor_vals[fn][code] = val

        # Build stock metadata for filtering
        for k in factor_cache:
            d, s, c = k
            if d == date and s == sn:
                stock_data.setdefault(c, {})
        for code in list(stock_data.keys()):
            df = pd.read_pickle(f"{CACHE}/{code}.pkl")
            if date not in df.index: continue
            row = df.loc[date]
            cs = df['close']; i = df.index.get_loc(date)
            stock_data[code] = {
                'pb': row.get('pb',0), 'pe': row.get('pe',20),
                'close': row.get('close',0),
                'amount': row.get('close',0)*row.get('volume',0),
                'roc_250': float(cs.iloc[i]/cs.iloc[max(0,i-250)]-1) if i>=250 else 0,
                'ma_200': float(cs.iloc[max(0,i-200):i+1].mean()) if i>=200 else 999999,
            }
            prices[code] = row.get('close', 0)
            if i > 0:
                returns_day[code] = float(cs.iloc[i]/cs.iloc[i-1]-1) if cs.iloc[i-1]>0 else 0
            else:
                returns_day[code] = 0.0

        # Run selection (ranking)
        vc = cfg["filter_fn"](stock_data) if cfg["filter_fn"] else set(stock_data.keys())
        fv_f = {fn: {c:v for c,v in vals.items() if c in vc}
                for fn, vals in factor_vals.items()}
        mc = {c: stock_data[c]['amount'] for c in vc}
        fv_n = {fn: neutralizer.neutralize(vals, mc) for fn, vals in fv_f.items()}
        scores = scorer.score(fv_n)
        rankings = scorer.select_top_k(scores, k=50)  # Top 50 for pool mgmt

        # FI-PPO signals: 用真实 V7 模型对池内股票做推理
        fi_signals = {}
        pool_codes = set(pm.stocks.keys())  # 所有跟踪中的股票
        for code in pool_codes:
            if code not in stock_data: continue
            df = pd.read_pickle(f"{CACHE}/{code}.pkl")
            for c in ['open','high','low','close','volume','pe','pb']:
                if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
            if date not in df.index: continue
            i = df.index.get_loc(date)
            if i < WINDOW: continue

            # 构建状态 + FI-PPO推理 (复用预加载模型)
            eng = FactorEngine(cfg["factors"])
            sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
            env = TradingEnv(df.iloc[:i+1], eng, sb, Denoiser(method="none"),
                             window_size=60, enable_short=True, market_ctx=market_ctx)
            env.reset(); env.idx = min(i, len(df)-1)
            state = env._get_state()
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = agents[sn]['model'].get_action(s, deterministic=True)
            fi_signals[code] = float(a.squeeze().numpy())

        # Portfolio daily step
        result = pm.daily_step(date, rankings, fi_signals, prices, returns_day)
        if result['events']:
            print(f"    {date.date()} {result['events']}", flush=True)

    # Compute performance
    perf = pm.performance()
    results[sn] = perf

    csi = market_ctx.csi300
    csi_t = csi[(csi.index >= START) & (csi.index <= END)]
    csi_ret = float(csi_t['close'].iloc[-1]/csi_t['close'].iloc[0]-1) if len(csi_t)>1 else 0
    print(f"\n  [{sn}] Ret={perf['total_return']:+.1%} Sharpe={perf['sharpe']:.2f} "
          f"DD={perf['max_drawdown']:.1%} {perf['n_days']}d")
    print(f"  CSI300: {csi_ret:+.1%}")

print(f"\n{'='*60}")
print(f"  PORTFOLIO RESULTS (2021-2026, {len(dates)}d)")
csi_ret = float(csi_t['close'].iloc[-1]/csi_t['close'].iloc[0]-1) if len(csi_t)>1 else 0
csi_sr = float(csi_t['close'].pct_change().dropna().mean()/
              (csi_t['close'].pct_change().dropna().std()+1e-10)*np.sqrt(252))
print(f"  CSI300: {csi_ret:+.1%} Sharpe={csi_sr:.2f}")
for sn in ["Value-Defensive", "Quality-Offensive"]:
    p = results[sn]
    print(f"  {sn}: Ret={p['total_return']:+.1%} Sharpe={p['sharpe']:.2f} DD={p['max_drawdown']:.1%}")
print("="*60)
