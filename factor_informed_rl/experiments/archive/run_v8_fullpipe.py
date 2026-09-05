"""
V8: 选股×交易全链路回测
2025-12-31 选股 → 2026-01-01~2026-06-30 FI-PPO交易 → 对比CSI300
"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time, os

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter, quality_filter

TOTAL = 200000
CACHE = "d:/JoinQuant/quant_env/data_cache/csi300"
MODEL_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
IND_PATH = "d:/JoinQuant/quant_env/data_cache/csi300_industry.pkl"

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "filter_fn": value_filter,
        "train_stock": "600519",
    },
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "filter_fn": quality_filter,
        "train_stock": "600276",
    },
    "Bare PPO": {
        "factors": ["roc_20","rank_20","std_20","pb_ratio","corr_20"],
        "filter_fn": None,
        "train_stock": "600519",
    },
}

market_ctx = MarketContext()
ind_map = pd.read_pickle(IND_PATH).to_dict()
neutralizer = BarraNeutralizer(ind_map)
scorer = FactorScorer()

# ── Step 1: Selection at 2025-12-31 ──
print("=" * 65)
print("  V8 FULL PIPELINE: Select @2025-12-31 → Trade 2026H1")
print("=" * 65)

SEL_DATE = pd.Timestamp("2025-12-31")
TEST_START = "2026-01-01"
TEST_END = "2026-06-30"

# 收集所有股票在2025-12-31的因子快照
print(f"\n[1/3] Selection @ {SEL_DATE.date()}...")
WINDOW = 60
stock_info = {}
factor_snaps = {}
for strat_name, cfg in STRATEGIES.items():
    factor_snaps[strat_name] = {fn: {} for fn in cfg["factors"]}

for f in os.listdir(CACHE):
    if not f.endswith('.pkl'): continue
    code = f.replace('.pkl','')
    df = pd.read_pickle(f"{CACHE}/{f}")
    for c in ['open','high','low','close','volume','pe','pb']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
    if SEL_DATE not in df.index: continue
    i = df.index.get_loc(SEL_DATE)
    if i < WINDOW: continue

    row = df.iloc[i]
    close_s = df['close']
    stock_data = {
        'pb': row.get('pb', 0), 'pe': row.get('pe', 20),
        'close': row.get('close', 0),
        'amount': row.get('close', 0) * row.get('volume', 0),
        'roc_250': float(close_s.iloc[i]/close_s.iloc[max(0,i-250)] - 1) if i>=250 else 0,
        'ma_200': float(close_s.iloc[max(0,i-200):i+1].mean()) if i>=200 else 999999,
        'mc': row.get('close',100) * row.get('volume',1e6),
    }
    stock_info[code] = stock_data

    # Compute factors for each strategy
    ohlcv = df.iloc[i-WINDOW:i+1][['open','high','low','close','volume']].values.astype(np.float64)
    for strat_name, cfg in STRATEGIES.items():
        engine = FactorEngine(cfg["factors"])
        factors = engine.compute_factors(ohlcv, ohlcv[:,4],
            pb_value=row.get('pb'), pe_percentile=row.get('pe_percentile', 0.5))
        for k, v in factors.items():
            factor_snaps[strat_name][k][code] = float(np.nan_to_num(np.asarray(v), nan=0.0))

print(f"  Indexed {len(stock_info)} stocks at {SEL_DATE.date()}")
mkt_caps = {c: d['mc'] for c, d in stock_info.items()}

# Selection results
selections = {}
for strat_name, cfg in STRATEGIES.items():
    # Hard filter
    if cfg["filter_fn"]:
        valid = cfg["filter_fn"](stock_info)
    else:
        valid = set(stock_info.keys())

    # Neutralize + Score
    fv = factor_snaps[strat_name]
    fv_filt = {fn: {c:v for c,v in vals.items() if c in valid} for fn, vals in fv.items()}
    fv_neut = {}
    for fn, vals in fv_filt.items():
        fv_neut[fn] = neutralizer.neutralize(vals, mkt_caps)
    scores = scorer.score(fv_neut)
    top6 = scorer.select_top_k(scores, k=6)
    selections[strat_name] = [c for c, _ in top6]

    inds = [ind_map.get(c, 'Other')[:8] for c, _ in top6]
    print(f"\n  [{strat_name}] Selected ({len(valid)}→6):")
    for i, (code, score) in enumerate(top6[:6], 1):
        print(f"    {i}. {code} ({ind_map.get(code,'?')[:8]:8s}) score={score:.4f}")

# ── Step 2: Train FI-PPO on selected stocks (train: 2015-2020, test: 2026H1) ──
print(f"\n[2/3] Training & Backtesting {TEST_START}→{TEST_END}...")

all_results = []
for strat_name, cfg in STRATEGIES.items():
    selected = selections[strat_name]
    use_fi = (strat_name != "Bare PPO")

    for code in selected:
        # Load data (from original cache, not CSI300 subset)
        orig_path = f"d:/JoinQuant/quant_env/data_cache/baostock_{code}.pkl"
        if not os.path.exists(orig_path):
            print(f"    [{strat_name}] {code}: no cached data, skipping")
            continue

        df = pd.read_pickle(orig_path)
        for c in ['open','high','low','close','volume','pe','pb','turn']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        if 'pe_percentile' in cfg["factors"]:
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)

        train_df = df[df.index.year <= 2020]
        test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
        if len(train_df) < 400 or len(test_df) < 50: continue

        # Train
        engine = FactorEngine(cfg["factors"])
        sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
        env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])

        fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                                warmup_steps=TOTAL//4) if use_fi else None
        trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                             n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)
        t0 = time.time()
        trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

        # Test
        test_env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                              window_size=60, enable_short=True, market_ctx=market_ctx)
        state, _ = test_env.reset(); done = False; rets = []
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = model.get_action(s, deterministic=True)
            state, r, terminated, truncated, info = test_env.step(float(a.squeeze().numpy()))
            done = terminated or truncated; rets.append(r)

        rets = np.array(rets); cum = (1+rets).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = float(np.min((cum-peak)/peak)) if len(cum)>0 else 0
        sr = float(rets.mean()/(np.std(rets)+1e-10)*np.sqrt(252))
        tr = float(cum[-1]-1)
        bh = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)
        alive = info['total_value'] > 50000

        all_results.append({
            'strategy': strat_name, 'code': code,
            'return': tr, 'sharpe': sr, 'max_dd': dd,
            'trades': info['trade_count'], 'alive': alive,
            'bh_ret': bh, 'time': time.time()-t0,
        })

# ── Step 3: Aggregate ──
print(f"\n{'='*65}")
print("  V8 RESULTS (2026H1, 6 stocks per strategy)")
print("=" * 65)

# CSI300 benchmark
csi300 = market_ctx.csi300
csi_test = csi300[(csi300.index >= TEST_START) & (csi300.index <= TEST_END)]
csi_bh = float(csi_test['close'].iloc[-1] / csi_test['close'].iloc[0] - 1)
csi_sr = float(csi_test['close'].pct_change().dropna().mean() /
               (csi_test['close'].pct_change().dropna().std() + 1e-10) * np.sqrt(252))
print(f"  CSI300: Ret={csi_bh:+.1%} Sharpe={csi_sr:.2f}")

df_results = pd.DataFrame(all_results)
for strat_name in ["Value-Defensive", "Quality-Offensive", "Bare PPO"]:
    sub = df_results[df_results['strategy'] == strat_name]
    if len(sub) == 0: continue
    print(f"\n  [{strat_name}] ({len(sub)} stocks)")
    for _, r in sub.iterrows():
        print(f"    {r['code']}: Ret={r['return']:+.1%} Sharpe={r['sharpe']:.2f} "
              f"DD={r['max_dd']:.1%} T={r['trades']} {'OK' if r['alive'] else 'DEAD'}")
    print(f"    AVG: Ret={sub['return'].mean():+.1%} Sharpe={sub['sharpe'].mean():.2f} "
          f"DD={sub['max_dd'].mean():.1%} T={sub['trades'].mean():.0f} "
          f"Beat CSI300: {(sub['sharpe']>csi_sr).sum()}/{len(sub)}")

print("=" * 65)
