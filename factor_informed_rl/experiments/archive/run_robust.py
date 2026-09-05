"""鲁棒性测试 — 12只股票 × 3个时间窗口"""
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

STOCKS = {
    "600519":"茅台","000858":"五粮液","000333":"美的","600276":"恒瑞",
    "600887":"伊利","002415":"海康","600030":"中信证券",
    "000651":"格力","601166":"兴业银行","600585":"海螺水泥",
    "000002":"万科A","601888":"中国中免",
}

# 3个时间窗口: train→test
SPLITS = [
    ("2015-2018→2019-2020", lambda df: (df[df.index.year <= 2018], df[(df.index.year >= 2019) & (df.index.year <= 2020)])),
    ("2017-2020→2021-2022", lambda df: (df[(df.index.year >= 2017) & (df.index.year <= 2020)], df[(df.index.year >= 2021) & (df.index.year <= 2022)])),
    ("2019-2022→2023-2025", lambda df: (df[(df.index.year >= 2019) & (df.index.year <= 2022)], df[df.index.year >= 2023])),
]

TOTAL = 200000
cache = "d:/JoinQuant/quant_env/data_cache"
market_ctx = MarketContext()

all_runs = []
for code, name in STOCKS.items():
    path = f"{cache}/baostock_{code}.pkl"
    if not os.path.exists(path): continue
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

    for split_name, split_fn in SPLITS:
        train_df, test_df = split_fn(df)
        if len(train_df) < 400 or len(test_df) < 100: continue
        bh_ret = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)

        for use_fi, label in [(False,"Bare"), (True,"FI")]:
            engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
            sb = StateBuilder(window_size=60, market_dim=11)
            env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                             window_size=60, enable_short=True, market_ctx=market_ctx)
            model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
            fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                                    warmup_steps=TOTAL//4) if use_fi else None
            trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                                 n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)

            t0 = time.time()
            trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

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
            alive = info['total_value'] > 50000

            all_runs.append({
                'stock': name, 'code': code, 'split': split_name,
                'use_fi': use_fi, 'label': label,
                'return': tr, 'sharpe': sr, 'max_dd': dd,
                'trades': info['trade_count'], 'alive': alive,
                'bh_ret': bh_ret, 'time': time.time()-t0,
            })
            print(f"[{name}|{split_name}] {label:5s}: Ret={tr:+.1%} Sharpe={sr:.2f} "
                  f"DD={dd:.1%} Trades={info['trade_count']} {'OK' if alive else 'DEAD'} "
                  f"({all_runs[-1]['time']:.0f}s)", flush=True)

# Summary
df_r = pd.DataFrame(all_runs)
bare = df_r[~df_r['use_fi']]
fi   = df_r[df_r['use_fi']]

print("\n" + "=" * 65)
print(f"  ROBUSTNESS RESULTS — {len(bare)} runs total")
print("=" * 65)

dead_b = (~bare['alive']).sum(); dead_f = (~fi['alive']).sum()
print(f"  Dead: Bare={dead_b}/{len(bare)}  FI={dead_f}/{len(fi)}")
print(f"  Profitable: Bare={(bare['return']>0).sum()}/{len(bare)}  FI={(fi['return']>0).sum()}/{len(fi)}")

# FI win rate per run
wins = (fi['sharpe'].values > bare['sharpe'].values).sum()
print(f"  FI wins Sharpe: {wins}/{len(bare)}")

print(f"\n  {'Metric':<18} {'Bare PPO':>14} {'FI PPO':>14} {'Improvement':>14}")
for col, fmt in [('sharpe','.2f'),('return','.1%'),('max_dd','.1%'),('trades','.0f')]:
    bm, fm = bare[col].mean(), fi[col].mean()
    if fmt == '.0f': print(f"  {col:<18} {bm:>14.0f} {fm:>14.0f} {fm-bm:>+14.0f}")
    else: print(f"  {col:<18} {bm:>14.2f} {fm:>14.2f} {fm-bm:>+14.2f}")

# Per-split analysis
for sn in df_r['split'].unique():
    sub = df_r[df_r['split']==sn]
    sb = sub[~sub['use_fi']]; sf = sub[sub['use_fi']]
    print(f"\n  [{sn}] FI wins: {(sf['sharpe'].values>sb['sharpe'].values).sum()}/{len(sb)} "
          f"Bare Sharpe={sb['sharpe'].mean():.2f} FI Sharpe={sf['sharpe'].mean():.2f}")

print("=" * 65)
