"""多股票实验 — 400K步 × 8只A股"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')

import pandas as pd, numpy as np, torch, time, os

from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

STOCKS = {
    "600519": "贵州茅台", "000001": "平安银行", "000858": "五粮液",
    "600036": "招商银行", "000333": "美的集团", "600276": "恒瑞医药",
    "000725": "京东方A", "601318": "中国平安",
}
TOTAL = 150000
cache = "d:/JoinQuant/quant_env/data_cache"

def run_stock(code, name, use_fi):
    path = f"{cache}/baostock_{code}.pkl"
    if not os.path.exists(path):
        return None

    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb']:
        df[c] = df[c].ffill().bfill().fillna(0)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]
    if len(train_df) < 500 or len(test_df) < 200:
        return None

    engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
    denoiser = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=engine.factor_names)

    env = TradingEnv(train_df, engine, sb, denoiser, window_size=60,
                     initial_capital=100000, enable_short=True)

    model = PPOActorCritic(sb.state_dim, action_dim=1, hidden_dims=[256,128,64])

    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if use_fi else None

    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=6, batch_size=256, device="cpu")

    t0 = time.time()
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)
    elapsed = time.time() - t0
    print(f"    [{name}] {elapsed:.0f}s", flush=True)

    # Test
    test_env = TradingEnv(test_df, engine, sb, denoiser, window_size=60,
                          initial_capital=100000, enable_short=True)
    state, _ = test_env.reset()
    done = False; rets = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = test_env.step(float(a.squeeze().numpy()))
        done = terminated or truncated
        rets.append(r)

    rets = np.array(rets)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak)) if len(cum) > 0 else 0
    sr = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
    dn = rets[rets < 0]
    so = float(rets.mean() / (np.std(dn) + 1e-10) * np.sqrt(252)) if len(dn) > 1 else sr
    tr = float(cum[-1] - 1)
    ar = float((1 + tr) ** (252 / max(len(rets), 1)) - 1)
    bh = float(test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)

    return {'code': code, 'name': name, 'use_fi': use_fi,
            'return': tr, 'ann_ret': ar, 'sharpe': sr, 'sortino': so,
            'max_dd': dd, 'trades': info['trade_count'],
            'long_pct': info.get('long_position',0), 'short_pct': info.get('short_position',0),
            'bh_ret': bh, 'time': elapsed}

# Main
print("=" * 65)
print("  MULTI-STOCK EXPERIMENT — 8 A-shares × 400K steps")
print("=" * 65)

all_results = []
for code, name in STOCKS.items():
    path = f"{cache}/baostock_{code}.pkl"
    if not os.path.exists(path):
        print(f"\n[{code} {name}] Data not found, downloading first...")
        continue

    test_ret = pd.read_pickle(path)['close']
    test_ret = test_ret[test_ret.index.year >= 2022]
    if len(test_ret) < 200:
        continue
    test_ret = test_ret.iloc[-1] / test_ret.iloc[0] - 1

    print(f"\n[{code} {name}] Bare PPO...", end=" ", flush=True)
    r_bare = run_stock(code, name, False)
    print(f"FI PPO...", end=" ", flush=True)
    r_fi = run_stock(code, name, True)

    if r_bare and r_fi:
        all_results.extend([r_bare, r_fi])
        delta_sr = r_fi['sharpe'] - r_bare['sharpe']
        print(f"\n{code} {name}: Test BH={test_ret:+.0%}")
        print(f"  Bare:   Ret={r_bare['return']:+.1%} Sharpe={r_bare['sharpe']:.2f} "
              f"Sortino={r_bare['sortino']:.2f} DD={r_bare['max_dd']:.1%} "
              f"Trades={r_bare['trades']} L={r_bare['long_pct']:.0%} S={r_bare['short_pct']:.0%}")
        print(f"  FI:     Ret={r_fi['return']:+.1%} Sharpe={r_fi['sharpe']:.2f} "
              f"Sortino={r_fi['sortino']:.2f} DD={r_fi['max_dd']:.1%} "
              f"Trades={r_fi['trades']} L={r_fi['long_pct']:.0%} S={r_fi['short_pct']:.0%}")
        print(f"  Delta:  Sharpe {delta_sr:+.2f} | DD {r_fi['max_dd']-r_bare['max_dd']:+.1%} "
              f"| Trades {r_fi['trades']-r_bare['trades']:+d}")

# Aggregated summary
print("\n" + "=" * 65)
print("  AGGREGATED SUMMARY")
print("=" * 65)

df_r = pd.DataFrame(all_results)
bare = df_r[~df_r['use_fi']]
fi = df_r[df_r['use_fi']]

print(f"\n  Stocks tested: {len(bare)}")
print(f"\n  {'Metric':<18} {'Bare PPO (avg)':>16} {'FI PPO (avg)':>16} {'Improvement':>14}")
print(f"  {'-'*64}")

for col, label, fmt in [
    ('return', 'Total Return', '.1%'),
    ('sharpe', 'Sharpe', '.2f'),
    ('sortino', 'Sortino', '.2f'),
    ('max_dd', 'Max Drawdown', '.1%'),
    ('trades', 'Trade Count', '.0f'),
]:
    b_mean, f_mean = bare[col].mean(), fi[col].mean()
    delta = f_mean - b_mean
    b_s = f"{b_mean:{fmt}}" if fmt != '.0f' else f"{b_mean:.0f}"
    f_s = f"{f_mean:{fmt}}" if fmt != '.0f' else f"{f_mean:.0f}"
    d_s = f"{delta:+{fmt}}" if fmt != '.0f' else f"{delta:+.0f}"
    print(f"  {label:<18} {b_s:>16} {f_s:>16} {d_s:>14}")

# Win count
wins = (fi['sharpe'].values > bare['sharpe'].values).sum()
wins_dd = (fi['max_dd'].values > bare['max_dd'].values).sum()  # less negative = better
wins_trade = (fi['trades'].values >= bare['trades'].values).sum()

print(f"\n  FI wins (Sharpe): {wins}/{len(bare)} stocks")
print(f"  FI wins (MaxDD):  {wins_dd}/{len(bare)} stocks")
print(f"  FI wins (Trades): {wins_trade}/{len(bare)} stocks")
print(f"  Avg training time: {bare['time'].mean():.0f}s + {fi['time'].mean():.0f}s")
print("=" * 65)
