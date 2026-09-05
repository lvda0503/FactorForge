"""A股真实数据实验 — 贵州茅台 600519"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')

import numpy as np
import pandas as pd
import torch
import os

from factor_informed_rl.config import cfg
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

# Load data
path = "d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl"
if not os.path.exists(path):
    print("Downloading data first...")
    import subprocess
    subprocess.run([sys.executable, "download_adata.py"], cwd=os.path.dirname(__file__))

df = pd.read_pickle(path)
print(f"Loaded Maotai: {len(df)} days, {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Total period return: {df['close'].iloc[-1]/df['close'].iloc[0]-1:.1%}")

# Split: train(2015-2021), val(2022), test(2023-2025)
n = len(df)
train_end = int(n * 0.65)
val_end = int(n * 0.80)
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]
print(f"Train: {train_df.index[0].date()}~{train_df.index[-1].date()} ({len(train_df)} days)")
print(f"Val:   {val_df.index[0].date()}~{val_df.index[-1].date()} ({len(val_df)} days)")
print(f"Test:  {test_df.index[0].date()}~{test_df.index[-1].date()} ({len(test_df)} days)")
print(f"Test period return: {test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1:.1%}")

def run_experiment(name, use_fl, total_steps):
    engine = FactorEngine(cfg.factor.factors)
    denoiser = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)

    env = TradingEnv(train_df, engine, sb, denoiser,
                     window_size=60, position_sizes=(0.0, 0.5, 1.0),
                     initial_capital=100000)

    model = PPOActorCritic(env.observation_space.shape[0],
                           env.action_space.n, hidden_dims=[256, 128, 64])

    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=total_steps // 4) if use_fl else None

    trainer = PPOTrainer(model, engine, fl,
                         lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=8, batch_size=256, device="cpu")

    print(f"\n[{name}] Training {total_steps:,} steps...")
    result = trainer.train(env, total_timesteps=total_steps, n_steps=1024, verbose=True)

    # Test
    test_env = TradingEnv(test_df, engine, sb, denoiser,
                          window_size=60, position_sizes=(0.0, 0.5, 1.0),
                          initial_capital=100000)
    state, _ = test_env.reset()
    done = False
    returns = []
    equities = [100000]
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = test_env.step(a.item())
        done = terminated or truncated
        returns.append(r)
        equities.append(info['total_value'])

    rets = np.array(returns)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = np.min((cum - peak) / peak)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
    total_ret = float(cum[-1] - 1)
    ann_ret = float((1 + total_ret) ** (252 / len(rets)) - 1)
    sortino_ret = rets[rets < 0]
    sortino = float(np.mean(rets) / (np.std(sortino_ret) + 1e-10) * np.sqrt(252)) if len(sortino_ret) > 1 else sharpe

    bh_ret = float(test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)
    bh_sharpe = float(np.mean(test_df['close'].pct_change().dropna()) /
                      (np.std(test_df['close'].pct_change().dropna()) + 1e-10) * np.sqrt(252))

    return {
        'name': name, 'total_return': total_ret, 'ann_return': ann_ret,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': float(dd),
        'trades': info.get('trade_count', 0), 'final_value': info.get('total_value', 0),
        'benchmark_return': bh_ret, 'benchmark_sharpe': bh_sharpe,
        'training_time': result['training_time'],
    }

# Run both
print("\n" + "=" * 60)
print("  A-SHARE EXPERIMENT — Kweichow Moutai (600519)")
print("  200K steps per experiment")
print("=" * 60)

r1 = run_experiment("Bare PPO", False, 200000)
r2 = run_experiment("Factor-Informed PPO", True, 200000)

# Results
print("\n" + "=" * 65)
print("  RESULTS — Kweichow Moutai 600519 (Test: 2022~2025)")
print("=" * 65)
print(f"  Benchmark (Buy&Hold): Return={r1['benchmark_return']:.1%}, Sharpe={r1['benchmark_sharpe']:.2f}")
print()
print(f"  {'Metric':<22} {'Bare PPO':>12} {'Factor-Informed':>16} {'Delta':>10}")
print(f"  {'-'*60}")
for key, label, fmt in [
    ('total_return', 'Total Return', '.1%'),
    ('ann_return', 'Ann Return', '.1%'),
    ('sharpe', 'Sharpe', '.2f'),
    ('sortino', 'Sortino', '.2f'),
    ('max_dd', 'Max Drawdown', '.1%'),
    ('trades', 'Trade Count', 'd'),
]:
    v1, v2 = r1[key], r2[key]
    if fmt == 'd':
        print(f"  {label:<22} {v1:>12} {v2:>16} {v2-v1:>+10d}")
    elif fmt == '.1%':
        print(f"  {label:<22} {v1:>12.1%} {v2:>16.1%}")
    elif fmt == '.2f':
        print(f"  {label:<22} {v1:>12.2f} {v2:>16.2f} {v2-v1:>+10.2f}")

si = (r2['sharpe'] - r1['sharpe']) / (abs(r1['sharpe']) + 1e-6) * 100
print(f"\n  Sharpe improvement: {si:+.1f}%")
print(f"  Training: Bare={r1['training_time']:.0f}s, FI={r2['training_time']:.0f}s")
print("=" * 65)
