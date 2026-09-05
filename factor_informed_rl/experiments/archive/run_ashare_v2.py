"""A股实验 v2 — 连续动作 + 做空 + 下行周期训练"""
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
df = pd.read_pickle(path)

print(f"Moutai: {len(df)} days ({df.index[0].date()} -> {df.index[-1].date()})")
print(f"Total Return: {df['close'].iloc[-1]/df['close'].iloc[0]-1:.1%}")

# 数据验证
for col in ['open','high','low','close','volume','pe','pb']:
    n_nan = df[col].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {col} has {n_nan} NaN values, fixing...")
        df[col] = df[col].ffill().bfill().fillna(0)
assert df[['open','high','low','close','volume']].isna().sum().sum() == 0, "Price data has NaN"
print("  Data validated: all NaN fixed")

# 按年显示收益
for y in range(2015, 2026):
    yd = df[df.index.year == y]
    if len(yd) > 0:
        ret = yd['close'].iloc[-1] / yd['close'].iloc[0] - 1
        print(f"  {y}: {ret:+.1%} ({len(yd)}d)")

# Split: 训练2015-2020 (含股灾+熊市), 验证2021, 测试2022-2025
train_mask = df.index.year <= 2020
val_mask = df.index.year == 2021
test_mask = df.index.year >= 2022

train_df = df[train_mask]
val_df = df[val_mask]
test_df = df[test_mask]

print(f"\nTrain: {train_df.index[0].date()}~{train_df.index[-1].date()} ({len(train_df)}d) "
      f"ret={train_df['close'].iloc[-1]/train_df['close'].iloc[0]-1:.1%}")
print(f"  Down periods: 2015 crash, 2016 circuit breaker, 2018 bear, 2020 COVID")
print(f"Val:   {val_df.index[0].date()}~{val_df.index[-1].date()} ({len(val_df)}d)")
print(f"Test:  {test_df.index[0].date()}~{test_df.index[-1].date()} ({len(test_df)}d) "
      f"ret={test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1:.1%}")

# 连续动作配置
ENABLE_SHORT = True
CONTINUOUS = True
TOTAL_STEPS = 200000

def run_experiment(name, use_fl):
    engine = FactorEngine(cfg.factor.factors)
    denoiser = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)

    env = TradingEnv(
        train_df, engine, sb, denoiser,
        window_size=60, initial_capital=100000,
        enable_short=ENABLE_SHORT,
        commission=0.001, slippage=0.001,
    )

    action_dim = 1  # 连续动作
    model = PPOActorCritic(
        env.observation_space.shape[0], action_dim=action_dim,
        hidden_dims=[256, 128, 64],
    )

    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL_STEPS // 4) if use_fl else None

    trainer = PPOTrainer(model, engine, fl,
                         lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=8, batch_size=256, device="cpu")

    print(f"\n[{name}] Training {TOTAL_STEPS:,} steps (continuous, short={'yes' if ENABLE_SHORT else 'no'})...")
    result = trainer.train(env, total_timesteps=TOTAL_STEPS, n_steps=1024, verbose=True)

    # Test
    test_env = TradingEnv(
        test_df, engine, sb, denoiser,
        window_size=60, initial_capital=100000,
        enable_short=ENABLE_SHORT,
    )
    state, _ = test_env.reset()
    done = False
    returns = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = test_env.step(a.item())
        done = terminated or truncated
        returns.append(r)

    rets = np.array(returns)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak)) if len(cum) > 0 else 0.0
    mean_r = np.mean(rets); std_r = np.std(rets)
    sharpe = float(mean_r / (std_r + 1e-10) * np.sqrt(252))
    total_ret = float(cum[-1] - 1)
    ann_ret = float((1 + total_ret) ** (252 / max(len(rets), 1)) - 1)

    # 下行风险
    dn = rets[rets < 0]
    sortino = float(mean_r / (np.std(dn) + 1e-10) * np.sqrt(252)) if len(dn) > 1 else sharpe

    bh_ret = float(test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)
    bh_r = test_df['close'].pct_change().dropna()
    bh_sharpe = float(bh_r.mean() / (bh_r.std() + 1e-10) * np.sqrt(252))

    return {
        'name': name, 'total_return': total_ret, 'ann_return': ann_ret,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': dd,
        'trades': info.get('trade_count', 0),
        'long_pct': info.get('long_position', 0),
        'short_pct': info.get('short_position', 0),
        'final_value': info.get('total_value', 0),
        'bh_ret': bh_ret, 'bh_sharpe': bh_sharpe,
        'time': result['training_time'],
    }

# ── Run ──
print("\n" + "=" * 65)
print("  MAOTAI 600519 — Continuous + Short — 200K steps")
print("=" * 65)

r_none = run_experiment("Bare PPO", False)
r_fi = run_experiment("Factor-Informed PPO", True)

print("\n" + "=" * 65)
print("  RESULTS (Test: 2022-2025, Maotai down -8.9%)")
print("=" * 65)
print(f"  Benchmark Buy&Hold: {r_none['bh_ret']:.1%} | Sharpe: {r_none['bh_sharpe']:.2f}")
print()
print(f"  {'Metric':<22} {'Bare PPO':>14} {'Factor-Informed':>16} {'Delta':>10}")
print(f"  {'-'*62}")
for key, label in [
    ('total_return', 'Total Return'), ('ann_return', 'Ann Return'),
    ('sharpe', 'Sharpe'), ('sortino', 'Sortino'),
    ('max_dd', 'Max Drawdown'), ('trades', 'Trade Count'),
    ('long_pct', 'Avg Long %'), ('short_pct', 'Avg Short %'),
]:
    v1, v2 = r_none[key], r_fi[key]
    if key == 'trades':
        print(f"  {label:<22} {v1:>14} {v2:>16} {v2-v1:>+10d}")
    elif key in ('long_pct', 'short_pct'):
        print(f"  {label:<22} {v1:>13.0%} {v2:>15.0%}")
    elif key in ('total_return', 'ann_return', 'max_dd'):
        print(f"  {label:<22} {v1:>13.1%} {v2:>15.1%}")
    else:
        print(f"  {label:<22} {v1:>14.2f} {v2:>16.2f} {v2-v1:>+10.2f}")

si = (r_fi['sharpe'] - r_none['sharpe']) / (abs(r_none['sharpe']) + 1e-6) * 100
print(f"\n  Sharpe improvement: {si:+.1f}%")
print(f"  Time: Bare={r_none['time']:.0f}s  FI={r_fi['time']:.0f}s")
print("=" * 65)
