"""快速实验运行 — 5万步训练，对比裸PPO vs Factor-Informed PPO"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')

import numpy as np
import pandas as pd
import torch

from factor_informed_rl.config import cfg
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer


def make_data(n=1200):
    """生成带趋势+反转模式的合成数据"""
    np.random.seed(42)
    log_ret = np.random.randn(n) * 0.014
    # 牛市段 (agent应该学到做多)
    log_ret[200:350] += 0.004
    log_ret[500:600] -= 0.005  # 熊市
    log_ret[700:850] += 0.003  # 慢牛
    log_p = np.cumsum(log_ret)
    close = np.exp(log_p) * 100

    rng = np.abs(np.random.randn(n) * close * 0.012)
    o = close - np.random.randn(n) * rng * 0.3
    h = np.maximum(o, close) + np.abs(np.random.randn(n)) * rng * 0.3
    l = np.minimum(o, close) - np.abs(np.random.randn(n)) * rng * 0.3
    v = np.abs(np.random.randn(n) * 1e7 + 5e7)
    dates = pd.date_range('2015-01-01', periods=n, freq='B')

    return pd.DataFrame({
        'open': o, 'high': h, 'low': l, 'close': close,
        'volume': v, 'pe': 20.0, 'pb': 4.5
    }, index=dates)


def run_one(use_factor_loss, label, df_train, df_test, total_steps=50000):
    engine = FactorEngine(cfg.factor.factors)
    denoiser = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)

    env = TradingEnv(df_train, engine, sb, denoiser,
                     window_size=60, position_sizes=(0.0, 0.5, 1.0),
                     initial_capital=100000)

    model = PPOActorCritic(env.observation_space.shape[0],
                           env.action_space.n, hidden_dims=[128, 64])

    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=total_steps // 4) if use_factor_loss else None

    trainer = PPOTrainer(model, engine, fl,
                         lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=5, batch_size=128, device="cpu")

    print(f"  [{label}] Training {total_steps} steps...")
    trainer.train(env, total_timesteps=total_steps, n_steps=512, verbose=False)

    # Test
    test_env = TradingEnv(df_test, engine, sb, denoiser,
                          window_size=60, position_sizes=(0.0, 0.5, 1.0),
                          initial_capital=100000)
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
    dd = (cum - peak) / peak
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
    total_ret = float(cum[-1] - 1)
    max_dd = float(np.min(dd))
    trades = info.get('trade_count', 0)

    return {'label': label, 'total_return': total_ret, 'sharpe': sharpe,
            'max_dd': max_dd, 'trades': trades,
            'final_value': info.get('total_value', 100000),
            'use_factor_loss': use_factor_loss}


# ── Main ──
print("=" * 60)
print("  Factor-Informed RL — Quick Experiment (50K steps)")
print("=" * 60)

df = make_data(1200)
df_train = df.iloc[:800]
df_test = df.iloc[800:]

print(f"  Train: {len(df_train)} days, Test: {len(df_test)} days")
print(f"  State dim: {cfg.env.window_size + len(cfg.factor.factors) + 3}")
print(f"  Total timesteps per experiment: 50,000\n")

# Run both
TOTAL_STEPS = 200000
r1 = run_one(False, "Bare PPO         ", df_train, df_test, TOTAL_STEPS)
print(f"    -> Return: {r1['total_return']:.2%}, Sharpe: {r1['sharpe']:.2f}, "
      f"MaxDD: {r1['max_dd']:.1%}, Trades: {r1['trades']}")

r2 = run_one(True,  "Factor-Informed  ", df_train, df_test, TOTAL_STEPS)
print(f"    -> Return: {r2['total_return']:.2%}, Sharpe: {r2['sharpe']:.2f}, "
      f"MaxDD: {r2['max_dd']:.1%}, Trades: {r2['trades']}")

# Comparison
print("\n" + "=" * 60)
print("  RESULTS COMPARISON")
print("=" * 60)
print(f"  {'Metric':<22} {'Bare PPO':>12} {'Factor-Informed':>16} {'Delta':>10}")
print(f"  {'-'*60}")
for key, label, fmt in [
    ('total_return', 'Total Return', '.1%'),
    ('sharpe', 'Sharpe', '.2f'),
    ('max_dd', 'Max Drawdown', '.1%'),
    ('trades', 'Trade Count', 'd'),
]:
    v1 = r1[key]; v2 = r2[key]
    if fmt == 'd':
        s1 = str(v1); s2 = str(v2); d = v2 - v1
        print(f"  {label:<22} {s1:>12} {s2:>16} {d:>+10d}")
    elif fmt == '.1%':
        s1 = f"{v1:.1%}"; s2 = f"{v2:.1%}"
        print(f"  {label:<22} {s1:>12} {s2:>16}")
    elif fmt == '.2f':
        s1 = f"{v1:.2f}"; s2 = f"{v2:.2f}"
        d = v2 - v1
        print(f"  {label:<22} {s1:>12} {s2:>16} {d:>+10.2f}")

si = (r2['sharpe'] - r1['sharpe']) / (abs(r1['sharpe']) + 1e-6) * 100
print(f"\n  Sharpe improvement: {si:+.1f}%")
print("=" * 60)
