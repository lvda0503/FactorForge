"""用合成数据验证完整pipeline (不依赖外部网络)"""
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
from factor_informed_rl.evaluation.metrics import compute_all_metrics

def generate_synthetic_data(n_days=1000):
    """生成类真实股票数据 (带趋势+波动+均值回归)"""
    np.random.seed(42)

    # 对数价格 = 随机游走 + 均值回归
    log_returns = np.random.randn(n_days) * 0.015 - 0.0001  # 日波动1.5%, 微负漂移
    # 加入一些趋势
    log_returns[200:400] += 0.003  # 牛市
    log_returns[600:750] -= 0.004  # 熊市
    log_prices = np.cumsum(log_returns)
    close = np.exp(log_prices) * 100

    # OHLC
    daily_range = np.abs(np.random.randn(n_days) * close * 0.01)
    open_price = close - np.random.randn(n_days) * daily_range * 0.3
    high = np.maximum(open_price, close) + np.abs(np.random.randn(n_days)) * daily_range * 0.3
    low = np.minimum(open_price, close) - np.abs(np.random.randn(n_days)) * daily_range * 0.3
    volume = np.abs(np.random.randn(n_days) * 1e7 + 5e7)

    dates = pd.date_range('2015-01-01', periods=n_days, freq='B')

    df = pd.DataFrame({
        'open': open_price, 'high': high, 'low': low,
        'close': close, 'volume': volume,
        'pe': np.full(n_days, 25.0), 'pb': np.full(n_days, 5.0),
    }, index=dates)

    return df

print("Generating synthetic stock data...")
df = generate_synthetic_data(1000)
print(f"  {len(df)} days")

train_df = df.iloc[:700]
val_df = df.iloc[700:850]
test_df = df.iloc[850:]
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 环境
engine = FactorEngine(cfg.factor.factors)
denoiser = Denoiser(method="none")
sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)

train_env = TradingEnv(
    train_df, engine, sb, denoiser, window_size=60,
    position_sizes=(0.0, 0.5, 1.0), initial_capital=100000,
)

# 模型
model = PPOActorCritic(
    input_dim=train_env.observation_space.shape[0],
    action_dim=train_env.action_space.n,
    hidden_dims=[128, 64],
)

# === EXPERIMENT 1: Bare PPO (no factor loss) ===
print("\n[EXP 1] Bare PPO (no factor constraints)...")
trainer_bare = PPOTrainer(
    model, engine, factor_loss=None,
    lr_actor=3e-4, lr_critic=1e-3,
    n_epochs=3, batch_size=64, device="cpu",
)
result_bare = trainer_bare.train(
    train_env, total_timesteps=5000, n_steps=256, verbose=False
)

# 测试
test_env = TradingEnv(
    test_df, engine, sb, denoiser, window_size=60,
    position_sizes=(0.0, 0.5, 1.0), initial_capital=100000,
)
state, _ = test_env.reset()
done = False
while not done:
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        a, _, _ = model.get_action(s, deterministic=True)
    state, r, terminated, truncated, info_bare = test_env.step(a.item())
    done = terminated or truncated

print(f"  Test Return: {info_bare.get('total_return', 0):.2%}")
print(f"  Sharpe: {info_bare.get('sharpe', 0):.2f}")
print(f"  MaxDD: {info_bare.get('max_drawdown', 0):.1%}")
print(f"  Trades: {info_bare.get('trade_count', 0)}")

# === EXPERIMENT 2: Factor-Informed PPO ===
print("\n[EXP 2] Factor-Informed PPO...")

# 重新创建环境 (新引擎)
engine2 = FactorEngine(cfg.factor.factors)
train_env2 = TradingEnv(
    train_df, engine2, sb, denoiser, window_size=60,
    position_sizes=(0.0, 0.5, 1.0), initial_capital=100000,
)

model2 = PPOActorCritic(
    input_dim=train_env2.observation_space.shape[0],
    action_dim=train_env2.action_space.n,
    hidden_dims=[128, 64],
)

fi_loss = FactorInformedLoss(
    engine2, lambda_ic=0.1, lambda_ortho=0.05,
    warmup_steps=500,  # 快速进入约束阶段
)

trainer_fi = PPOTrainer(
    model2, engine2, fi_loss,
    lr_actor=3e-4, lr_critic=1e-3,
    n_epochs=3, batch_size=64, device="cpu",
)
result_fi = trainer_fi.train(
    train_env2, total_timesteps=5000, n_steps=256, verbose=False
)

# 测试
test_env2 = TradingEnv(
    test_df, engine2, sb, denoiser, window_size=60,
    position_sizes=(0.0, 0.5, 1.0), initial_capital=100000,
)
state, _ = test_env2.reset()
done = False
while not done:
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        a, _, _ = model2.get_action(s, deterministic=True)
    state, r, terminated, truncated, info_fi = test_env2.step(a.item())
    done = terminated or truncated

print(f"  Test Return: {info_fi.get('total_return', 0):.2%}")
print(f"  Sharpe: {info_fi.get('sharpe', 0):.2f}")
print(f"  MaxDD: {info_fi.get('max_drawdown', 0):.1%}")
print(f"  Trades: {info_fi.get('trade_count', 0)}")

# 对比
print("\n" + "=" * 55)
print("  ABLATION RESULTS (5K steps, synthetic data)")
print("=" * 55)
print(f"  {'Metric':<20} {'Bare PPO':>12} {'Factor-Informed':>15}")
print(f"  {'-'*47}")
for metric, bare_val, fi_val in [
    ("Total Return", f"{info_bare.get('total_return',0):.2%}", f"{info_fi.get('total_return',0):.2%}"),
    ("Sharpe", f"{info_bare.get('sharpe',0):.2f}", f"{info_fi.get('sharpe',0):.2f}"),
    ("Max Drawdown", f"{info_bare.get('max_drawdown',0):.1%}", f"{info_fi.get('max_drawdown',0):.1%}"),
    ("Trades", str(info_bare.get('trade_count',0)), str(info_fi.get('trade_count',0))),
]:
    print(f"  {metric:<20} {bare_val:>12} {fi_val:>15}")

print("\n=== PIPELINE VERIFIED ===")
