"""快速烟雾测试 — 验证完整pipeline可运行"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')

import numpy as np
import torch
from factor_informed_rl.config import cfg, FactorInformedConfig

# 快速测试配置
cfg.ppo.total_timesteps = 5000
cfg.ppo.n_steps = 256
cfg.ppo.n_epochs = 3
cfg.ppo.batch_size = 64
cfg.factor_loss.warmup_steps = 1000
cfg.data.start_date = '2023-01-01'
cfg.data.end_date = '2024-12-31'

from factor_informed_rl.data.loader import DataLoader
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

print("Loading AAPL data...")
loader = DataLoader(source="yfinance")
df = loader.load(cfg.data.ticker, cfg.data.start_date, cfg.data.end_date)
print(f"  {len(df)} days: {df.index[0].date()} -> {df.index[-1].date()}")

train_df, val_df, test_df = loader.split_data(df, 0.7, 0.15)
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 创建环境
engine = FactorEngine(cfg.factor.factors)
denoiser = Denoiser(method="none")
sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)

train_env = TradingEnv(
    train_df, engine, sb, denoiser, window_size=60,
    position_sizes=cfg.env.position_sizes,
    initial_capital=cfg.env.initial_capital,
)

# 创建模型
model = PPOActorCritic(
    input_dim=train_env.observation_space.shape[0],
    action_dim=train_env.action_space.n,
    hidden_dims=[128, 64],
)

# Factor-Informed Loss
fi_loss = FactorInformedLoss(
    engine,
    lambda_ic=0.1, lambda_ortho=0.05,
    warmup_steps=cfg.factor_loss.warmup_steps,
)

# 训练
print("\nTraining Factor-Informed PPO (quick)...")
trainer = PPOTrainer(
    model, engine, fi_loss,
    lr_actor=3e-4, lr_critic=1e-3,
    n_epochs=3, batch_size=64, device="cpu",
)
result = trainer.train(train_env, total_timesteps=5000, n_steps=256, verbose=True)

print(f"\nTraining complete: {result['total_steps']} steps, "
      f"{result['training_time']:.1f}s, {result['total_episodes']} episodes")

# 测试评估
print("\nTest evaluation...")
test_env = TradingEnv(
    test_df, engine, sb, denoiser, window_size=60,
    position_sizes=cfg.env.position_sizes,
    initial_capital=cfg.env.initial_capital,
)

state = test_env.reset()
done = False
while not done:
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        a, _, _ = model.get_action(s, deterministic=True)
    state, r, terminated, truncated, info = test_env.step(a.item())
    done = terminated or truncated

print(f"  Total Return: {info.get('total_return', 0):.2%}")
print(f"  Sharpe: {info.get('sharpe', 0):.2f}")
print(f"  Max Drawdown: {info.get('max_drawdown', 0):.1%}")
print(f"  Trades: {info.get('trade_count', 0)}")

print("\n=== END-TO-END PIPELINE WORKS ===")
