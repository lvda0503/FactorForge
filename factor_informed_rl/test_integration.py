"""集成测试 — 验证所有模块可正常导入和运行"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')

import numpy as np
import torch

print('1. Testing imports...')
from factor_informed_rl.config import cfg
from factor_informed_rl.data.loader import DataLoader
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.buffer import RolloutBuffer
print('[PASS] All imports')

print('\n2. Testing config...')
print(f'   Ticker: {cfg.data.ticker}, Source: {cfg.data.source}')
print(f'   Factors: {cfg.factor.factors}')
print(f'   State dim: {cfg.env.window_size + len(cfg.factor.factors) + 3}')
print('[PASS]')

print('\n3. Testing factor engine...')
engine = FactorEngine(cfg.factor.factors, ic_window=120)
prices = np.random.randn(60, 4).cumsum(axis=0) + 100
prices[:, 3] = prices[:, 3].clip(min=1)
volume = np.abs(np.random.randn(60) * 1000 + 5000)
factors = engine.compute_factors(prices, volume, pb_value=3.5)
for name, val in factors.items():
    print(f'   {name}: {val:.4f}')
print('[PASS]')

print('\n4. Testing state builder...')
sb = StateBuilder(window_size=60, factor_names=cfg.factor.factors)
state = sb.build(
    price_window=np.random.randn(60, 5).astype(np.float32),
    close_denoised=np.random.randn(60).astype(np.float32),
    factors=factors, position=0.5, cash_ratio=0.5, unrealized_pnl=0.02
)
print(f'   State shape: {state.shape}, dim: {sb.state_dim}')
assert state.shape == (sb.state_dim,), f"Expected {sb.state_dim}, got {state.shape}"
print('[PASS]')

print('\n5. Testing model creation...')
model = PPOActorCritic(input_dim=sb.state_dim, action_dim=3, hidden_dims=[128, 64])
dummy = torch.FloatTensor(state).unsqueeze(0)
action, log_prob, value = model.get_action(dummy)
print(f'   Action: {action.item()}, LogProb: {log_prob.item():.4f}, Value: {value.item():.4f}')
print('[PASS]')

print('\n6. Testing factor-informed loss...')
fi_loss = FactorInformedLoss(factor_engine=engine, lambda_ic=0.1, lambda_ortho=0.05)
loss_info = fi_loss.forward(global_step=100000)
print(f'   IC loss: {loss_info["ic_loss"].item():.4f}')
print(f'   Ortho loss: {loss_info["ortho_loss"].item():.4f}')
print(f'   Total: {loss_info["total_factor_loss"].item():.4f}')
print('[PASS]')

print('\n7. Testing buffer...')
buf = RolloutBuffer(n_steps=128, state_dim=sb.state_dim)
buf.add(state, 1, -0.5, 0.01, 10.5, False)
buf.compute_gae(last_value=10.2)
print(f'   Buffer filled: {buf.idx}, Advantage[0]: {buf.advantages[0]:.4f}')
print('[PASS]')

print('\n8. Testing denoiser...')
d = Denoiser(method="ma", window=5)
noisy = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5
clean = d.denoise(noisy)
print(f'   Input std: {noisy.std():.3f}, Output std: {clean.std():.3f}')
print('[PASS]')

print('\n' + '='*50)
print('  ALL INTEGRATION TESTS PASSED')
print('='*50)
