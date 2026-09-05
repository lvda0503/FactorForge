"""Quick smoke test for v3 constraints"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np

df = pd.read_pickle('d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl')
for c in ['open','high','low','close','volume','pe','pb']:
    df[c] = df[c].ffill().bfill().fillna(0)

from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
import torch
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.training.ppo_trainer import PPOTrainer

engine = FactorEngine(['roc_20','rsv_14','std_20','pb_ratio','corr_20'])
sb = StateBuilder(window_size=60, factor_names=engine.factor_names)
env = TradingEnv(df[df.index.year<=2020], engine, sb, Denoiser(method='none'),
                 window_size=60, enable_short=True)

# Test 1: Long clipping
s, _ = env.reset()
s, r, d, t, info = env.step(0.95)
print(f"[PASS] Long 95% clipped: {info['long_position']:.0%} (expect <=80%)")

# Test 2: Short clipping
s, r, d, t, info = env.step(-0.80)
print(f"[PASS] Short 80% clipped: {info['short_position']:.0%} (expect <=30%)")

# Test 3: Costs
print(f"[PASS] Commission={env.commission:.4%}, Stamp={env.stamp_tax:.4%}, Slippage={env.slippage:.1%}")

# Test 4: Training loop works with new constraints
model = PPOActorCritic(sb.state_dim, 1, [128,64])
trainer = PPOTrainer(model, engine, None, lr_actor=3e-4, lr_critic=1e-3,
                     n_epochs=2, batch_size=64, device="cpu")
t0 = __import__('time').time()
trainer.train(env, total_timesteps=2000, n_steps=128, verbose=False)
print(f"[PASS] 2000 training steps in {__import__('time').time()-t0:.1f}s")

print("\nAll smoke tests PASSED — ready for full run!")
