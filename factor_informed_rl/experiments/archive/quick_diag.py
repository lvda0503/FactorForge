"""Quick diagnostic for A-share experiment"""
import sys, time
sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np

t0 = time.time()
df = pd.read_pickle('d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl')
for c in ['open','high','low','close','volume','pe','pb']:
    df[c] = df[c].ffill().bfill().fillna(0)
print(f"[{time.time()-t0:.1f}s] Loaded {len(df)} days")

# Split
train_df = df[df.index.year <= 2020]
test_df = df[df.index.year >= 2022]
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.env.trading_env import TradingEnv

engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
denoiser = Denoiser(method="none")
sb = StateBuilder(window_size=60, factor_names=engine.factor_names)
print(f"State dim: {sb.state_dim}")

env = TradingEnv(train_df, engine, sb, denoiser, window_size=60,
                 initial_capital=100000, enable_short=True)

# Single step test
state, _ = env.reset()
print(f"State shape: {state.shape}, any NaN: {np.isnan(state).any()}")
a = 0.5
s, r, d, t, info = env.step(a)
print(f"Step: reward={r:.4f}, value={info['total_value']:.0f}, done={d}")
a2 = -0.3
s2, r2, d2, t2, info2 = env.step(a2)
print(f"Short step: reward={r2:.4f}, value={info2['total_value']:.0f}")

# Model test
import torch
from factor_informed_rl.models.actor_critic import PPOActorCritic
model = PPOActorCritic(sb.state_dim, action_dim=1, hidden_dims=[128,64])
st = torch.FloatTensor(state).unsqueeze(0)
act, lp, val = model.get_action(st)
print(f"Action: {act.item():.3f}, Value: {val.item():.3f}")

# 1000 training steps
from factor_informed_rl.training.ppo_trainer import PPOTrainer
trainer = PPOTrainer(model, engine, None, lr_actor=3e-4, lr_critic=1e-3,
                     n_epochs=2, batch_size=64, device="cpu")
t1 = time.time()
trainer.train(env, total_timesteps=1000, n_steps=128, verbose=False)
print(f"[{time.time()-t1:.1f}s] 1000 training steps complete")

# Time estimate
steps_per_sec = 1000 / (time.time() - t1)
est_200k = 200000 / steps_per_sec
print(f"\nSpeed: {steps_per_sec:.0f} steps/sec")
print(f"Estimated 200K steps: {est_200k:.0f}s ({est_200k/60:.1f} min)")
print("\nAll diagnostics PASSED!")
