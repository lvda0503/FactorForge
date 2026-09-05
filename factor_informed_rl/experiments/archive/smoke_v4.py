"""Smoke test v4"""
import sys; sys.path.insert(0, 'd:/JoinQuant/quant_env')
import pandas as pd, numpy as np, torch, time

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.training.ppo_trainer import PPOTrainer

ctx = MarketContext()
print(f"CSI300: {len(ctx.csi300)} days")

df = pd.read_pickle('d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl')
for c in ['open','high','low','close','volume','pe','pb','turn']:
    if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
sb = StateBuilder(window_size=60, market_dim=ctx.feature_dim)
print(f"State dim: {sb.state_dim} (expect 79)")

env = TradingEnv(df[df.index.year<=2020], engine, sb, Denoiser(method="none"),
                 window_size=60, enable_short=True, market_ctx=ctx)
s, _ = env.reset()
print(f"State shape: {s.shape}")
info = env.step(0.5)[4]
print(f"Step OK: reward={env.step(0.3)[1]:.4f} pos={info['long_position']:.0%}")

model = PPOActorCritic(sb.state_dim, 1, [128,64])
trainer = PPOTrainer(model, engine, None, lr_actor=3e-4, lr_critic=1e-3,
                     n_epochs=2, batch_size=64, device="cpu")
t0 = time.time()
trainer.train(env, total_timesteps=2000, n_steps=128, verbose=False)
print(f"Training OK: {time.time()-t0:.1f}s")
print("ALL GOOD")
