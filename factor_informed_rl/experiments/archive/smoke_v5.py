"""Smoke V5"""
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
df = pd.read_pickle('d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl')
for c in ['open','high','low','close','volume','pe','pb']:
    if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

sb = StateBuilder(window_size=60, market_dim=11)
env = TradingEnv(df[df.index.year<=2020],
    FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"]),
    sb, Denoiser(method="none"), window_size=60, enable_short=True, market_ctx=ctx)

s, _ = env.reset()
s, r, d, t, info = env.step(0.5)
print("Long 0.5:", f"reward={r:.5f}", f"long={info['long_position']:.0%}")
s, r, d, t, info = env.step(-0.08)
print("Short 0.08:", f"reward={r:.5f}", f"short={info['short_position']:.0%}")
print(f"Short cap={env.max_short_pct:.0f} StopLoss={env.stop_loss_pct:.0f}")

model = PPOActorCritic(sb.state_dim, 1, [128,64])
trainer = PPOTrainer(model, FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"]),
                     None, lr_actor=3e-4, lr_critic=1e-3, n_epochs=1, batch_size=64,
                     entropy_coef=0.03, device="cpu")
t0 = time.time()
trainer.train(env, total_timesteps=1000, n_steps=128, verbose=False)
print(f"Training OK: {time.time()-t0:.1f}s")
print("SMOKE PASSED - starting V5 full run")
