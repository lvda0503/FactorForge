"""Quick single-stock run — 250K steps, FI vs Bare"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

STOCKS = {"000858":"五粮液","000333":"美的","600276":"恒瑞"}
TOTAL = 250000
cache = "d:/JoinQuant/quant_env/data_cache"
market_ctx = MarketContext()

for code, name in STOCKS.items():
    df = pd.read_pickle(f"{cache}/baostock_{code}.pkl")
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]
    bh_ret = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)

    for use_fi, label in [(False,"Bare"), (True,"FI")]:
        engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
        sb = StateBuilder(window_size=60, market_dim=11)
        env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
        fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                                warmup_steps=TOTAL//4) if use_fi else None
        trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                             n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)

        t0 = time.time()
        trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

        test_env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                              window_size=60, enable_short=True, market_ctx=market_ctx)
        state, _ = test_env.reset(); done = False; rets = []
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = model.get_action(s, deterministic=True)
            state, r, terminated, truncated, info = test_env.step(float(a.squeeze().numpy()))
            done = terminated or truncated; rets.append(r)

        rets = np.array(rets)
        cum = (1+rets).cumprod(); peak = np.maximum.accumulate(cum)
        dd = float(np.min((cum-peak)/peak)) if len(cum)>0 else 0
        sr = float(rets.mean()/(np.std(rets)+1e-10)*np.sqrt(252))
        tr = float(cum[-1]-1)
        alive = "OK" if info['total_value']>50000 else "DEAD"

        print(f"[{name}] {label:5s}: Ret={tr:+.1%} Sharpe={sr:.2f} DD={dd:.1%} "
              f"Trades={info['trade_count']} [{alive}] ({time.time()-t0:.0f}s)", flush=True)

    print(flush=True)
