"""V4 — 市场环境 + 硬风控 + 真实费率 — 4 stocks × 200K"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time, os

from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer
from factor_informed_rl.data.market_context import MarketContext

STOCKS = {"600519": "茅台", "000858": "五粮液", "000333": "美的", "600276": "恒瑞"}
TOTAL = 200000
cache = "d:/JoinQuant/quant_env/data_cache"

# 下载沪深300指数
print("Loading CSI300 index...", end=" ", flush=True)
market_ctx = MarketContext()
print(f"OK ({len(market_ctx.csi300)} days)", flush=True)

def run_stock(code, name, use_fi):
    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns:
            df[c] = df[c].ffill().bfill().fillna(0)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]

    engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
    d = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=engine.factor_names,
                      market_dim=market_ctx.feature_dim)

    env = TradingEnv(train_df, engine, sb, d, window_size=60,
                     enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(sb.state_dim, action_dim=1, hidden_dims=[256,128,64])
    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if use_fi else None
    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=6, batch_size=256, device="cpu")
    t0 = time.time()
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

    test_env = TradingEnv(test_df, engine, sb, d, window_size=60,
                          enable_short=True, market_ctx=market_ctx)
    state, _ = test_env.reset()
    done = False; rets = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = test_env.step(float(a.squeeze().numpy()))
        done = terminated or truncated; rets.append(r)

    rets = np.array(rets)
    cum = (1 + rets).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak)) if len(cum) > 0 else 0
    sr = float(rets.mean() / (np.std(rets)+1e-10) * np.sqrt(252))
    dn = rets[rets < 0]
    so = float(rets.mean() / (np.std(dn)+1e-10) * np.sqrt(252)) if len(dn)>1 else sr
    tr = float(cum[-1] - 1)
    bh = float(test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)

    return {'code':code,'name':name,'use_fi':use_fi,'return':tr,'sharpe':sr,
            'sortino':so,'max_dd':dd,'trades':info['trade_count'],
            'final_value':info['total_value'], 'bh_ret':bh, 'time':time.time()-t0}

print("=" * 65)
print("  V4 — Market Context + Hard Controls — 4 stocks × 200K")
print(f"  State dim: {StateBuilder(market_dim=11).state_dim} (60 price + 5 factor + 3 portfolio + 11 market)")
print("=" * 65)

all_results = []
for code, name in STOCKS.items():
    bh_r = pd.read_pickle(f"{cache}/baostock_{code}.pkl")['close']
    bh_ret = bh_r[bh_r.index.year >= 2022].iloc[-1] / bh_r[bh_r.index.year >= 2022].iloc[0] - 1

    print(f"\n[{code} {name}] BH={bh_ret:+.0%}", flush=True)
    print(f"  Bare PPO...", end=" ", flush=True)
    r1 = run_stock(code, name, False)
    print(f"{r1['time']:.0f}s", flush=True)
    print(f"  FI PPO...", end=" ", flush=True)
    r2 = run_stock(code, name, True)
    print(f"{r2['time']:.0f}s", flush=True)
    all_results.extend([r1, r2])

    alive1 = "OK" if r1['final_value'] > 50000 else "DEAD"
    alive2 = "OK" if r2['final_value'] > 50000 else "DEAD"
    print(f"  Bare[{alive1}]: Ret={r1['return']:+.1%} Sharpe={r1['sharpe']:.2f} "
          f"DD={r1['max_dd']:.1%} T={r1['trades']}", flush=True)
    print(f"  FI  [{alive2}]: Ret={r2['return']:+.1%} Sharpe={r2['sharpe']:.2f} "
          f"DD={r2['max_dd']:.1%} T={r2['trades']}", flush=True)
    delta_sr = r2['sharpe'] - r1['sharpe']
    print(f"  Delta: Sharpe {delta_sr:+.2f} | DD {r2['max_dd']-r1['max_dd']:+.1%}"
          f" | T {r2['trades']-r1['trades']:+d}", flush=True)

print("\n" + "=" * 65)
print("  SUMMARY (V4 — with market context)")
print("=" * 65)
bare = [r for r in all_results if not r['use_fi']]
fi   = [r for r in all_results if r['use_fi']]

dead_bare = sum(1 for r in bare if r['final_value'] < 50000)
dead_fi   = sum(1 for r in fi   if r['final_value'] < 50000)
wins = sum(1 for b, f in zip(bare, fi) if f['sharpe'] > b['sharpe'])

print(f"  Bankruptcy: Bare={dead_bare}/4  FI={dead_fi}/4")
print(f"  FI wins (Sharpe): {wins}/4")

for label, rows in [("Bare PPO", bare), ("FI PPO", fi)]:
    print(f"  {label}: Ret={np.mean([r['return'] for r in rows]):+.1%} "
          f"Sharpe={np.mean([r['sharpe'] for r in rows]):.2f} "
          f"DD={np.mean([r['max_dd'] for r in rows]):.1%} "
          f"Trades={np.mean([r['trades'] for r in rows]):.0f}")
print("=" * 65)
