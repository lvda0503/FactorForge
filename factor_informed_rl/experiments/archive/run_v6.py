"""V6 — 15 factors + 涨跌停 + 流动性过滤 + 400K steps"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time, os

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

# 15因子 — 6个维度各2-3个
FACTORS = [
    # 趋势 (3): 短期+长期动量+斜率
    "roc_20", "roc_60", "beta_20",
    # 位置 (3): KDJ RSV + 排名 + 下分位
    "rsv_14", "rank_20", "qtld_20",
    # 波动率 (3): 短期+中期+长期
    "std_10", "std_20", "std_60",
    # 价值 (2): PB + PE分位
    "pb_ratio", "pe_percentile",
    # 量价 (2): 相关+均量
    "corr_20", "vma_20",
    # Aroon (2): 趋势强度
    "imax_20", "imxd_20",
]

STOCKS = {"600519":"茅台","000858":"五粮液","000333":"美的","600276":"恒瑞"}
TOTAL = 400000
cache = "d:/JoinQuant/quant_env/data_cache"
market_ctx = MarketContext()

def run_stock(code, name, use_fi):
    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

    # 计算 PE 分位因子（PE 越低越好，类似 E/P）
    df['pe_percentile'] = (df['pe'].expanding(min_periods=60).rank(pct=True)
                           if 'pe' in df.columns else 0.5).fillna(0.5)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]

    engine = FactorEngine(FACTORS, ic_window=120)
    sb = StateBuilder(window_size=60, factor_names=FACTORS, market_dim=11)

    env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if use_fi else None

    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=8, batch_size=256, device="cpu", entropy_coef=0.03)

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

    rets = np.array(rets); cum = (1+rets).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum-peak)/peak)) if len(cum)>0 else 0
    sr = float(rets.mean()/(np.std(rets)+1e-10)*np.sqrt(252))
    tr = float(cum[-1]-1)
    alive = info['total_value'] > 50000
    bh = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)
    # 统计涨跌停命中次数
    limit_count = sum(1 for k,v in info.items() if 'limit' in str(k))

    return {'code':code,'name':name,'use_fi':use_fi,'return':tr,'sharpe':sr,
            'sortino':0,'max_dd':dd,'trades':info['trade_count'],
            'alive':alive,'bh_ret':bh,'time':time.time()-t0,'state_dim':sb.state_dim}

print("=" * 65)
print("  V6 — 15 Factors + Limit Up/Down + Liquidity + 400K")
print(f"  Factors: {FACTORS}")
print(f"  State dim: {StateBuilder(factor_names=FACTORS, market_dim=11).state_dim}")
print("=" * 65)

all_results = []
for code, name in STOCKS.items():
    bh_ret = pd.read_pickle(f"{cache}/baostock_{code}.pkl")['close']
    bh_ret = bh_ret[bh_ret.index.year >= 2022].iloc[-1]/bh_ret[bh_ret.index.year >= 2022].iloc[0]-1

    print(f"\n[{code} {name}] BH={bh_ret:+.0%}", flush=True)
    print(f"  Bare...", end=" ", flush=True)
    r1 = run_stock(code, name, False)
    print(f"{r1['time']:.0f}s", flush=True)
    print(f"  FI...", end=" ", flush=True)
    r2 = run_stock(code, name, True)
    print(f"{r2['time']:.0f}s", flush=True)
    all_results.extend([r1, r2])

    a1 = "OK" if r1['alive'] else "DEAD"
    a2 = "OK" if r2['alive'] else "DEAD"
    print(f"  Bare[{a1}]: Ret={r1['return']:+.1%} Sharpe={r1['sharpe']:.2f} "
          f"DD={r1['max_dd']:.1%} T={r1['trades']}", flush=True)
    print(f"  FI  [{a2}]: Ret={r2['return']:+.1%} Sharpe={r2['sharpe']:.2f} "
          f"DD={r2['max_dd']:.1%} T={r2['trades']}", flush=True)
    d = r2['sharpe']-r1['sharpe']
    print(f"  Delta: Sharpe {d:+.2f} | DD {r2['max_dd']-r1['max_dd']:+.1%}"
          f" | T {r2['trades']-r1['trades']:+d}", flush=True)

print("\n" + "=" * 65)
print("  SUMMARY (V6 — 15 factors + limits + 400K)")
print("=" * 65)
bare = [r for r in all_results if not r['use_fi']]
fi   = [r for r in all_results if r['use_fi']]
dead_b = sum(1 for r in bare if not r['alive'])
dead_f = sum(1 for r in fi   if not r['alive'])
wins   = sum(1 for b,f in zip(bare,fi) if f['sharpe']>b['sharpe'])
print(f"  Alive: Bare={4-dead_b}/4  FI={4-dead_f}/4")
print(f"  FI wins Sharpe: {wins}/4")
for label, rows in [("Bare PPO", bare), ("FI PPO", fi)]:
    print(f"  {label}: Ret={np.mean([r['return'] for r in rows]):+.1%} "
          f"Sharpe={np.mean([r['sharpe'] for r in rows]):.2f} "
          f"DD={np.mean([r['max_dd'] for r in rows]):.1%} "
          f"T={np.mean([r['trades'] for r in rows]):.0f}")
print(f"  State dim: {r1['state_dim']}")
print("=" * 65)
