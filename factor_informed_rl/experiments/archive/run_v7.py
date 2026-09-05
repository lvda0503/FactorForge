"""V7 — Value-Defensive vs Quality-Offensive vs Bare PPO vs CSI300"""
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

# 两套策略因子
STRATEGIES = {
    "Value-Defensive": [
        "pb_ratio",         # 截面估值 — BP=1/PB (Graham 1934)
        "pe_percentile",    # 时序估值 — PE历史分位 (PIT-safe)
        "rank_20",          # 短期排名 — 当前位置 (替代RSV14, 更鲁棒)
        "std_60",           # 低波动 — 防御 (Black 1972)
        "corr_20",          # 量价配合 — 确认资金跟进
    ],
    "Quality-Offensive": [
        "roc_60",           # 长期动量 — 趋势确认 (Jegadeesh 1993)
        "beta_20",          # 趋势斜率 — 速度与方向
        "rsqr_20",          # 拟合度 R² — 防假突破
        "vma_20",           # 放量确认 — 量在价先
        "std_20",           # 适度波动 — 有波动才有Alpha
    ],
}

BARE_FACTORS = [
    "roc_20", "rsv_14", "std_20", "pb_ratio", "corr_20"
]

STOCKS = {"600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui"}
TOTAL = 300000
cache = "d:/JoinQuant/quant_env/data_cache"
market_ctx = MarketContext()

# 沪深300基准收益
csi300 = market_ctx.csi300
csi_test = csi300[csi300.index.year >= 2022]
CSI300_BH = float(csi_test['close'].iloc[-1] / csi_test['close'].iloc[0] - 1)
CSI300_RETS = csi_test['close'].pct_change().dropna()
CSI300_SHARPE = float(CSI300_RETS.mean() / (CSI300_RETS.std() + 1e-10) * np.sqrt(252))
CSI300_DD = float(np.min((csi_test['close'].cumprod() /
    np.maximum.accumulate(csi_test['close'].cumprod()) - 1)))

def run_experiment(name, factors, code, stock_name, bare=False):
    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    if 'pe_percentile' in factors:
        # PIT-safe: expanding窗口只用到当前日期及之前的数据
        df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]

    engine = FactorEngine(factors, ic_window=120)
    sb = StateBuilder(window_size=60, factor_names=factors, market_dim=11)

    env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])

    use_fi = not bare
    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if use_fi else None

    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=8, batch_size=256, device="cpu", entropy_coef=0.03)
    t0 = time.time()
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

    # 保存模型权重
    save_dir = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{name}_{code}_{'' if bare else 'fi'}.pt"
    torch.save({'model_state': model.state_dict(), 'factors': factors,
                'state_dim': sb.state_dim, 'bare': bare}, save_path)

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

    return {'name':name,'factors':factors,'bare':bare,'code':code,'stock':stock_name,
            'return':tr,'sharpe':sr,'max_dd':dd,'trades':info['trade_count'],
            'alive':alive,'bh_ret':bh,'time':time.time()-t0,'state_dim':sb.state_dim}

print("=" * 65)
print("  V7 — Value-Defensive vs Quality-Offensive vs Bare PPO")
print(f"  CSI300 BH={CSI300_BH:+.1%} Sharpe={CSI300_SHARPE:.2f} DD={CSI300_DD:.1%}")
print("=" * 65)

all_results = []
for code, sname in STOCKS.items():
    bh_r = pd.read_pickle(f"{cache}/baostock_{code}.pkl")['close']
    bh_ret = bh_r[bh_r.index.year >= 2022].iloc[-1]/bh_r[bh_r.index.year >= 2022].iloc[0]-1

    print(f"\n[{code} {sname}] BH={bh_ret:+.0%}", flush=True)

    # 1. Bare PPO (baseline)
    print(f"  Bare PPO...", end=" ", flush=True)
    r_bare = run_experiment("Bare PPO", BARE_FACTORS,
                            code, sname, bare=True)
    all_results.append(r_bare)
    print(f"{r_bare['time']:.0f}s Ret={r_bare['return']:+.1%} Sharpe={r_bare['sharpe']:.2f} "
          f"DD={r_bare['max_dd']:.1%} T={r_bare['trades']}", flush=True)

    # 2. Value-Defensive FI-PPO
    print(f"  Value-Def...", end=" ", flush=True)
    r_val = run_experiment("Value-Defensive", STRATEGIES["Value-Defensive"],
                           code, sname, bare=False)
    all_results.append(r_val)
    print(f"{r_val['time']:.0f}s Ret={r_val['return']:+.1%} Sharpe={r_val['sharpe']:.2f} "
          f"DD={r_val['max_dd']:.1%} T={r_val['trades']}", flush=True)

    # 3. Quality-Offensive FI-PPO
    print(f"  Quality-Off...", end=" ", flush=True)
    r_qual = run_experiment("Quality-Offensive", STRATEGIES["Quality-Offensive"],
                            code, sname, bare=False)
    all_results.append(r_qual)
    print(f"{r_qual['time']:.0f}s Ret={r_qual['return']:+.1%} Sharpe={r_qual['sharpe']:.2f} "
          f"DD={r_qual['max_dd']:.1%} T={r_qual['trades']}", flush=True)

# Aggregate summary
print("\n" + "=" * 70)
print("  V7 AGGREGATE RESULTS (Test: 2022-2025)")
print(f"  CSI300 Benchmark: Ret={CSI300_BH:+.1%} Sharpe={CSI300_SHARPE:.2f} DD={CSI300_DD:.1%}")
print("=" * 70)
print(f"  {'Strategy':<22} {'Avg Ret':>10} {'Avg Sharpe':>12} {'Avg DD':>10} {'Avg Trades':>11} {'Alive':>7} {'Wins':>6}")
print(f"  {'-'*78}")

for label in ["Bare PPO", "Value-Defensive", "Quality-Offensive"]:
    rows = [r for r in all_results if r['name'] == label]
    print(f"  {label:<22} {np.mean([r['return'] for r in rows]):>+9.1%} "
          f"{np.mean([r['sharpe'] for r in rows]):>12.2f} "
          f"{np.mean([r['max_dd'] for r in rows]):>10.1%} "
          f"{np.mean([r['trades'] for r in rows]):>11.0f} "
          f"{sum(r['alive'] for r in rows):>6d}/4 "
          f"{sum(1 for r in rows if r['sharpe'] > CSI300_SHARPE):>6d}/4")

# Head-to-head comparison
print(f"\n  Head-to-Head vs Bare PPO:")
bare_rows = {r['code']: r for r in all_results if r['name'] == 'Bare PPO'}
for label in ["Value-Defensive", "Quality-Offensive"]:
    strat_rows = {r['code']: r for r in all_results if r['name'] == label}
    wins = sum(1 for code in bare_rows if strat_rows[code]['sharpe'] > bare_rows[code]['sharpe'])
    deltas = [strat_rows[code]['sharpe'] - bare_rows[code]['sharpe'] for code in bare_rows]
    print(f"    {label}: wins {wins}/4, avg delta={np.mean(deltas):+.2f}")

print("=" * 70)
