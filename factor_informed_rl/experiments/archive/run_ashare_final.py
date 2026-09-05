"""A股实验 v2 Final — 连续动作 + 做空 + 200K步"""
import sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time

from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

# Load + clean
df = pd.read_pickle('d:/JoinQuant/quant_env/data_cache/baostock_600519.pkl')
for c in ['open','high','low','close','volume','pe','pb']:
    df[c] = df[c].ffill().bfill().fillna(0)

train_df = df[df.index.year <= 2020]
test_df = df[df.index.year >= 2022]

print("="*60)
print("  Maotai 600519 — Continuous + Short")
print(f"  Train: {train_df.index[0].date()}~{train_df.index[-1].date()} ({len(train_df)}d)")
print(f"  Train ret: {train_df['close'].iloc[-1]/train_df['close'].iloc[0]-1:.1%}")
print(f"  Test: {test_df.index[0].date()}~{test_df.index[-1].date()} ({len(test_df)}d)")
print(f"  Test ret: {test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1:.1%}")
print("="*60)

TOTAL = 200000

def run(label, use_fi):
    engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
    denoiser = Denoiser(method="none")
    sb = StateBuilder(window_size=60, factor_names=engine.factor_names)

    env = TradingEnv(train_df, engine, sb, denoiser, window_size=60,
                     initial_capital=100000, enable_short=True)

    model = PPOActorCritic(sb.state_dim, action_dim=1, hidden_dims=[256,128,64])

    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if use_fi else None

    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=6, batch_size=256, device="cpu")

    t0 = time.time()
    print(f"\n[{label}] Training {TOTAL:,} steps...")
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)
    elapsed = time.time() - t0

    # Evaluate
    test_env = TradingEnv(test_df, engine, sb, denoiser, window_size=60,
                          initial_capital=100000, enable_short=True)
    state, _ = test_env.reset()
    done = False
    rets = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = test_env.step(float(a.squeeze().numpy()))
        done = terminated or truncated
        rets.append(r)

    rets = np.array(rets)
    cum = np.cumprod(1 + rets)
    dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
    sr = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
    dn = rets[rets < 0]
    so = float(rets.mean() / (np.std(dn) + 1e-10) * np.sqrt(252)) if len(dn) > 1 else sr
    tr = float(cum[-1] - 1)
    ar = float((1 + tr) ** (252 / max(len(rets), 1)) - 1)

    bh = float(test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)
    bh_sr = float(test_df['close'].pct_change().dropna().mean() /
                  (test_df['close'].pct_change().dropna().std() + 1e-10) * np.sqrt(252))

    print(f"  [{label}] {elapsed:.0f}s | Return={tr:.1%} Sharpe={sr:.2f} "
          f"Sortino={so:.2f} DD={dd:.1%} Trades={info['trade_count']}")
    print(f"    Long={info['long_position']:.0%} Short={info['short_position']:.0%}")

    return {'label': label, 'return': tr, 'ann_ret': ar, 'sharpe': sr,
            'sortino': so, 'max_dd': dd, 'trades': info['trade_count'],
            'bh_ret': bh, 'bh_sharpe': bh_sr, 'time': elapsed}

r1 = run("Bare PPO         ", False)
r2 = run("Factor-Informed  ", True)

print("\n" + "=" * 62)
print(f"  {'Metric':<20} {'Bare PPO':>12} {'FI PPO':>12} {'Buy&Hold':>12}")
print(f"  {'-'*56}")
for k, lab in [('return','Total Return'),('sharpe','Sharpe'),('sortino','Sortino'),
                ('max_dd','Max Drawdown'),('trades','Trades')]:
    v1, v2, bh = r1[k], r2[k], r1.get('bh_ret',0) if k=='return' else r1.get('bh_sharpe',0) if k in ('sharpe','sortino') else 0
    if k == 'trades':
        print(f"  {lab:<20} {v1:>12} {v2:>12}")
    else:
        s1 = f"{v1:.1%}" if k in ('return','max_dd') else f"{v1:.2f}"
        s2 = f"{v2:.1%}" if k in ('return','max_dd') else f"{v2:.2f}"
        print(f"  {lab:<20} {s1:>12} {s2:>12}")

si = (r2['sharpe'] - r1['sharpe']) / (abs(r1['sharpe']) + 1e-6) * 100
print(f"\n  Sharpe improvement: {si:+.1f}%")
print(f"  Time: {r1['time']:.0f}s + {r2['time']:.0f}s = {r1['time']+r2['time']:.0f}s")
print("=" * 62)
