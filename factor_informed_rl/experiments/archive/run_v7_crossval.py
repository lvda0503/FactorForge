"""V7 Cross-Validation — Train on 4 stocks, test on ALL 12 stocks"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time, os, pickle

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

STRATEGIES = {
    "Value-Defensive": ["pb_ratio","pe_percentile","rsv_14","std_60","corr_20"],
    "Quality-Offensive": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
    "Bare PPO": ["roc_20","rsv_14","std_20","pb_ratio","corr_20"],
}

TRAIN_STOCKS = {"600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui"}
ALL_STOCKS = {
    "600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui",
    "600887":"Yili","002415":"Hikvision","600030":"CITIC Sec","000651":"Gree",
    "601166":"Ind Bank","600585":"Conch","000002":"Vanke","601888":"CDFG",
}
TOTAL = 200000
cache = "d:/JoinQuant/quant_env/data_cache"
save_dir = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
os.makedirs(save_dir, exist_ok=True)
market_ctx = MarketContext()

def train_one(code, stock_name, factors, name, bare=False):
    """Train and save model"""
    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    if 'pe_percentile' in factors:
        df['pe_percentile'] = df['pe'].rank(pct=True).fillna(0.5)

    train_df = df[df.index.year <= 2020]
    engine = FactorEngine(factors, ic_window=120)
    sb = StateBuilder(window_size=60, factor_names=factors, market_dim=11)

    env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4) if not bare else None
    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)
    t0 = time.time()
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

    # Save
    save_path = f"{save_dir}/{name}_{code}.pt"
    torch.save({'model_state': model.state_dict(), 'factors': factors,
                'state_dim': sb.state_dim, 'bare': bare, 'stock_name': stock_name}, save_path)
    return {'name': name, 'code': code, 'state_dim': sb.state_dim, 'time': time.time()-t0, 'save_path': save_path}

def test_one(code, stock_name, model_state, factors, state_dim):
    """Test a saved model on one stock (no training)"""
    path = f"{cache}/baostock_{code}.pkl"
    if not os.path.exists(path): return None
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    if 'pe_percentile' in factors:
        df['pe_percentile'] = df['pe'].rank(pct=True).fillna(0.5)

    test_df = df[df.index.year >= 2022]
    if len(test_df) < 100: return None

    engine = FactorEngine(factors, ic_window=120)
    sb = StateBuilder(window_size=60, factor_names=factors, market_dim=11)

    env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(state_dim, 1, [256,128,64])
    model.load_state_dict(model_state)
    model.eval()

    state, _ = env.reset(); done = False; rets = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        state, r, terminated, truncated, info = env.step(float(a.squeeze().numpy()))
        done = terminated or truncated; rets.append(r)

    rets = np.array(rets); cum = (1+rets).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum-peak)/peak)) if len(cum)>0 else 0
    sr = float(rets.mean()/(np.std(rets)+1e-10)*np.sqrt(252))
    tr = float(cum[-1]-1)
    alive = info['total_value'] > 50000
    bh = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)

    return {'code':code,'stock':stock_name,'return':tr,'sharpe':sr,'max_dd':dd,
            'trades':info['trade_count'],'alive':alive,'bh_ret':bh}

# ── Phase 1: Train on 4 core stocks ──
print("=" * 65)
print("  V7 CROSS-VALIDATION — Train 4, Test 12")
print("=" * 65)

saved_models = []
for code, sname in TRAIN_STOCKS.items():
    print(f"\n[{code} {sname}] Training...", flush=True)
    for strat_name, factors in STRATEGIES.items():
        bare = (strat_name == "Bare PPO")
        print(f"  {strat_name}...", end=" ", flush=True)
        r = train_one(code, sname, factors, strat_name, bare=bare)
        saved_models.append(r)
        print(f"{r['time']:.0f}s", flush=True)

# ── Phase 2: Test ALL models on ALL 12 stocks ──
print(f"\n{'='*65}")
print("  Phase 2: Testing ALL models on ALL 12 stocks")
print("=" * 65)

all_test_results = []
for sm in saved_models:
    ckpt = torch.load(sm['save_path'])
    model_state = ckpt['model_state']
    factors = ckpt['factors']
    state_dim = ckpt['state_dim']

    for code, sname in ALL_STOCKS.items():
        r = test_one(code, sname, model_state, factors, state_dim)
        if r:
            r['train_stock'] = sm['stock_name']
            r['strategy'] = sm['name']
            all_test_results.append(r)

# ── Aggregate ──
df = pd.DataFrame(all_test_results)
CSI300_BH = float(market_ctx.csi300[market_ctx.csi300.index.year >= 2022]['close'].iloc[-1] /
                  market_ctx.csi300[market_ctx.csi300.index.year >= 2022]['close'].iloc[0] - 1)

print(f"\n{'='*70}")
print("  CROSS-VALIDATION RESULTS (12 stocks, Train→Test: 2015-2020→2022-2025)")
print(f"  CSI300 Benchmark: {CSI300_BH:+.1%}")
print("=" * 70)

# In-sample vs Out-of-sample
for strat_name in ["Bare PPO", "Value-Defensive", "Quality-Offensive"]:
    st_df = df[df['strategy'] == strat_name]
    # Split by whether stock was in training set
    in_sample = st_df[st_df['code'].isin(TRAIN_STOCKS.keys())]
    out_sample = st_df[~st_df['code'].isin(TRAIN_STOCKS.keys())]

    print(f"\n  [{strat_name}]")
    print(f"    In-sample  (4 stocks):   Sharpe={in_sample['sharpe'].mean():.2f} "
          f"Ret={in_sample['return'].mean():+.1%} Alive={in_sample['alive'].sum()}/4 "
          f"DD={in_sample['max_dd'].mean():.1%}")
    print(f"    Out-sample (8 stocks):   Sharpe={out_sample['sharpe'].mean():.2f} "
          f"Ret={out_sample['return'].mean():+.1%} Alive={out_sample['alive'].sum()}/8 "
          f"DD={out_sample['max_dd'].mean():.1%}")
    print(f"    ALL 12 stocks:           Sharpe={st_df['sharpe'].mean():.2f} "
          f"Ret={st_df['return'].mean():+.1%} Alive={st_df['alive'].sum()}/12 "
          f"DD={st_df['max_dd'].mean():.1%}")
    print(f"    Profitable: {(st_df['return']>0).sum()}/12  "
          f"Beat CSI300: {(st_df['sharpe']>0).sum()}/12")

# Best strategy win-rate
print(f"\n  Per-strategy pairwise vs Bare PPO (all 12 stocks):")
bare_df = df[df['strategy'] == 'Bare PPO']
for strat_name in ["Value-Defensive", "Quality-Offensive"]:
    strat_df = df[df['strategy'] == strat_name]
    wins = 0
    for code in ALL_STOCKS:
        b = bare_df[bare_df['code'] == code]['sharpe']
        s = strat_df[strat_df['code'] == code]['sharpe']
        if len(b) > 0 and len(s) > 0 and s.values[0] > b.values[0]:
            wins += 1
    delta_sr = (strat_df['sharpe'].mean() - bare_df['sharpe'].mean())
    print(f"    {strat_name}: wins {wins}/12, delta Sharpe={delta_sr:+.2f}")

print("=" * 70)
