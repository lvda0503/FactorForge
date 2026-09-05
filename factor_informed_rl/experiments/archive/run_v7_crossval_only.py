"""V7 Cross-Validation — Pure inference (no training)"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, os, glob

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic

ALL_STOCKS = {
    "600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui",
    "600887":"Yili","002415":"Hikvision","600030":"CITIC Sec","000651":"Gree",
    "601166":"Ind Bank","600585":"Conch","000002":"Vanke","601888":"CDFG",
}
MODEL_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
cache = "d:/JoinQuant/quant_env/data_cache"
market_ctx = MarketContext()

def test_one(code, stock_name, model_path):
    ckpt = torch.load(model_path)
    factors = ckpt['factors']
    state_dim = ckpt['state_dim']

    path = f"{cache}/baostock_{code}.pkl"
    if not os.path.exists(path): return None
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    if 'pe_percentile' in factors:
        df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)

    test_df = df[df.index.year >= 2022]
    if len(test_df) < 100: return None

    engine = FactorEngine(factors, ic_window=120)
    sb = StateBuilder(window_size=60, factor_names=factors, market_dim=11)
    env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state'])
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
    train_stock = os.path.basename(model_path).split('_')[1]

    return {'code':code,'stock':stock_name,'return':tr,'sharpe':sr,'max_dd':dd,
            'trades':info['trade_count'],'alive':alive,'bh_ret':bh,
            'train_stock':train_stock,'model_file':os.path.basename(model_path)}

model_files = sorted(glob.glob(f"{MODEL_DIR}/*.pt"))
print(f"Found {len(model_files)} saved models")
print("Testing each model on all 12 stocks...\n")

all_results = []
for mf in model_files:
    name = os.path.basename(mf).replace('.pt','')
    for code, sname in ALL_STOCKS.items():
        r = test_one(code, sname, mf)
        if r:
            r['model_name'] = name
            all_results.append(r)
    print(f"  [{name}] 12 stocks tested", flush=True)

# Aggregate
df = pd.DataFrame(all_results)
TRAIN_CODES = {"600519","000858","000333","600276"}

CSI300_BH = float(market_ctx.csi300[market_ctx.csi300.index.year >= 2022]['close'].iloc[-1] /
                  market_ctx.csi300[market_ctx.csi300.index.year >= 2022]['close'].iloc[0] - 1)

# Group by strategy type
strategies = {
    "Value-Defensive": [m for m in model_files if "Value-Defensive" in m],
    "Quality-Offensive": [m for m in model_files if "Quality-Offensive" in m],
    "Bare PPO": [m for m in model_files if "Bare PPO" in m],
}

print(f"\n{'='*70}")
print("  CROSS-VALIDATION: Train 4 → Test 12 (Pure OOS)")
print(f"  CSI300 Benchmark: {CSI300_BH:+.1%}")
print("=" * 70)

for strat_name, mfiles in strategies.items():
    st_df = df[df['model_file'].isin([os.path.basename(m) for m in mfiles])]
    in_sample  = st_df[st_df['code'].isin(TRAIN_CODES)]
    out_sample = st_df[~st_df['code'].isin(TRAIN_CODES)]

    print(f"\n  [{strat_name}] ({len(mfiles)} models × 12 stocks = {len(st_df)} tests)")
    print(f"    In-sample  (4 stocks):   Sharpe={in_sample['sharpe'].mean():.2f} "
          f"Ret={in_sample['return'].mean():+.1%} Alive={in_sample['alive'].sum()}/4 "
          f"DD={in_sample['max_dd'].mean():.1%}")
    print(f"    Out-sample (8 stocks):   Sharpe={out_sample['sharpe'].mean():.2f} "
          f"Ret={out_sample['return'].mean():+.1%} Alive={out_sample['alive'].sum()}/8 "
          f"DD={out_sample['max_dd'].mean():.1%}")
    print(f"    ALL 12 stocks:           Sharpe={st_df['sharpe'].mean():.2f} "
          f"Ret={st_df['return'].mean():+.1%} Alive={st_df['alive'].sum()}/12 "
          f"DD={st_df['max_dd'].mean():.1%}")
    print(f"    Profitable: {(st_df['return']>0).sum()}/{len(st_df)}  "
          f"Beat CSI300: {(st_df['sharpe']>0).sum()}/{len(st_df)}")

# Head-to-head on ALL 12 stocks
print(f"\n  Head-to-Head vs Bare PPO (averaged across 4 training models):")
for strat_name in ["Value-Defensive", "Quality-Offensive"]:
    s_files = strategies[strat_name]
    b_files = strategies["Bare PPO"]
    wins = 0; deltas = []
    for sf, bf in zip(sorted(s_files), sorted(b_files)):
        sdf = df[df['model_file'] == os.path.basename(sf)]
        bdf = df[df['model_file'] == os.path.basename(bf)]
        common = set(sdf['code']) & set(bdf['code'])
        for code in common:
            ss = sdf[sdf['code']==code]['sharpe'].values[0]
            bs = bdf[bdf['code']==code]['sharpe'].values[0]
            if ss > bs: wins += 1
            deltas.append(ss - bs)
    print(f"    {strat_name}: wins {wins}/{len(deltas)}, avg delta Sharpe={np.mean(deltas):+.2f}")

print("=" * 70)
