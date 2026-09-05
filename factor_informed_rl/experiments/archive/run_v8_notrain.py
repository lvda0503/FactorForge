"""
V8: 选股@2025-12-31 → 加载V7模型 → 回测2026H1 (不重新训练!)
"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, time, os

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter, quality_filter

MODEL_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
CACHE = "d:/JoinQuant/quant_env/data_cache/csi300"
IND_PATH = "d:/JoinQuant/quant_env/data_cache/csi300_industry.pkl"
SEL_DATE = pd.Timestamp("2025-12-31")
TEST_START, TEST_END = "2026-01-01", "2026-06-30"
WINDOW = 60

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "filter_fn": value_filter,
        "model_file": "Value-Defensive_600519_fi.pt",
    },
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "filter_fn": quality_filter,
        "model_file": "Quality-Offensive_600276_fi.pt",
    },
    "Bare PPO": {
        "factors": ["roc_20","rank_20","std_20","pb_ratio","corr_20"],
        "filter_fn": None,
        "model_file": "Bare PPO_600519_.pt",
    },
}

market_ctx = MarketContext()
ind_map = pd.read_pickle(IND_PATH).to_dict()
neutralizer = BarraNeutralizer(ind_map)
scorer = FactorScorer()

# ── Step 1: Selection @ 2025-12-31 ──
print("="*60)
print("  V8 NOTRAIN: Select@2025-12-31 → V7 models → 2026H1")
print("="*60)

stock_info = {}
factor_snaps = {s: {fn: {} for fn in cfg["factors"]} for s, cfg in STRATEGIES.items()}

for f in os.listdir(CACHE):
    if not f.endswith('.pkl'): continue
    code = f.replace('.pkl','')
    df = pd.read_pickle(f"{CACHE}/{f}")
    for c in ['open','high','low','close','volume','pe','pb']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
    if SEL_DATE not in df.index: continue
    i = df.index.get_loc(SEL_DATE)
    if i < WINDOW: continue
    row = df.iloc[i]; cs = df['close']
    stock_info[code] = {
        'pb': row.get('pb',0), 'pe': row.get('pe',20),
        'close': row.get('close',0),
        'amount': row.get('close',0) * row.get('volume',0),
        'roc_250': float(cs.iloc[i]/cs.iloc[max(0,i-250)]-1) if i>=250 else 0,
        'ma_200': float(cs.iloc[max(0,i-200):i+1].mean()) if i>=200 else 999999,
    }
    ohlcv = df.iloc[i-WINDOW:i+1][['open','high','low','close','volume']].values.astype(np.float64)
    for sn, cfg in STRATEGIES.items():
        eng = FactorEngine(cfg["factors"])
        fv = eng.compute_factors(ohlcv, ohlcv[:,4], pb_value=row.get('pb'),
                                 pe_percentile=row.get('pe_percentile',0.5))
        for k, v in fv.items():
            if k in factor_snaps[sn]:  # 只存这个策略需要的因子
                factor_snaps[sn][k][code] = float(np.nan_to_num(np.asarray(v), nan=0.0))

print(f"  {len(stock_info)} stocks @{SEL_DATE.date()}")

selections = {}
for sn, cfg in STRATEGIES.items():
    vc = cfg["filter_fn"](stock_info) if cfg["filter_fn"] else set(stock_info.keys())
    fv = factor_snaps[sn]
    fv_f = {fn:{c:v for c,v in vals.items() if c in vc} for fn,vals in fv.items()}
    fv_n = {fn: neutralizer.neutralize(vals, {c:stock_info[c]['amount'] for c in vc})
            for fn,vals in fv_f.items()}
    scores = scorer.score(fv_n)
    top6 = scorer.select_top_k(scores, 6)
    selections[sn] = [c for c,_ in top6]
    print(f"  [{sn}]: {len(vc)}→6 selected")

# ── Step 2: Load V7 models + Test on selected stocks ──
print(f"\n[Testing {TEST_START}→{TEST_END}]")

all_results = []
for sn, cfg in STRATEGIES.items():
    ckpt = torch.load(f"{MODEL_DIR}/{cfg['model_file']}")
    state_dim = ckpt['state_dim']

    # 统一使用 Value-Defensive 选股池 (公平对比)
    codes_to_test = selections["Value-Defensive"]
    for code in codes_to_test:
        # Load stock data (from original cache or CSI300)
        for d in [f"d:/JoinQuant/quant_env/data_cache/baostock_{code}.pkl", f"{CACHE}/{code}.pkl"]:
            if os.path.exists(d): break
        else: continue
        df = pd.read_pickle(d)
        for c in ['open','high','low','close','volume','pe','pb']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        if 'pe_percentile' in cfg["factors"]:
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
        # 环境需要至少 window_size + 30 行才能初始化和产生足够交易信号
        if len(test_df) < WINDOW + 30: continue
        # 确保 OHLCV 列完整
        if test_df[['close','high','low','open','volume']].isna().any().any(): continue

        engine = FactorEngine(cfg["factors"])
        sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
        env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        try:
            state, _ = env.reset(); done = False; rets = []
            while not done:
                s = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    a, _, _ = model.get_action(s, deterministic=True)
                state, r, terminated, truncated, info = env.step(float(a.squeeze().numpy()))
                done = terminated or truncated; rets.append(r)
        except Exception as e:
            print(f"    [{sn}] {code}: ERROR {str(e)[:60]}")
            continue

        rets = np.array(rets); cum = (1+rets).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = float(np.min((cum-peak)/peak)) if len(cum)>0 else 0
        sr = float(rets.mean()/(np.std(rets)+1e-10)*np.sqrt(252))
        tr = float(cum[-1]-1)
        bh = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)
        alive = info['total_value'] > 50000

        all_results.append({'strategy':sn,'code':code,'return':tr,'sharpe':sr,
                           'max_dd':dd,'trades':info['trade_count'],'alive':alive,'bh_ret':bh})

# ── Results ──
csi300 = market_ctx.csi300
csi_t = csi300[(csi300.index >= TEST_START) & (csi300.index <= TEST_END)]
if len(csi_t) < 2:
    csi_bh = 0.0; csi_sr = 0.0
else:
    csi_bh = float(csi_t['close'].iloc[-1]/csi_t['close'].iloc[0]-1)
    csi_sr = float(csi_t['close'].pct_change().dropna().mean()/
                  (csi_t['close'].pct_change().dropna().std()+1e-10)*np.sqrt(252))
print(f"\n{'='*60}")
print(f"  CSI300: {csi_bh:+.1%} Sharpe={csi_sr:.2f}")
print("="*60)

df_r = pd.DataFrame(all_results)
for sn in ["Value-Defensive","Quality-Offensive","Bare PPO"]:
    sub = df_r[df_r['strategy']==sn]
    if len(sub)==0: continue
    print(f"\n  [{sn}] ({len(sub)} stocks, V7 model, no retraining)")
    for _, r in sub.iterrows():
        print(f"    {r['code']}: Ret={r['return']:+.1%} Sharpe={r['sharpe']:.2f} "
              f"DD={r['max_dd']:.1%} T={r['trades']} {'OK' if r['alive'] else 'DEAD'}")
    print(f"    AVG: Ret={sub['return'].mean():+.1%} Sharpe={sub['sharpe'].mean():.2f} "
          f"DD={sub['max_dd'].mean():.1%} BeatCSI300={(sub['sharpe']>csi_sr).sum()}/{len(sub)}")
print("="*60)
