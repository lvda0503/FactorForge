"""V8 独立选股 + 交易明细CSV"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, os

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
SEL_DATE = pd.Timestamp("2021-01-04")
OUT_DIR  = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/trade_logs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "filter_fn": value_filter, "model": "Value-Defensive_600519_fi.pt",
    },
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "filter_fn": quality_filter, "model": "Quality-Offensive_600276_fi.pt",
    },
    "Bare PPO": {
        "factors": ["roc_20","rank_20","std_20","pb_ratio","corr_20"],
        "filter_fn": None, "model": "Bare PPO_600519_.pt",
    },
}

market_ctx = MarketContext()
ind_map = pd.read_pickle(IND_PATH).to_dict()
neutralizer = BarraNeutralizer(ind_map)
scorer = FactorScorer()

# ── Step 1: Each strategy selects its own pool ──
print(f"Selection @ {SEL_DATE.date()} (PIT-safe)")
stock_info = {}
factor_snaps = {s: {fn: {} for fn in cfg["factors"]} for s, cfg in STRATEGIES.items()}

for f in os.listdir(CACHE):
    if not f.endswith('.pkl'): continue
    code = f.replace('.pkl','')
    df = pd.read_pickle(f"{CACHE}/{f}")[lambda x: x.index <= SEL_DATE]
    if len(df) < 61: continue
    for c in ['open','high','low','close','volume','pe','pb']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
    df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
    if SEL_DATE not in df.index: continue
    i = df.index.get_loc(SEL_DATE); row = df.iloc[i]; cs = df['close']
    stock_info[code] = {
        'pb': row.get('pb',0), 'pe': row.get('pe',20),
        'close': row.get('close',0), 'amount': row.get('close',0)*row.get('volume',0),
        'roc_250': float(cs.iloc[i]/cs.iloc[max(0,i-250)]-1) if i>=250 else 0,
        'ma_200': float(cs.iloc[max(0,i-200):i+1].mean()) if i>=200 else 999999,
    }
    ohlcv = df.iloc[i-60:i+1][['open','high','low','close','volume']].values.astype(np.float64)
    for sn, cfg in STRATEGIES.items():
        eng = FactorEngine(cfg["factors"])
        fv = eng.compute_factors(ohlcv, ohlcv[:,4], pb_value=row.get('pb'),
                                 pe_percentile=row.get('pe_percentile',0.5))
        for k, v in fv.items():
            if k in factor_snaps[sn]:
                factor_snaps[sn][k][code] = float(np.nan_to_num(np.asarray(v), nan=0.0))

selections = {}
for sn, cfg in STRATEGIES.items():
    vc = cfg["filter_fn"](stock_info) if cfg["filter_fn"] else set(stock_info.keys())
    fv = factor_snaps[sn]
    fv_f = {fn:{c:v for c,v in vals.items() if c in vc} for fn,vals in fv.items()}
    mc = {c:stock_info[c]['amount'] for c in vc}
    fv_n = {fn: neutralizer.neutralize(vals, mc) for fn,vals in fv_f.items()}
    scores = scorer.score(fv_n)
    selections[sn] = [c for c,_ in scorer.select_top_k(scores, 6)]
    names = [ind_map.get(c,'?')[:6] for c in selections[sn]]
    print(f"  [{sn}] {len(vc)}→6: {list(zip(selections[sn], names))}")

# ── Step 2: Backtest each strategy on its own pool ──
all_summary = []
for sn, cfg in STRATEGIES.items():
    ckpt = torch.load(f"{MODEL_DIR}/{cfg['model']}")
    trade_logs = []

    for code in selections[sn]:
        for loc in [f"d:/JoinQuant/quant_env/data_cache/baostock_{code}.pkl", f"{CACHE}/{code}.pkl"]:
            if os.path.exists(loc): break
        else: continue
        df = pd.read_pickle(loc)
        for c in ['open','high','low','close','volume','pe','pb']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        if 'pe_percentile' in cfg["factors"]:
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        test_df = df[(df.index >= '2021-01-01') & (df.index <= '2026-06-30')]
        if len(test_df) < 91: continue

        engine = FactorEngine(cfg["factors"])
        sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
        env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
        model.load_state_dict(ckpt['model_state']); model.eval()

        try:
            state, _ = env.reset(); done = False; rets = []
            prev_pos = 0.0
            while not done:
                s = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, _, _ = model.get_action(s, deterministic=True)
                action = float(action_tensor.squeeze().numpy())
                state, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated; rets.append(r)
                # Log trade if position changed
                curr_pos = info.get('long_position',0) - info.get('short_position',0)
                if abs(curr_pos - prev_pos) > 0.005:
                    trade_logs.append({
                        'strategy': sn, 'code': code,
                        'date': test_df.index[env.idx-1] if env.idx < len(test_df) else test_df.index[-1],
                        'action': round(action, 4),
                        'net_pos': round(curr_pos, 4),
                        'price': round(info.get('current_price',0), 2),
                        'total_value': round(info['total_value'], 2),
                        'pnl_pct': round(r*100, 4),
                    })
                prev_pos = curr_pos
        except Exception as e:
            print(f"    [{sn}] {code}: ERROR {str(e)[:60]}")
            continue

        rets_arr = np.array(rets); cum = (1+rets_arr).cumprod()
        dd = float(np.min(cum/np.maximum.accumulate(cum)-1)) if len(cum)>0 else 0
        sr = float(rets_arr.mean()/(np.std(rets_arr)+1e-10)*np.sqrt(252))
        tr = float(cum[-1]-1)
        bh = float(test_df['close'].iloc[-1]/test_df['close'].iloc[0]-1)
        all_summary.append({'strategy':sn,'code':code,'return':tr,'sharpe':sr,
                          'max_dd':dd,'trades':info['trade_count'],'alive':info['total_value']>50000,'bh_ret':bh})

    # Save CSV for this strategy
    df_log = pd.DataFrame(trade_logs)
    csv_path = f"{OUT_DIR}/{sn}_trades.csv"
    df_log.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  [{sn}] {len(trade_logs)} trades saved → {csv_path}")

# ── Summary ──
df_s = pd.DataFrame(all_summary)
csi = market_ctx.csi300
csi_t = csi[(csi.index>='2021-01-01')&(csi.index<='2026-06-30')]
csi_bh = float(csi_t['close'].iloc[-1]/csi_t['close'].iloc[0]-1)
csi_sr = float(csi_t['close'].pct_change().dropna().mean()/
              (csi_t['close'].pct_change().dropna().std()+1e-10)*np.sqrt(252))

print(f"\n{'='*65}")
print(f"  CSI300 (2021-2026): {csi_bh:+.1%} Sharpe={csi_sr:.2f}")
print(f"{'='*65}")
for sn in ["Value-Defensive","Quality-Offensive","Bare PPO"]:
    sub = df_s[df_s['strategy']==sn]
    if len(sub)==0: continue
    print(f"\n  [{sn}] ({len(sub)} stocks — own selection pool)")
    for _, r in sub.iterrows():
        print(f"    {r['code']}: Ret={r['return']:+.1%} Sharpe={r['sharpe']:.2f} "
              f"DD={r['max_dd']:.1%} T={r['trades']} {'OK' if r['alive'] else 'DEAD'}")
    print(f"    AVG: Ret={sub['return'].mean():+.1%} Sharpe={sub['sharpe'].mean():.2f} "
          f"DD={sub['max_dd'].mean():.1%}")
print(f"\n  CSV files: {OUT_DIR}/")
print("="*65)
