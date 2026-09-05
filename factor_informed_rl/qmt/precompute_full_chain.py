"""
Full-chain precompute: monthly stock selection + daily FI-PPO actions.
Output: D:\data\full_chain.json → QMT replays selection + trading.

Usage:
  cd D:\JoinQuant\quant_env
  python factor_informed_rl\qmt\precompute_full_chain.py
"""
import json, os, sys, numpy as np, pandas as pd, torch, pickle
sys.path.insert(0, r'D:\JoinQuant\quant_env')

from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter

MODEL_DIR = r"D:\JoinQuant\quant_env\factor_informed_rl\experiments\paper\v7_models"
CACHE_DIR = r"D:\JoinQuant\quant_env\data_cache\csi300"
IND_PATH  = r"D:\JoinQuant\quant_env\data_cache\csi300_industry.pkl"
OUTPUT    = r"D:\data\full_chain.json"

MODEL_FILE = "Value-Defensive_600519_fi.pt"
FACTORS = ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"]
TOP_K = 5         # stocks per month
REBALANCE_DAY = 1 # rebalance on Nth trading day of month

STOCK_CACHE = {}  # {code: DataFrame}


def load_stock_cache():
    """Load all CSI300 stock DataFrames."""
    if not os.path.exists(CACHE_DIR):
        return
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.pkl'):
            code = f.replace('.pkl','')
            df = pd.read_pickle(os.path.join(CACHE_DIR, f))
            for c in ['open','high','low','close','volume','pe','pb']:
                if c in df.columns:
                    df[c] = df[c].ffill().bfill().fillna(0)
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
            STOCK_CACHE[code] = df
    print(f"[Precompute] Loaded {len(STOCK_CACHE)} stocks from cache")


def get_trading_days(df, start, end):
    """Get list of trading days in date range."""
    idx = df.index[(df.index >= start) & (df.index <= end)]
    return sorted(set(str(d.date()) for d in idx))


def month_boundaries(trading_days):
    """Group trading days by month, return [(month_key, [days])]."""
    months = {}
    for d in trading_days:
        m = d[:7]  # "2021-01"
        months.setdefault(m, []).append(d)
    return [(m, days) for m, days in sorted(months.items())]


def main():
    print("[Precompute] Loading model...")
    ckpt = torch.load(f"{MODEL_DIR}/{MODEL_FILE}", map_location='cpu')
    sb = StateBuilder(window_size=60, factor_names=FACTORS, market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state']); model.eval()

    print("[Precompute] Loading stock cache...")
    load_stock_cache()

    # Init selection components (same as run_today.py)
    from factor_informed_rl.stock_selection.hard_filter import value_filter
    ind_map = pd.read_pickle(IND_PATH).to_dict() if os.path.exists(IND_PATH) else {}
    neutralizer = BarraNeutralizer(ind_map)
    scorer = FactorScorer()

    # Get all trading days from a reference stock
    ref_df = STOCK_CACHE.get('600036')
    if ref_df is None:
        print("[Precompute] FATAL: no reference stock data")
        return
    all_days = get_trading_days(ref_df, '2021-01-01', '2026-06-30')
    months_list = month_boundaries(all_days)
    print(f"[Precompute] {len(months_list)} months, {len(all_days)} trading days")

    # ── Phase 1: Monthly stock selection ──
    print("[Precompute] Phase 1: Monthly stock selection...")
    monthly_selection = {}

    for month_key, days in months_list:
        sel_date = days[min(REBALANCE_DAY-1, len(days)-1)]
        print(f"  {month_key} ({sel_date}): computing factors for all stocks...")

        # Compute factor values + stock data for ALL stocks at sel_date
        fv_dict = {}  # {factor_name: {code: value}}
        stock_info = {}
        mkt_caps = {}
        count = 0

        for code, df in STOCK_CACHE.items():
            if sel_date not in df.index:
                continue
            i = df.index.get_loc(sel_date)
            if i < 60:
                continue
            window = df.iloc[i-60:i+1]
            ohlcv = window[['open','high','low','close','volume']].values.astype(np.float64)
            row = df.iloc[i]
            engine = FactorEngine(FACTORS)
            fv = engine.compute_factors(ohlcv, ohlcv[:,4],
                                        pb_value=row.get('pb'),
                                        pe_percentile=row.get('pe_percentile',0.5))
            for fn, vals in fv.items():
                if fn not in fv_dict:
                    fv_dict[fn] = {}
                fv_dict[fn][code] = float(np.nan_to_num(np.asarray(vals), nan=0.0))

            close_s = df['close']
            stock_info[code] = {
                'pb': row.get('pb',0), 'pe': row.get('pe',20),
                'close': row.get('close',0),
                'amount': row.get('close',0)*row.get('volume',0),
                'roc_250': float(close_s.iloc[i]/close_s.iloc[max(0,i-250)]-1) if i>=250 else 0,
                'ma_200': float(close_s.iloc[max(0,i-200):i+1].mean()) if i>=200 else 1e9,
            }
            mkt_caps[code] = row.get('close', 100) * row.get('volume', 1e6)
            count += 1

        # Hard filter → Barra → Score → Top-K
        valid = value_filter(stock_info)
        fv_filt = {fn: {c: v for c, v in vals.items() if c in valid}
                   for fn, vals in fv_dict.items()}
        fv_neut = {fn: neutralizer.neutralize(vals, mkt_caps)
                   for fn, vals in fv_filt.items()}
        scores = scorer.score(fv_neut)
        top = scorer.select_top_k(scores, k=TOP_K)

        selected = [code for code, score in top]
        monthly_selection[month_key] = {
            "rebalance_date": sel_date,
            "stocks": selected,
        }
        print(f"    {count} stocks computed, {len(valid)} passed filter, selected: {selected}")

    # ── Phase 2: FI-PPO actions for selected stocks ──
    print("[Precompute] Phase 2: FI-PPO actions per stock...")
    mkt_zeros = np.zeros(11, dtype=np.float32)
    all_actions = {}  # {stock_code: {date: {"a": ..., "c": ...}}}

    for month_key, info in monthly_selection.items():
        for code in info["stocks"]:
            if code in all_actions:
                continue  # Already computed

            df = STOCK_CACHE.get(code)
            if df is None:
                print(f"  {code}: no data, skip")
                continue

            print(f"  Computing {code}...")
            actions = {}
            engine = FactorEngine(FACTORS)

            for i in range(60, len(df)):
                date_str = str(df.index[i].date())
                if date_str < '2021-01-01':
                    continue
                window = df.iloc[i-60:i+1]
                ohlcv = window[['open','high','low','close','volume']].values.astype(np.float64)
                close_price = float(ohlcv[-1][3])

                pb_val = window['pb'].iloc[-1] if 'pb' in window.columns else None
                pe_pct = window['pe_percentile'].iloc[-1] if 'pe_percentile' in window.columns else 0.5
                factors = engine.compute_factors(ohlcv, ohlcv[:,4],
                                                 pb_value=pb_val, pe_percentile=pe_pct)
                state = sb.build(price_window=ohlcv, close_denoised=ohlcv[:,3],
                                 factors=factors, position=0.0, cash_ratio=0.5,
                                 unrealized_pnl=0.0, market_features=mkt_zeros)
                s = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = model.get_action(s, deterministic=True)

                actions[date_str] = {
                    "a": round(float(action.squeeze().numpy()), 6),
                    "c": round(close_price, 2)
                }
            all_actions[code] = actions
            print(f"    {len(actions)} dates")

    # ── Assemble output ──
    output = {
        "config": {
            "strategy": "Value-Defensive",
            "factors": FACTORS,
            "model": MODEL_FILE,
            "top_k": TOP_K,
            "rebalance_day": REBALANCE_DAY,
        },
        "monthly": monthly_selection,
        "actions": all_actions,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(output, f)

    total_actions = sum(len(v) for v in all_actions.values())
    unique_stocks = len(all_actions)
    print(f"\n[Precompute] Done!")
    print(f"  Months: {len(monthly_selection)}")
    print(f"  Unique stocks: {unique_stocks}")
    print(f"  Total actions: {total_actions}")
    print(f"  Output: {OUTPUT}")


if __name__ == '__main__':
    main()
