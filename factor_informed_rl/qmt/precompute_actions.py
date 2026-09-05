"""
Pre-compute FI-PPO actions for all stocks x all dates.
Output: D:\data\actions.json  →  QMT backtest looks up this table.

Usage:
  cd D:\JoinQuant\quant_env
  python factor_informed_rl\qmt\precompute_actions.py
"""
import json, os, sys, numpy as np, pandas as pd, torch

sys.path.insert(0, r'D:\JoinQuant\quant_env')

from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.preprocessing.factor_engine import FactorEngine

MODEL_DIR = r"D:\JoinQuant\quant_env\factor_informed_rl\experiments\paper\v7_models"
CACHE_DIR = r"D:\JoinQuant\quant_env\data_cache\csi300"
OUTPUT    = r"D:\data\actions.json"

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "model": "Value-Defensive_600519_fi.pt",
    },
}

STOCKS = ["600036", "000858", "600276", "000333"]


def main():
    # Load model
    cfg = STRATEGIES["Value-Defensive"]
    ckpt = torch.load(f"{MODEL_DIR}/{cfg['model']}", map_location='cpu')
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256, 128, 64])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[Precompute] Model loaded, state_dim={sb.state_dim}")

    # Load stock cache
    stock_data = {}
    for code in STOCKS:
        path = f"{CACHE_DIR}/{code}.pkl"
        if os.path.exists(path):
            df = pd.read_pickle(path)
            for c in ['open','high','low','close','volume','pe','pb']:
                if c in df.columns:
                    df[c] = df[c].ffill().bfill().fillna(0)
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
            stock_data[code] = df
            print(f"  {code}: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")
        else:
            print(f"  {code}: NO DATA")

    # Pre-compute actions
    actions = {}
    mkt_zeros = np.zeros(11, dtype=np.float32)

    for code, df in stock_data.items():
        print(f"[Precompute] Processing {code}...")
        actions[code] = {}
        engine = FactorEngine(cfg["factors"])

        for i in range(60, len(df)):
            date_str = str(df.index[i].date())
            window = df.iloc[i-60:i+1]
            ohlcv = window[['open','high','low','close','volume']].values.astype(np.float64)
            close_price = float(ohlcv[-1][3])

            # Factor computation
            pb_val = window['pb'].iloc[-1] if 'pb' in window.columns else None
            pe_pct = window['pe_percentile'].iloc[-1] if 'pe_percentile' in window.columns else 0.5
            factors = engine.compute_factors(ohlcv, ohlcv[:, 4],
                                             pb_value=pb_val, pe_percentile=pe_pct)

            # State building
            state = sb.build(
                price_window=ohlcv, close_denoised=ohlcv[:, 3],
                factors=factors, position=0.0, cash_ratio=0.5,
                unrealized_pnl=0.0, market_features=mkt_zeros)

            # Inference
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(s, deterministic=True)

            close_price = float(ohlcv[-1][3])
            actions[code][date_str] = {
                "a": round(float(action.squeeze().numpy()), 6),
                "c": round(close_price, 2)
            }

        print(f"  {len(actions[code])} dates computed")

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(actions, f)
    print(f"\n[Precompute] Done! Saved to {OUTPUT}")
    print(f"  Total entries: {sum(len(v) for v in actions.values())}")


if __name__ == '__main__':
    main()
