"""
文件通信桥 — FI-PPO推理服务 (文件模式)
====================================
替代 HTTP server, 通过文件系统与QMT通信。

QMT侧:  写入 request  →  D:/data/requests/{stock}.json
本机侧: 轮询 requests  →  推理  →  写入 D:/data/responses/{stock}.json
QMT侧:  读取 response → 执行下单

日频交易, 文件IO延迟(<50ms)完全可以忽略。
"""
import json, os, time, torch, numpy as np, pandas as pd, sys, pickle
sys.path.insert(0, r'D:\JoinQuant\quant_env')

from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter

# ── 配置 ──
MODEL_DIR = r"D:\JoinQuant\quant_env\factor_informed_rl\experiments\paper\v7_models"
REQ_DIR   = r"D:\data\qmt_requests"
RESP_DIR  = r"D:\data\qmt_responses"
SS_CACHE  = r"D:\JoinQuant\quant_env\data_cache\csi300"
IND_PATH  = r"D:\JoinQuant\quant_env\data_cache\csi300_industry.pkl"
os.makedirs(REQ_DIR, exist_ok=True)
os.makedirs(RESP_DIR, exist_ok=True)

STRATEGIES = {
    "Value-Defensive": {
        "path": f"{MODEL_DIR}/Value-Defensive_600519_fi.pt",
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
    },
    "Quality-Offensive": {
        "path": f"{MODEL_DIR}/Quality-Offensive_600276_fi.pt",
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
    },
}

# ── 加载模型 + 数据 ──
print("[Bridge] Loading models...")
models = {}
for name, cfg in STRATEGIES.items():
    ckpt = torch.load(cfg["path"], map_location='cpu')
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state']); model.eval()
    models[name] = {'model':model, 'state_builder':sb, 'factors':cfg["factors"]}
    print(f"  {name} (state_dim={sb.state_dim})")

print("[Bridge] Loading market context...")
market_ctx = MarketContext()
ind_map = pd.read_pickle(IND_PATH).to_dict() if os.path.exists(IND_PATH) else {}
neutralizer = BarraNeutralizer(ind_map); scorer = FactorScorer()

print("[Bridge] Loading stock cache...")
stock_cache = {}
if os.path.exists(SS_CACHE):
    for f in os.listdir(SS_CACHE):
        if f.endswith('.pkl'):
            code = f.replace('.pkl','')
            df = pd.read_pickle(os.path.join(SS_CACHE, f))
            for c in ['open','high','low','close','volume','pe','pb']:
                if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
            stock_cache[code] = df
    print(f"  {len(stock_cache)} stocks cached")

# ── 处理函数 ──

def _handle_predict(req):
    stock = req['stock']
    strategy = req['strategy']
    ohlcv = np.array(req['ohlcv'], dtype=np.float64)
    position = float(req.get('position', 0))
    cash_ratio = float(req.get('cash_ratio', 0.5))
    unrealized_pnl = float(req.get('unrealized_pnl', 0))

    if strategy not in models:
        return {'action': 0.0}
    m = models[strategy]

    # 因子计算
    engine = FactorEngine(m['factors'])
    factors = engine.compute_factors(ohlcv, ohlcv[:,4],
                                     pb_value=None, pe_percentile=0.5)

    # 市场环境
    mkt_feat = _get_market_features()

    # 构建状态 → 推理
    sb = m['state_builder']
    state = sb.build(price_window=ohlcv, close_denoised=ohlcv[:,3],
                     factors=factors, position=position,
                     cash_ratio=cash_ratio, unrealized_pnl=unrealized_pnl,
                     market_features=mkt_feat)
    s = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = m['model'].get_action(s, deterministic=True)
    return {'action': round(float(action.squeeze().numpy()), 6)}


def _handle_select(req):
    strategy = req.get('strategy', 'Value-Defensive')
    top_k = req.get('top_k', 8)
    today = pd.Timestamp.now().normalize()

    if not stock_cache:
        return {'stocks': []}

    factors = STRATEGIES.get(strategy, STRATEGIES['Value-Defensive'])['factors']
    sd = {}; fv_dict = {fn: {} for fn in factors}

    for code, df in stock_cache.items():
        df_t = df[df.index <= today]
        if len(df_t) < 61 or today not in df_t.index: continue
        i = df_t.index.get_loc(today)
        row = df_t.iloc[i]; cs = df_t['close']
        ohlcv = df_t.iloc[i-60:i+1][['open','high','low','close','volume']].values.astype(np.float64)
        eng = FactorEngine(factors)
        fv = eng.compute_factors(ohlcv, ohlcv[:,4],
                                 pb_value=row.get('pb'),
                                 pe_percentile=row.get('pe_percentile',0.5))
        for fn, vals in fv.items():
            fv_dict[fn][code] = float(np.nan_to_num(np.asarray(vals), nan=0.0))
        sd[code] = {'pb':row.get('pb',0),'pe':row.get('pe',20),
                    'close':row.get('close',0),
                    'amount':row.get('close',0)*row.get('volume',0)}

    vc = value_filter(sd)
    fv_f = {fn:{c:v for c,v in vals.items() if c in vc} for fn,vals in fv_dict.items()}
    mc = {c:sd[c]['amount'] for c in vc}
    fv_n = {fn:neutralizer.neutralize(vals,mc) for fn,vals in fv_f.items()}
    scores = scorer.score(fv_n)
    top = scorer.select_top_k(scores, k=top_k)
    stocks = [f"{c}.SH" if c.startswith('6') else f"{c}.SZ" for c,_ in top]
    return {'stocks': stocks, 'date': str(today.date())}


def _handle_market():
    csi = market_ctx.csi300
    if csi is not None and len(csi) > 1:
        last = csi['close'].iloc[-1]
        ma200 = float(csi['close'].rolling(200).mean().iloc[-1])
        return {'csi300': float(last), 'ma200': float(ma200),
                'is_bear': last < ma200}
    return {'error': 'CSI300 not available'}


def _get_market_features():
    try:
        csi = market_ctx.csi300
        if csi is not None and len(csi) > 20:
            csi_close = csi['close'].iloc[-1]
            csi_ma200 = float(csi['close'].rolling(200).mean().iloc[-1])
            is_bear = 1 if csi_close < csi_ma200 else 0
            ret_5d = float(csi['close'].pct_change(5).iloc[-1])
            ret_20d = float(csi['close'].pct_change(20).iloc[-1])
            vol_20d = float(csi['close'].pct_change().rolling(20).std().iloc[-1])
            feat = np.array([ret_5d, ret_20d, vol_20d, 0,0, is_bear, 0,0, 0,0,0], dtype=np.float32)
            return np.nan_to_num(feat, 0.0)
    except: pass
    return np.zeros(11, dtype=np.float32)


# ── 主循环: 轮询请求文件 ──
if __name__ == '__main__':
    print(f"[Bridge] Ready. Watching {REQ_DIR}")
    print(f"[Bridge] Press Ctrl+C to stop\n")
    while True:
        try:
            files = sorted(os.listdir(REQ_DIR))
        except:
            time.sleep(1)
            continue

        for fname in files:
            fpath = os.path.join(REQ_DIR, fname)
            if not fname.endswith('.json'):
                continue

            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    req = json.load(f)
                os.remove(fpath)
            except (json.JSONDecodeError, PermissionError):
                time.sleep(0.1)
                continue

            action_type = req.get('type', 'predict')
            resp = {}

            try:
                if action_type == 'predict':
                    resp = _handle_predict(req)
                elif action_type == 'select':
                    resp = _handle_select(req)
                elif action_type == 'market':
                    resp = _handle_market()
            except Exception as e:
                resp = {'action': 0.0, 'error': str(e)}

            resp_file = os.path.join(RESP_DIR, fname)
            try:
                with open(resp_file, 'w', encoding='utf-8') as f:
                    json.dump(resp, f)
            except:
                alt_name = fname.replace('.json', f'_r_{int(time.time()*1000)%10000}.json')
                with open(os.path.join(RESP_DIR, alt_name), 'w', encoding='utf-8') as f:
                    json.dump(resp, f)

        time.sleep(0.2)
