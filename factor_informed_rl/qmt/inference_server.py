"""
FI-PPO 本地推理服务
===================
接收 QMT 发来的 OHLCV 数据 → 构建状态 → FI-PPO 推理 → 返回 action

启动:
  cd D:\JoinQuant\quant_env
  python -m factor_informed_rl.qmt.inference_server

默认端口: 8899
"""
import sys
sys.path.insert(0, r'D:\JoinQuant\quant_env')

import json, torch, numpy as np, os, pickle, pandas as pd
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter

# ═══ 加载模型 ═══
MODEL_DIR = r"D:\JoinQuant\quant_env\factor_informed_rl\experiments\paper\v7_models"

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

# 沪深300行情
market_ctx = MarketContext()
# 行业映射
IND_PATH = "D:/JoinQuant/quant_env/data_cache/csi300_industry.pkl"
ind_map = pd.read_pickle(IND_PATH).to_dict() if os.path.exists(IND_PATH) else {}
neutralizer = BarraNeutralizer(ind_map); scorer = FactorScorer()
# 预加载所有股票日线 (用于选股)
CSI300_CACHE = "D:/JoinQuant/quant_env/data_cache/csi300"
stock_cache = {}
if os.path.exists(CSI300_CACHE):
    for f in os.listdir(CSI300_CACHE):
        if f.endswith('.pkl'):
            code = f.replace('.pkl','')
            df = pd.read_pickle(f"{CSI300_CACHE}/{f}")
            for c in ['open','high','low','close','volume','pe','pb']:
                if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
            stock_cache[code] = df
    print(f"[Server] Stock cache: {len(stock_cache)} stocks")

models = {}
for name, cfg in STRATEGIES.items():
    ckpt = torch.load(cfg["path"], map_location='cpu')
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    models[name] = {'model': model, 'state_builder': sb, 'factors': cfg["factors"]}
    print(f"[Server] Loaded {name} (state_dim={sb.state_dim})")


class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            request = json.loads(body)

            # /predict — FI-PPO推理
            if self.path == '/predict':
                self._handle_predict(request)
            # /select — 选股池更新
            elif self.path == '/select':
                self._handle_select(request)
            # /market — 市场环境查询
            elif self.path == '/market':
                self._handle_market(request)
            else:
                self._respond({'error': f'Unknown endpoint: {self.path}'})
        except Exception as e:
            self._respond({'error': str(e)})

    def _handle_predict(self, req):
        stock = req['stock']
        strategy = req['strategy']
        ohlcv = np.array(req['ohlcv'], dtype=np.float64)
        position = float(req['position'])
        cash_ratio = float(req['cash_ratio'])
        unrealized_pnl = float(req['unrealized_pnl'])

        if strategy not in models:
            self._respond({'action': 0.0, 'error': f'Unknown: {strategy}'})
            return

        m = models[strategy]

        # 因子 + 状态
        engine = FactorEngine(m['factors'])
        factors = engine.compute_factors(ohlcv, ohlcv[:,4],
                                         pb_value=None, pe_percentile=0.5)
        sb = m['state_builder']
        close = ohlcv[:, 3]

        # 真实市场环境
        mkt_feat = self._get_market_features()

        state = sb.build(
            price_window=ohlcv, close_denoised=close,
            factors=factors, position=position,
            cash_ratio=cash_ratio, unrealized_pnl=unrealized_pnl,
            market_features=mkt_feat)

        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = m['model'].get_action(s, deterministic=True)

        self._respond({'action': round(float(action.squeeze().numpy()), 6)})

    def _handle_select(self, req):
        """Value策略日频选股 — 返回Top-K股票代码"""
        strategy = req.get('strategy', 'Value-Defensive')
        top_k = req.get('top_k', 8)
        today = pd.Timestamp(datetime.now().date())

        if not stock_cache:
            self._respond({'stocks': [], 'error': 'Stock cache not loaded'})
            return

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
                        'amount':row.get('close',0)*row.get('volume',0),
                        'roc_250':float(cs.iloc[i]/cs.iloc[max(0,i-250)]-1) if i>=250 else 0,
                        'ma_200':float(cs.iloc[max(0,i-200):i+1].mean()) if i>=200 else 1e9}

        vc = value_filter(sd)
        fv_f = {fn:{c:v for c,v in vals.items() if c in vc} for fn,vals in fv_dict.items()}
        mc = {c:sd[c]['amount'] for c in vc}
        fv_n = {fn:neutralizer.neutralize(vals,mc) for fn,vals in fv_f.items()}
        scores = scorer.score(fv_n)
        top = scorer.select_top_k(scores, k=top_k)
        stocks = [f"{c}.SH" if c.startswith('6') else f"{c}.SZ" for c,_ in top]

        self._respond({'stocks': stocks, 'date': str(today.date())})

    def _handle_market(self, req):
        """返回当前市场状态"""
        csi = market_ctx.csi300
        if csi is not None and len(csi) > 1:
            last = csi['close'].iloc[-1]
            ma200 = float(csi['close'].rolling(200).mean().iloc[-1])
            is_bear = last < ma200
            self._respond({'csi300': float(last), 'ma200': float(ma200),
                          'is_bear': is_bear, 'ret_5d': float(csi['close'].pct_change(5).iloc[-1])})
        else:
            self._respond({'error': 'CSI300 data not available'})

    def _get_market_features(self):
        """构建11维市场环境特征"""
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
        except:
            pass
        return np.zeros(11, dtype=np.float32)

    def _respond(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        pass  # 禁止访问日志


if __name__ == '__main__':
    PORT = 8899
    server = HTTPServer(('127.0.0.1', PORT), InferenceHandler)
    print(f"[Server] FI-PPO inference server running on http://127.0.0.1:{PORT}")
    print(f"[Server] Available strategies: {list(models.keys())}")
    print(f"[Server] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Stopped")
