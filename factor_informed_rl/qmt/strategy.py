"""
QMT 模拟盘策略 — FI-PPO 单股交易
=================================
移植要点:
  1. 模型加载: torch.load V7/V9 .pt → 纯推理, 不上传模型文件到QMT服务器
  2. 数据: QMT的 get_market_data_ex() → 转为我们的 OHLCV DataFrame
  3. 因子: FactorEngine 保持不变
  4. 风控: 硬编码在策略中, 不依赖回测环境
  5. 下单: passorder() 替代 env.step()

QMT上下文变量 (由平台注入):
  C: Context对象 — 账户/持仓/现金
  P: PassOrder对象 — 下单
  xtdata: 行情数据
"""
import numpy as np
import pandas as pd
import torch
import os
import sys

# QMT中需要手动设置路径
MODEL_DIR = "D:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"

# 策略配置
STRATEGY_CONFIG = {
    "Value-Defensive": {
        "model_file": "Value-Defensive_600519_fi.pt",
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
    },
    "Quality-Offensive": {
        "model_file": "Quality-Offensive_600276_fi.pt",
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
    },
}

class FIFPPO_QMT:
    """FI-PPO QMT 策略适配器"""

    def __init__(self, strategy_name="Value-Defensive",
                 initial_capital=1_000_000, max_position=0.18,
                 stop_loss=-0.08, max_long=0.80, max_short=0.10):
        self.name = strategy_name
        cfg = STRATEGY_CONFIG[strategy_name]
        self.factors = cfg["factors"]

        # 风控参数 (硬编码, 不依赖回测环境)
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.max_long = max_long
        self.max_short = max_short
        self.stop_loss = stop_loss

        # 佣金 (QMT实际费率)
        self.commission = 0.00025   # 万2.5
        self.stamp_tax = 0.0005     # 卖出0.05%
        self.slippage = 0.001       # 预留冲击

        # 模型 — 延迟加载 (避免init时阻塞QMT)
        self.model = None
        self.state_builder = None
        self.model_path = f"{MODEL_DIR}/{cfg['model_file']}"
        self.model_loaded = False

        # 每只股票的状态追踪
        self.stock_state = {}  # {code: {entry_price, shares, position_pct, max_value, trades}}

    def load_model(self):
        """加载FI-PPO模型 (首次运行时调用)"""
        if self.model_loaded:
            return

        sys.path.insert(0, "D:/JoinQuant/quant_env")
        from factor_informed_rl.models.actor_critic import PPOActorCritic
        from factor_informed_rl.env.state_builder import StateBuilder

        ckpt = torch.load(self.model_path, map_location='cpu')
        self.state_dim = ckpt['state_dim']

        self.state_builder = StateBuilder(
            window_size=60, factor_names=self.factors, market_dim=11)

        self.model = PPOActorCritic(self.state_dim, 1, [256, 128, 64])
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.model_loaded = True

        print(f"[FI-PPO] Model loaded: {self.name}, state_dim={self.state_dim}")

    # ── QMT 生命周期 ──
    def init(self, C):
        """QMT初始化回调"""
        self.load_model()
        self.C = C
        print(f"[FI-PPO] Strategy initialized: {self.name}")

    def handle_bar(self, C, stock_code):
        """
        QMT逐K线回调 — 每只股票每个bar触发
        我们要在这里:
          1. 拉历史数据 → 构建状态
          2. FI-PPO推理 → 得到动作
          3. 检查风控 → 执行下单
        """
        if not self.model_loaded:
            return

        # 1. 拉取历史数据 (60天窗口 + 当天)
        history = self._get_history(C, stock_code, days=61)
        if history is None or len(history) < 61:
            return

        # 2. 计算因子值
        from factor_informed_rl.preprocessing.factor_engine import FactorEngine
        engine = FactorEngine(self.factors)
        ohlcv = history[['open','high','low','close','volume']].values.astype(np.float64)

        # PE/PB (QMT提供)
        pb_value = self._get_financial(stock_code, 'pb')
        pe_percentile = self._get_pe_percentile(stock_code, history)

        factors = engine.compute_factors(ohlcv, ohlcv[:,4],
                                         pb_value=pb_value, pe_percentile=pe_percentile)

        # 3. 构建状态
        close_denoised = ohlcv[:, 3]  # QMT可直接用, 后续加去噪

        # 获取当前持仓
        position = self._get_position(C, stock_code)
        cash_ratio = self._get_cash_ratio(C)
        unrealized_pnl = self._get_unrealized_pnl(C, stock_code)

        # 市场环境 (简化版, QMT可获取指数行情)
        mkt_features = self._get_market_context(C)

        state = self.state_builder.build(
            price_window=ohlcv, close_denoised=close_denoised,
            factors=factors, position=position,
            cash_ratio=cash_ratio, unrealized_pnl=unrealized_pnl,
            market_features=mkt_features)

        # 4. FI-PPO推理
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.model.get_action(s, deterministic=True)
        target_position = float(action.squeeze().numpy())

        # 5. 风控裁剪
        target_position = self._apply_risk_controls(
            C, stock_code, target_position, history)

        # 6. 执行调仓
        self._execute_trade(C, stock_code, target_position, history)

    # ── QMT数据获取 ──
    def _get_history(self, C, stock_code, days=61):
        """拉取历史K线"""
        try:
            # QMT API: get_market_data_ex
            data = C.get_market_data_ex(
                fields=['open','high','low','close','volume'],
                stock_list=[stock_code],
                period='1d',
                count=days,
                dividend_type='front'  # 前复权
            )
            if data is None or stock_code not in data:
                return None

            df = data[stock_code]
            df.columns = ['open','high','low','close','volume']
            return df
        except Exception as e:
            print(f"[FI-PPO] Data error {stock_code}: {e}")
            return None

    def _get_financial(self, stock_code, field):
        """获取财务数据"""
        try:
            # QMT通过 get_market_data_ex 的 pe/pb 字段获取
            data = self.C.get_market_data_ex(
                fields=[field], stock_list=[stock_code],
                period='1d', count=1, dividend_type='front')
            if data and stock_code in data:
                return float(data[stock_code].iloc[-1, 0])
        except:
            pass
        return None

    def _get_pe_percentile(self, stock_code, history):
        """PE历史分位 (PIT-safe)"""
        # 简化: 用最近250天PE的百分位
        try:
            data = self.C.get_market_data_ex(
                fields=['pe'], stock_list=[stock_code],
                period='1d', count=250, dividend_type='front')
            if data and stock_code in data:
                pe_vals = data[stock_code].dropna().values.flatten()
                current_pe = pe_vals[-1]
                return float(np.mean(pe_vals < current_pe)) if len(pe_vals)>0 else 0.5
        except:
            pass
        return 0.5

    def _get_market_context(self, C):
        """获取市场环境 (简化版)"""
        # QMT: 获取沪深300指数行情
        # 简化实现 — 模拟盘阶段可跳过, 实盘时补充
        return np.zeros(11, dtype=np.float32)

    # ── 风控层 ──
    def _get_position(self, C, stock_code):
        """当前仓位比例"""
        try:
            pos = C.get_trade_detail_data(
                stock_code, 'stock', 'position')
            if pos and len(pos) > 0:
                market_value = pos[0].m_dVolume * self._get_price(stock_code)
                return market_value / max(self._get_total_value(C), 1)
        except:
            pass
        return 0.0

    def _get_cash_ratio(self, C):
        try: return C.account.cash / C.account.total_asset
        except: return 0.5

    def _get_total_value(self, C):
        try: return C.account.total_asset
        except: return self.initial_capital

    def _get_price(self, stock_code):
        return float(self.C.get_full_tick([stock_code])[stock_code]['lastPrice'])

    def _get_unrealized_pnl(self, C, stock_code):
        try:
            pos = C.get_trade_detail_data(stock_code, 'stock', 'position')
            return pos[0].m_dProfitLossPct / 100 if pos and len(pos)>0 else 0.0
        except:
            return 0.0

    def _apply_risk_controls(self, C, stock_code, action, history):
        """硬风控裁剪"""
        # 仓位限制
        action = np.clip(action, -self.max_short, self.max_long)

        # 止损: 如果浮动亏损 > 8%, 强制清仓
        pnl = self._get_unrealized_pnl(C, stock_code)
        if pnl < self.stop_loss and self._get_position(C, stock_code) > 0:
            return 0.0  # 清仓做多

        # 回撤保护
        total = self._get_total_value(C)
        if total < self.initial_capital * 0.8:
            action = min(action, 0.0)  # 只允许减仓

        return action

    def _execute_trade(self, C, stock_code, target_pos, history):
        """执行调仓 — QMT下单"""
        current_pos = self._get_position(C, stock_code)
        delta = target_pos - current_pos

        if abs(delta) < 0.01:  # 变化<1%, 不交易
            return

        price = history['close'].iloc[-1]
        total_value = self._get_total_value(C)
        target_amount = delta * total_value

        if target_amount > 0:
            # 买入
            buy_price = price * (1 + self.slippage)
            volume = int(target_amount / buy_price / 100) * 100  # 100股整数倍
            if volume > 0:
                passorder(23, 1101, C.account, stock_code,
                         5, -1, volume, C.order_name,
                         'buy', C)
        elif target_amount < 0:
            # 卖出
            sell_price = price * (1 - self.slippage)
            cur_shares = self._get_shares(C, stock_code)
            sell_volume = min(cur_shares, int(abs(target_amount) / sell_price / 100) * 100)
            if sell_volume > 0:
                passorder(24, 1101, C.account, stock_code,
                         5, -1, sell_volume, C.order_name,
                         'sell', C)

    def _get_shares(self, C, stock_code):
        try:
            pos = C.get_trade_detail_data(stock_code, 'stock', 'position')
            return pos[0].m_nVolume if pos and len(pos)>0 else 0
        except:
            return 0

    # ── QMT 退出回调 ──
    def stop(self, C):
        print(f"[FI-PPO] Strategy stopped: {self.name}")


# ═══════════════════════════════════════════════════════════
# QMT 入口函数 (QMT框架自动调用这两个函数)
# ═══════════════════════════════════════════════════════════

def init(C):
    """QMT策略入口 — 初始化"""
    # 选择策略: 修改这里切换 Value-Defensive / Quality-Offensive
    STRATEGY = "Value-Defensive"

    C.strategy = FIFPPO_QMT(strategy_name=STRATEGY)
    C.strategy.init(C)

    # 设置股票池 — 从选股模块获取, 或手动指定
    C.stock_list = ["600519.SH", "000858.SZ", "000333.SZ",
                    "600276.SH", "600887.SH", "002415.SZ"]

    # 定时任务: 每天14:55运行 (收盘前5分钟)
    C.run_time("handle_bar", "1d", "2021-01-01", "14:55:00")

    print(f"[QMT] Strategy initialized, stock_list={C.stock_list}")


def handle_bar(C):
    """QMT主循环 — 每根K线触发"""
    for stock in C.stock_list:
        C.strategy.handle_bar(C, stock)
