"""
交易环境 v2 — 连续动作 + 做空
==============================
单股票 RL 交易环境。不依赖 gym。

动作空间: 连续 [-1, +1]
  -1.0  = 满仓做空
   0.0  = 空仓
  +1.0  = 满仓做多

做空逻辑:
  借入股票卖出 → 现金增加 → 负债 = 借入时市值
  每日盯市: 做空盈亏 = -(当前价/借入价 - 1) × 做空金额
  保证金: 做空金额的 X% 被锁定，不可用于做多
  融券费: 每日做空市值 × 日费率

State: 分层设计
  价格层: 去噪后的收益率序列 (60维)
  因子层: 5个因子值 (5维)
  组合层: [净仓位, 现金比, 浮动盈亏] (3维)
  总计: 68维
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .state_builder import StateBuilder
from ..preprocessing.factor_engine import FactorEngine
from ..preprocessing.denoiser import Denoiser
from ..data.market_context import MarketContext


class BaseEnv:
    def __init__(self):
        self.action_space = None
        self.observation_space = None
    def reset(self, seed=None):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError


class DiscreteSpace:
    def __init__(self, n: int): self.n = n

class BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low; self.high = high; self.shape = shape; self.dtype = dtype


class TradingEnv(BaseEnv):
    """连续动作 + 做空的交易环境

    参数:
        df: 价格数据 (open,high,low,close,volume,pe,pb)
        factor_engine: 因子引擎
        state_builder: 状态构建器
        denoiser: 降噪器
        window_size: 历史窗口
        initial_capital: 初始资金
        enable_short: 是否启用做空
        short_margin: 做空保证金比例
        short_borrow_cost: 融券日费率
        commission: 手续费率
        slippage: 滑点率
    """

    def __init__(self, df, factor_engine, state_builder, denoiser,
                 window_size=60, initial_capital=100000.0,
                 enable_short=True,
                 max_long_pct=0.80, max_short_pct=0.10, stop_loss_pct=0.08,
                 commission=0.00025, stamp_tax=0.0005, slippage=0.001,
                 short_margin=0.5, short_borrow_cost=0.0001,
                 market_ctx=None):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self._dates = df.index          # 保留原始日期索引
        self.factor_engine = factor_engine
        self.state_builder = state_builder
        self.denoiser = denoiser

        self.window_size = window_size
        self.initial_capital = initial_capital
        self.enable_short = enable_short

        # 硬风控参数
        self.max_long_pct = max_long_pct      # 多头仓位上限 80%
        self.max_short_pct = max_short_pct    # 空头仓位上限 30%
        self.stop_loss_pct = stop_loss_pct    # 单方向止损线 15%

        # A股真实费率
        self.commission = commission           # 万2.5 = 0.00025
        self.stamp_tax = stamp_tax             # 卖出印花税 0.05% = 0.0005
        self.slippage = slippage               # 滑点 0.1%
        self.short_margin = short_margin
        self.short_borrow_cost = short_borrow_cost
        self.market_ctx = market_ctx

        # PE/PB 历史分位预计算
        self._pe_hist = df['pe'].dropna().values if 'pe' in df.columns else np.array([20.0])
        self._pb_hist = df['pb'].dropna().values if 'pb' in df.columns else np.array([5.0])

        # 连续动作空间
        self.action_space = BoxSpace(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = BoxSpace(
            low=-np.inf, high=np.inf, shape=(state_builder.state_dim,))

        # 内部状态
        self.idx = window_size

        # 多头
        self.long_shares = 0.0
        self.long_cost = 0.0
        # 空头
        self.short_shares = 0.0       # 借入并卖出的股数
        self.short_liability = 0.0    # 空头负债 = 借入时的市值
        self.short_margin_locked = 0.0

        self.cash = initial_capital
        self.total_value = initial_capital

        self.episode_returns = []
        self.trade_count = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.idx = self.window_size + np.random.randint(0, min(20, len(self.df) - self.window_size - 1))
        self.long_shares = 0.0
        self.long_cost = 0.0
        self.short_shares = 0.0
        self.short_liability = 0.0
        self.short_margin_locked = 0.0
        self.cash = self.initial_capital
        self.total_value = self.initial_capital
        self.episode_returns = []
        self.trade_count = 0
        self._last_trade_count = 0
        self._steps_since_last_trade = 0
        self.factor_engine.reset()
        return self._get_state(), {}

    def step(self, action: float):
        """
        执行一步交易

        Args:
            action: 连续值 [-1, +1]
              -1.0 = 满仓做空, 0.0 = 空仓, +1.0 = 满仓做多

        Returns:
            state, reward, terminated, truncated, info
        """
        action = float(np.clip(action, -1.0, 1.0))

        # Agent在t日收盘后决策, 次日(t+1)开盘执行, 收盘估值
        next_idx = self.idx + 1
        exec_price = self._get_open(next_idx)
        current_price = self._get_price(next_idx)
        prev_total = self.total_value

        # ── 涨跌停检查 ──
        if next_idx >= 1:
            prev_close = self._get_price(next_idx - 1)
            is_limit_up   = (prev_close > 0 and exec_price >= prev_close * 1.098)
            is_limit_down = (prev_close > 0 and exec_price <= prev_close * 0.902)
        else:
            is_limit_up = False; is_limit_down = False

        # ── 流动性过滤 ──
        volume = self.df.iloc[min(next_idx, len(self.df)-1)].get('volume', 0)
        min_volume = 1e6
        low_liquidity = (volume < min_volume)

        # ── 止损检查 ──
        if self.long_shares > 0 and self.long_cost > 0:
            long_loss = (current_price / self.long_cost - 1)
            if long_loss < -self.stop_loss_pct:
                self._close_long(current_price)
                self.trade_count += 1
        if self.short_shares > 0 and self.short_liability > 0:
            short_loss = -(current_price * self.short_shares / self.short_liability - 1)
            if short_loss < -self.stop_loss_pct:
                self._close_short(current_price)
                self.trade_count += 1

        # ── 计算目标仓位 (硬约束裁剪) ──
        available_capital = self.cash - self.short_margin_locked

        if action >= 0:
            target_long_pct = min(action, self.max_long_pct)    # ← 硬约束: 最多80%
            target_short_pct = 0.0
        else:
            target_short_pct = min(abs(action), self.max_short_pct)  # ← 硬约束: 最多30%
            target_long_pct = 0.0

        # 破产保护: 如果净值低于初始的50%, 强制空仓
        if self.total_value < self.initial_capital * 0.5:
            target_long_pct = 0.0
            target_short_pct = 0.0

        # 涨停→禁止买入  跌停→禁止卖出  流动性差→禁止交易
        if low_liquidity:
            target_long_pct = self._current_position()
            target_short_pct = 0.0
        else:
            if is_limit_up and target_long_pct > 0:
                target_long_pct = 0.0  # 涨停买不了
            if is_limit_down and target_short_pct > 0:
                target_short_pct = 0.0  # 跌停卖不了
            if is_limit_up and self.long_shares > 0:
                # 涨停但持有 → 可以卖出但不能买入
                target_long_pct = max(target_long_pct, 0.0)

        buy_signal = (target_long_pct > 0 and abs(action - self._current_position()) > 0.03)
        short_signal = (target_short_pct > 0 and self.enable_short)

        # ── 平掉对冲仓位 (用开盘价执行) ──
        if buy_signal and self.short_shares > 0:
            self._close_short(exec_price)
            self.trade_count += 1
        if short_signal and self.long_shares > 0:
            self._close_long(exec_price)
            self.trade_count += 1

        # ── 做多 (用开盘价执行) ──
        if buy_signal:
            target_long_value = available_capital * target_long_pct
            current_long_value = self.long_shares * current_price
            delta = target_long_value - current_long_value

            if abs(delta) > available_capital * 0.01:
                self.trade_count += 1
                buy_price = exec_price * (1 + self.slippage)
                if delta > 0:
                    cost = delta * (1 + self.commission)
                    if cost <= self.cash:
                        self.long_shares += delta / buy_price
                        self.cash -= delta
                        self.long_cost = (self.long_cost + buy_price) / 2 if self.long_cost > 0 else buy_price
                else:
                    shares_to_sell = min(abs(delta) / exec_price, self.long_shares)
                    self.cash += shares_to_sell * exec_price * (1 - self.slippage) * (1 - self.commission - self.stamp_tax)
                    self.long_shares -= shares_to_sell

        # ── 做空 (用开盘价执行) ──
        if short_signal and self.enable_short:
            target_short_value = available_capital * target_short_pct
            current_short_liability = self.short_liability

            if target_short_value > 0 and self.short_shares == 0:
                self.trade_count += 1
                sell_price = exec_price * (1 - self.slippage)
                short_value = target_short_value

                # 卖空得现金，同时锁定保证金
                self.short_shares = short_value / sell_price
                self.short_liability = self.short_shares * sell_price
                self.cash += self.short_liability * (1 - self.commission - self.stamp_tax)
                self.short_margin_locked = self.short_liability * self.short_margin

        # ── 持仓观望时平仓 (用开盘价执行) ──
        if not buy_signal and not short_signal:
            if abs(action) < 0.05:
                if self.long_shares > 0:
                    self._close_long(exec_price)
                    self.trade_count += 1
                if self.short_shares > 0:
                    self._close_short(exec_price)
                    self.trade_count += 1

        # ── 盯市: 用收盘价估值 ──
        long_value = self.long_shares * current_price
        short_mtm = 0.0
        if self.short_shares > 0:
            # 做空盈亏: 借入价 - 当前价
            short_mtm = self.short_liability - (self.short_shares * current_price)
            # 融券费
            short_mtm -= self.short_liability * self.short_borrow_cost

        self.total_value = self.cash + long_value + short_mtm

        # ── 奖励: PnL + 基本面锚定 + 活动激励 ──
        self.total_value = max(self.total_value, 1000.0)
        pnl_reward = np.log(self.total_value / (prev_total + 1e-10))
        reward = float(np.nan_to_num(pnl_reward, nan=0.0))

        # 1. 基本面锚定: PB低位做多 / PB高位做空 → 奖励
        if 'pb' in self.df.columns:
            pb_val = self.df.iloc[min(self.idx, len(self.df)-1)]['pb']
            if hasattr(self, '_pb_hist') and len(self._pb_hist) > 0:
                pb_pct = float(np.mean(self._pb_hist < pb_val))
                pos = self._current_position()
                if pb_pct < 0.3 and pos > 0:
                    reward += 0.003 * pos           # 便宜做多加奖励
                if pb_pct > 0.7 and pos < 0:
                    reward += 0.003 * abs(pos)      # 贵做空加奖励
                if pb_pct > 0.85 and pos > 0.5:
                    reward -= 0.008                  # 极高还在重仓做多→惩罚

        # 2. 活动激励: 调仓有奖励, 长期不动有微惩罚
        made_trade = self.trade_count > (self._last_trade_count if hasattr(self, '_last_trade_count') else 0)
        if made_trade:
            reward += 0.002
            self._steps_since_last_trade = 0
        else:
            self._steps_since_last_trade = getattr(self, '_steps_since_last_trade', 0) + 1
            if self._steps_since_last_trade > 100:
                reward -= 0.0001
        self._last_trade_count = self.trade_count

        reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0))
        self.episode_returns.append(reward)

        # ── 移到下一步 ──
        self.idx += 1

        terminated = self.idx >= len(self.df) - 2  # 需要 next_idx=self.idx+1 有效
        truncated = False

        info = {
            'long_position': self.long_shares * current_price / self.total_value if self.total_value > 0 else 0,
            'short_position': self.short_liability / self.total_value if self.total_value > 0 else 0,
            'total_value': self.total_value,
            'cash': self.cash,
            'trade_count': self.trade_count,
            'current_price': current_price,
            'factor_values': self._get_current_factors(),
        }

        if terminated:
            rets = np.array(self.episode_returns)
            info['total_return'] = float(self.total_value / self.initial_capital - 1)
            info['sharpe'] = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)) if len(rets) > 1 else 0.0
            cum = np.cumprod(1 + rets)
            peak = np.maximum.accumulate(cum)
            info['max_drawdown'] = float(np.min((cum - peak) / peak)) if len(cum) > 0 else 0.0

        state = self._get_state()
        if np.isnan(state).any():
            state = np.nan_to_num(state, nan=0.0)
        return state, reward, terminated, truncated, info

    def _close_long(self, price):
        if self.long_shares > 0:
            sell_price = price * (1 - self.slippage)
            # A股卖出: 佣金 + 印花税
            self.cash += self.long_shares * sell_price * (1 - self.commission - self.stamp_tax)
            self.long_shares = 0.0
            self.long_cost = 0.0

    def _close_short(self, price):
        if self.short_shares > 0:
            buyback_price = price * (1 + self.slippage)
            # 买回: 只有佣金 (印花税卖方付, 我们是买方)
            buyback_cost = self.short_shares * buyback_price * (1 + self.commission)
            self.cash -= buyback_cost
            self.short_shares = 0.0
            self.short_liability = 0.0
            self.short_margin_locked = 0.0

    def _current_position(self):
        """返回当前净仓位 [-1, +1]"""
        long_val = self.long_shares * self._get_price(self.idx)
        short_val = self.short_liability
        total = self.total_value
        if total <= 0:
            return 0.0
        return (long_val - short_val) / total

    def _get_state(self):
        idx = min(max(self.idx, self.window_size), len(self.df) - 1)
        price_window = self.df.iloc[idx - self.window_size:idx]
        ohlcv = price_window[['open','high','low','close','volume']].values.astype(np.float64)

        denoiser_input = ohlcv[:, 3]
        close_denoised = self.denoiser.denoise(denoiser_input)

        pb_value = self.df.iloc[idx]['pb'] if 'pb' in self.df.columns else None
        pe_pct_val = self.df.iloc[idx].get('pe_percentile', 0.5) if 'pe_percentile' in self.df.columns else 0.5
        factors = self.factor_engine.compute_factors(ohlcv, ohlcv[:, 4], pb_value,
                                                      pe_percentile=pe_pct_val)

        long_val = self.long_shares * ohlcv[-1, 3]
        short_val = self.short_liability
        total = self.total_value if self.total_value > 0 else 1.0
        net_position = (long_val - short_val) / total
        cash_ratio = self.cash / total
        unrealized = 0.0
        if self.long_shares > 0 and self.long_cost > 0:
            unrealized = ohlcv[-1, 3] / self.long_cost - 1
        elif self.short_shares > 0 and self.short_liability > 0:
            unrealized = -(ohlcv[-1, 3] * self.short_shares / self.short_liability - 1)

        # 市场环境特征
        mkt_feat = np.zeros(11, dtype=np.float32)
        if self.market_ctx is not None:
            date = self._dates[idx]
            pe_pct = self._compute_percentile(self.df.iloc[idx].get('pe',np.nan), 'pe') if 'pe' in self.df.columns else 0.5
            pb_pct = self._compute_percentile(self.df.iloc[idx]['pb'], 'pb')
            turnover = self.df.iloc[idx].get('turn', None) if 'turn' in self.df.columns else None
            mkt_feat = self.market_ctx.compute(date, self.df,
                pe_percentile=pe_pct, pb_percentile=pb_pct,
                turnover_ratio=turnover)

        state = self.state_builder.build(
            price_window=ohlcv, close_denoised=close_denoised,
            factors=factors, position=net_position,
            cash_ratio=cash_ratio, unrealized_pnl=np.clip(unrealized, -1.0, 1.0),
            market_features=mkt_feat)
        return state.astype(np.float32)

    def _compute_percentile(self, value, which='pe'):
        hist = getattr(self, f'_{which}_hist', None)
        if hist is None or len(hist) == 0 or np.isnan(value):
            return 0.5
        return float(np.mean(hist < value))

    def _get_price(self, idx):
        return float(self.df.iloc[min(idx, len(self.df) - 1)]['close'])

    def _get_open(self, idx):
        return float(self.df.iloc[min(idx, len(self.df) - 1)]['open'])

    def _get_current_factors(self):
        return {name: state.values for name, state in self.factor_engine.states.items()}
