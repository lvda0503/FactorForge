"""
分层状态空间构建器 v4
====================
将原始数据组织为Agent可用的状态向量。

四层状态:
  Layer 1 (价格):   去噪后的标准化收益率序列 [60维]
  Layer 2 (因子):   因子值 [5维]
  Layer 3 (组合):   [仓位, 现金比, 浮动盈亏] [3维]
  Layer 4 (市场):   大盘+资金+日历+波动+估值 [11维]

总计: 60 + 5 + 3 + 11 = 79维
"""
import numpy as np
from typing import Dict, Optional


class StateBuilder:
    """分层状态构建器

    Args:
        window_size: 价格窗口大小 (如 60)
        factor_names: 因子名称列表
        use_price_returns: 用收益率代替原始价格 (推荐)
        use_normalized_factors: 是否标准化因子 (Z-score, 需要足够历史)
    """

    def __init__(
        self,
        window_size: int = 60,
        factor_names: Optional[list] = None,
        use_price_returns: bool = True,
        use_normalized_factors: bool = False,
        market_dim: int = 11,  # v4: 市场环境维度
    ):
        self.window_size = window_size
        self.factor_names = factor_names or [
            "roc_20", "rsv_14", "std_20", "pb_ratio", "corr_20"
        ]
        self.use_price_returns = use_price_returns
        self.use_normalized_factors = use_normalized_factors
        self.market_dim = market_dim

        self.factor_means: Dict[str, float] = {}
        self.factor_stds: Dict[str, float] = {}

        n_price = window_size
        n_factors = len(self.factor_names)
        n_portfolio = 3

        self.state_dim = n_price + n_factors + n_portfolio + market_dim

    def build(
        self,
        price_window: np.ndarray,
        close_denoised: np.ndarray,
        factors: Dict[str, float],
        position: float,
        cash_ratio: float,
        unrealized_pnl: float,
        market_features: np.ndarray = None,
    ) -> np.ndarray:
        """
        构建完整状态向量

        Args:
            price_window: (window_size, 5) OHLCV原始值
            close_denoised: (window_size,) 去噪后的收盘价
            factors: {factor_name: value}
            position: 当前仓位 [0, 1]
            cash_ratio: 现金比例 [0, 1]
            unrealized_pnl: 浮动盈亏率
            market_features: (11,) 市场环境特征

        Returns:
            一维状态向量 (state_dim,)
        """
        parts = []

        # --- Layer 1: 价格层 ---
        if self.use_price_returns:
            close = np.maximum(close_denoised, 1e-8)
            if len(close) >= 2:
                log_returns = np.diff(np.log(close))
                if len(log_returns) < self.window_size:
                    padded = np.zeros(self.window_size)
                    padded[-len(log_returns):] = log_returns
                    parts.append(padded)
                else:
                    parts.append(log_returns[-self.window_size:])
            else:
                parts.append(np.zeros(self.window_size))
        else:
            close = price_window[:, 3]
            normed = close / (close[-1] + 1e-10) - 1.0
            if len(normed) >= self.window_size:
                parts.append(normed[-self.window_size:])
            else:
                padded = np.zeros(self.window_size)
                padded[-len(normed):] = normed
                parts.append(padded)

        # --- Layer 2: 因子层 ---
        factor_vec = np.zeros(len(self.factor_names))
        for i, name in enumerate(self.factor_names):
            val = factors.get(name, 0.0)
            if self.use_normalized_factors:
                if name in self.factor_means and name in self.factor_stds:
                    val = (val - self.factor_means[name]) / (
                        self.factor_stds[name] + 1e-10)
            factor_vec[i] = val
        parts.append(factor_vec)

        # --- Layer 3: 组合层 ---
        parts.append(np.array([
            position,
            cash_ratio,
            np.clip(unrealized_pnl, -1.0, 1.0)
        ]))

        # --- Layer 4: 市场环境层 ---
        if market_features is not None and len(market_features) == self.market_dim:
            parts.append(market_features.astype(np.float32))
        else:
            parts.append(np.zeros(self.market_dim, dtype=np.float32))

        state = np.concatenate(parts)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        state = np.clip(state, -100.0, 100.0)

        return state.astype(np.float32)
