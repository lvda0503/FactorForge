"""
因子计算引擎 + IC实时监控
=========================
5因子: ROC20, RSV14, STD20, PB, CORR20
每个因子自动追踪滚动IC，用于Factor-Informed Loss。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class FactorState:
    """单个因子的运行时状态"""
    values: np.ndarray          # 最新因子值 (标量，当前时刻)
    history: deque               # 历史因子值 (用于IC计算)
    ic_history: deque            # 历史IC序列
    current_ic: float = 0.0     # 当前滚动IC
    current_ic_ir: float = 0.0  # 当前IC_IR

    def update_ic(self, forward_return: float, ic_window: int = 120):
        """用最新的前向收益更新IC估计"""
        self.history.append(forward_return)
        if len(self.history) > ic_window:
            self.history.popleft()

        # 计算Spearman Rank IC
        if len(self.history) >= 30:
            # 用历史因子值和历史前向收益计算
            f_vals = np.array([h[0] for h in list(self.history)[-ic_window:]])
            r_vals = np.array([h[1] for h in list(self.history)[-ic_window:]])

            # 简化: 用因子值变化方向 vs 收益方向的一致性
            f_diff = np.diff(f_vals)
            r_diff = r_vals[1:]  # 对齐
            if len(f_diff) > 1 and np.std(f_diff) > 1e-10:
                ic = np.corrcoef(f_diff, r_diff)[0, 1]
                if not np.isnan(ic):
                    self.current_ic = ic
                    self.ic_history.append(ic)
                    if len(self.ic_history) > ic_window:
                        self.ic_history.popleft()

        # 计算 IC_IR
        ics = list(self.ic_history)
        if len(ics) >= 30 and np.std(ics) > 1e-10:
            self.current_ic_ir = np.mean(ics) / np.std(ics)

    def record_value(self, value: float):
        """记录因子值"""
        self.values = value


class FactorEngine:
    """
    因子计算引擎

    负责:
    1. 从价格/成交量数据计算5个因子值
    2. 追踪每个因子的滚动IC
    3. 检测因子衰减
    4. 计算因子间相关性

    用法:
        engine = FactorEngine(ic_window=120)
        factor_dict = engine.compute_factors(price_window, volume_window)
        # 在训练循环中每步更新IC:
        engine.update_ic(forward_return_dict)
    """

    def __init__(self, factor_names: List[str] = None, ic_window: int = 120):
        self.ic_window = ic_window

        if factor_names is None:
            factor_names = ["roc_20", "rsv_14", "std_20", "pb_ratio", "corr_20"]

        self.factor_names = factor_names
        self.states: Dict[str, FactorState] = {
            name: FactorState(
                values=0.0,
                history=deque(maxlen=ic_window),
                ic_history=deque(maxlen=ic_window),
            )
            for name in factor_names
        }

        # 存储最近的因子值序列 (用于计算正交性)
        self.recent_factor_values: Dict[str, deque] = {
            name: deque(maxlen=500) for name in factor_names
        }

    def compute_factors(self, price_window: np.ndarray,
                        volume_window: np.ndarray,
                        pb_value: Optional[float] = None,
                        pe_percentile: Optional[float] = None) -> Dict[str, float]:
        """
        从价格和成交量窗口计算因子值

        Args:
            price_window: (window_size, 4) OHLC数据
            volume_window: (window_size,) 成交量数据
            pb_value: PB比率 (外部提供，因为需要财务数据)

        Returns:
            {factor_name: factor_value}
        """
        close = price_window[:, 3]   # close在索引3
        high = price_window[:, 1]
        low = price_window[:, 2]
        volume = volume_window

        factors = {}

        # --- ROC20: 20日动量 ---
        if len(close) >= 20:
            factors["roc_20"] = close[-1] / close[-20] - 1
        else:
            factors["roc_20"] = 0.0

        # --- RSV14: 14日价格位置 (KDJ的RSV) ---
        if len(high) >= 14:
            h_max = np.max(high[-14:])
            l_min = np.min(low[-14:])
            rsv = (close[-1] - l_min) / (h_max - l_min + 1e-8)
            factors["rsv_14"] = np.clip(rsv, 0, 1)
        else:
            factors["rsv_14"] = 0.5

        # --- STD20: 20日波动率 ---
        if len(close) >= 20:
            returns = np.diff(close[-21:]) / (close[-21:-1] + 1e-8)
            factors["std_20"] = np.std(returns)
        else:
            factors["std_20"] = 0.0

        # --- PB: 市净率 (外部提供) ---
        if pb_value is not None and not np.isnan(pb_value) and pb_value > 0:
            factors["pb_ratio"] = 1.0 / max(pb_value, 0.01)  # BP = 1/PB
        else:
            factors["pb_ratio"] = 0.0

        # --- PE percentile (外部提供) ---
        if "pe_percentile" in self.factor_names:
            factors["pe_percentile"] = float(pe_percentile) if pe_percentile is not None else 0.5

        # --- ROC60: 60日动量 ---
        if "roc_60" in self.factor_names:
            factors["roc_60"] = close[-1] / close[-60] - 1 if len(close) >= 60 else 0.0

        # --- BETA20: 20日线性回归斜率 ---
        if "beta_20" in self.factor_names:
            if len(close) >= 20:
                x = np.arange(20); y = close[-20:]
                beta = (np.sum(x*y) - 20*np.mean(x)*np.mean(y)) / (np.sum(x**2) - 20*np.mean(x)**2 + 1e-10)
                factors["beta_20"] = float(beta / (close[-1] + 1e-10))
            else:
                factors["beta_20"] = 0.0

        # --- RANK20: 20日价格排名 ---
        if "rank_20" in self.factor_names:
            factors["rank_20"] = float(np.mean(close[-20:] < close[-1])) if len(close) >= 20 else 0.5

        # --- QTLD20: 20日收盘价20%分位数 ---
        if "qtld_20" in self.factor_names:
            factors["qtld_20"] = float(np.percentile(close[-20:], 20) / (close[-1] + 1e-10)) if len(close) >= 20 else 1.0

        # --- STD10: 10日波动率 ---
        if "std_10" in self.factor_names:
            if len(close) >= 11:
                r = np.diff(close[-11:]) / (close[-11:-1] + 1e-8)
                factors["std_10"] = float(np.std(r))
            else:
                factors["std_10"] = 0.0

        # --- STD60: 60日波动率 ---
        if "std_60" in self.factor_names:
            if len(close) >= 61:
                r = np.diff(close[-61:]) / (close[-61:-1] + 1e-8)
                factors["std_60"] = float(np.std(r))
            else:
                factors["std_60"] = 0.0

        # --- VMA20: 20日均量比 ---
        if "vma_20" in self.factor_names:
            factors["vma_20"] = float(np.mean(volume[-20:]) / (volume[-1] + 1e-10)) if len(volume) >= 20 else 1.0

        # --- IMAX20: 距20日高点天数比 ---
        if "imax_20" in self.factor_names:
            if len(high) >= 20:
                max_idx = np.argmax(high[-20:])
                factors["imax_20"] = float((19 - max_idx) / 20)
            else:
                factors["imax_20"] = 0.5

        # --- RSQR20: 20日趋势拟合度 R² ---
        if "rsqr_20" in self.factor_names:
            if len(close) >= 20:
                x = np.arange(20); y = close[-20:]
                ss_tot = np.sum((y - np.mean(y))**2)
                if ss_tot > 1e-10:
                    beta = (np.sum(x*y) - 20*np.mean(x)*np.mean(y)) / (np.sum(x**2) - 20*np.mean(x)**2 + 1e-10)
                    alpha = np.mean(y) - beta * np.mean(x)
                    y_pred = alpha + beta * x
                    ss_res = np.sum((y - y_pred)**2)
                    factors["rsqr_20"] = float(1 - ss_res / ss_tot)
                else:
                    factors["rsqr_20"] = 0.0
            else:
                factors["rsqr_20"] = 0.0

        # --- IMXD20: IMAX - IMIN 差值 ---
        if "imxd_20" in self.factor_names:
            if len(high) >= 20 and len(low) >= 20:
                max_idx = np.argmax(high[-20:])
                min_idx = np.argmin(low[-20:])
                factors["imxd_20"] = float((max_idx - min_idx) / 20)
            else:
                factors["imxd_20"] = 0.0

        # --- CORR20: 20日量价相关性 ---
        if len(close) >= 21 and len(volume) >= 21:
            price_changes = np.diff(close[-21:])
            vol_changes = np.diff(volume[-21:])
            if np.std(price_changes) > 1e-10 and np.std(vol_changes) > 1e-10:
                corr = np.corrcoef(price_changes, vol_changes)[0, 1]
                factors["corr_20"] = corr if not np.isnan(corr) else 0.0
            else:
                factors["corr_20"] = 0.0
        else:
            factors["corr_20"] = 0.0

        # NaN处理
        for k in factors:
            factors[k] = float(np.nan_to_num(factors[k], nan=0.0, posinf=1.0, neginf=-1.0))

        # 记录因子值
        for name, value in factors.items():
            if name in self.states:
                self.states[name].record_value(value)
                self.recent_factor_values[name].append(value)

        return factors

    def get_current_ics(self) -> Dict[str, float]:
        """获取所有因子的当前IC"""
        return {name: state.current_ic for name, state in self.states.items()}

    def get_current_ic_irs(self) -> Dict[str, float]:
        """获取所有因子的当前IC_IR"""
        return {name: state.current_ic_ir for name, state in self.states.items()}

    def get_factor_correlation_matrix(self) -> np.ndarray:
        """计算因子间的相关性矩阵 (基于最近500个因子值)"""
        n = len(self.factor_names)
        corr_matrix = np.eye(n)

        factor_arrays = {}
        for name in self.factor_names:
            vals = list(self.recent_factor_values[name])
            if len(vals) >= 30:
                factor_arrays[name] = np.array(vals)
            else:
                factor_arrays[name] = np.zeros(1)

        for i, f1 in enumerate(self.factor_names):
            for j, f2 in enumerate(self.factor_names):
                if i < j:
                    a1 = factor_arrays[f1]
                    a2 = factor_arrays[f2]
                    min_len = min(len(a1), len(a2))
                    if min_len >= 30:
                        corr = np.corrcoef(a1[-min_len:], a2[-min_len:])[0, 1]
                        corr_matrix[i, j] = corr_matrix[j, i] = (
                            corr if not np.isnan(corr) else 0.0
                        )

        return corr_matrix

    def detect_decay(self, threshold: float = 0.005) -> List[str]:
        """检测哪些因子正在衰减 (IC在最近缩短)"""
        decayed = []
        for name, state in self.states.items():
            ics = list(state.ic_history)
            if len(ics) >= 60:
                recent_ic = np.mean(ics[-30:])
                older_ic = np.mean(ics[-60:-30])
                if older_ic - recent_ic > threshold:
                    decayed.append(name)
        return decayed

    def reset(self):
        """重置所有状态 (每个episode开始时调用)"""
        for state in self.states.values():
            state.values = 0.0
            state.current_ic = 0.0
            state.current_ic_ir = 0.0
