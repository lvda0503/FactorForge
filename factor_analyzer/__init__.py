"""
FactorAnalyzer — 一站式因子分析模块
====================================
用法:
    from factor_analyzer import FactorAnalyzer

    fa = FactorAnalyzer(factor_df, prices_df)
    fa.full_report()              # 控制台完整报告
    fa.full_report_html('report.html')  # HTML 报告

    # 分维度调用
    fa.prediction.report()        # IC/IR/Fama-MacBeth
    fa.returns.report()           # 分组收益/多空/FF Alpha
    fa.stability.report()         # 换手率/自相关/回撤
    fa.uniqueness.report()        # 相关性/VIF/边际贡献
    fa.advanced.report()          # PBO/Deflated Sharpe/状态

输入格式:
    factor_df:  MultiIndex DataFrame, index=(date, asset), 至少包含一列因子值
    prices_df:  Wide DataFrame, index=date, columns=asset (可选，也可以直接用 returns_df)
    returns_df: Wide DataFrame, index=date, columns=asset (日收益率)
    groupby:    dict, {asset: group_name}  行业分组
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

from .prediction import PredictionAnalyzer
from .returns import ReturnsAnalyzer
from .stability import StabilityAnalyzer
from .uniqueness import UniquenessAnalyzer
from .advanced import AdvancedAnalyzer
from .visualization import FactorVisualizer
from .reports import ReportGenerator


class FactorAnalyzer:
    """因子分析主入口

    Parameters
    ----------
    factor_df : pd.DataFrame (MultiIndex)
        因子数据，index=(date, asset)，至少一列因子值。列名若为'factor'则自动识别。
    prices_df : pd.DataFrame, optional
        价格数据，index=date, columns=asset。如果提供，自动计算收益率。
    returns_df : pd.DataFrame, optional
        日收益率数据，index=date, columns=asset。与prices_df二选一。
    groupby : dict, optional
        行业分组，{asset: group_name}
    periods : tuple, default (1, 5, 10, 20)
        前向收益周期（天）
    quantiles : int, default 5
        分位数分组数
    factor_name : str, optional
        因子名称，如果不提供则自动从数据中推断
    """

    def __init__(
        self,
        factor_df: pd.DataFrame,
        prices_df: Optional[pd.DataFrame] = None,
        returns_df: Optional[pd.DataFrame] = None,
        groupby: Optional[Dict] = None,
        periods: Tuple[int, ...] = (1, 5, 10, 20),
        quantiles: int = 5,
        factor_name: Optional[str] = None,
    ):
        self.factor_df = factor_df
        self.prices_df = prices_df
        self.periods = periods
        self.quantiles = quantiles
        self.groupby = groupby

        # 推断因子名称
        if factor_name:
            self.factor_name = factor_name
        elif isinstance(factor_df, pd.DataFrame) and len(factor_df.columns) == 1:
            self.factor_name = factor_df.columns[0]
        else:
            self.factor_name = 'factor'

        # 准备数据
        self._prepare_data(returns_df)

        # 初始化子分析器
        self.prediction = PredictionAnalyzer(self)
        self.returns = ReturnsAnalyzer(self)
        self.stability = StabilityAnalyzer(self)
        self.uniqueness = UniquenessAnalyzer(self)
        self.advanced = AdvancedAnalyzer(self)
        self.visualizer = FactorVisualizer(self)
        self.reporter = ReportGenerator(self)

    def _prepare_data(self, returns_df=None):
        """数据预处理：对齐因子和收益"""
        if returns_df is not None:
            self.returns_df = returns_df
        elif self.prices_df is not None:
            self.returns_df = self.prices_df.pct_change().dropna(how='all')
        else:
            raise ValueError("必须提供 prices_df 或 returns_df 之一")

        # 确保因子是 Series
        if isinstance(self.factor_df, pd.DataFrame):
            if len(self.factor_df.columns) == 1:
                self.factor_series = self.factor_df.iloc[:, 0]
            elif self.factor_name in self.factor_df.columns:
                self.factor_series = self.factor_df[self.factor_name]
            else:
                self.factor_series = self.factor_df.iloc[:, 0]
                self.factor_name = self.factor_df.columns[0]
        else:
            self.factor_series = self.factor_df

        self.factor_series.name = self.factor_name

        # 计算前向收益
        self._compute_forward_returns()

    def _compute_forward_returns(self):
        """计算多周期前向收益"""
        factor_dates = set(self.factor_series.index.get_level_values(0))
        ret_dates = self.returns_df.index

        # 找到共同日期
        common_dates = sorted(factor_dates & set(ret_dates))

        forward_returns = {}
        for p in self.periods:
            fwd = self.returns_df.shift(-p)
            # 多日累计收益
            if p > 1:
                cum_ret = (1 + self.returns_df).rolling(p).apply(
                    lambda x: np.prod(1 + x) - 1, raw=True
                ).shift(-p)
                # 简化处理：用pct_change
                fwd_vals = self.returns_df.shift(-1)
            else:
                fwd_vals = self.returns_df.shift(-1)

            # 对齐到因子日期
            stacked = pd.DataFrame(index=self.factor_series.index)
            for date in common_dates:
                if date in fwd_vals.index:
                    row = fwd_vals.loc[date]
                    for asset in stacked.loc[date].index:
                        if asset in row.index:
                            stacked.loc[(date, asset), f'{p}D'] = row[asset]

            forward_returns[f'{p}D'] = stacked[f'{p}D']

        self.forward_returns = pd.DataFrame(forward_returns, index=self.factor_series.index)

    def quick_scan(self) -> pd.DataFrame:
        """快速扫描：返回核心指标摘要表"""
        ic_stats = self.prediction.ic_summary()
        ret_stats = self.returns.quantile_summary()
        stab_stats = self.stability.summary()

        results = {
            'IC_mean': ic_stats.get('IC_mean', np.nan),
            'Rank_IC': ic_stats.get('Rank_IC_mean', np.nan),
            'IC_IR': ic_stats.get('IC_IR', np.nan),
            'IC_pos_ratio': ic_stats.get('IC_pos_ratio', np.nan),
            'Long_Short_ann': ret_stats.get('long_short_ann', np.nan),
            'Top_ann': ret_stats.get('top_ann', np.nan),
            'Bottom_ann': ret_stats.get('bottom_ann', np.nan),
            'Sharpe': ret_stats.get('sharpe', np.nan),
            'Max_DD': stab_stats.get('max_dd', np.nan),
            'Turnover_q1': stab_stats.get('turnover_top', np.nan),
            'Turnover_q5': stab_stats.get('turnover_bottom', np.nan),
            'Autocorr': stab_stats.get('autocorr', np.nan),
        }
        return pd.Series(results).round(4)

    def full_report(self):
        """控制台完整报告"""
        self.reporter.console_report()

    def full_report_html(self, path: str = 'factor_report.html'):
        """HTML 完整报告"""
        self.reporter.html_report(path)
        return path

    def plot_ic_ts(self):
        """绘制 IC 时间序列"""
        return self.visualizer.plot_ic_ts()

    def plot_ic_heatmap(self):
        """绘制 IC 月度热力图"""
        return self.visualizer.plot_ic_heatmap()

    def plot_quantile_returns(self):
        """绘制分组收益"""
        return self.visualizer.plot_quantile_returns()

    def plot_turnover(self):
        """绘制换手率"""
        return self.visualizer.plot_turnover()

    def plot_correlation_matrix(self, other_factors: Optional[pd.DataFrame] = None):
        """绘制相关性矩阵"""
        return self.visualizer.plot_correlation_matrix(other_factors)

    def plot_all(self):
        """生成全部图表"""
        return self.visualizer.plot_all()
