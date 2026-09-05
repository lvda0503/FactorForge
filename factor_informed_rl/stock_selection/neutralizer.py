"""
Barra 风险中性化 — 剔除行业和市值的影响
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class BarraNeutralizer:
    """行业 + 市值中性化

    对每个因子做截面回归取残差:
      factor_i = β_1 × log_mktcap + Σ β_j × industry_dummy_j + ε_i
      neutralized_i = ε_i  (残差 = 纯因子暴露)

    用法:
        bn = BarraNeutralizer(industry_map)
        bn.fit()  # 可选, 不必须
        factors_neutral = bn.neutralize(factor_dict, market_caps, date)
    """

    def __init__(self, industry_map: Dict[str, str] = None):
        """
        Args:
            industry_map: {code: "银行", "000001": "银行", ...}
        """
        self.industry_map = industry_map or {}

    def neutralize(
        self,
        factor_values: Dict[str, float],
        market_caps: Dict[str, float],
        min_stocks: int = 30,
    ) -> Dict[str, float]:
        """
        对一组股票的因子值做中性化

        Args:
            factor_values: {code: raw_factor_value}
            market_caps:    {code: market_cap}
            min_stocks:     截面最少股票数, 否则返回原始值

        Returns:
            {code: neutralized_factor_value}
        """
        codes = sorted(set(factor_values.keys()) &
                       set(market_caps.keys()))
        if len(codes) < min_stocks:
            return factor_values

        # 构建设计矩阵
        y = np.array([factor_values[c] for c in codes])
        n = len(codes)

        # 市值
        log_mkt = np.array([np.log(max(market_caps.get(c, 1e10), 1e8)) for c in codes])
        log_mkt = (log_mkt - log_mkt.mean()) / (log_mkt.std() + 1e-10)

        # 行业虚拟变量
        industries = sorted(set(self.industry_map.get(c, 'Other') for c in codes))
        n_ind = len(industries)
        ind_map = {ind: i for i, ind in enumerate(industries)}

        X_cols = [np.ones(n), log_mkt]
        X_cols += [
            np.array([1.0 if self.industry_map.get(c, 'Other') == ind else 0.0
                      for c in codes])
            for ind in industries[:-1]  # 留一个做基准
        ]
        X = np.column_stack(X_cols)

        # OLS 求残差
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
        except np.linalg.LinAlgError:
            residuals = y - y.mean()

        # Z-score 标准化残差
        if residuals.std() > 1e-10:
            residuals = (residuals - residuals.mean()) / residuals.std()

        return {c: float(residuals[i]) for i, c in enumerate(codes)}

    def batch_neutralize(
        self,
        factor_df: pd.DataFrame,   # MultiIndex (date, code) or dict of Series
        market_caps: pd.Series,    # MultiIndex (date, code)
    ) -> pd.DataFrame:
        """批量截面中性化"""
        if isinstance(factor_df, dict):
            factor_df = pd.DataFrame(factor_df)

        dates = sorted(set(factor_df.index.get_level_values(0)))
        result = {}

        for date in dates:
            f_vals = factor_df.loc[date].to_dict() if hasattr(factor_df.loc[date], 'to_dict') else {}
            m_vals = {}
            if isinstance(market_caps, pd.Series):
                try:
                    m_vals = market_caps.loc[date].to_dict()
                except:
                    pass

            neutral = self.neutralize(f_vals, m_vals)
            for code, val in neutral.items():
                result[(date, code)] = val

        idx = pd.MultiIndex.from_tuples(result.keys(), names=['date', 'code'])
        return pd.Series(result.values(), index=idx, name='neutralized')
