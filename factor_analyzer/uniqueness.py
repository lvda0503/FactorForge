"""
独特性分析模块
==============
因子相关性 / VIF (方差膨胀因子) / 边际贡献 / 正交性检验
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class UniquenessAnalyzer:
    def __init__(self, fa):
        self.fa = fa

    def correlation_with(self, other_factors: pd.DataFrame) -> pd.DataFrame:
        """计算本因子与其他因子的相关性矩阵"""
        factor = self.fa.factor_series
        all_factors = other_factors.copy()

        if self.fa.factor_name not in all_factors.columns:
            all_factors[self.fa.factor_name] = factor

        # 按日期对齐，计算横截面平均相关
        dates = sorted(set(factor.index.get_level_values(0)))
        corr_results = {}

        for f1 in all_factors.columns:
            corr_results[f1] = {}
            for f2 in all_factors.columns:
                cors = []
                for date in dates:
                    if date not in all_factors.index.get_level_values(0):
                        continue
                    v1 = all_factors.loc[date, f1].dropna() if isinstance(all_factors.index, pd.MultiIndex) else all_factors.loc[date]
                    v2 = all_factors.loc[date, f2].dropna() if isinstance(all_factors.index, pd.MultiIndex) else all_factors.loc[date]
                    # 简化：直接用所有日期的合并数据
                # 简化版本：合并全部数据计算
                s1 = all_factors[f1].dropna() if not isinstance(all_factors.index, pd.MultiIndex) else all_factors.xs('2024-01-01', level=0) if '2024-01-01' in all_factors.index.get_level_values(0) else all_factors.groupby(level=1).mean().iloc[:, all_factors.columns.get_loc(f1)]

        # 更简单实用的做法：用因子日截面值计算平均相关
        return self._practical_corr(other_factors)

    def _practical_corr(self, other_factors: pd.DataFrame) -> pd.DataFrame:
        """实用的因子相关计算：逐日计算截面相关后取均值"""
        factor = self.fa.factor_series
        dates = sorted(set(factor.index.get_level_values(0)))

        # 收集每日所有因子的截面值
        all_cols = list(other_factors.columns) + [self.fa.factor_name]
        daily_data = {col: {} for col in all_cols}

        for date in dates:
            f_val = factor.loc[date].dropna() if date in factor.index.get_level_values(0) else None
            if f_val is not None:
                daily_data[self.fa.factor_name][date] = f_val

            if isinstance(other_factors.index, pd.MultiIndex):
                for col in other_factors.columns:
                    if date in other_factors.index.get_level_values(0):
                        v = other_factors.loc[date, col].dropna()
                        if len(v) > 0:
                            daily_data[col][date] = v

        # 计算平均相关矩阵
        corr_matrix = pd.DataFrame(np.eye(len(all_cols)), index=all_cols, columns=all_cols)
        for i, f1 in enumerate(all_cols):
            for j, f2 in enumerate(all_cols):
                if i >= j:
                    continue
                cors = []
                common_dates = set(daily_data[f1].keys()) & set(daily_data[f2].keys())
                for d in common_dates:
                    s1 = daily_data[f1][d]
                    s2 = daily_data[f2][d]
                    common = s1.index.intersection(s2.index)
                    if len(common) < 30:
                        continue
                    cors.append(np.corrcoef(s1[common], s2[common])[0, 1])
                if cors:
                    corr_matrix.loc[f1, f2] = corr_matrix.loc[f2, f1] = np.mean(cors)

        return corr_matrix.round(4)

    def vif(self, other_factors: pd.DataFrame) -> pd.Series:
        """计算本因子对其他因子的 VIF (方差膨胀因子)"""
        factor = self.fa.factor_series
        dates = sorted(set(factor.index.get_level_values(0)))

        # 收集面板数据
        y_vals, X_rows = [], []
        for date in dates:
            if date not in factor.index.get_level_values(0):
                continue
            f = factor.loc[date].dropna()
            row = pd.DataFrame(index=f.index)
            row[self.fa.factor_name] = f

            if isinstance(other_factors.index, pd.MultiIndex):
                for col in other_factors.columns:
                    if date in other_factors.index.get_level_values(0):
                        row[col] = other_factors.loc[date, col]
            else:
                for col in other_factors.columns:
                    row[col] = other_factors[col]

            valid = row.dropna()
            if len(valid) < 30:
                continue
            y_vals.extend(valid[self.fa.factor_name].values)
            X_rows.extend(valid[list(other_factors.columns)].values)

        if len(y_vals) < 100:
            return pd.Series({'VIF': np.nan, 'R2': np.nan, 'note': '数据不足'})

        X = np.array(X_rows)
        y = np.array(y_vals)

        try:
            coeff = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
            y_pred = np.column_stack([np.ones(len(X)), X]) @ coeff
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            vif = 1 / (1 - r2) if r2 < 1 else np.inf
        except:
            r2 = np.nan
            vif = np.nan

        return pd.Series({
            'VIF': round(vif, 2),
            'R2': round(r2, 4),
            'multicollinear': '严重' if vif > 10 else '中度' if vif > 5 else '良好'
        })

    def marginal_contribution(self, other_factors: pd.DataFrame, period: str = '1D') -> Dict:
        """计算加入本因子后的边际 IC 提升"""
        fwd = self.fa.forward_returns[period]
        factor = self.fa.factor_series
        dates = sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0)))

        # 基准：只用其他因子
        base_ics = []
        full_ics = []

        for date in dates:
            r = fwd.loc[date].dropna()

            # 构建特征矩阵
            row = pd.DataFrame(index=r.index)
            row[self.fa.factor_name] = factor.loc[date]

            if isinstance(other_factors.index, pd.MultiIndex):
                for col in other_factors.columns:
                    if date in other_factors.index.get_level_values(0):
                        row[col] = other_factors.loc[date, col]

            valid = row.dropna()
            if len(valid) < 30:
                continue

            # 基准模型（不含本因子）
            X_base = valid[list(other_factors.columns)].values
            # 完整模型（含本因子）
            X_full = np.column_stack([X_base, valid[self.fa.factor_name].values])

            y = r[valid.index].values

            try:
                b_coeff = np.linalg.lstsq(np.column_stack([np.ones(X_base.shape[0]), X_base]), y, rcond=None)[0]
                b_pred = np.column_stack([np.ones(X_base.shape[0]), X_base]) @ b_coeff
                base_ic = np.corrcoef(b_pred, y)[0, 1]
                base_ics.append(base_ic)

                f_coeff = np.linalg.lstsq(np.column_stack([np.ones(X_full.shape[0]), X_full]), y, rcond=None)[0]
                f_pred = np.column_stack([np.ones(X_full.shape[0]), X_full]) @ f_coeff
                full_ic = np.corrcoef(f_pred, y)[0, 1]
                full_ics.append(full_ic)
            except:
                continue

        if not base_ics:
            return {'error': '数据不足'}

        base_mean = np.mean(base_ics)
        full_mean = np.mean(full_ics)
        delta = full_mean - base_mean

        return {
            'base_IC': round(base_mean, 6),
            'full_IC': round(full_mean, 6),
            'delta_IC': round(delta, 6),
            'improvement': f'{delta/base_mean:.1%}' if abs(base_mean) > 0 else 'N/A',
            'valuable': delta > 0.002  # IC提升超过0.002视为有价值
        }

    def report(self) -> str:
        lines = [
            "=" * 60,
            f"  🔗 独特性分析 — {self.fa.factor_name}",
            "=" * 60,
            "  (需要传入 other_factors 参数进行完整分析)",
            "  使用: fa.uniqueness.correlation_with(other_factors)",
            "       fa.uniqueness.vif(other_factors)",
            "       fa.uniqueness.marginal_contribution(other_factors)",
            "=" * 60,
        ]
        return '\n'.join(lines)
