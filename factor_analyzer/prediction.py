"""
预测力分析模块
==============
IC / Rank IC / IC_IR / IC衰减 / Fama-MacBeth 回归
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class PredictionAnalyzer:
    def __init__(self, fa):
        self.fa = fa

    # ── IC / Rank IC ──────────────────────────────────────
    def compute_ic(self, period: str = '1D', method: str = 'spearman') -> pd.Series:
        """逐日计算IC

        Parameters
        ----------
        period : str, 前向收益周期列名, 如 '1D' '5D' '10D'
        method : str, 'pearson' or 'spearman'

        Returns
        -------
        pd.Series, index=date, 每日IC值
        """
        fwd = self.fa.forward_returns[period]
        factor = self.fa.factor_series

        ic_series = {}
        for date in sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0))):
            f = factor.loc[date].dropna()
            r = fwd.loc[date].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                continue
            if method == 'spearman':
                ic_series[date] = stats.spearmanr(f[common], r[common])[0]
            else:
                ic_series[date] = stats.pearsonr(f[common], r[common])[0]

        return pd.Series(ic_series, name=f'IC_{period}')

    def ic_summary(self, period: str = '1D') -> Dict:
        """IC 汇总统计"""
        ic = self.compute_ic(period)
        rank_ic = self.compute_ic(period, method='spearman')

        n = len(ic.dropna())
        ic_mean = ic.mean()
        ic_std = ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        rank_ic_mean = rank_ic.mean()
        rank_ic_std = rank_ic.std()
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0

        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if n > 1 else 1

        return {
            'n_obs': n,
            'IC_mean': round(ic_mean, 6),
            'IC_std': round(ic_std, 6),
            'IC_IR': round(ic_ir, 4),
            'IC_pos_ratio': round((ic > 0).mean(), 4),
            'IC_skew': round(ic.skew(), 4),
            'IC_kurtosis': round(ic.kurtosis(), 4),
            't_stat': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'Rank_IC_mean': round(rank_ic_mean, 6),
            'Rank_IC_std': round(rank_ic_std, 6),
            'Rank_IC_IR': round(rank_ic_ir, 4),
        }

    # ── IC 衰减曲线 ───────────────────────────────────────
    def ic_decay(self) -> pd.DataFrame:
        """多周期IC衰减分析"""
        results = {}
        for period in self.fa.forward_returns.columns:
            ic = self.compute_ic(period)
            results[period] = {
                'IC_mean': ic.mean(),
                'IC_IR': ic.mean() / ic.std() if ic.std() > 0 else 0,
                'IC_pos_ratio': (ic > 0).mean(),
            }
        df = pd.DataFrame(results).T
        df['half_life'] = np.nan
        # 估算半衰期：IC降到峰值一半时的周期
        if len(df) > 1:
            peak = df['IC_mean'].abs().max()
            for i, (period, row) in enumerate(df.iterrows()):
                if abs(row['IC_mean']) < peak / 2:
                    days = int(period.replace('D', ''))
                    df.at[period, 'half_life'] = days
                    break
        return df.round(6)

    # ── Fama-MacBeth 回归 ─────────────────────────────────
    def fama_macbeth(self, period: str = '1D', controls: pd.DataFrame = None) -> Dict:
        """Fama-MacBeth 两阶段回归

        Stage 1 (横截面): 每天回归 future_return = α + β·factor + γ·controls
        Stage 2 (时间序列): 对所有天的 β 做 t 检验

        Parameters
        ----------
        controls : pd.DataFrame, 控制变量 (MultiIndex, 同factor格式)
        """
        factor = self.fa.factor_series
        fwd = self.fa.forward_returns[period]

        betas = []
        for date in sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0))):
            f = factor.loc[date].dropna()
            r = fwd.loc[date].dropna()
            common = f.index.intersection(r.index)

            if controls is not None:
                ctrl_vals = controls.loc[date]
                common = common.intersection(ctrl_vals.index)

            if len(common) < 50:
                continue

            X = pd.DataFrame({'factor': f[common].values})
            if controls is not None:
                for col in controls.columns:
                    X[col] = ctrl_vals.loc[common, col].values
            X = np.column_stack([np.ones(len(X)), X.values])

            try:
                beta = np.linalg.lstsq(X, r[common].values, rcond=None)[0]
                betas.append({'date': date, 'alpha': beta[0], 'beta_factor': beta[1]})
            except:
                continue

        if not betas:
            return {'error': '回归失败，数据不足'}

        df = pd.DataFrame(betas).set_index('date')
        n = len(df)

        results = {}
        for col in ['alpha', 'beta_factor']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            t_val = mean_val / (std_val / np.sqrt(n)) if std_val > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=n-1)) if n > 1 else 1
            results[col] = {
                'mean': round(mean_val, 6),
                'std': round(std_val, 6),
                't_stat': round(t_val, 4),
                'p_value': round(p_val, 6),
                'significant': abs(t_val) > 2.0,
            }

        return results

    # ── 月度 IC 热力图数据 ─────────────────────────────────
    def monthly_ic(self, period: str = '1D') -> pd.DataFrame:
        """按月聚合IC，返回年月矩阵"""
        ic = self.compute_ic(period)
        ic.index = pd.to_datetime(ic.index)
        monthly = ic.resample('ME').mean()
        matrix = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'IC': monthly.values
        }).pivot_table(values='IC', index='year', columns='month', aggfunc='mean')
        return matrix

    # ── 报告 ──────────────────────────────────────────────
    def report(self) -> str:
        """控制台报告"""
        ic_s = self.ic_summary('1D')
        decay = self.ic_decay()
        fm = self.fama_macbeth('1D')

        lines = [
            "=" * 60,
            f"  [CHART] 预测力分析 — {self.fa.factor_name}",
            "=" * 60,
            "",
            "── IC 分析 (1日前向收益) ──",
            f"  IC均值(pearson):     {ic_s['IC_mean']:.4f}",
            f"  IC标准差:            {ic_s['IC_std']:.4f}",
            f"  IC_IR:               {ic_s['IC_IR']:.2f}  {'[PASS]' if abs(ic_s['IC_IR']) > 0.5 else '[WARN]' if abs(ic_s['IC_IR']) > 0.3 else '[FAIL]'}",
            f"  Rank IC均值:         {ic_s['Rank_IC_mean']:.4f}",
            f"  Rank IC_IR:          {ic_s['Rank_IC_IR']:.2f}",
            f"  IC>0比例:            {ic_s['IC_pos_ratio']:.1%}  {'[PASS]' if ic_s['IC_pos_ratio'] > 0.55 else '[WARN]'}",
            f"  t统计量:             {ic_s['t_stat']:.2f}  {'[PASS] 显著' if abs(ic_s['t_stat']) > 2.0 else '[FAIL] 不显著'}",
            f"  p值:                 {ic_s['p_value']:.4f}",
            "",
            "── IC 衰减 ──",
        ]
        for _, row in decay.iterrows():
            lines.append(f"  {_:5s}: IC={row['IC_mean']:+.4f}  IR={row['IC_IR']:+.3f}")

        if 'error' not in fm:
            lines += [
                "",
                "── Fama-MacBeth 回归 ──",
                f"  因子beta均值:  {fm['beta_factor']['mean']:.6f}",
                f"  t统计量:       {fm['beta_factor']['t_stat']:.2f}  {'[PASS] 显著' if fm['beta_factor']['significant'] else '[FAIL]'}",
                f"  alpha均值:     {fm['alpha']['mean']:.6f}",
                f"  alpha的t:      {fm['alpha']['t_stat']:.2f}",
            ]

        lines.append("=" * 60)
        return '\n'.join(lines)
