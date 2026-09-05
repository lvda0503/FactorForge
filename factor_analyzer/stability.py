"""
稳定性分析模块
==============
换手率 / 因子自相关 / 最大回撤 / Calmar / 滚动窗口稳定性
"""
import numpy as np
import pandas as pd
from typing import Dict


class StabilityAnalyzer:
    def __init__(self, fa):
        self.fa = fa

    def turnover(self, period: int = 1) -> pd.DataFrame:
        """分位数组换手率

        定义为：本期分组内不在上期分组内的股票占比
        """
        factor = self.fa.factor_series
        q = self.fa.quantiles

        # 按日分组
        q_by_date = {}
        dates = sorted(set(factor.index.get_level_values(0)))
        for date in dates:
            vals = factor.loc[date].dropna()
            if len(vals) < q * 5:
                continue
            try:
                q_result = pd.qcut(vals, q, labels=range(1, q + 1), duplicates='drop')
                q_by_date[date] = q_result
            except:
                continue

        dates = sorted(q_by_date.keys())
        turnover_results = {g: {} for g in range(1, q + 1)}

        for i in range(period, len(dates)):
            prev_date = dates[i - period]
            curr_date = dates[i]

            prev_q = q_by_date[prev_date]
            curr_q = q_by_date[curr_date]

            for g in range(1, q + 1):
                prev_set = set(prev_q[prev_q == g].index)
                curr_set = set(curr_q[curr_q == g].index)
                if len(curr_set) == 0:
                    continue
                # 新进入该组的股票占比
                new_entries = len(curr_set - prev_set)
                turnover = new_entries / len(curr_set)
                turnover_results[g][curr_date] = turnover

        df = pd.DataFrame(turnover_results).sort_index()
        return df

    def autocorrelation(self, period: int = 1) -> pd.Series:
        """因子排名自相关"""
        factor = self.fa.factor_series
        dates = sorted(set(factor.index.get_level_values(0)))

        autocorrs = {}
        for i in range(period, len(dates)):
            prev = factor.loc[dates[i - period]].dropna()
            curr = factor.loc[dates[i]].dropna()
            common = prev.index.intersection(curr.index)
            if len(common) < 30:
                continue
            autocorrs[dates[i]] = np.corrcoef(prev[common].rank(), curr[common].rank())[0, 1]

        return pd.Series(autocorrs, name='autocorr')

    def max_drawdown(self, period: str = '1D') -> Dict:
        """最大回撤分析（基于多空组合）"""
        from .returns import ReturnsAnalyzer
        ra = ReturnsAnalyzer(self.fa)
        pf_ret = ra.factor_portfolio_returns(period)

        if len(pf_ret) < 10:
            return {'max_dd': np.nan, 'max_dd_days': np.nan, 'calmar': np.nan}

        cum = (1 + pf_ret).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # 回撤持续天数
        if pd.notna(max_dd_idx):
            peak_idx = running_max[:max_dd_idx].idxmax()
            dd_start = pd.to_datetime(peak_idx)
            dd_end = pd.to_datetime(max_dd_idx)
            dd_days = (dd_end - dd_start).days
            # 恢复天数
            recovery = cum[max_dd_idx:] >= running_max[max_dd_idx]
            recovery_days = recovery[recovery].index[0] if recovery.any() else None
            if recovery_days is not None:
                recovery_days = (pd.to_datetime(recovery_days) - dd_end).days
        else:
            dd_days = np.nan
            recovery_days = np.nan

        ann_ret = (1 + pf_ret.mean()) ** 252 - 1
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        return {
            'max_dd': round(max_dd, 4),
            'max_dd_days': dd_days,
            'recovery_days': recovery_days,
            'calmar': round(calmar, 4) if not np.isnan(calmar) else np.nan,
            'ann_return': round(ann_ret, 4),
        }

    def rolling_stability(self, window: int = 120, period: str = '1D') -> pd.DataFrame:
        """滚动窗口IC — 检测因子是否持续有效"""
        ic_series = self.fa.prediction.compute_ic(period)
        ic_series.index = pd.to_datetime(ic_series.index)

        rolling = pd.DataFrame({
            'IC_mean': ic_series.rolling(window).mean(),
            'IC_std': ic_series.rolling(window).std(),
        })
        rolling['IC_IR'] = rolling['IC_mean'] / rolling['IC_std']
        return rolling

    def summary(self) -> Dict:
        """稳定性汇总指标"""
        turnover_df = self.turnover(1)
        autocorr = self.autocorrelation(1)
        dd = self.max_drawdown('1D')

        top_turnover = turnover_df[self.fa.quantiles].mean().mean() if len(turnover_df) > 0 else np.nan
        bot_turnover = turnover_df[1].mean().mean() if len(turnover_df) > 0 else np.nan

        return {
            'turnover_top': round(top_turnover, 4),
            'turnover_bottom': round(bot_turnover, 4),
            'autocorr': round(autocorr.mean(), 4) if len(autocorr) > 0 else np.nan,
            'max_dd': dd['max_dd'],
            'max_dd_days': dd['max_dd_days'],
            'calmar': dd['calmar'],
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            "=" * 60,
            f"  [LOOP] 稳定性分析 — {self.fa.factor_name}",
            "=" * 60,
            "",
            f"  顶级换手率:     {s['turnover_top']:.1%}  {'[PASS] 低' if s['turnover_top'] < 0.3 else '[WARN] 高' if s['turnover_top'] < 0.5 else '[FAIL] 极高'}",
            f"  底级换手率:     {s['turnover_bottom']:.1%}",
            f"  排名自相关:     {s['autocorr']:.3f}  {'[PASS] 稳定' if 0.8 < s['autocorr'] < 0.98 else '[WARN]'}",
            f"  最大回撤:       {s['max_dd']:.1%}  (持续 {s['max_dd_days']} 天)" if not np.isnan(s['max_dd']) else "  最大回撤: N/A",
            f"  Calmar比率:     {s['calmar']:.2f}" if not np.isnan(s.get('calmar', np.nan)) else "",
            "=" * 60,
        ]
        return '\n'.join(lines)
