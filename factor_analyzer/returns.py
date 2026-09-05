"""
收益性分析模块
==============
分组收益 / 多空组合 / Fama-French Alpha / 信息比率
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict


class ReturnsAnalyzer:
    def __init__(self, fa):
        self.fa = fa

    def quantize_factor(self) -> pd.Series:
        """因子分位数分组"""
        factor = self.fa.factor_series
        q = self.fa.quantiles
        quantiles = {}
        for date in sorted(set(factor.index.get_level_values(0))):
            vals = factor.loc[date].dropna()
            if len(vals) < q * 5:
                continue
            labels = range(1, q + 1)
            try:
                q_result = pd.qcut(vals, q, labels=labels, duplicates='drop')
                for asset, label in q_result.items():
                    quantiles[(date, asset)] = label
            except:
                continue
        return pd.Series(quantiles, name='quantile')

    def quantile_summary(self, period: str = '1D') -> Dict:
        """分组收益汇总"""
        q_series = self.quantize_factor()
        fwd = self.fa.forward_returns[period]
        q = self.fa.quantiles

        common_idx = q_series.index.intersection(fwd.index)
        df = pd.DataFrame({'quantile': q_series.loc[common_idx], 'fwd_ret': fwd.loc[common_idx]})

        results = {}
        for g in range(1, q + 1):
            rets = df[df['quantile'] == g]['fwd_ret'].dropna()
            if len(rets) < 10:
                results[g] = {'mean': np.nan, 'std': np.nan, 'sharpe': np.nan, 'n': 0}
                continue
            mean_ret = rets.mean()
            std_ret = rets.std()
            # 年化
            ann_factor = 252 / int(period.replace('D', ''))
            ann_ret = (1 + mean_ret) ** ann_factor - 1
            ann_std = std_ret * np.sqrt(ann_factor)
            results[g] = {
                'mean': round(mean_ret, 6),
                'ann_ret': round(ann_ret, 4),
                'std': round(std_ret, 6),
                'sharpe': round(ann_ret / ann_std if ann_std > 0 else 0, 4),
                'n': int(len(rets)),
            }

        # 多空利差
        top_ret = df[df['quantile'] == q]['fwd_ret']
        bot_ret = df[df['quantile'] == 1]['fwd_ret']
        common_dates = set(top_ret.index.get_level_values(0)) & set(bot_ret.index.get_level_values(0))

        ls_rets = []
        for date in sorted(common_dates):
            t = top_ret.loc[date].mean()
            b = bot_ret.loc[date].mean()
            ls_rets.append(t - b)

        if ls_rets:
            ls_rets = np.array(ls_rets)
            ann_factor = 252 / int(period.replace('D', ''))
            ls_ann = (1 + ls_rets.mean()) ** ann_factor - 1
            ls_std = ls_rets.std() * np.sqrt(ann_factor)
            ls_sharpe = ls_ann / ls_std if ls_std > 0 else 0
            # 单调性检验
            q_rets = [results.get(g, {}).get('mean', np.nan) for g in range(1, q + 1)]
            monotonic = all(q_rets[i] <= q_rets[i+1] for i in range(len(q_rets)-1) if not np.isnan(q_rets[i]) and not np.isnan(q_rets[i+1]))

            results['long_short_ann'] = round(ls_ann, 4)
            results['long_short_sharpe'] = round(ls_sharpe, 4)
            results['monotonic'] = monotonic
            results['top_ann'] = results.get(q, {}).get('ann_ret', np.nan)
            results['bottom_ann'] = results.get(1, {}).get('ann_ret', np.nan)

        return results

    def factor_portfolio_returns(self, period: str = '1D') -> pd.Series:
        """因子加权多空组合日收益"""
        factor = self.fa.factor_series
        fwd = self.fa.forward_returns[period]
        q = self.fa.quantiles

        q_series = self.quantize_factor()

        daily_ret = {}
        for date in sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0))):
            f = factor.loc[date].dropna()
            r = fwd.loc[date].dropna()
            common = f.index.intersection(r.index)
            if len(common) < q * 3:
                continue

            # 做多高分位，做空低分位
            f_vals = f[common]
            f_rank = f_vals.rank(pct=True)
            weights = f_rank - 0.5  # 中心化, 多空组合
            weights = weights / weights.abs().sum()
            daily_ret[date] = (weights * r[common]).sum()

        return pd.Series(daily_ret, name='factor_portfolio')

    def ff_alpha(self, period: str = '1D') -> Dict:
        """Fama-French 风格 Alpha 分解（用市场因子简化版）"""
        pf_ret = self.factor_portfolio_returns(period)

        # 用等权市场收益作为 market factor 的代理
        fwd = self.fa.forward_returns[period]
        mkt_ret = {}
        for date in sorted(set(pf_ret.index) & set(fwd.index.get_level_values(0))):
            mkt_ret[date] = fwd.loc[date].mean()
        mkt = pd.Series(mkt_ret)

        common = sorted(set(pf_ret.index) & set(mkt.index))
        y = pf_ret[common]
        X = np.column_stack([np.ones(len(common)), mkt[common].values])

        try:
            coeff = np.linalg.lstsq(X, y.values, rcond=None)[0]
            alpha_daily = coeff[0]
            beta = coeff[1]

            resid = y.values - X @ coeff
            resid_std = resid.std()

            ann_factor = 252 / int(period.replace('D', ''))
            alpha_ann = (1 + alpha_daily) ** ann_factor - 1

            t_alpha = alpha_daily / (resid_std / np.sqrt(len(y))) if resid_std > 0 else 0
            p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=len(y)-2)) if len(y) > 2 else 1

            sharpe_ann = alpha_ann / (resid_std * np.sqrt(ann_factor)) if resid_std > 0 else 0

            return {
                'alpha_daily': round(alpha_daily, 6),
                'alpha_ann': round(alpha_ann, 4),
                'beta_mkt': round(beta, 4),
                't_alpha': round(t_alpha, 4),
                'p_alpha': round(p_alpha, 6),
                'significant': abs(t_alpha) > 2.0,
                'sharpe': round(sharpe_ann, 4),
            }
        except:
            return {'error': '回归失败'}

    def report(self) -> str:
        qs = self.quantile_summary('1D')
        ff = self.ff_alpha('1D')
        q = self.fa.quantiles

        lines = [
            "=" * 60,
            f"  [MONEY] 收益性分析 — {self.fa.factor_name}",
            "=" * 60,
            "",
            "── 分组收益 (1日前向) ──",
            f"  {'组别':<8} {'日均收益':>10} {'年化收益':>10} {'Sharpe':>8} {'样本数':>8}",
        ]
        for g in range(1, q + 1):
            r = qs.get(g, {})
            lines.append(f"  Q{g:<7} {r.get('mean',0):>10.4%} {r.get('ann_ret',0):>10.2%} {r.get('sharpe',0):>8.2f} {r.get('n',0):>8,d}")

        lines += [
            "",
            f"  Q{q}-Q1 多空年化: {qs.get('long_short_ann', 0):.2%}  {'[PASS] 显著' if qs.get('long_short_ann', 0) > 0.05 else ''}",
            f"  多空 Sharpe:     {qs.get('long_short_sharpe', 0):.2f}",
            f"  分组单调性:      {'[PASS] 单调' if qs.get('monotonic', False) else '[FAIL] 不单调'}",
            "",
            "── Fama-French Alpha ──",
        ]
        if 'error' not in ff:
            lines += [
                f"  Alpha(年化): {ff['alpha_ann']:.2%}",
                f"  Beta(市场):  {ff['beta_mkt']:.2f}",
                f"  t(alpha):    {ff['t_alpha']:.2f}  {'[PASS] 显著' if ff['significant'] else '[FAIL]'}",
                f"  Sharpe:      {ff['sharpe']:.2f}",
            ]
        lines.append("=" * 60)
        return '\n'.join(lines)
