"""
高级分析模块
============
Deflated Sharpe Ratio / PBO (回测过拟合概率) / 市场状态分析 / Bootstrapping
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


class AdvancedAnalyzer:
    def __init__(self, fa):
        self.fa = fa

    # ── Deflated Sharpe Ratio ──────────────────────────────
    def deflated_sharpe(self, period: str = '1D', n_trials: int = 100) -> Dict:
        """Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

        考虑多次试验后的真实显著性：如果试了N个因子，
        即使最好的那个 Sharpe=1.5 也可能只是运气。

        Parameters
        ----------
        n_trials : int, 假设一共测试过的因子数量（包括未采纳的）
        """
        from .returns import ReturnsAnalyzer
        ra = ReturnsAnalyzer(self.fa)
        pf_ret = ra.factor_portfolio_returns(period)

        if len(pf_ret) < 50:
            return {'error': '数据不足'}

        n = len(pf_ret)
        sr = pf_ret.mean() / pf_ret.std() * np.sqrt(252)

        # 偏度和峰度修正
        skew = stats.skew(pf_ret)
        kurt = stats.kurtosis(pf_ret)

        # Probabilistic Sharpe Ratio
        sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / n)
        psr = stats.norm.cdf(sr / sr_std) if sr_std > 0 else 0

        # Deflated Sharpe Ratio
        # E[max(SR)] ≈ √(2·ln(N)) × SR_std  (extreme value theory)
        expected_max = np.sqrt(2 * np.log(max(n_trials, 1))) * sr_std
        dsr = stats.norm.cdf((sr - expected_max) / sr_std) if sr_std > 0 else 0

        return {
            'sharpe_ann': round(sr, 4),
            'sharpe_std': round(sr_std, 4),
            'skew': round(skew, 4),
            'kurtosis': round(kurt, 4),
            'PSR': round(psr, 4),
            'DSR': round(dsr, 4),  # 考虑n_trials次试验的真实显著水平
            'n_trials': n_trials,
            'conclusion': '[PASS] 统计显著' if dsr > 0.95 else '[WARN] 边缘显著' if dsr > 0.80 else '[FAIL] 可能过拟合',
        }

    # ── PBO (Probability of Backtest Overfitting) ──────────
    def pbo(self, period: str = '1D', n_splits: int = 10, n_subsets: int = 16) -> Dict:
        """PBO 过拟合概率 (Bailey et al., 2017)

        原理：将样本随机分成训练集和测试集，
        如果训练集最优参数在测试集上的排名不稳定 → 过拟合

        简化实现：对时间序列做随机子集抽样
        """
        from .returns import ReturnsAnalyzer
        ra = ReturnsAnalyzer(self.fa)
        pf_ret = ra.factor_portfolio_returns(period)

        if len(pf_ret) < 100:
            return {'error': '数据不足（需要100+交易日）'}

        n = len(pf_ret)
        subset_size = n // 2

        # 生成随机子集
        np.random.seed(42)
        train_sharpe_rank = []
        test_sharpe_rank = []

        for _ in range(n_subsets * 10):
            idx = np.random.choice(n, subset_size, replace=False)
            train_idx = idx[:subset_size // 2]
            test_idx = idx[subset_size // 2:]

            train_sr = pf_ret.iloc[train_idx].mean() / pf_ret.iloc[train_idx].std()
            test_sr = pf_ret.iloc[test_idx].mean() / pf_ret.iloc[test_idx].std()
            train_sharpe_rank.append(train_sr)
            test_sharpe_rank.append(test_sr)

        # 计算 rank 相关性
        train_rank = stats.rankdata(train_sharpe_rank)
        test_rank = stats.rankdata(test_sharpe_rank)

        # PBO = 训练集最优在测试集中不在前50%的概率
        best_train_idx = np.argmax(train_sharpe_rank)
        pbo = 1 - test_rank[best_train_idx] / len(test_rank)

        return {
            'PBO': round(pbo, 4),
            'n_subsets': n_subsets * 10,
            'conclusion': '[PASS] 低过拟合风险' if pbo < 0.1 else '[WARN] 中等风险' if pbo < 0.3 else '[FAIL] 高风险（回测结果不可靠）',
        }

    # ── 市场状态分析 ───────────────────────────────────────
    def regime_analysis(self, period: str = '1D') -> pd.DataFrame:
        """不同市场状态下的因子表现

        状态定义：
          - UP:   市场20日均线上行
          - DOWN: 市场20日均线下行
          - HIGH_VOL: 市场20日波动率 > 历史中位数
          - LOW_VOL:  市场20日波动率 <= 历史中位数
        """
        factor = self.fa.factor_series
        fwd = self.fa.forward_returns[period]

        # 市场收益代理：每日截面均值
        dates = sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0)))
        mkt_ret = {}
        for d in dates:
            mkt_ret[d] = fwd.loc[d].mean()
        mkt = pd.Series(mkt_ret).sort_index()

        # 市场状态
        mkt_ma20 = mkt.rolling(20).mean()
        mkt_vol20 = mkt.rolling(20).std()
        med_vol = mkt_vol20.median()

        regimes = {
            'UP': mkt_ma20 > mkt_ma20.shift(1),
            'DOWN': mkt_ma20 <= mkt_ma20.shift(1),
            'HIGH_VOL': mkt_vol20 > med_vol,
            'LOW_VOL': mkt_vol20 <= med_vol,
        }

        results = {}
        for regime_name, regime_mask in regimes.items():
            regime_dates = set(regime_mask[regime_mask].index) & set(dates)
            ics = []
            for d in regime_dates:
                f = factor.loc[d].dropna()
                r = fwd.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 30:
                    continue
                ic = stats.spearmanr(f[common], r[common])[0]
                ics.append(ic)

            if ics:
                ics = np.array(ics)
                results[regime_name] = {
                    'IC_mean': round(ics.mean(), 6),
                    'IC_IR': round(ics.mean() / ics.std(), 4) if ics.std() > 0 else 0,
                    'IC_pos_ratio': round((ics > 0).mean(), 4),
                    'n_days': len(ics),
                }

        return pd.DataFrame(results).T

    # ── Bootstrapping 置信区间 ─────────────────────────────
    def bootstrap_confidence(self, period: str = '1D', n_boot: int = 1000) -> Dict:
        """Bootstrap 法估计 IC 和 Sharpe 的置信区间"""
        ic = self.fa.prediction.compute_ic(period).dropna()
        from .returns import ReturnsAnalyzer
        ra = ReturnsAnalyzer(self.fa)
        pf_ret = ra.factor_portfolio_returns(period).dropna()

        np.random.seed(42)
        ic_boot = []
        sr_boot = []

        for _ in range(n_boot):
            boot_idx = np.random.choice(len(ic), len(ic), replace=True)
            ic_boot.append(ic.iloc[boot_idx].mean())
            if len(pf_ret) > 0:
                boot_idx2 = np.random.choice(len(pf_ret), len(pf_ret), replace=True)
                r = pf_ret.iloc[boot_idx2]
                sr_boot.append(r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)

        ic_boot = np.array(ic_boot)
        sr_boot = np.array(sr_boot) if sr_boot else np.array([0])

        return {
            'IC_mean': round(ic.mean(), 6),
            'IC_95CI_low': round(np.percentile(ic_boot, 2.5), 6),
            'IC_95CI_high': round(np.percentile(ic_boot, 97.5), 6),
            'Sharpe_ann': round(pf_ret.mean() / pf_ret.std() * np.sqrt(252), 4) if pf_ret.std() > 0 else 0,
            'Sharpe_95CI_low': round(np.percentile(sr_boot, 2.5), 4),
            'Sharpe_95CI_high': round(np.percentile(sr_boot, 97.5), 4),
        }

    # ── 截面回归 (含控制变量) ───────────────────────────────
    def cross_sectional_regression(self, controls: Optional[pd.DataFrame] = None, period: str = '1D') -> Dict:
        """逐日截面回归，报告系数均值和显著性

        回归模型: future_return = α + β₁·factor + β₂·market_cap + β₃·industry + ε
        """
        factor = self.fa.factor_series
        fwd = self.fa.forward_returns[period]
        dates = sorted(set(factor.index.get_level_values(0)) & set(fwd.index.get_level_values(0)))

        betas = []
        for date in dates:
            f = factor.loc[date].dropna()
            r = fwd.loc[date].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                continue

            X = np.column_stack([np.ones(len(common)), f[common].values])
            if controls is not None and date in controls.index.get_level_values(0):
                ctrl = controls.loc[date]
                common2 = common.intersection(ctrl.index)
                if len(common2) >= 30:
                    X = np.column_stack([X, ctrl.loc[common2].values])
                    y = r[common2].values
                else:
                    y = r[common].values
            else:
                y = r[common].values

            try:
                b = np.linalg.lstsq(X[:len(y)], y, rcond=None)[0]
                betas.append({'date': date, 'alpha': b[0], 'beta': b[1]})
            except:
                continue

        if not betas:
            return {'error': '数据不足'}

        df = pd.DataFrame(betas).set_index('date')
        n = len(df)

        results = {}
        for col in ['beta', 'alpha']:
            mean_v = df[col].mean()
            std_v = df[col].std()
            t = mean_v / (std_v / np.sqrt(n)) if std_v > 0 else 0
            results[col] = {
                'mean': round(mean_v, 6),
                'std': round(std_v, 4),
                't': round(t, 2),
                'significant_5pct': abs(t) > 1.96,
            }

        return results

    def report(self, n_trials: int = 100) -> str:
        dsr = self.deflated_sharpe('1D', n_trials)
        pbo = self.pbo('1D')
        regime = self.regime_analysis('1D')
        boot = self.bootstrap_confidence('1D')

        lines = [
            "=" * 60,
            f"  [LAB] 高级分析 — {self.fa.factor_name}",
            "=" * 60,
            "",
            "── Deflated Sharpe Ratio ──",
            f"  年化Sharpe:       {dsr.get('sharpe_ann', 'N/A')}",
            f"  PSR:              {dsr.get('PSR', 'N/A')}  (Probabilistic SR)",
            f"  DSR:              {dsr.get('DSR', 'N/A')}  (Deflated, {n_trials}次试验)",
            f"  结论:             {dsr.get('conclusion', 'N/A')}",
            "",
            "── PBO (过拟合概率) ──",
        ]
        if 'error' not in pbo:
            lines += [
                f"  PBO:              {pbo['PBO']:.2%}",
                f"  结论:             {pbo['conclusion']}",
            ]
        else:
            lines.append(f"  {pbo['error']}")

        lines += [
            "",
            "── Bootstrap 95% 置信区间 ──",
            f"  IC:   [{boot['IC_95CI_low']:.4f}, {boot['IC_95CI_high']:.4f}]",
            f"  Sharpe: [{boot['Sharpe_95CI_low']:.2f}, {boot['Sharpe_95CI_high']:.2f}]",
            "",
            "── 市场状态分析 ──",
        ]
        for regime_name, row in regime.iterrows():
            lines.append(f"  {regime_name:12s}: IC={row['IC_mean']:+.4f}  IR={row['IC_IR']:+.3f}  N={int(row['n_days'])}")

        lines.append("=" * 60)
        return '\n'.join(lines)
