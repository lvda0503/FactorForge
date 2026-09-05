"""
可视化模块
==========
所有图表的生成。基于 matplotlib，无需额外依赖。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式，可无GUI运行
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional
import os

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

plt.style.use('seaborn-v0_8-darkgrid')


class FactorVisualizer:
    def __init__(self, fa):
        self.fa = fa

    # ── IC 时间序列 ───────────────────────────────────────
    def plot_ic_ts(self, period: str = '1D', save_path: Optional[str] = None):
        ic = self.fa.prediction.compute_ic(period).dropna()
        ic.index = pd.to_datetime(ic.index)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        # 上图: IC + 1月均线
        ax = axes[0]
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.plot(ic.index, ic.values, alpha=0.3, color='steelblue', linewidth=0.5, label='Daily IC')
        ma = ic.rolling(21).mean()
        ax.plot(ma.index, ma.values, color='darkorange', linewidth=1.5, label='21D MA')
        ax.fill_between(ic.index, 0, ic.values,
                         where=(ic.values > 0), alpha=0.15, color='green')
        ax.fill_between(ic.index, 0, ic.values,
                         where=(ic.values < 0), alpha=0.15, color='red')
        ax.set_title(f'IC Time Series — {self.fa.factor_name} ({period} Forward)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Information Coefficient')
        ax.legend(loc='upper right')

        # 下图: 累计IC
        ax2 = axes[1]
        cum_ic = ic.cumsum()
        ax2.fill_between(cum_ic.index, 0, cum_ic.values,
                          where=(cum_ic.values > 0), alpha=0.3, color='green')
        ax2.fill_between(cum_ic.index, 0, cum_ic.values,
                          where=(cum_ic.values < 0), alpha=0.3, color='red')
        ax2.plot(cum_ic.index, cum_ic.values, color='steelblue', linewidth=1)
        ax2.set_ylabel('Cumulative IC')
        ax2.set_xlabel('Date')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── IC 月度热力图 ─────────────────────────────────────
    def plot_ic_heatmap(self, period: str = '1D', save_path: Optional[str] = None):
        matrix = self.fa.prediction.monthly_ic(period)

        fig, ax = plt.subplots(figsize=(14, max(3, len(matrix) * 0.5)))
        im = ax.imshow(matrix.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)

        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(matrix)))
        ax.set_yticklabels(matrix.index)

        for i in range(len(matrix)):
            for j in range(12):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=8, color='black' if abs(val) < 0.06 else 'white')

        ax.set_title(f'Monthly IC Heatmap — {self.fa.factor_name}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='IC')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── 分组收益 ───────────────────────────────────────────
    def plot_quantile_returns(self, period: str = '1D', save_path: Optional[str] = None):
        from .returns import ReturnsAnalyzer
        ra = ReturnsAnalyzer(self.fa)
        qs = ra.quantile_summary(period)
        q = self.fa.quantiles

        groups = list(range(1, q + 1))
        ann_rets = [qs.get(g, {}).get('ann_ret', 0) * 100 for g in groups]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d73027'] + ['#fc8d59'] + ['#fee08b'] * (q - 4) + ['#91cf60'] + ['#1a9850']

        bars = ax.bar(range(q), ann_rets, color=colors, edgecolor='white', linewidth=1.2)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xticks(range(q))
        ax.set_xticklabels([f'Q{g} (Low)' if g == 1 else f'Q{g} (High)' if g == q else f'Q{g}' for g in groups])
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title(f'Quantile Returns — {self.fa.factor_name} ({period})', fontsize=13, fontweight='bold')

        for bar, val in zip(bars, ann_rets):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3 * (1 if val >= 0 else -1),
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

        ls_ann = qs.get('long_short_ann', 0) * 100
        ax.text(0.98, 0.95, f'Long-Short: {ls_ann:.1f}%\nMonotonic: {"✓" if qs.get("monotonic") else "✗"}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── 换手率 ─────────────────────────────────────────────
    def plot_turnover(self, save_path: Optional[str] = None):
        turnover_df = self.fa.stability.turnover(1)
        autocorr = self.fa.stability.autocorrelation(1)
        q = self.fa.quantiles

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 换手率时序
        for g in [1, q]:
            if g in turnover_df.columns:
                ax1.plot(turnover_df.index, turnover_df[g].rolling(21).mean(),
                         label=f'Q{g} (21D MA)', linewidth=1.5)
        ax1.set_title('Quantile Turnover', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Turnover Rate')
        ax1.legend()
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # 自相关时序
        ax2.plot(autocorr.index, autocorr.values, color='steelblue', alpha=0.5, linewidth=0.5)
        ax2.plot(autocorr.index, autocorr.rolling(21).mean(), color='darkorange', linewidth=1.5, label='21D MA')
        ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Ideal (0.9)')
        ax2.set_title('Factor Rank Autocorrelation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Autocorrelation')
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── IC 衰减 ────────────────────────────────────────────
    def plot_ic_decay(self, save_path: Optional[str] = None):
        decay = self.fa.prediction.ic_decay()

        fig, ax = plt.subplots(figsize=(10, 5))
        periods_days = [int(c.replace('D', '')) for c in decay.index]
        ax.plot(periods_days, decay['IC_mean'].abs(), 'o-', color='steelblue',
                linewidth=2, markersize=8, label='|IC|')
        ax.fill_between(periods_days, 0, decay['IC_mean'].abs(), alpha=0.15, color='steelblue')

        # 标注半衰期
        for i, (_, row) in enumerate(decay.iterrows()):
            if pd.notna(row.get('half_life')):
                ax.axvline(x=int(row['half_life']), color='red', linestyle='--', alpha=0.5)
                ax.text(int(row['half_life']), ax.get_ylim()[1] * 0.5,
                        f"Half-life ≈ {int(row['half_life'])} days",
                        rotation=90, verticalalignment='center', color='red')
                break

        ax.set_xlabel('Forward Period (Days)')
        ax.set_ylabel('|IC|')
        ax.set_title(f'IC Decay — {self.fa.factor_name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── 相关性矩阵 ─────────────────────────────────────────
    def plot_correlation_matrix(self, other_factors: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        if other_factors is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, '需要传入 other_factors 参数', ha='center', va='center',
                    fontsize=12, transform=ax.transAxes)
            plt.show()
            return fig

        corr_matrix = self.fa.uniqueness.correlation_with(other_factors)

        fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix) * 0.8),
                                         max(6, len(corr_matrix) * 0.7)))
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr_matrix.index, fontsize=9)

        # 标注数值
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black')

        ax.set_title(f'Factor Correlation Matrix', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # ── 全部图表 ───────────────────────────────────────────
    def plot_all(self, save_dir: Optional[str] = None):
        """生成全部图表"""
        figs = {}
        figs['ic_ts'] = self.plot_ic_ts(save_path=os.path.join(save_dir, 'ic_ts.png') if save_dir else None)
        plt.close(figs['ic_ts'])

        figs['ic_heatmap'] = self.plot_ic_heatmap(save_path=os.path.join(save_dir, 'ic_heatmap.png') if save_dir else None)
        plt.close(figs['ic_heatmap'])

        figs['ic_decay'] = self.plot_ic_decay(save_path=os.path.join(save_dir, 'ic_decay.png') if save_dir else None)
        plt.close(figs['ic_decay'])

        figs['quantile_returns'] = self.plot_quantile_returns(save_path=os.path.join(save_dir, 'quantile_returns.png') if save_dir else None)
        plt.close(figs['quantile_returns'])

        figs['turnover'] = self.plot_turnover(save_path=os.path.join(save_dir, 'turnover.png') if save_dir else None)
        plt.close(figs['turnover'])

        return figs
