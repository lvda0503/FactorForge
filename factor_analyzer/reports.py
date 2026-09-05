"""
报告生成模块
============
控制台报告 + HTML 报告
"""
import os
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd


class ReportGenerator:
    def __init__(self, fa):
        self.fa = fa

    def console_report(self):
        """打印完整控制台报告"""
        sections = [
            f"\n{'#' * 65}",
            f"#  FactorAnalyzer — 因子评估报告",
            f"#  因子: {self.fa.factor_name}",
            f"#  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"{'#' * 65}\n",
            self.fa.prediction.report(),
            self.fa.returns.report(),
            self.fa.stability.report(),
            self.fa.uniqueness.report(),
            self.fa.advanced.report(),
            f"\n{'#' * 65}",
            f"#  快速结论",
            f"{'#' * 65}",
            self._quick_conclusion(),
        ]
        report = '\n'.join(sections)
        print(report)
        return report

    def _quick_conclusion(self) -> str:
        """生成快速结论"""
        ic_s = self.fa.prediction.ic_summary('1D')
        ret_s = self.fa.returns.quantile_summary('1D')
        stab_s = self.fa.stability.summary()
        dsr = self.fa.advanced.deflated_sharpe('1D')

        score = 0
        items = []

        # IC
        ic_ir = abs(ic_s.get('IC_IR', 0))
        if ic_ir > 0.5:
            score += 3; items.append('[PASS]  IC_IR > 0.5: 预测力优秀')
        elif ic_ir > 0.3:
            score += 2; items.append('[WARN]  IC_IR > 0.3: 预测力合格')
        else:
            items.append('[FAIL]  IC_IR < 0.3: 预测力不足')

        # 多空
        ls = ret_s.get('long_short_ann', 0) or 0
        if ls > 0.10:
            score += 3; items.append('[PASS]  多空年化 > 10%: 收益显著')
        elif ls > 0.05:
            score += 2; items.append('[WARN]  多空年化 5-10%: 收益一般')
        else:
            items.append('[FAIL]  多空年化 < 5%: 收益偏弱')

        # 单调性
        if ret_s.get('monotonic', False):
            score += 1; items.append('[PASS]  分组单调: 因子区分度好')

        # 换手率
        turnover = stab_s.get('turnover_top', 1)
        if turnover < 0.3:
            score += 2; items.append('[PASS]  低换手: 交易成本可控')
        elif turnover < 0.5:
            score += 1; items.append('[WARN]  中换手: 需考虑成本')
        else:
            items.append('[FAIL]  高换手: 实盘收益将大打折扣')

        # DSR
        dsr_val = dsr.get('DSR', 0)
        if dsr_val > 0.95:
            score += 1; items.append('[PASS]  DSR 显著: 非过拟合')

        # 总评
        if score >= 8:
            grade = '[BEST]  A级 — 优秀因子，可直接用于组合'
        elif score >= 5:
            grade = '[OK]    B级 — 有潜力，建议优化或组合使用'
        elif score >= 3:
            grade = '[WEAK]  C级 — 勉强可用，需大幅改进'
        else:
            grade = '[DROP]   D级 — 不建议使用'

        lines = [f"  总分: {score}/10  |  等级: {grade}", ""] + [f"  {item}" for item in items]
        return '\n'.join(lines)

    def html_report(self, path: str = 'factor_report.html'):
        """生成 HTML 完整报告"""
        import matplotlib.pyplot as plt

        # 生成图表并转为 base64
        figs = {}
        try:
            figs['ic_ts'] = self._fig_to_base64(self.fa.visualizer.plot_ic_ts())
            plt.close('all')
        except: figs['ic_ts'] = ''
        try:
            figs['ic_heatmap'] = self._fig_to_base64(self.fa.visualizer.plot_ic_heatmap())
            plt.close('all')
        except: figs['ic_heatmap'] = ''
        try:
            figs['ic_decay'] = self._fig_to_base64(self.fa.visualizer.plot_ic_decay())
            plt.close('all')
        except: figs['ic_decay'] = ''
        try:
            figs['quantile'] = self._fig_to_base64(self.fa.visualizer.plot_quantile_returns())
            plt.close('all')
        except: figs['quantile'] = ''
        try:
            figs['turnover'] = self._fig_to_base64(self.fa.visualizer.plot_turnover())
            plt.close('all')
        except: figs['turnover'] = ''

        # 数据摘要
        ic_s = self.fa.prediction.ic_summary('1D')
        ret_s = self.fa.returns.quantile_summary('1D')
        stab_s = self.fa.stability.summary()
        decay = self.fa.prediction.ic_decay()
        regime = self.fa.advanced.regime_analysis('1D')
        dsr = self.fa.advanced.deflated_sharpe('1D')

        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>因子评估报告 — {self.fa.factor_name}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Microsoft YaHei',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#f5f5f5;color:#333}}
  h1{{border-bottom:3px solid #2c3e50;padding-bottom:10px}}
  h2{{color:#2c3e50;border-left:4px solid #3498db;padding-left:10px;margin-top:30px}}
  .card{{background:white;border-radius:8px;padding:20px;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin:10px 0}}
  .metric{{background:#f8f9fa;border-radius:6px;padding:12px;text-align:center}}
  .metric .val{{font-size:1.6em;font-weight:bold}}
  .metric .lbl{{font-size:.8em;color:#666;margin-top:4px}}
  .good{{color:#27ae60}} .warn{{color:#e67e22}} .bad{{color:#e74c3c}}
  table{{width:100%;border-collapse:collapse;margin:10px 0}}
  th{{background:#2c3e50;color:white;padding:8px 12px;text-align:left}}
  td{{padding:7px 12px;border-bottom:1px solid #eee}}
  tr:hover{{background:#f8f9fa}}
  img{{max-width:100%;border-radius:4px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,0.1)}}
  .conclusion{{background:linear-gradient(135deg,#2c3e50,#3498db);color:white;padding:20px;border-radius:8px;margin:20px 0}}
  .conclusion h2{{color:white;border:none;padding:0;margin-top:0}}
</style>
</head>
<body>
<h1>[OK]    因子评估报告: {self.fa.factor_name}</h1>
<p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 分位数: {self.fa.quantiles}组 | 前向周期: {self.fa.periods}</p>

<div class="conclusion">
<h2>快速结论</h2>
<pre style="font-size:1.05em;margin:0;white-space:pre-wrap">{self._quick_conclusion()}</pre>
</div>

<h2>[CHART] 核心指标</h2>
<div class="card">
<div class="grid">
<div class="metric"><div class="val good">{ic_s.get('IC_mean', 0):+.4f}</div><div class="lbl">IC 均值</div></div>
<div class="metric"><div class="val">{ic_s.get('IC_IR', 0):.2f}</div><div class="lbl">IC_IR</div></div>
<div class="metric"><div class="val">{ic_s.get('Rank_IC_mean', 0):+.4f}</div><div class="lbl">Rank IC</div></div>
<div class="metric"><div class="val">{ret_s.get('long_short_ann', 0)*100:.1f}%</div><div class="lbl">多空年化</div></div>
<div class="metric"><div class="val">{ret_s.get('long_short_sharpe', 0):.2f}</div><div class="lbl">多空 Sharpe</div></div>
<div class="metric"><div class="val">{stab_s.get('turnover_top', 0)*100:.0f}%</div><div class="lbl">顶级换手率</div></div>
<div class="metric"><div class="val">{stab_s.get('autocorr', 0):.3f}</div><div class="lbl">排名自相关</div></div>
<div class="metric"><div class="val">{stab_s.get('max_dd', 0)*100:.1f}%</div><div class="lbl">最大回撤</div></div>
</div>
</div>

<h2>[DOWN]  IC 衰减</h2>
<div class="card">{decay.to_html(float_format=lambda x: f'{{x:.4f}}')}</div>

<h2>[TEMP]  市场状态表现</h2>
<div class="card">{regime.to_html(float_format=lambda x: f'{{x:.4f}}')}</div>

<h2>[OK]    图表</h2>
<div class="card"><h3>IC 时间序列</h3>{f'<img src="data:image/png;base64,{{figs["ic_ts"]}}">' if figs['ic_ts'] else '<p>图表生成失败</p>'}</div>
<div class="card"><h3>IC 月度热力图</h3>{f'<img src="data:image/png;base64,{{figs["ic_heatmap"]}}">' if figs['ic_heatmap'] else ''}</div>
<div class="card"><h3>IC 衰减曲线</h3>{f'<img src="data:image/png;base64,{{figs["ic_decay"]}}">' if figs['ic_decay'] else ''}</div>
<div class="card"><h3>分组收益</h3>{f'<img src="data:image/png;base64,{{figs["quantile"]}}">' if figs['quantile'] else ''}</div>
<div class="card"><h3>换手率与自相关</h3>{f'<img src="data:image/png;base64,{{figs["turnover"]}}">' if figs['turnover'] else ''}</div>

<p style="text-align:center;color:#999;margin-top:30px">FactorAnalyzer · 自动生成 · {datetime.now().strftime('%Y-%m-%d')}</p>
</body></html>'''

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return path

    def _fig_to_base64(self, fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
