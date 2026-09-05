"""Drawdown root cause analysis"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, r'd:\JoinQuant\quant_env')
from factor_informed_rl.data.market_context import MarketContext

LOG_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/portfolio_logs"

for sn in ["Value-Defensive", "Quality-Offensive"]:
    df_daily = pd.read_csv(f"{LOG_DIR}/{sn}_daily.csv", parse_dates=['date'])
    df_events = pd.read_csv(f"{LOG_DIR}/{sn}_events.csv", parse_dates=['date'])

    vals = df_daily['total'].values
    cum = vals / vals[0]
    peak = np.maximum.accumulate(cum)

    # 找到最大回撤区间
    dd = cum / peak - 1
    dd_end = np.argmin(dd)
    dd_start = np.argmax(cum[:dd_end+1])
    dd_mid = np.argmin(cum[dd_start:dd_end+1]) + dd_start

    print(f"\n{'='*60}")
    print(f"  {sn} — Drawdown Analysis")
    print(f"{'='*60}")

    print(f"\n  最大回撤: {dd[dd_end]:.1%}")
    print(f"  峰值日期: {df_daily['date'].iloc[dd_start].date()} (val={vals[dd_start]:,.0f})")
    print(f"  谷底日期: {df_daily['date'].iloc[dd_mid].date()} (val={vals[dd_mid]:,.0f})")
    print(f"  恢复/结束: {df_daily['date'].iloc[dd_end].date()} (val={vals[dd_end]:,.0f})")

    # 关键: 回撤期间持有哪些股票?
    dd_phase_start = df_daily['date'].iloc[dd_start]
    dd_phase_mid   = df_daily['date'].iloc[dd_mid]
    dd_phase_end   = df_daily['date'].iloc[dd_end]

    for label, d1, d2 in [("上升期(峰值前1年)", dd_phase_start - pd.DateOffset(years=1), dd_phase_start),
                            ("回撤期(顶峰→谷底)", dd_phase_start, dd_phase_mid),
                            ("恢复期(谷底→结束)", dd_phase_mid, dd_phase_end)]:
        sub = df_events[(df_events['date']>=d1)&(df_events['date']<=d2)&(df_events['event'].isin(['BUY','SELL']))]
        if len(sub)==0: continue
        # 交易最活跃的股票
        top_codes = sub['code'].value_counts().head(8)
        buys = sub[sub['event']=='BUY'].groupby('code')['amount'].sum()
        sells = sub[sub['event']=='SELL'].groupby('code')['amount'].sum()
        net = pd.DataFrame({'buy':buys,'sell':sells}).fillna(0)
        net['net'] = net['buy'] - net['sell']
        print(f"\n  [{label}] ({d1.date()}→{d2.date()})")
        for code in top_codes.index[:5]:
            n = net.loc[code] if code in net.index else pd.Series({'buy':0,'sell':0,'net':0})
            print(f"    {code}: buy={n['buy']:,.0f} sell={n['sell']:,.0f} net={n['net']:+,.0f}")

    # 回撤期间市场环境
    csi = MarketContext().csi300
    for label, d1, d2 in [("上升期", dd_phase_start-pd.DateOffset(years=1), dd_phase_start),
                            ("回撤期", dd_phase_start, dd_phase_mid),
                            ("恢复期", dd_phase_mid, dd_phase_end)]:
        csi_sub = csi[(csi.index>=d1)&(csi.index<=d2)]
        if len(csi_sub)>1:
            csi_r = csi_sub['close'].iloc[-1]/csi_sub['close'].iloc[0]-1
            print(f"\n  CSI300 {label}: {csi_r:+.1%}")

    # 回撤期间 agent 行为
    dd_events = df_events[(df_events['date']>=dd_phase_start)&(df_events['date']<=dd_phase_end)]
    buys = len(dd_events[dd_events['event']=='BUY'])
    sells = len(dd_events[dd_events['event']=='SELL'])
    print(f"\n  Agent during DD: {buys} buys, {sells} sells (ratio={buys/max(sells,1):.1f})")

    # FI-PPO信号分析
    if 'fi_action' in dd_events.columns:
        fi_buy = dd_events[dd_events['event']=='BUY']['fi_action'].dropna()
        fi_sell = dd_events[dd_events['event']=='SELL']['fi_action'].dropna()
        print(f"  FI-PPO buy signal avg: {fi_buy.mean():+.3f}")
        print(f"  FI-PPO sell signal avg: {fi_sell.mean():+.3f}")

print("\n分析完成")
