"""V10: Two strategies + detail logging"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, os, pickle
from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.stock_selection.neutralizer import BarraNeutralizer
from factor_informed_rl.stock_selection.scorer import FactorScorer
from factor_informed_rl.stock_selection.hard_filter import value_filter, quality_filter

market_ctx = MarketContext()
ind_map = pd.read_pickle("d:/JoinQuant/quant_env/data_cache/csi300_industry.pkl").to_dict()
neutralizer = BarraNeutralizer(ind_map); scorer = FactorScorer()
with open("d:/JoinQuant/quant_env/data_cache/csi300_factors.pkl",'rb') as f:
    fc = pickle.load(f)

CACHE = "d:/JoinQuant/quant_env/data_cache/csi300"
MODEL_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v7_models"
START, END = "2021-01-01", "2026-06-30"
dates = sorted(set(d for (d,_,_) in fc if pd.Timestamp(START)<=d<=pd.Timestamp(END)))
OUT_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/portfolio_logs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_STOCKS, TOP_N = 6, 200
OBS, BLD, EXT = 5, 8, 10
EXIT_RANK, MIN_HOLD = 30, 60
MAX_POS_BULL = 0.18; MAX_POS_BEAR = 0.08; BUILD_SPEED = 1.0

STRATEGIES = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "filter_fn": value_filter,"model":"Value-Defensive_600519_fi.pt"},
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "filter_fn": quality_filter,"model":"Quality-Offensive_600276_fi.pt"},
}

print("Loading stock data...", end=" ", flush=True)
stock_dfs = {}
for f in os.listdir(CACHE):
    if f.endswith('.pkl'):
        code = f.replace('.pkl','')
        df = pd.read_pickle(f"{CACHE}/{f}")
        for c in ['open','high','low','close','volume','pe','pb']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        stock_dfs[code] = df
print(f"{len(stock_dfs)} stocks")

for sn, cfg in STRATEGIES.items():
    print(f"\n{'='*60}\n  {sn}\n{'='*60}")

    ckpt = torch.load(f"{MODEL_DIR}/{cfg['model']}")
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    model.load_state_dict(ckpt['model_state']); model.eval()

    # Cache price windows
    price_windows = {}
    for code, df in stock_dfs.items():
        close = df['close'].values; w = {}
        for i in range(60, len(df)): w[df.index[i]] = close[i-60:i+1]
        price_windows[code] = w

    total = 1_000_000.0; cash = 1_000_000.0; holding = {}; pool = {}
    log = []; events = []
    cumulative_cost = 0.0; cumulative_turnover = 0.0

    for day_i, date in enumerate(dates):
        # ── 市场状态(修复1+2: 真实CSI300 + 动态仓位) ──
        csi = market_ctx.csi300
        csi_close = csi.loc[date]['close'] if date in csi.index else 4000
        csi_ma200 = float(csi['close'].rolling(200).mean().loc[date]) if date in csi.index and csi['close'].rolling(200).mean().loc[date]>0 else csi_close
        csi_ret_5d = float(csi['close'].pct_change(5).loc[date]) if date in csi.index else 0
        csi_ret_20d = float(csi['close'].pct_change(20).loc[date]) if date in csi.index else 0
        csi_vol_20d = float(csi['close'].pct_change().rolling(20).std().loc[date]) if date in csi.index else 0.01
        is_bear = csi_close < csi_ma200
        MAX_POS = MAX_POS_BEAR if is_bear else MAX_POS_BULL
        mkt_features = np.array([csi_ret_5d, csi_ret_20d, csi_vol_20d,
                                  0,0, 0 if not is_bear else 1, 0,0, 0,0,0], dtype=np.float32)
        mkt_features = np.nan_to_num(mkt_features, 0)

        # ── 修复3: 组合回撤止损 ──
        peak_so_far = max(v for v in [r['total'] for r in log]) if log else total
        portfolio_dd = (total - peak_so_far) / max(peak_so_far, 1)
        if portfolio_dd < -0.20:
            MAX_POS = min(MAX_POS, 0.06)  # 回撤>20% → 仓位上限压到6%

        # Selection
        sd = {}; fv_dict = {fn: {} for fn in cfg["factors"]}
        for (d,s,c), fv in fc.items():
            if d==date and s==sn:
                for fn,v in fv.items(): fv_dict[fn][c] = v
        for c in fv_dict[cfg["factors"][0]]:
            df = stock_dfs.get(c)
            if df is None or date not in df.index: continue
            row = df.loc[date]; cs = df['close']; i = df.index.get_loc(date)
            sd[c] = {'close':row.get('close',0),
                     'amount':row.get('close',0)*row.get('volume',0),
                     'pb':row.get('pb',0),'pe':row.get('pe',20),
                     'roc_250':float(cs.iloc[i]/cs.iloc[max(0,i-250)]-1) if i>=250 else 0,
                     'ma_200':float(cs.iloc[max(0,i-200):i+1].mean()) if i>=200 else 1e9}

        vc = cfg["filter_fn"](sd)
        fv_f = {fn:{c:v for c,v in vals.items() if c in vc} for fn,vals in fv_dict.items()}
        mc = {c:sd[c]['amount'] for c in vc}
        fv_n = {fn:neutralizer.neutralize(vals,mc) for fn,vals in fv_f.items()}
        scores = scorer.score(fv_n)
        rankings = scorer.select_top_k(scores, k=TOP_N)
        rank_map = {c:i for i,(c,_) in enumerate(rankings,1)}

        # Exit
        for code in list(pool.keys()):
            st = pool[code]
            if st['state']=='ACTIVE' and rank_map.get(code,999)>EXIT_RANK and st['day']>MIN_HOLD:
                st['state']='EXITING'; st['exit_day']=0
                events.append({'date':date,'code':code,'event':'START_EXIT','price':sd.get(code,{}).get('close',0),'reason':f'rank={rank_map.get(code,999)}'})

        # Entry
        pool_count = sum(1 for s in pool.values() if s['state'] in ('WATCHING','BUILDING','ACTIVE'))
        if pool_count < MAX_STOCKS:
            for code, _ in rankings[:50]:
                if code not in pool:
                    pool[code] = {'state':'WATCHING','day':0,'position':0.0,'returns':[],'entry_date':date}
                    pool_count += 1
                    events.append({'date':date,'code':code,'event':'START_WATCH','price':sd.get(code,{}).get('close',0),'reason':'selected'})
                    if pool_count >= MAX_STOCKS: break

        # FI-PPO signals
        fi_signals = {}
        for code in pool:
            if code not in sd: continue
            win = price_windows.get(code, {}).get(date)
            if win is None: continue
            factors_vals = {fn: fv_dict[fn].get(code, 0.0) for fn in cfg["factors"]}
            log_rets = np.diff(np.log(np.maximum(win, 1e-8)))
            price_feat = np.zeros(60); price_feat[-len(log_rets):] = log_rets[-60:]
            factor_feat = np.array([float(factors_vals[f]) for f in cfg["factors"]])
            pos = holding.get(code,0) * sd[code]['close'] / max(total, 1)
            cash_pct = cash / max(total, 1)
            pnl = (sd[code]['close'] / max(sd[code].get('close', 1), 1) - 0.5) if holding.get(code,0)>0 else 0.0
            state_vec = np.concatenate([price_feat, factor_feat, [pos, cash_pct, pnl], mkt_features])
            state_vec = np.nan_to_num(state_vec, nan=0.0)
            s = torch.FloatTensor(state_vec).unsqueeze(0)
            with torch.no_grad(): a, _, _ = model.get_action(s, deterministic=True)
            fi_signals[code] = float(a.squeeze().numpy())

        # Advance states
        for code, st in list(pool.items()):
            ret_today = 0.0
            if code in sd and date in stock_dfs[code].index:
                idx = stock_dfs[code].index.get_loc(date)
                if idx > 0: ret_today = float(stock_dfs[code]['close'].iloc[idx]/stock_dfs[code]['close'].iloc[idx-1]-1)

            if st['state'] == 'WATCHING':
                st['day']+=1; st['returns'].append(ret_today)
                if st['day']>=OBS:
                    obs=st['returns'][-OBS:]
                    if np.mean(obs)>-0.003 and np.min(obs)>-0.10:
                        st['state']='BUILDING'; st['build_day']=0
                        events.append({'date':date,'code':code,'event':'START_BUILD','price':sd.get(code,{}).get('close',0),'reason':'passed observation'})
                    else: del pool[code]

            elif st['state'] == 'BUILDING':
                st['build_day']+=1
                price=sd.get(code,{}).get('close',0)
                daily_build = (MAX_POS/BLD) * BUILD_SPEED
                if price>0 and cash >= daily_build*total:
                    amt = daily_build * total; cost = amt * 0.00025
                    new_shares = amt / price * (1 - 0.00025)
                    cash -= amt; cumulative_cost += cost
                    holding[code] = holding.get(code,0) + new_shares
                    st['position'] += daily_build
                    cumulative_turnover += amt / total
                    events.append({'date':date,'code':code,'event':'BUY','price':price,'amount':amt,'cost':cost,'position':st['position'],'fi_action':fi_signals.get(code,0)})
                if st['build_day']>=BLD or st['position']>=MAX_POS*0.95:
                    st['state']='ACTIVE'; st['day']=0
                    events.append({'date':date,'code':code,'event':'ACTIVE','price':price,'position':st['position']})

            elif st['state'] == 'ACTIVE':
                st['day']+=1
                fi = fi_signals.get(code, 0.0); price = sd.get(code,{}).get('close',0)
                cur_shares = holding.get(code,0)
                adjust = fi * 0.03 * total
                new_target = np.clip(st['position'] + adjust/max(total,1), 0.0, MAX_POS)
                target_value = new_target * total
                cur_value = cur_shares * price; delta = (target_value - cur_value) * 0.15
                if abs(delta) > 1000:
                    if delta > 0 and cash >= delta:
                        cost = delta * 0.00025
                        new_shares = delta / price * (1 - 0.00025)
                        cash -= delta; holding[code] = cur_shares + new_shares
                        st['position'] = new_target; cumulative_cost += cost
                        cumulative_turnover += delta / total
                        events.append({'date':date,'code':code,'event':'BUY','price':price,'amount':delta,'cost':cost,'position':st['position'],'fi_action':fi})
                    elif delta < 0:
                        tax = abs(delta) * 0.00075  # commission + stamp
                        proceeds = abs(delta) * (1 - 0.00075)
                        cash += proceeds; cumulative_cost += tax
                        holding[code] = max(0, cur_shares + delta/price)
                        st['position'] = new_target
                        cumulative_turnover += abs(delta) / total
                        events.append({'date':date,'code':code,'event':'SELL','price':price,'amount':abs(delta),'cost':tax,'position':st['position'],'fi_action':fi})

            elif st['state'] == 'EXITING':
                st['exit_day']+=1
                cur_shares = holding.get(code,0); price = sd.get(code,{}).get('close',0)
                if cur_shares > 0 and price > 0:
                    sell_s = cur_shares * 0.25; amt = sell_s * price
                    tax = amt * 0.00075; cash += amt - tax
                    cumulative_cost += tax; cumulative_turnover += amt / total
                    holding[code] = cur_shares - sell_s
                    events.append({'date':date,'code':code,'event':'SELL','price':price,'amount':amt,'cost':tax,'position':0,'fi_action':-1,'reason':'EXITING'})
                if st['exit_day']>EXT or holding.get(code,0)<=0:
                    events.append({'date':date,'code':code,'event':'DONE','price':price,'reason':'exit complete'})
                    del pool[code]; holding.pop(code, None)

        # Mark-to-market
        equity = sum(holding.get(c,0) * sd.get(c,{}).get('close',0) for c in holding)
        total = cash + equity; log.append({'date':date,'total':total,'cash':cash,'equity':equity,'n_holding':len(holding),'n_active':sum(1 for s in pool.values() if s['state']=='ACTIVE')})

        if day_i % 200 == 0:
            act=sum(1 for s in pool.values() if s['state']=='ACTIVE')
            print(f"  {date.date()} total={total:,.0f} active={act}", flush=True)

    # Save
    df_events = pd.DataFrame(events)
    df_events.to_csv(f"{OUT_DIR}/{sn}_events.csv", index=False, encoding='utf-8-sig')
    df_log = pd.DataFrame(log)
    df_log.to_csv(f"{OUT_DIR}/{sn}_daily.csv", index=False, encoding='utf-8-sig')

    vals=np.array([r['total'] for r in log]); rets=np.diff(vals)/vals[:-1]; cum=vals/vals[0]
    dd=float(np.min(cum/np.maximum.accumulate(cum)-1))
    sr=float(rets.mean()/(rets.std()+1e-10)*np.sqrt(252)) if len(rets)>1 else 0
    tr=float(cum[-1]-1)
    n_stocks = len(set(e['code'] for e in events if e['event'] in ('BUY','SELL')))
    buy_events = sum(1 for e in events if e['event']=='BUY')
    sell_events = sum(1 for e in events if e['event']=='SELL')
    avg_turnover = cumulative_turnover / len(log) if log else 0

    csi=market_ctx.csi300; csi_t=csi[(csi.index>=START)&(csi.index<=END)]
    csi_ret=float(csi_t['close'].iloc[-1]/csi_t['close'].iloc[0]-1)

    print(f"\n  [{sn}] Ret={tr:+.1%} Sharpe={sr:.2f} DD={dd:.1%}")
    print(f"  CSI300: {csi_ret:+.1%} | Stocks traded: {n_stocks} | Trade events: {buy_events+sell_events}")
    print(f"  Total costs: {cumulative_cost:,.0f} | Avg daily turnover: {avg_turnover*100:.2f}%")
    print(f"  Events saved: {OUT_DIR}/{sn}_events.csv")
    print(f"  Daily log saved: {OUT_DIR}/{sn}_daily.csv")

print("\nDone!")
