"""
组合管理器 v2 — 简化状态机, 修复进出bug
"""
import numpy as np
from typing import Dict, List, Tuple

class PortfolioManager:
    """单状态机组合管理

    每只股票的状态：WATCHING → BUILDING → ACTIVE → EXITING → DONE
    """

    def __init__(self, max_stocks=6, total_capital=1_000_000,
                 observe_days=5, build_days=8, exit_days=6,
                 max_position=0.18, top_n=50):
        self.max_stocks = max_stocks
        self.capital = total_capital
        self.cash = total_capital
        self.observe_days = observe_days
        self.build_days = build_days
        self.exit_days = exit_days
        self.max_position = max_position
        self.top_n = top_n

        # 统一状态: {code: {state, day, position, target, ...}}
        self.stocks = {}
        self.daily_returns = []

    @property
    def total_value(self):
        return self.cash + sum(s.get('value', 0) for s in self.stocks.values())

    def daily_step(self, date, rankings, fi_signals, prices, returns):
        """
        rankings: [(code, score), ...] 全市场排名, 取其前 top_n
        fi_signals: {code: float} FI-PPO action ∈ [-1, +1]
        prices: {code: close_price}
        returns: {code: daily_return}
        """
        rank_map = {code: i for i, (code, _) in enumerate(rankings[:self.top_n], 1)}
        pool_count = sum(1 for s in self.stocks.values()
                        if s['state'] in ('WATCHING','BUILDING','ACTIVE'))

        # 检查应退出
        for code, st in list(self.stocks.items()):
            if st['state'] == 'EXITING':
                self._do_exit(code, st, fi_signals.get(code, -0.5), prices)
            elif st['state'] == 'ACTIVE':
                rank = rank_map.get(code, 999)
                if rank > 24 and st['day'] > 20:  # 排名掉出前24且持有>20天, 触发退出
                    st['state'] = 'EXITING'; st['day'] = 0

        # 检查应进入
        for code, score in rankings[:self.top_n]:
            if code not in self.stocks and pool_count < self.max_stocks:
                if code not in prices: continue
                self.stocks[code] = {'state': 'WATCHING', 'day': 0, 'position': 0.0,
                                     'value': 0.0, 'entry_price': 0.0, 'returns': []}

        # 推进各状态
        events = []
        for code, st in list(self.stocks.items()):
            if st['state'] == 'WATCHING':
                self._do_watch(code, st, returns.get(code, 0))
                if st['day'] >= self.observe_days:
                    rets = st['returns'][-self.observe_days:]
                    if np.mean(rets) > -0.003 and np.min(rets) > -0.08:  # 观察期表现合格
                        st['state'] = 'BUILDING'; st['day'] = 0
                        events.append(f"BUILD: {code}")
                    else:
                        del self.stocks[code]  # 观察不合格, 放弃

            elif st['state'] == 'BUILDING':
                action = fi_signals.get(code, 0.3)
                self._do_build(code, st, action, prices)
                if st['day'] >= self.build_days or st['position'] >= self.max_position * 0.9:
                    st['state'] = 'ACTIVE'; st['day'] = 0
                    events.append(f"ACTIVE: {code}")

            elif st['state'] == 'ACTIVE':
                action = fi_signals.get(code, 0.0)
                self._do_trade(code, st, action, prices)

            elif st['state'] == 'EXITING':
                if st['position'] < 0.002:
                    del self.stocks[code]
                    events.append(f"DONE: {code}")

        # 估值
        portfolio_value = self.cash
        for code, st in self.stocks.items():
            if st['position'] > 0 and code in prices:
                st['value'] = st['position'] * self.capital
                portfolio_value += st['value']
            else:
                st['value'] = 0.0

        self.daily_returns.append(portfolio_value / max(self.capital, 1) - 1)
        self.capital = portfolio_value

        return {'value': portfolio_value, 'active': active_count,
                'n_stocks': len(self.stocks), 'events': events}

    def _do_watch(self, code, st, ret):
        st['day'] += 1
        st['returns'].append(ret)

    def _do_build(self, code, st, action, prices):
        st['day'] += 1
        if code not in prices: return
        price = prices[code]
        # 每日建仓量 = 目标仓位 / 建仓天数, 受FI-PPO调节
        daily_target = self.max_position / self.build_days
        fi_factor = max(0.15, min(1.0, abs(action)))  # FI-PPO控制建仓速度
        buy_pct = daily_target * fi_factor

        cost = buy_pct * self.capital * 1.00025
        if cost <= self.cash:
            self.cash -= buy_pct * self.capital
            st['position'] += buy_pct
            st['value'] = st['position'] * self.capital
            st['entry_price'] = price

    def _do_trade(self, code, st, action, prices):
        if code not in prices: return
        st['day'] += 1
        price = prices[code]
        current_pct = st['position']
        # FI-PPO action → 仓位调整 (最多 ±3%/天)
        delta = np.clip(action * 0.03, -0.03, 0.03)
        delta = np.clip(delta, -current_pct, self.max_position - current_pct)

        if abs(delta) > 0.002:
            if delta > 0:
                cost = delta * self.capital * 1.00025
                if cost <= self.cash:
                    self.cash -= delta * self.capital
                    st['position'] += delta
            else:
                proceeds = abs(delta) * self.capital * (1 - 0.00025 - 0.0005)
                self.cash += proceeds
                st['position'] += delta

    def _do_exit(self, code, st, action, prices):
        st['day'] += 1
        if code not in prices: return
        current = st['position']
        # 每日减持20%, FI-PPO控制节奏
        reduce = min(current * 0.25, current)
        fi_factor = 0.3 + 0.7 * abs(min(action, 0))
        sell_pct = reduce * fi_factor

        if sell_pct > 0.001:
            proceeds = sell_pct * self.capital * (1 - 0.00025 - 0.0005)
            self.cash += proceeds
            st['position'] -= sell_pct

    def performance(self):
        if not self.daily_returns: return {}
        r = np.array(self.daily_returns)
        c = np.cumprod(1+r)
        dd = float(np.min(c/np.maximum.accumulate(c)-1))
        sr = float(r.mean()/(r.std()+1e-10)*np.sqrt(252))
        return {'total_return': float(c[-1]-1), 'sharpe': sr, 'max_drawdown': dd, 'n_days': len(r)}
