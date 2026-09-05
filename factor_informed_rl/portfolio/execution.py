"""
建仓/清仓管理器 — 控制节奏, 防冲击
"""
import numpy as np
from typing import Dict, Optional


class EntryManager:
    """建仓管理器

    流程:
      Day 0-4: 观察期 (只看不买, 跟踪波动和偏移)
      Day 5-12: 建仓期 (每天买入 target/ramp_days)
      Day 13+: 转为核心持仓, FI-PPO 自由交易
    """

    def __init__(self, observe_days=5, ramp_days=8,
                 max_position=0.20, daily_cash_ratio=0.10):
        self.observe_days = observe_days
        self.ramp_days = ramp_days
        self.max_position = max_position
        self.daily_cash_ratio = daily_cash_ratio
        self.entry_state = {}  # {code: {day, target, built, observed_returns}}

    def init_entry(self, code: str, target_position: float):
        self.entry_state[code] = {
            'day': 0, 'target': target_position, 'built': 0.0,
            'observed_returns': [], 'observing': True, 'building': False, 'active': False,
        }

    def step(self, code: str, daily_return: float, fi_ppo_action: float,
             available_cash_pct: float) -> Optional[float]:
        """
        每日推进建仓进度

        Returns:
            None = 今天不动 (观察期)
            float = 目标仓位变化 (正=买入, 负=卖出)
        """
        if code not in self.entry_state:
            return None
        es = self.entry_state[code]
        es['day'] += 1
        es['observed_returns'].append(daily_return)

        # 观察期: 收集数据, 不操作
        if es['observing']:
            if es['day'] >= self.observe_days:
                # 检查观察期表现
                obs_rets = es['observed_returns'][-self.observe_days:]
                avg_ret = np.mean(obs_rets)
                max_dd = np.min(obs_rets)
                # 如果观察期跌超10%或平均负收益→放弃
                if max_dd < -0.10 or avg_ret < -0.005:
                    del self.entry_state[code]
                    return None  # 放弃这只
                # 通过观察, 开始建仓
                es['observing'] = False
                es['building'] = True
            else:
                return None

        # 建仓期
        if es['building']:
            daily_target = min(
                es['target'] / self.ramp_days,           # 等分建仓
                available_cash_pct * self.daily_cash_ratio,  # 现金限制
            )
            # FI-PPO 只控制当天的建仓幅度 (0 = 不建, +1 = 全量建)
            adjusted = daily_target * max(0.1, fi_ppo_action)
            es['built'] += adjusted

            if es['day'] >= self.observe_days + self.ramp_days or es['built'] >= es['target'] * 0.95:
                es['building'] = False
                es['active'] = True
            return adjusted

        return 0.0


class ExitManager:
    """清仓管理器

    流程:
      Day 0-N: 每日减持剩余仓位的 daily_reduce_pct
      清仓完毕后从状态中移除
    """

    def __init__(self, exit_days=6, daily_reduce_pct=0.20):
        self.exit_days = exit_days
        self.daily_reduce_pct = daily_reduce_pct
        self.exit_state = {}  # {code: {day, remaining, initial}}

    def init_exit(self, code: str, current_position: float):
        self.exit_state[code] = {
            'day': 0, 'remaining': current_position, 'initial': current_position,
        }

    def step(self, code: str, fi_ppo_action: float) -> Optional[float]:
        """
        每日减持

        Returns:
            float = 目标仓位变化 (负数=减持)
        """
        if code not in self.exit_state:
            return None
        es = self.exit_state[code]
        es['day'] += 1

        # 计算减持量
        reduce_base = es['remaining'] * self.daily_reduce_pct
        # FI-PPO 控制节奏: action=-1→加速减持, action=0→缓慢减持
        fi_factor = 0.5 + 0.5 * abs(min(fi_ppo_action, 0))  # [0.5, 1.0]
        reduce = -reduce_base * fi_factor

        es['remaining'] += reduce

        # 接近清零或超时 → 全清
        if es['day'] > self.exit_days or es['remaining'] < 0.005:
            final = -es['remaining']
            del self.exit_state[code]
            return final

        return reduce
