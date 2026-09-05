"""
股票池管理器 — 每日滚动选股 + 调仓队列
"""
import pandas as pd, numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class StockStatus:
    """单只股票在池中的状态"""
    code: str
    active: bool = False           # 是否核心持仓
    entering: bool = False         # 是否在建仓期
    exiting: bool = False          # 是否在退出期
    observing: bool = False        # 是否在观察期
    rank: int = 999                # 最近排名
    entry_day: int = 0             # 进入第几天
    exit_day: int = 0              # 退出第几天
    observe_day: int = 0           # 观察第几天
    target_position: float = 0.0   # 目标仓位
    last_signal: str = ""          # 最近状态变化原因


class PoolManager:
    """每日选股 + 调仓队列管理

    保守逻辑:
    - 每天最多换 1 只股票
    - 排名 ≤ threshold 才考虑进入
    - 排名 > 2×threshold 触发退出
    - 已进入退出队列的, 不因排名回升而取消
    """

    def __init__(self, max_stocks=6, enter_rank=8, exit_rank=15,
                 daily_max_change=1):
        self.max_stocks = max_stocks
        self.enter_rank = enter_rank
        self.exit_rank = exit_rank
        self.daily_max_change = daily_max_change

        # 状态追踪
        self.status: Dict[str, StockStatus] = {}

    def update_daily(self, rankings: List[Tuple[str, float]],
                     current_positions: Dict[str, float]) -> Tuple[List, List]:
        """
        每日更新

        Args:
            rankings: [(code, score), ...] 全市场排名
            current_positions: {code: position_pct} 当前持仓

        Returns:
            entering: [code, ...]  今天进入观察/建仓的股票
            exiting:  [code, ...]  今天开始退出的股票
        """
        entering = []
        exiting = []

        rank_map = {code: rank for rank, (code, _) in enumerate(rankings, 1)}

        # 更新现有持仓的排名
        for code, status in self.status.items():
            status.rank = rank_map.get(code, 999)

        # 找出退出的: 排名 > exit_rank
        for code in current_positions:
            if code not in self.status:
                self.status[code] = StockStatus(code=code, active=True)
            st = self.status[code]
            if st.exiting:
                continue  # 已在退出, 不逆转
            rank = rank_map.get(code, 999)
            if rank > self.exit_rank:
                st.exiting = True
                st.exit_day = 0
                st.active = False
                st.last_signal = f"rank={rank}>{self.exit_rank}"
                exiting.append(code)

        # 找出进入的: 排名 ≤ enter_rank 且不在池中
        candidates = []
        for code, score in rankings[:self.enter_rank]:
            if code not in current_positions and code not in [s for s in self.status.values() if s.entering or s.observing]:
                candidates.append(code)

        # 限制每日调仓数
        active_count = sum(1 for s in self.status.values() if s.active or s.entering or s.observing)
        exiting_count = sum(1 for s in self.status.values() if s.exiting)
        net = active_count - exiting_count

        slots = self.max_stocks - net
        if slots > 0 and candidates:
            # 只取排名最高的几个
            for code in candidates[:min(slots, self.daily_max_change)]:
                if code not in self.status:
                    self.status[code] = StockStatus(code=code, rank=rank_map.get(code, 999))
                self.status[code].observing = True
                self.status[code].observe_day = 0
                self.status[code].last_signal = f"new entry, rank={rank_map.get(code, 999)}"
                entering.append(code)

        return entering, exiting

    def active_codes(self) -> List[str]:
        return [c for c, s in self.status.items() if s.active]

    def entering_codes(self) -> List[str]:
        return [c for c, s in self.status.items() if s.entering or s.observing]

    def exiting_codes(self) -> List[str]:
        return [c for c, s in self.status.items() if s.exiting]
