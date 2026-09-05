"""
组合管理器配置
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class PoolConfig:
    max_stocks: int = 6                  # 最大持仓数
    daily_max_change: int = 1            # 每日最多调入/调出 1 只
    rank_enter_threshold: int = 8        # 排名 ≤8 可进入
    rank_exit_threshold: int = 15        # 排名 >15 触发退出

@dataclass
class EntryConfig:
    observe_days: int = 5               # 观察期 (确认无异常)
    ramp_days: int = 8                  # 建仓天数 (每天建1/N)
    max_init_position: float = 0.20      # 单票最大仓位 20%
    entry_cash_ratio: float = 0.10      # 每天只投可用现金的10%

@dataclass
class ExitConfig:
    exit_days: int = 6                  # 退出天数
    daily_reduce_pct: float = 0.20      # 每天减持剩余仓位的20%
    min_cash_buffer: float = 0.10       # 现金不低于10%

@dataclass
class PortfolioConfig:
    total_capital: float = 1_000_000.0
    pool: PoolConfig = field(default_factory=PoolConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
