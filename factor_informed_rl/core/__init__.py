"""FactorRL 核心抽象层 — 策略基类、注册机制、统一配置。"""
from .config import StrategyConfig, SelectionConfig
from .registry import register_strategy, get_strategy, list_strategies
from .strategy import Strategy

__all__ = [
    "Strategy",
    "StrategyConfig",
    "SelectionConfig",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
