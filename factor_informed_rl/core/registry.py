"""
策略注册表 — 装饰器式注册机制。

用法:
    from factor_informed_rl.core import register_strategy

    @register_strategy("my_strategy")
    class MyStrategy(Strategy):
        ...
"""
from typing import Dict, Type

_STRATEGIES: Dict[str, Type] = {}


def register_strategy(name: str):
    """装饰器：将策略类注册到全局注册表。

    Args:
        name: 策略唯一标识符（小写蛇形命名，如 "value_defensive"）
    """
    def decorator(cls):
        if name in _STRATEGIES:
            raise ValueError(f"Strategy '{name}' already registered")
        _STRATEGIES[name] = cls
        return cls
    return decorator


def get_strategy(name: str):
    """按名称获取策略类。"""
    if name not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise KeyError(f"Strategy '{name}' not found. Available: {available}")
    return _STRATEGIES[name]


def list_strategies() -> Dict[str, Type]:
    """返回所有已注册策略 {name: class}。"""
    return dict(_STRATEGIES)
