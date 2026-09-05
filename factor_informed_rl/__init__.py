"""
FactorRL — Factor-Informed Reinforcement Learning for Quantitative Trading
==========================================================================
PINN-inspired factor constraints for PPO-based trading agents.

导入本包时自动注册内置策略（value_defensive / quality_offensive）。
"""
# 先导入 core（避免循环依赖），再导入 strategies（触发注册）
from . import core  # noqa: F401
from . import strategies  # noqa: F401

__version__ = "0.1.0"

__all__ = ["core", "strategies"]
