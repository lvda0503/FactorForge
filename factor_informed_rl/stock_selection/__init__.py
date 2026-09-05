"""
Stock Selection Module — 独立选股层
====================================
与 RL 交易层零耦合。通过两个接口交互:
  1. StockSelector.select(date, factor_values) → stock_list
  2. stock_list → FI-PPO trading agents

Pipeline: 硬过滤 → Barra中性化 → 非线性变换 → IC_IR加权 → Top-K
"""
from .config import SelectionConfig, cfg
from .pipeline import StockSelector
from .universe import UniverseFilter
from .neutralizer import BarraNeutralizer
from .factor_transform import FactorTransformer, SplineTransformer, PolynomialTransformer, QuantileTransformer, TreeBinTransformer
from .scorer import FactorScorer
