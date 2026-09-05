"""
统一策略配置 — 所有可调参数集中到一个 dataclass。
用户自定义策略时只需实例化这个配置，无需改内部代码。
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrategyConfig:
    """策略完整配置。

    一个 StrategyConfig 完整描述一个可训练/可回测的策略。
    """
    # ── 标识 ──
    name: str = "custom_strategy"

    # ── 因子（FactorEngine 现成 20+ 因子，任意组合）──
    factors: List[str] = field(default_factory=list)

    # ── 风控 ──
    max_long: float = 0.80        # 最大做多仓位
    max_short: float = 0.10       # 最大做空仓位
    stop_loss: float = 0.08       # 止损阈值（比例）
    initial_capital: float = 1_000_000.0
    enable_short: bool = True

    # ── 交易成本 ──
    commission: float = 0.00025   # 佣金
    stamp_tax: float = 0.0005     # 印花税（卖出）
    slippage: float = 0.001       # 滑点

    # ── 状态空间 ──
    window_size: int = 60         # 价格窗口
    market_dim: int = 11          # 市场环境维度

    # ── 模型结构 ──
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # ── FI-PPO 因子损失 ──
    use_factor_loss: bool = True  # False → 裸 PPO
    lambda_ic: float = 0.1        # IC 约束权重
    lambda_ortho: float = 0.05    # 正交性约束权重
    warmup_steps: int = 50_000    # 课程学习：前 N 步不加约束

    # ── 训练超参 ──
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    n_epochs: int = 10
    batch_size: int = 256

    # ── 数据切分 ──
    train_end_year: int = 2020    # 训练集截止年份（含）
    test_start_year: int = 2022   # 测试集起始年份（含）

    # ── 市场环境 ──
    market_context: Optional[object] = None  # 延迟加载，避免循环导入


@dataclass
class SelectionConfig:
    """选股配置（独立于 RL 策略）。"""
    factors: List[str] = field(default_factory=list)
    top_k: int = 10
    rebalance_freq: str = "monthly"
    universe: str = "csi300"
