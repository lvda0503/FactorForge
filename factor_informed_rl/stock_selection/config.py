"""
选股模块配置 — 独立于 RL 交易层
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class UniverseConfig:
    """股票池过滤配置"""
    min_daily_amount: float = 1e8      # 最小日成交额 (1亿)
    exclude_st: bool = True             # 排除ST
    exclude_suspended: bool = True       # 排除停牌
    exclude_new_listed: int = 60        # 排除上市不足N天的新股

@dataclass
class BarraConfig:
    """Barra 中性化配置"""
    enable: bool = True
    neutralize_industry: bool = True     # 行业中性化
    neutralize_size: bool = True         # 市值中性化
    industry_field: str = "sw_industry"  # 申万一级行业

@dataclass
class TransformConfig:
    """非线性变换配置"""
    method: str = "spline"               # "spline"|"polynomial"|"quantile"|"treebin"|"auto"
    # Spline
    spline_df: int = 6
    spline_degree: int = 3
    # Polynomial
    poly_max_degree: int = 3
    poly_include_inverse: bool = True
    # TreeBin
    tree_max_leaves: int = 15
    tree_max_depth: int = 4
    # Auto selection
    min_ic_improvement: float = 0.005    # IC至少提升0.005才采用变换

@dataclass
class ScorerConfig:
    """打分配置"""
    method: str = "ic_ir_weighted"        # "ic_ir_weighted"|"equal"|"custom"
    ic_window: int = 120                  # IC/IC_IR 计算窗口
    top_k: int = 10                       # 选取股票数量
    rebalance_freq: str = "monthly"       # 调仓频率: "daily"|"weekly"|"monthly"

@dataclass
class SelectionConfig:
    """选股总配置"""
    universe: str = "csi300"              # 候选池: "csi300"|"csi500"|"all"
    universe_cfg: UniverseConfig = field(default_factory=UniverseConfig)
    barra: BarraConfig = field(default_factory=BarraConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)

    # 价值策略因子 (Graham 1934 + Fama-French 1993 价值维度)
    value_factors: List[str] = field(default_factory=lambda: [
        "pb_ratio",         # 截面估值 (1/PB, Graham 1934)
        "pe_percentile",    # 时序估值 (PE历史分位, PIT-safe)
        "rank_20",          # 短期价格排名 (替代RSV14, 更鲁棒)
        "std_60",           # 低波动 (Black 1972, 防御属性)
        "corr_20",          # 量价配合确认
    ])
    value_weights: Optional[Dict[str, float]] = None

    # 质量策略因子
    quality_factors: List[str] = field(default_factory=lambda: [
        "roc_60", "beta_20", "rsqr_20", "vma_20", "std_20"
    ])
    quality_weights: Optional[Dict[str, float]] = None

cfg = SelectionConfig()
