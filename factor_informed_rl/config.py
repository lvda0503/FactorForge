"""
全局配置中心
============
所有超参数、路径、开关集中管理。
修改实验只需改这个文件。
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class DataConfig:
    """数据配置"""
    source: str = "yfinance"          # "yfinance" | "baostock" | "custom_csv"
    ticker: str = "AAPL"              # 美股: AAPL, MSFT, GOOGL  A股: sh.600519
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"
    train_split: float = 0.70        # 前70%训练
    val_split: float = 0.15          # 中间15%验证
    test_split: float = 0.15         # 后15%测试

@dataclass
class FactorConfig:
    """因子配置"""
    factors: List[str] = field(default_factory=lambda: [
        "roc_20",        # 20日动量 (趋势)
        "rank_20",       # 20日价格排名 (位置, 替代RSV14)
        "std_20",        # 20日波动率 (风险)
        "pb_ratio",      # 市净率 (价值)
        "corr_20",       # 20日量价相关 (量价健康度)
    ])
    ic_window: int = 120              # IC计算窗口 (交易日)
    ic_decay_threshold: float = 0.005 # IC衰减检测阈值

@dataclass
class EnvConfig:
    """环境配置"""
    window_size: int = 60             # 历史价格窗口
    action_dim: int = 1              # 连续动作: [-1, +1]
    enable_short: bool = True        # 启用做空
    short_margin: float = 0.5        # 做空保证金比例 (50%现金锁定)
    short_borrow_cost: float = 0.0005  # 融券日费率 0.05%
    initial_capital: float = 100_000.0
    commission: float = 0.001         # 手续费率 0.1%
    slippage: float = 0.001           # 滑点 0.1%

@dataclass
class PPOConfig:
    """PPO算法配置"""
    # 网络结构
    actor_hidden: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 128, 64])

    # PPO超参数
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99              # 折扣因子
    gae_lambda: float = 0.95         # GAE参数
    clip_epsilon: float = 0.2        # PPO裁剪范围
    entropy_coef: float = 0.03       # 熵正则系数 (提升探索)
    value_coef: float = 0.5          # 价值损失系数
    max_grad_norm: float = 0.5       # 梯度裁剪

    # 训练配置
    n_steps: int = 2048               # 每次收集的经验步数
    n_epochs: int = 10               # 每批经验复用次数
    batch_size: int = 256            # 小批量大小
    total_timesteps: int = 200_000   # 总训练步数

    # 评估
    eval_freq: int = 20_000          # 每多少步评估一次
    n_eval_episodes: int = 5

@dataclass
class FactorInformedConfig:
    """因子信息损失配置 (核心创新)"""
    enable: bool = True               # 是否启用因子约束 (消融实验开关)

    # IC约束
    lambda_ic: float = 0.1           # IC约束权重
    ic_check_freq: int = 1000        # 每多少步检测一次IC
    min_pos_ic: float = -0.01        # IC低于此值开始惩罚

    # 正交性约束
    lambda_ortho: float = 0.05       # 正交性约束权重
    ortho_check_freq: int = 5000     # 每多少步检测正交性
    max_corr: float = 0.7            # 相关性高于此值开始惩罚

    # 课程学习
    warmup_steps: int = 50_000       # 前N步不加因子约束 (让RL自由探索)
    lambda_growth_rate: float = 1.2  # warmup后每步 λ 增长率

@dataclass
class ExperimentConfig:
    """实验总配置"""
    name: str = "factor_informed_rl_mve"
    seed: int = 42
    device: str = "cpu"               # "cpu" | "cuda"
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"

    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    factor_loss: FactorInformedConfig = field(default_factory=FactorInformedConfig)

    # 消融实验变量
    ablation: str = "full"            # "full" | "no_factor_loss" | "no_factors"

# 全局配置实例
cfg = ExperimentConfig()
