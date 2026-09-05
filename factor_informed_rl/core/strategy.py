"""
Strategy 抽象基类 — 用户继承此类实现自定义策略。

一个 Strategy 封装了从"数据 → 环境 → 模型 → 训练 → 回测"的完整流程。
用户只需指定因子列表和超参，即可复用全部基础设施。
"""
import sys
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from .config import StrategyConfig


class Strategy(ABC):
    """策略抽象基类。

    子类必须实现:
      - build_env(df): 用配置构建交易环境
      - build_model(): 构建 PPO 模型

    可选覆盖:
      - train(df): 自定义训练流程
      - backtest(df): 自定义回测流程
    """

    config: StrategyConfig

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or self._default_config()

    @classmethod
    def _default_config(cls) -> StrategyConfig:
        """默认配置（子类通常覆盖 config 类属性）。"""
        cfg = getattr(cls, "config", None)
        if isinstance(cfg, StrategyConfig):
            return cfg
        return StrategyConfig(name=cls.__name__)

    # ── 子类必须实现 ──
    @abstractmethod
    def build_env(self, df: pd.DataFrame):
        """用数据 + 配置构建交易环境。"""

    @abstractmethod
    def build_model(self):
        """构建 PPO Actor-Critic 模型。"""

    # ── 默认实现（可覆盖）──
    def load_data(self, code: str, cache_dir: str) -> pd.DataFrame:
        """加载单只股票数据并预处理（含 PIT-safe 因子）。"""
        path = f"{cache_dir}/baostock_{code}.pkl"
        df = pd.read_pickle(path)
        for c in ['open', 'high', 'low', 'close', 'volume', 'pe', 'pb', 'turn']:
            if c in df.columns:
                df[c] = df[c].ffill().bfill().fillna(0)
        # PIT-safe: expanding 窗口，避免未来函数
        if 'pe_percentile' in self.config.factors:
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        return df

    def train_test_split(self, df: pd.DataFrame):
        """按年份切分训练集/测试集。"""
        train = df[df.index.year <= self.config.train_end_year]
        test = df[df.index.year >= self.config.test_start_year]
        return train, test

    def train(self, df: pd.DataFrame, stock_name: str = "") -> str:
        """默认训练流程。返回模型保存路径。"""
        from factor_informed_rl.training.ppo_trainer import PPOTrainer
        from factor_informed_rl.models.factor_loss import FactorInformedLoss
        import torch

        train_df, _ = self.train_test_split(df)
        env = self.build_env(train_df)
        model = self.build_model()

        factor_engine = env.factor_engine

        # FI-PPO 损失（可选）
        if self.config.use_factor_loss:
            factor_loss = FactorInformedLoss(
                factor_engine,
                lambda_ic=self.config.lambda_ic,
                lambda_ortho=self.config.lambda_ortho,
                warmup_steps=self.config.warmup_steps,
            )
        else:
            factor_loss = None

        trainer = PPOTrainer(
            model=model,
            factor_engine=factor_engine,
            factor_loss=factor_loss,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
        )

        trainer.train(env, total_steps=1_000_000)
        save_path = f"models/{self.config.name}.pt"
        torch.save({'model_state': model.state_dict()}, save_path)
        return save_path

    def backtest(self, df: pd.DataFrame) -> dict:
        """默认回测流程。返回绩效指标字典。"""
        _, test_df = self.train_test_split(df)
        env = self.build_env(test_df)
        model = self.build_model()

        # 加载训练好的权重（如果有）
        import os
        import torch
        save_path = f"models/{self.config.name}.pt"
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state'])

        # 简单回测循环
        state = env.reset()
        done = False
        total_return = 0.0
        while not done:
            import numpy as np
            with torch.no_grad():
                action, _ = model.get_action(torch.FloatTensor(state).unsqueeze(0), deterministic=True)
            state, reward, done, info = env.step(float(action.squeeze().numpy()))
            total_return += reward

        return {
            'strategy': self.config.name,
            'total_return': total_return,
            'final_equity': getattr(env, 'portfolio_value', None),
        }

    # ── 便捷方法 ──
    def __repr__(self):
        return f"<Strategy {self.config.name}: factors={self.config.factors}>"
