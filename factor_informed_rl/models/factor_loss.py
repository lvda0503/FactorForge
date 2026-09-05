"""
Factor-Informed Loss — 核心创新
=================================
类比PINN (Physics-Informed Neural Networks):
  PINN:  L_total = L_data + λ_pde * L_PDE_residual
  本工作: L_total = L_PPO  + λ_ic * L_factor_IC + λ_ortho * L_factor_ortho

两个约束项:
  1. L_factor_IC:
     惩罚因子IC为负的情况
     → 保证agent的决策逻辑隐含地保持因子预测力

  2. L_factor_ortho:
     惩罚因子间高度相关的暴露
     → 鼓励agent利用多样化的因子信号

课程学习:
  warmup_steps内不加约束 → RL自由探索
  warmup后逐步增加 λ → 收紧约束
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from ..preprocessing.factor_engine import FactorEngine


class FactorInformedLoss(nn.Module):
    """因子信息正则化损失

    参数:
        lambda_ic: IC约束权重
        lambda_ortho: 正交性约束权重
        min_pos_ic: IC低于此阈值开始惩罚
        max_corr: 相关性高于此阈值开始惩罚
        warmup_steps: 前N步不施加约束
        lambda_growth_rate: warmup后λ的增长速率
    """

    def __init__(
        self,
        factor_engine: FactorEngine,
        lambda_ic: float = 0.1,
        lambda_ortho: float = 0.05,
        min_pos_ic: float = -0.01,
        max_corr: float = 0.7,
        warmup_steps: int = 50_000,
        lambda_growth_rate: float = 1.2,
    ):
        super().__init__()
        self.factor_engine = factor_engine
        self.lambda_ic_init = lambda_ic
        self.lambda_ortho_init = lambda_ortho
        self.min_pos_ic = min_pos_ic
        self.max_corr = max_corr
        self.warmup_steps = warmup_steps
        self.lambda_growth_rate = lambda_growth_rate

        # 当前 λ (随训练步数增长)
        self.current_lambda_ic = 0.0
        self.current_lambda_ortho = 0.0
        self.global_step = 0

        # 统计
        self.latest_ic_loss = 0.0
        self.latest_ortho_loss = 0.0
        self.latest_total_loss = 0.0

    def forward(self, global_step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        计算因子约束损失

        Returns:
            {
                'ic_loss': IC约束损失 (标量tensor),
                'ortho_loss': 正交性约束损失 (标量tensor),
                'total_factor_loss': 加权总和,
                'current_ics': 当前各因子IC的均值,
                'max_pairwise_corr': 最大因子间相关性,
            }
        """
        if global_step is not None:
            self.global_step = global_step

        # 更新 λ (课程学习)
        self._update_lambdas()

        # --- 1. 计算 IC 约束损失 ---
        ic_loss = self._compute_ic_loss()

        # --- 2. 计算正交性约束损失 ---
        ortho_loss = self._compute_ortho_loss()

        # --- 3. 加权总损失 ---
        total = self.current_lambda_ic * ic_loss + self.current_lambda_ortho * ortho_loss

        # 存储
        self.latest_ic_loss = ic_loss.item()
        self.latest_ortho_loss = ortho_loss.item()
        self.latest_total_loss = total.item()

        # 获取当前因子统计
        ics = self.factor_engine.get_current_ics()
        corr_matrix = self.factor_engine.get_factor_correlation_matrix()
        max_corr = np.max(np.abs(corr_matrix - np.eye(len(corr_matrix))))

        return {
            'ic_loss': ic_loss,
            'ortho_loss': ortho_loss,
            'total_factor_loss': total,
            'current_ic_mean': float(np.mean(list(ics.values()))),
            'max_pairwise_corr': float(max_corr),
            'lambda_ic': self.current_lambda_ic,
            'lambda_ortho': self.current_lambda_ortho,
        }

    def _compute_ic_loss(self) -> torch.Tensor:
        """计算IC约束损失

        L_IC = mean(relu(-IC - threshold))
        IC越负，惩罚越大。IC为正时不惩罚。
        """
        ics = self.factor_engine.get_current_ics()

        if not ics:
            return torch.tensor(0.0)

        ic_values = torch.tensor(
            list(ics.values()), dtype=torch.float32
        )

        # 只有IC < min_pos_ic 时施加惩罚
        # relu(-IC - min_pos_ic): IC=-0.05, threshold=-0.01 → relu(0.05-0.01)=0.04
        violations = torch.relu(-ic_values - self.min_pos_ic)

        # L2惩罚 (平滑)
        ic_loss = torch.mean(violations ** 2)

        return ic_loss

    def _compute_ortho_loss(self) -> torch.Tensor:
        """计算正交性约束损失

        L_ortho = mean(relu(|corr| - max_corr))
        因子间绝对相关性越高于阈值，惩罚越大。
        """
        corr_matrix = self.factor_engine.get_factor_correlation_matrix()

        if corr_matrix is None or corr_matrix.shape[0] <= 1:
            return torch.tensor(0.0)

        corr_tensor = torch.tensor(corr_matrix, dtype=torch.float32)
        n = corr_tensor.shape[0]

        # 只取上三角 (排除对角线)
        triu_idx = torch.triu_indices(n, n, offset=1)
        pair_corrs = corr_tensor[triu_idx[0], triu_idx[1]]

        # 只有 |corr| > max_corr 时才惩罚
        violations = torch.relu(torch.abs(pair_corrs) - self.max_corr)
        ortho_loss = torch.mean(violations ** 2)

        return ortho_loss

    def _update_lambdas(self):
        """课程学习: 逐步收紧约束"""
        if self.global_step < self.warmup_steps:
            # Warmup阶段: 不加约束
            self.current_lambda_ic = 0.0
            self.current_lambda_ortho = 0.0
        else:
            # Warmup后: 逐步增加 λ
            steps_after_warmup = self.global_step - self.warmup_steps
            # 指数增长，但有上限
            growth = min(
                self.lambda_growth_rate ** (steps_after_warmup / 10_000),
                10.0  # 最多增长到初始值的10倍
            )
            self.current_lambda_ic = self.lambda_ic_init * growth
            self.current_lambda_ortho = self.lambda_ortho_init * growth

    def get_stats(self) -> Dict[str, float]:
        """返回当前损失统计 (用于日志)"""
        ics = self.factor_engine.get_current_ics()
        ic_irs = self.factor_engine.get_current_ic_irs()
        decayed = self.factor_engine.detect_decay()

        return {
            'ic_loss': self.latest_ic_loss,
            'ortho_loss': self.latest_ortho_loss,
            'total_factor_loss': self.latest_total_loss,
            'mean_ic': float(np.mean(list(ics.values()))) if ics else 0.0,
            'mean_ic_ir': float(np.mean(list(ic_irs.values()))) if ic_irs else 0.0,
            'lambda_ic': self.current_lambda_ic,
            'lambda_ortho': self.current_lambda_ortho,
            'decayed_factors': decayed,
        }
