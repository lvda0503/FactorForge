"""
Rollout Buffer (PPO经验缓冲区)
===============================
存储 agent 与环境交互产生的一系列 transitions，
用于后续的 PPO 多轮更新。

结构:
  states:      (n_steps, state_dim)
  actions:     (n_steps,)
  log_probs:   (n_steps,)
  rewards:     (n_steps,)
  values:      (n_steps,)
  dones:       (n_steps,)
  advantages:  (n_steps,)  — GAE计算后填充
  returns:     (n_steps,)  — 折扣回报
"""
import torch
import numpy as np
from typing import Optional


class RolloutBuffer:
    def __init__(self, n_steps: int, state_dim: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.device = device

        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, 1), dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)

        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, state, action, log_prob, reward, value, done):
        """添加一个 transition (连续动作版)"""
        if self.idx >= self.n_steps:
            self.full = True
            return
        self.states[self.idx] = state
        self.actions[self.idx] = action  # float [1]
        self.log_probs[self.idx] = log_prob
        self.rewards[self.idx] = reward
        self.values[self.idx] = value
        self.dones[self.idx] = float(done)
        self.idx += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """使用 GAE (Generalized Advantage Estimation) 计算 advantages 和 returns

        GAE公式:
          δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t = δ_t + γλ * (1 - done_t) * A_{t+1}

        Returns: R_t = A_t + V_t
        """
        advantages = np.zeros_like(self.rewards)

        gae = 0.0
        for t in reversed(range(self.idx)):
            if t == self.idx - 1:
                next_value = last_value
                next_done = False
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD误差
            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - next_done)
                - self.values[t]
            )

            # GAE累加
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + self.values[:self.idx]

        # 标准化 advantages (稳定训练)
        if self.idx > 1:
            self.advantages = (
                (self.advantages - np.mean(self.advantages))
                / (np.std(self.advantages) + 1e-8)
            )

    def get_batches(self, batch_size: int) -> list:
        """生成随机小批量的索引"""
        indices = np.random.permutation(self.idx)
        start_idx = 0
        batches = []

        while start_idx < self.idx:
            end_idx = min(start_idx + batch_size, self.idx)
            batch_indices = indices[start_idx:end_idx]
            batches.append(batch_indices)
            start_idx = end_idx

        return batches

    def get_batch_data(self, indices: np.ndarray) -> dict:
        """根据索引提取批量数据 (转为torch tensor)"""
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'advantages': torch.FloatTensor(self.advantages[indices]).to(self.device),
            'returns': torch.FloatTensor(self.returns[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
        }
