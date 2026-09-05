"""
PPO Actor-Critic v2 — 连续动作空间
====================================
Actor输出 Gaussian 分布的 mean + log_std (不是Categorical)
Critic输出 V(s) 不变

动作采样: action = tanh(mean + std * noise)  → 自动约束到 [-1, 1]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.network = nn.Sequential(*layers)
        self.output_dim = prev
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))

    def forward(self, x):
        return self.network(x)


class GaussianActor(nn.Module):
    """连续动作Actor — 输出 Gaussian (mean, log_std)

    Actor(state) → (mean, log_std) → 采样 action = tanh(mean + std * noise)
    使用 tanh 将动作压缩到 [-1, 1]
    """

    def __init__(self, feature_dim, action_dim=1, log_std_init=-0.5):
        super().__init__()
        self.action_dim = action_dim
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        orthogonal_init(self.mean_head, gain=0.01)

    def forward(self, features):
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, features, deterministic=False):
        mean, std = self.forward(features)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.clamp(torch.nan_to_num(std, nan=0.5), min=1e-4, max=10.0)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # 从 Gaussian 采样
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()  # reparameterized sampling
            action = torch.tanh(raw_action)
            # 计算 tanh 变换后的 log_prob
            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def evaluate(self, features, actions):
        """给定特征和动作，返回 log_prob + entropy (用于PPO更新)"""
        mean, std = self.forward(features)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.clamp(torch.nan_to_num(std, nan=0.5), min=1e-4, max=10.0)
        dist = torch.distributions.Normal(mean, std)

        # 反推 raw_action: atanh(actions)
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * torch.log((1 + actions_clamped) / (1 - actions_clamped + 1e-6))

        log_prob = dist.log_prob(raw_actions)
        log_prob -= torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)
        orthogonal_init(self.head, gain=1.0)

    def forward(self, features):
        return self.head(features).squeeze(-1)


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic — 连续动作版

    State → FeatureExtractor → GaussianActor → action ∈ [-1, +1]
                             → Critic → V(s)
    """

    def __init__(self, input_dim, action_dim=1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.feature_extractor = FeatureExtractor(input_dim, hidden_dims)
        self.actor = GaussianActor(self.feature_extractor.output_dim, action_dim)
        self.critic = Critic(self.feature_extractor.output_dim)

    def forward(self, state):
        """返回 (action_mean, value) — 兼容训练器"""
        features = self.feature_extractor(state)
        mean, _ = self.actor(features)
        value = self.critic(features)
        return mean, value

    def get_action(self, state, deterministic=False):
        features = self.feature_extractor(state)
        action, log_prob = self.actor.get_action(features, deterministic)
        value = self.critic(features)
        return action, log_prob, value

    def evaluate(self, state, action):
        features = self.feature_extractor(state)
        log_prob, entropy = self.actor.evaluate(features, action)
        value = self.critic(features)
        return log_prob, entropy, value
