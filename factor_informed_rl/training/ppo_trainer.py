"""
手写 PPO 训练器
================
核心训练循环，包含:
  - Rollout收集
  - GAE计算
  - 多轮PPO更新
  - 因子信息损失注入 (消融实验开关)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
import time

from .buffer import RolloutBuffer
from ..models.actor_critic import PPOActorCritic
from ..models.factor_loss import FactorInformedLoss
from ..preprocessing.factor_engine import FactorEngine


class PPOTrainer:
    """PPO训练器 (手写实现)

    参数:
        model: PPO Actor-Critic 网络
        factor_engine: 因子计算引擎
        factor_loss: 因子信息损失模块 (可为None → 裸PPO)
        lr_actor: Actor学习率
        lr_critic: Critic学习率
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_epsilon: PPO裁剪范围
        entropy_coef: 熵正则系数
        value_coef: 价值损失系数
        max_grad_norm: 梯度裁剪
        n_epochs: 每批经验复用次数
        batch_size: 小批量大小
        device: 训练设备
    """

    def __init__(
        self,
        model: PPOActorCritic,
        factor_engine: FactorEngine,
        factor_loss: Optional[FactorInformedLoss] = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.factor_engine = factor_engine
        self.factor_loss = factor_loss
        self.device = device

        # 优化器 (Actor和Critic分开优化是标准做法)
        self.actor_optimizer = optim.Adam(
            model.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = optim.Adam(
            model.critic.parameters(), lr=lr_critic
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # 训练统计
        self.global_step = 0
        self.episode_count = 0
        self.stats_history: List[Dict] = []

    def collect_rollout(
        self,
        env,
        buffer: RolloutBuffer,
        n_steps: int,
    ) -> float:
        """收集一批经验

        Returns:
            episode平均收益
        """
        state, _ = env.reset()
        episode_rewards = []
        total_episode_reward = 0.0

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.model.get_action(state_tensor)

            # 连续动作: action shape (batch, 1) → float
            action_val = action.squeeze(-1).cpu().numpy()[0]

            next_state, reward, terminated, truncated, info = env.step(
                float(action_val)
            )
            total_episode_reward += reward

            buffer.add(
                state=state,
                action=np.array([action_val], dtype=np.float32),
                log_prob=log_prob.item() if log_prob is not None else 0.0,
                reward=reward,
                value=value.item(),
                done=terminated or truncated,
            )

            state = next_state

            if terminated or truncated:
                episode_rewards.append(total_episode_reward)
                total_episode_reward = 0.0
                state, _ = env.reset()

            self.global_step += 1

        # 计算最后一个状态的value (用于GAE)
        if not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value = self.model(state_tensor)
            last_value = last_value.item()
        else:
            last_value = 0.0

        if buffer.idx > 0:
            buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        if episode_rewards:
            self.episode_count += len(episode_rewards)
            return np.mean(episode_rewards)
        return 0.0

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO的多轮更新

        对同一批经验做 n_epochs 次遍历，每次用小批量更新。
        """
        epoch_stats = defaultdict(list)

        for _ in range(self.n_epochs):
            batches = buffer.get_batches(self.batch_size)

            for batch_indices in batches:
                batch = buffer.get_batch_data(batch_indices)

                # --- 计算 PPO 损失 ---
                log_prob, entropy, values = self.model.evaluate(
                    batch['states'], batch['actions']
                )

                # PPO clipped surrogate loss
                ratio = torch.exp(log_prob - batch['log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch['advantages']
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                if hasattr(self, '_use_value_clipping') and self._use_value_clipping:
                    value_pred_clipped = batch['values'] + torch.clamp(
                        values - batch['values'],
                        -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss_1 = (values - batch['returns']) ** 2
                    value_loss_2 = (value_pred_clipped - batch['returns']) ** 2
                    critic_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    critic_loss = 0.5 * ((values - batch['returns']) ** 2).mean()

                # 熵损失 (鼓励探索)
                entropy_loss = -entropy.mean()

                # --- 因子信息损失 (核心创新) ---
                factor_info = {}
                if self.factor_loss is not None:
                    factor_info = self.factor_loss.forward(self.global_step)
                    factor_total = factor_info['total_factor_loss']
                else:
                    factor_total = torch.tensor(0.0, device=self.device)

                # --- 总损失 ---
                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                    + factor_total  # ← 因子约束项
                )

                # 跳过NaN batch
                if torch.isnan(total_loss):
                    continue

                # --- 反向传播 ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # 记录
                epoch_stats['actor_loss'].append(actor_loss.item())
                epoch_stats['critic_loss'].append(critic_loss.item())
                epoch_stats['entropy'].append(entropy.mean().item())
                epoch_stats['total_loss'].append(total_loss.item())
                epoch_stats['approx_kl'].append(
                    ((ratio - 1) - torch.log(ratio)).mean().item()
                )
                if self.factor_loss is not None:
                    epoch_stats['factor_loss'].append(
                        factor_total.item()
                    )

        # 汇总
        summary = {
            key: np.mean(vals) for key, vals in epoch_stats.items()
        }

        # 加入因子统计
        if self.factor_loss is not None:
            factor_stats = self.factor_loss.get_stats()
            for k, v in factor_stats.items():
                if isinstance(v, (int, float, np.floating)):
                    summary[f'factor_{k}'] = float(v)

        self.stats_history.append(summary)
        return summary

    def train(
        self,
        env,
        total_timesteps: int,
        n_steps: int = 2048,
        eval_env=None,
        eval_freq: int = 20_000,
        verbose: bool = True,
    ) -> Dict:
        """完整训练循环"""
        buffer = RolloutBuffer(n_steps, env.observation_space.shape[0], self.device)
        all_eval_results = []
        start_time = time.time()

        while self.global_step < total_timesteps:
            buffer.reset()

            # 1. 收集经验
            avg_ep_reward = self.collect_rollout(env, buffer, n_steps)

            # 2. PPO更新
            update_stats = self.update(buffer)

            # 3. 日志
            if verbose and self.global_step % 10_000 < n_steps:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / (elapsed + 1e-6)
                print(
                    f"Step {self.global_step:>8d}/{total_timesteps} | "
                    f"EpReward: {avg_ep_reward:>7.2f} | "
                    f"ActorLoss: {update_stats.get('actor_loss', 0):.4f} | "
                    f"Steps/s: {steps_per_sec:.0f}"
                )
                if self.factor_loss is not None:
                    fs = self.factor_loss.get_stats()
                    print(
                        f"  Factor: mean_IC={fs['mean_ic']:.4f} | "
                        f"IC_loss={fs['ic_loss']:.4f} | "
                        f"λ_ic={fs['lambda_ic']:.4f}"
                    )

            # 4. 评估
            if eval_env is not None and self.global_step % eval_freq < n_steps:
                eval_results = self.evaluate(eval_env)
                eval_results['step'] = self.global_step
                all_eval_results.append(eval_results)

                if verbose:
                    print(
                        f"  ── EVAL ── "
                        f"Return: {eval_results['total_return']:.2%} | "
                        f"Sharpe: {eval_results['sharpe']:.2f} | "
                        f"MaxDD: {eval_results['max_drawdown']:.1%} | "
                        f"Trades: {eval_results['trade_count']}"
                    )

        return {
            'stats_history': self.stats_history,
            'eval_results': all_eval_results,
            'total_steps': self.global_step,
            'total_episodes': self.episode_count,
            'training_time': time.time() - start_time,
        }

    def evaluate(self, env, n_episodes: int = 3) -> Dict:
        """评估当前策略"""
        total_returns = []
        sharpes = []
        max_dds = []
        trade_counts = []
        final_values = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_return = 0.0
            returns_list = []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.model.get_action(
                        state_tensor, deterministic=True
                    )
                action_val = float(action.squeeze(-1).cpu().numpy()[0])
                state, reward, terminated, truncated, info = env.step(action_val)
                done = terminated or truncated
                episode_return += reward
                returns_list.append(reward)

            total_returns.append(info.get('total_return', episode_return))
            sharpes.append(info.get('sharpe', self._calc_sharpe(returns_list)))
            max_dds.append(info.get('max_drawdown', self._calc_max_dd(returns_list)))
            trade_counts.append(info.get('trade_count', 0))
            final_values.append(info.get('total_value', 0))

        return {
            'total_return': float(np.mean(total_returns)),
            'total_return_std': float(np.std(total_returns)),
            'sharpe': float(np.mean(sharpes)),
            'max_drawdown': float(np.mean(max_dds)),
            'trade_count': float(np.mean(trade_counts)),
            'final_value': float(np.mean(final_values)),
        }

    def _calc_sharpe(self, returns: list) -> float:
        if len(returns) < 2:
            return 0.0
        rets = np.array(returns)
        return float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))

    def _calc_max_dd(self, returns: list) -> float:
        if not returns:
            return 0.0
        cum = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cum)
        return float(np.min((cum - running_max) / running_max))

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'stats_history': self.stats_history,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        self.stats_history = checkpoint['stats_history']
