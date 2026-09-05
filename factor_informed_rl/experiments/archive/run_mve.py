"""
最小可行实验 (MVE)
===================
消融实验: 裸PPO vs Factor-Informed PPO

运行:
    cd d:\JoinQuant\quant_env
    python -m factor_informed_rl.experiments.run_mve

输出:
    logs/                          # 训练日志
    checkpoints/                   # 模型权重
    results_comparison.csv         # 结果对比表
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from factor_informed_rl.config import cfg, DataConfig, PPOConfig, FactorInformedConfig, EnvConfig
from factor_informed_rl.data.loader import DataLoader
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer
from factor_informed_rl.evaluation.metrics import compute_all_metrics, compare_experiments


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_env(
    df: pd.DataFrame,
    factor_engine: FactorEngine,
    env_cfg: EnvConfig,
    window_size: int = 60,
    denoiser_method: str = "none",
):
    """创建交易环境"""
    state_builder = StateBuilder(
        window_size=window_size,
        factor_names=factor_engine.factor_names,
    )
    denoiser = Denoiser(method=denoiser_method)

    env = TradingEnv(
        df=df,
        factor_engine=factor_engine,
        state_builder=state_builder,
        denoiser=denoiser,
        window_size=window_size,
        position_sizes=env_cfg.position_sizes,
        initial_capital=env_cfg.initial_capital,
        commission=env_cfg.commission,
        slippage=env_cfg.slippage,
    )
    return env


def run_experiment(
    experiment_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_factor_loss: bool,
    cfg,
):
    """运行一次完整实验

    Args:
        use_factor_loss: True = Factor-Informed PPO, False = 裸PPO

    Returns:
        train_stats, eval_results, test_metrics
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Factor-Informed Loss: {use_factor_loss}")
    print(f"{'='*60}\n")

    set_seed(cfg.seed)

    # 1. 因子引擎
    factor_engine = FactorEngine(
        factor_names=cfg.factor.factors,
        ic_window=cfg.factor.ic_window,
    )

    # 2. 环境
    train_env = create_env(train_df, factor_engine, cfg.env, denoiser_method="none")
    val_env = create_env(val_df, factor_engine, cfg.env, denoiser_method="none")
    test_env = create_env(test_df, factor_engine, cfg.env, denoiser_method="none")

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Train days: {len(train_df)}, Val days: {len(val_df)}, Test days: {len(test_df)}")

    # 3. 模型
    model = PPOActorCritic(
        input_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=cfg.ppo.actor_hidden,
    )

    # 4. 因子损失 (核心创新)
    factor_loss = None
    if use_factor_loss:
        factor_loss = FactorInformedLoss(
            factor_engine=factor_engine,
            lambda_ic=cfg.factor_loss.lambda_ic,
            lambda_ortho=cfg.factor_loss.lambda_ortho,
            min_pos_ic=cfg.factor_loss.min_pos_ic,
            max_corr=cfg.factor_loss.max_corr,
            warmup_steps=cfg.factor_loss.warmup_steps,
            lambda_growth_rate=cfg.factor_loss.lambda_growth_rate,
        )

    # 5. 训练器
    trainer = PPOTrainer(
        model=model,
        factor_engine=factor_engine,
        factor_loss=factor_loss,
        lr_actor=cfg.ppo.lr_actor,
        lr_critic=cfg.ppo.lr_critic,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_epsilon=cfg.ppo.clip_epsilon,
        entropy_coef=cfg.ppo.entropy_coef,
        value_coef=cfg.ppo.value_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        n_epochs=cfg.ppo.n_epochs,
        batch_size=cfg.ppo.batch_size,
        device=cfg.device,
    )

    # 6. 训练
    print("  Starting training...")
    result = trainer.train(
        env=train_env,
        total_timesteps=cfg.ppo.total_timesteps,
        n_steps=cfg.ppo.n_steps,
        eval_env=val_env,
        eval_freq=cfg.ppo.eval_freq,
        verbose=True,
    )

    # 7. 测试集评估
    print("\n  Running test evaluation...")
    test_metrics_list = []
    for _ in range(10):  # 跑10次取平均
        state, _ = test_env.reset()
        done = False
        episode_returns = []
        equity = [test_env.initial_capital]
        trades = []
        factor_exposures = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, deterministic=True)

            prev_position = test_env.position
            state, reward, terminated, truncated, info = test_env.step(action.item())
            done = terminated or truncated

            episode_returns.append(reward)
            equity.append(info['total_value'])

            # 记录交易
            if test_env.position != prev_position:
                trades.append({
                    'step': len(episode_returns),
                    'action': action.item(),
                    'price': info.get('current_price', 0),
                    'position': test_env.position,
                    'pnl': reward,
                    'factor_values': info.get('factor_values', {}),
                })

            # 记录因子暴露
            if info.get('factor_values'):
                factor_exposures.append(list(info['factor_values'].values()))

        equity = np.array(equity)
        returns = np.array(episode_returns)
        fe_matrix = np.array(factor_exposures) if factor_exposures else None

        metrics = compute_all_metrics(returns, equity, trades, fe_matrix)
        test_metrics_list.append(metrics)

    # 汇总测试指标
    test_metrics = {}
    for key in test_metrics_list[0].keys():
        vals = [m[key] for m in test_metrics_list if key in m]
        test_metrics[key] = float(np.mean(vals)) if vals else 0.0
        test_metrics[f'{key}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0

    test_metrics['experiment'] = experiment_name
    test_metrics['use_factor_loss'] = use_factor_loss

    return result, test_metrics


def main():
    print("=" * 60)
    print("  Factor-Informed RL — Minimal Viable Experiment")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 数据加载 ──
    print("\n[1/4] Loading data...")
    loader = DataLoader(source=cfg.data.source)
    df = loader.load(
        ticker=cfg.data.ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
    )
    print(f"  Loaded {len(df)} days: {df.index[0].date()} → {df.index[-1].date()}")

    train_df, val_df, test_df = loader.split_data(
        df, cfg.data.train_split, cfg.data.val_split
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 实验 1: 裸PPO (无因子约束) ──
    print("\n[2/4] Running Bare PPO (baseline)...")
    bare_results, bare_test = run_experiment(
        experiment_name="Bare_PPO",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        use_factor_loss=False,
        cfg=cfg,
    )

    # ── 实验 2: Factor-Informed PPO ──
    print("\n[3/4] Running Factor-Informed PPO...")
    fi_results, fi_test = run_experiment(
        experiment_name="Factor_Informed_PPO",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        use_factor_loss=True,
        cfg=cfg,
    )

    # ── 结果对比 ──
    print("\n[4/4] Results Comparison")
    print("=" * 60)

    comparison = compare_experiments(bare_test, fi_test)
    print("\n" + comparison.to_string())

    # 保存结果
    os.makedirs(cfg.save_dir, exist_ok=True)
    comparison.to_csv(os.path.join(cfg.save_dir, "results_comparison.csv"))

    # 保存详细指标
    pd.DataFrame([bare_test, fi_test]).to_csv(
        os.path.join(cfg.save_dir, "detailed_metrics.csv"), index=False
    )

    print(f"\n  Results saved to: {cfg.save_dir}/")
    print("=" * 60)

    # 简单结论
    bare_sharpe = bare_test.get('sharpe', 0)
    fi_sharpe = fi_test.get('sharpe', 0)
    delta = (fi_sharpe - bare_sharpe) / (abs(bare_sharpe) + 1e-6) * 100

    print(f"\n  Summary:")
    print(f"    Bare PPO Sharpe:            {bare_sharpe:.4f}")
    print(f"    Factor-Informed PPO Sharpe:  {fi_sharpe:.4f}")
    print(f"    Improvement:                 {delta:+.1f}%")
    print(f"    Bare PPO MaxDD:              {bare_test.get('max_drawdown', 0):.1%}")
    print(f"    Factor-Informed PPO MaxDD:    {fi_test.get('max_drawdown', 0):.1%}")

    return bare_test, fi_test, comparison


if __name__ == "__main__":
    main()
