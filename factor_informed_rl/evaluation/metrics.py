"""
评估指标模块
============
Sharpe / Sortino / Calmar / 最大回撤 / 胜率 / 盈亏比 / 因子暴露统计
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_all_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    trades: List[Dict],
    factor_exposures: np.ndarray = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """计算所有评估指标

    Args:
        returns: 日收益序列
        equity_curve: 净值曲线
        trades: 交易记录 [{date, action, price, position, pnl}, ...]
        factor_exposures: (n_days, n_factors) 因子暴露矩阵 (可选)
        risk_free_rate: 无风险利率 (年化)

    Returns:
        Dict[str, float]: 所有指标
    """
    metrics = {}

    n = len(returns)
    if n < 2:
        return metrics

    # --- 收益指标 ---
    total_return = equity_curve[-1] / equity_curve[0] - 1
    ann_return = (1 + total_return) ** (252 / n) - 1
    ann_volatility = np.std(returns) * np.sqrt(252)
    metrics['total_return'] = float(total_return)
    metrics['ann_return'] = float(ann_return)
    metrics['ann_volatility'] = float(ann_volatility)

    # --- 风险调整收益 ---
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    metrics['sharpe'] = float(sharpe)

    # Sortino (只用下行波动)
    downside = returns[returns < 0]
    if len(downside) > 1:
        sortino = np.mean(excess_returns) / (np.std(downside) + 1e-10) * np.sqrt(252)
    else:
        sortino = sharpe
    metrics['sortino'] = float(sortino)

    # --- 回撤 ---
    max_dd, max_dd_days = _compute_max_drawdown(equity_curve)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    metrics['max_drawdown'] = float(max_dd)
    metrics['max_dd_days'] = max_dd_days
    metrics['calmar'] = float(calmar)

    # --- 交易统计 ---
    if trades:
        win_trades = [t for t in trades if t.get('pnl', 0) > 0]
        loss_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = len(win_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in loss_trades])) if loss_trades else 1
        profit_factor = (
            (avg_win * len(win_trades)) / (avg_loss * len(loss_trades) + 1e-10)
        )
        n_trades = len(trades)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        n_trades = 0

    metrics['win_rate'] = float(win_rate)
    metrics['avg_win'] = float(avg_win)
    metrics['avg_loss'] = float(avg_loss)
    metrics['profit_factor'] = float(profit_factor)
    metrics['n_trades'] = n_trades

    # --- 因子暴露统计 (可选) ---
    if factor_exposures is not None:
        for i in range(factor_exposures.shape[1]):
            col = factor_exposures[:, i]
            metrics[f'factor_{i}_mean'] = float(np.mean(col))
            metrics[f'factor_{i}_std'] = float(np.std(col))
            metrics[f'factor_{i}_autocorr'] = float(
                np.corrcoef(col[:-1], col[1:])[0, 1]
            ) if len(col) > 1 else 0.0

    return metrics


def _compute_max_drawdown(equity: np.ndarray) -> Tuple[float, int]:
    """计算最大回撤和持续天数"""
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)

    # 回撤持续天数
    end_idx = np.argmin(drawdown)
    if end_idx > 0:
        start_idx = np.argmax(equity[:end_idx])
        dd_days = end_idx - start_idx
    else:
        dd_days = 0

    return float(max_dd), dd_days


def compare_experiments(
    bare_results: Dict,
    factor_informed_results: Dict,
) -> pd.DataFrame:
    """对比裸PPO和Factor-Informed PPO的实验结果"""
    comparison = pd.DataFrame({
        'Metric': [
            'Total Return', 'Ann Return', 'Ann Volatility', 'Sharpe', 'Sortino',
            'Max Drawdown', 'Calmar',
            'Win Rate', 'Profit Factor', 'N Trades',
        ],
        'Bare PPO': [
            f"{bare_results.get('total_return', 0):.2%}",
            f"{bare_results.get('ann_return', 0):.2%}",
            f"{bare_results.get('ann_volatility', 0):.2%}",
            f"{bare_results.get('sharpe', 0):.2f}",
            f"{bare_results.get('sortino', 0):.2f}",
            f"{bare_results.get('max_drawdown', 0):.1%}",
            f"{bare_results.get('calmar', 0):.2f}",
            f"{bare_results.get('win_rate', 0):.1%}",
            f"{bare_results.get('profit_factor', 0):.2f}",
            str(bare_results.get('n_trades', 0)),
        ],
        'Factor-Informed PPO': [
            f"{factor_informed_results.get('total_return', 0):.2%}",
            f"{factor_informed_results.get('ann_return', 0):.2%}",
            f"{factor_informed_results.get('ann_volatility', 0):.2%}",
            f"{factor_informed_results.get('sharpe', 0):.2f}",
            f"{factor_informed_results.get('sortino', 0):.2f}",
            f"{factor_informed_results.get('max_drawdown', 0):.1%}",
            f"{factor_informed_results.get('calmar', 0):.2f}",
            f"{factor_informed_results.get('win_rate', 0):.1%}",
            f"{factor_informed_results.get('profit_factor', 0):.2f}",
            str(factor_informed_results.get('n_trades', 0)),
        ],
    })

    return comparison.set_index('Metric')
