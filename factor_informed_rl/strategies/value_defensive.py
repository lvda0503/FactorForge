"""
Value-Defensive 策略 — 防御型价值风格。

因子（Graham 1934 + Fama-French 1993 价值维度）:
  - pb_ratio      截面估值（1/PB）
  - pe_percentile 时序估值（PE 历史分位，PIT-safe）
  - rank_20       短期价格排名
  - std_60        低波动（防御属性）
  - corr_20       量价配合确认

参考实现，展示如何用 Strategy 基类定义一个完整策略。
"""
from factor_informed_rl.core import Strategy, StrategyConfig, register_strategy
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic


@register_strategy("value_defensive")
class ValueDefensiveStrategy(Strategy):
    config = StrategyConfig(
        name="value_defensive",
        factors=[
            "pb_ratio",
            "pe_percentile",
            "rank_20",
            "std_60",
            "corr_20",
        ],
        max_long=0.80,
        max_short=0.10,
        stop_loss=0.08,
        # 价值策略偏防御，止损更紧
        lambda_ic=0.1,
        lambda_ortho=0.05,
    )

    def __init__(self, config=None):
        super().__init__(config or self.config)

    def build_env(self, df):
        engine = FactorEngine(self.config.factors, ic_window=120)
        sb = StateBuilder(
            window_size=self.config.window_size,
            factor_names=self.config.factors,
            market_dim=self.config.market_dim,
        )
        return TradingEnv(
            df, engine, sb, Denoiser(method="none"),
            window_size=self.config.window_size,
            initial_capital=self.config.initial_capital,
            enable_short=self.config.enable_short,
            max_long_pct=self.config.max_long,
            max_short_pct=self.config.max_short,
            stop_loss_pct=self.config.stop_loss,
            commission=self.config.commission,
            stamp_tax=self.config.stamp_tax,
            slippage=self.config.slippage,
        )

    def build_model(self):
        sb = StateBuilder(
            window_size=self.config.window_size,
            factor_names=self.config.factors,
            market_dim=self.config.market_dim,
        )
        return PPOActorCritic(sb.state_dim, 1, self.config.hidden_dims)
