"""
Quality-Offensive 策略 — 进攻型质量/动量风格。

因子（动量 + 趋势确认维度）:
  - roc_60  长期动量（趋势确认）
  - beta_20 趋势斜率（速度与方向）
  - rsqr_20 拟合度 R²（防假突破）
  - vma_20  放量确认（量在价先）
  - std_20  适度波动（有波动才有 Alpha）

参考实现，展示同一框架下换因子列表即换策略风格。
"""
from factor_informed_rl.core import Strategy, StrategyConfig, register_strategy
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic


@register_strategy("quality_offensive")
class QualityOffensiveStrategy(Strategy):
    config = StrategyConfig(
        name="quality_offensive",
        factors=[
            "roc_60",
            "beta_20",
            "rsqr_20",
            "vma_20",
            "std_20",
        ],
        max_long=0.80,
        max_short=0.10,
        stop_loss=0.08,
        # 进攻型策略，因子约束可稍强以控制回撤
        lambda_ic=0.15,
        lambda_ortho=0.08,
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
