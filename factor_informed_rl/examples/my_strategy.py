"""
用户自定义策略示例 — 三步造一个策略。

1. 继承 Strategy，定义 config（因子 + 超参）
2. 用 @register_strategy 注册
3. 实现 build_env / build_model（或复用基类默认）

运行:
  python -m factor_informed_rl list --include factor_informed_rl.examples.my_strategy
"""
from factor_informed_rl.core import Strategy, StrategyConfig, register_strategy
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic


@register_strategy("my_momentum")
class MyMomentumStrategy(Strategy):
    """动量 + 情绪因子策略。

    示例：加入情绪因子 sentiment_1d，展示如何混合新因子。
    """
    config = StrategyConfig(
        name="my_momentum",
        factors=[
            "roc_20",          # 短期动量
            "std_60",          # 波动率
            "pb_ratio",        # 估值
            "sentiment_1d",    # 情绪因子（需先跑 sentiment 模块）
        ],
        max_long=0.60,         # 更保守的仓位
        stop_loss=0.06,        # 更紧的止损
        lambda_ic=0.08,
        lambda_ortho=0.03,
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
        )

    def build_model(self):
        sb = StateBuilder(
            window_size=self.config.window_size,
            factor_names=self.config.factors,
            market_dim=self.config.market_dim,
        )
        return PPOActorCritic(sb.state_dim, 1, self.config.hidden_dims)
