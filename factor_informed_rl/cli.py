"""
FactorRL 统一命令行入口。

用法:
  python -m factor_informed_rl list                          # 列出已注册策略
  python -m factor_informed_rl train --strategy value_defensive --stock 600519
  python -m factor_informed_rl backtest --strategy value_defensive --stock 600519
  python -m factor_informed_rl show --strategy value_defensive  # 查看配置

用户自定义策略（examples/my_strategy.py）:
  python -m factor_informed_rl list --include examples.my_strategy
"""
import sys
import argparse
import importlib

# 导入内置策略（触发注册）
import factor_informed_rl.strategies  # noqa: F401
from factor_informed_rl.core import list_strategies, get_strategy
from factor_informed_rl import paths


def cmd_list(args):
    """列出所有已注册策略。"""
    if args.include:
        for mod in args.include.split(','):
            importlib.import_module(mod.strip())

    strategies = list_strategies()
    if not strategies:
        print("没有已注册的策略。请导入策略模块。")
        return

    print(f"{'名称':<25} {'因子数':<6} {'最大做多':<8} {'最大做空':<8}")
    print("-" * 60)
    for name, cls in strategies.items():
        cfg = cls.config
        print(f"{name:<25} {len(cfg.factors):<6} {cfg.max_long:<8} {cfg.max_short:<8}")


def cmd_show(args):
    """查看策略配置详情。"""
    cls = get_strategy(args.strategy)
    cfg = cls.config
    print(f"策略: {cfg.name}")
    print(f"因子: {cfg.factors}")
    print(f"风控: long={cfg.max_long} short={cfg.max_short} stop_loss={cfg.stop_loss}")
    print(f"FI-PPO: lambda_ic={cfg.lambda_ic} lambda_ortho={cfg.lambda_ortho} warmup={cfg.warmup_steps}")
    print(f"模型: hidden={cfg.hidden_dims}")


def cmd_train(args):
    """训练策略。"""
    cls = get_strategy(args.strategy)
    strategy = cls()
    df = strategy.load_data(args.stock, args.cache)
    print(f"[Train] {strategy.config.name} on {args.stock}")
    path = strategy.train(df, stock_name=args.stock)
    print(f"[Train] Saved to {path}")


def cmd_backtest(args):
    """回测策略。"""
    cls = get_strategy(args.strategy)
    strategy = cls()
    df = strategy.load_data(args.stock, args.cache)
    print(f"[Backtest] {strategy.config.name} on {args.stock}")
    result = strategy.backtest(df)
    for k, v in result.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="FactorRL 统一入口")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="列出策略")
    p_list.add_argument("--include", help="额外导入的模块，逗号分隔")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="查看策略配置")
    p_show.add_argument("--strategy", required=True)
    p_show.set_defaults(func=cmd_show)

    p_train = sub.add_parser("train", help="训练策略")
    p_train.add_argument("--strategy", required=True)
    p_train.add_argument("--stock", default="600519")
    p_train.add_argument("--cache", default=paths.DATA_DIR)
    p_train.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest", help="回测策略")
    p_bt.add_argument("--strategy", required=True)
    p_bt.add_argument("--stock", default="600519")
    p_bt.add_argument("--cache", default=paths.DATA_DIR)
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
