"""
V9: 按风格混合重训 FI-PPO Agent
Value Agent: 茅台(牛) + 浦发(价值) + 万科(熊)
Quality Agent: 恒瑞(质量) + 立讯(成长) + 美的(震荡)
"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import pandas as pd, numpy as np, torch, os

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

CONFIG = {
    "Value-Defensive": {
        "factors": ["pb_ratio","pe_percentile","rank_20","std_60","corr_20"],
        "stocks": ["600519", "600000", "000002"],  # 茅台+浦发+万科
    },
    "Quality-Offensive": {
        "factors": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
        "stocks": ["600276", "002475", "000333"],  # 恒瑞+立讯+美的
    },
}

TOTAL = 200000
OUT_DIR = "d:/JoinQuant/quant_env/factor_informed_rl/experiments/paper/v9_models"
os.makedirs(OUT_DIR, exist_ok=True)
market_ctx = MarketContext()

for sn, cfg in CONFIG.items():
    print(f"\n{'='*60}")
    print(f"  Training {sn} — {cfg['stocks']}")
    print(f"  Factors: {cfg['factors']}")
    print(f"{'='*60}")

    all_train_dfs = []
    for code in cfg["stocks"]:
        for loc in [f"d:/JoinQuant/quant_env/data_cache/baostock_{code}.pkl",
                    f"d:/JoinQuant/quant_env/data_cache/csi300/{code}.pkl"]:
            if os.path.exists(loc): break
        else: continue
        df = pd.read_pickle(loc)
        for c in ['open','high','low','close','volume','pe','pb']:
            if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)
        if 'pe_percentile' in cfg["factors"]:
            df['pe_percentile'] = df['pe'].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        train = df[df.index.year <= 2020]
        if len(train) > 300:
            all_train_dfs.append(train)
            ret = train['close'].iloc[-1]/train['close'].iloc[0]-1
            print(f"  {code}: {len(train)}d train, ret={ret:+.0%}")

    # 合并训练集 (交替拼接, 保留日期索引)
    min_len = min(len(d) for d in all_train_dfs)
    train_chunks = []
    for d in all_train_dfs:
        train_chunks.append(d.iloc[:min_len])
    train_df = pd.concat(train_chunks)

    print(f"  Merged train set: {len(train_df)} days")

    engine = FactorEngine(cfg["factors"])
    sb = StateBuilder(window_size=60, factor_names=cfg["factors"], market_dim=11)
    env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                     window_size=60, enable_short=True, market_ctx=market_ctx)
    model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
    fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                            warmup_steps=TOTAL//4)

    trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                         n_epochs=8, batch_size=256, device="cpu", entropy_coef=0.03)
    import time; t0 = time.time()
    trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)
    elapsed = time.time()-t0

    # Save
    save_path = f"{OUT_DIR}/{sn}.pt"
    torch.save({'model_state': model.state_dict(), 'factors': cfg["factors"],
                'state_dim': sb.state_dim, 'stocks': cfg["stocks"]}, save_path)
    print(f"  Saved → {save_path} ({elapsed:.0f}s)")
    print(f"  State dim: {sb.state_dim}, Model params: {sum(p.numel() for p in model.parameters()):,}")

print(f"\nDone! Models saved to {OUT_DIR}/")
