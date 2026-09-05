"""
Generate equity curves and position over time for paper figures.
Re-evaluates trained models or generates from cached results.
"""
import sys; sys.path.insert(0, r'd:\JoinQuant\quant_env')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.0,
})

OUTPUT = os.path.dirname(__file__) + '/figures'
os.makedirs(OUTPUT, exist_ok=True)

C_BARE = '#4472C4'; C_FI = '#ED7D31'; C_BH = '#A5A5A5'
C_LONG = '#27AE60'; C_SHORT = '#E74C3C'
STOCKS = {"600519":"Moutai","000858":"Wuliangye","000333":"Midea","600276":"Hengrui"}
TOTAL = 200000
cache = "d:/JoinQuant/quant_env/data_cache"

from factor_informed_rl.data.market_context import MarketContext
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.preprocessing.denoiser import Denoiser
from factor_informed_rl.env.trading_env import TradingEnv
from factor_informed_rl.env.state_builder import StateBuilder
from factor_informed_rl.models.actor_critic import PPOActorCritic
from factor_informed_rl.models.factor_loss import FactorInformedLoss
from factor_informed_rl.training.ppo_trainer import PPOTrainer

market_ctx = MarketContext()

def test_with_tracking(env, model, test_df):
    """Run test episode and record all states"""
    state, _ = env.reset()
    done = False
    records = []
    while not done:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = model.get_action(s, deterministic=True)
        action_val = float(a.squeeze().numpy())
        prev_pos = env._current_position()
        state, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated
        records.append({
            'date': test_df.index[env.idx - 1] if env.idx < len(test_df) else test_df.index[-1],
            'total_value': info['total_value'],
            'long_pct': info.get('long_position', 0),
            'short_pct': info.get('short_position', 0),
            'net_pos': env._current_position(),
            'action': action_val,
            'trade': env.trade_count,
        })
    return pd.DataFrame(records).set_index('date')


def generate_one_stock(code, name):
    """Train + record equity curves for one stock"""
    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]

    results = {}
    for use_fi, label in [(False, "Bare"), (True, "FI")]:
        engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
        sb = StateBuilder(window_size=60, market_dim=11)

        env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
        fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                                warmup_steps=TOTAL//4) if use_fi else None
        trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                             n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)
        trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

        test_env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                              window_size=60, enable_short=True, market_ctx=market_ctx)
        records = test_with_tracking(test_env, model, test_df)
        results[label] = records
        print(f"  [{name}] {label}: final_value={records['total_value'].iloc[-1]:.0f}")

    # Buy & Hold
    bh_equity = test_df['close'] / test_df['close'].iloc[0] * 100000

    return results, bh_equity


# ================================================================
# Figure 9: 4-panel equity curves  (2x2 subplot)
# ================================================================
def fig9_equity_curves():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (code, name) in zip(axes.flatten(), STOCKS.items()):
        results, bh = generate_one_stock(code, name)

        bare_equity = results["Bare"]['total_value']
        fi_equity   = results["FI"]['total_value']

        # Normalize to start at 100
        bare_norm = bare_equity / bare_equity.iloc[0] * 100
        fi_norm   = fi_equity / fi_equity.iloc[0] * 100
        bh_norm   = bh / bh.iloc[0] * 100

        ax.plot(bare_equity.index, bare_norm, color=C_BARE, linewidth=1.2, label='Vanilla PPO', alpha=0.9)
        ax.plot(fi_equity.index,   fi_norm,   color=C_FI,   linewidth=1.5, label='FI-PPO (Ours)', alpha=0.9)
        ax.plot(bh.index,          bh_norm,   color=C_BH,   linewidth=0.8, label='Buy & Hold', linestyle='--', alpha=0.7)

        ax.axhline(y=100, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_title(f'{name} ({code})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Equity (Base=100)')
        ax.legend(frameon=False, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        # Annotate final value
        final_bare = bare_norm.iloc[-1]
        final_fi   = fi_norm.iloc[-1]
        ax.text(0.98, 0.95, f'FI: {final_fi:.0f}\nBare: {final_bare:.0f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Figure 9: Equity Curves — FI-PPO vs Vanilla PPO vs Buy & Hold (2022–2025)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT}/fig9_equity_curves.pdf', format='pdf')
    fig.savefig(f'{OUTPUT}/fig9_equity_curves.png', format='png')
    plt.close()
    print('[OK] Figure 9: Equity Curves')


# ================================================================
# Figure 10: Position over time for one representative stock (Moutai)
# ================================================================
def fig10_positions():
    code, name = "600519", "Moutai"

    path = f"{cache}/baostock_{code}.pkl"
    df = pd.read_pickle(path)
    for c in ['open','high','low','close','volume','pe','pb','turn']:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0)

    train_df = df[df.index.year <= 2020]
    test_df = df[df.index.year >= 2022]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9),
                                         gridspec_kw={'height_ratios': [2, 1.5, 1.5]})

    results = {}
    for use_fi, label, color in [(False, "Bare", C_BARE), (True, "FI", C_FI)]:
        engine = FactorEngine(["roc_20","rsv_14","std_20","pb_ratio","corr_20"])
        sb = StateBuilder(window_size=60, market_dim=11)
        env = TradingEnv(train_df, engine, sb, Denoiser(method="none"),
                         window_size=60, enable_short=True, market_ctx=market_ctx)
        model = PPOActorCritic(sb.state_dim, 1, [256,128,64])
        fl = FactorInformedLoss(engine, lambda_ic=0.1, lambda_ortho=0.05,
                                warmup_steps=TOTAL//4) if use_fi else None
        trainer = PPOTrainer(model, engine, fl, lr_actor=3e-4, lr_critic=1e-3,
                             n_epochs=6, batch_size=256, device="cpu", entropy_coef=0.03)
        trainer.train(env, total_timesteps=TOTAL, n_steps=1024, verbose=False)

        test_env = TradingEnv(test_df, engine, sb, Denoiser(method="none"),
                              window_size=60, enable_short=True, market_ctx=market_ctx)
        records = test_with_tracking(test_env, model, test_df)
        results[label] = records

    # Top panel: FI net position
    fi_rec = results["FI"]
    ax1.fill_between(fi_rec.index, 0, fi_rec['net_pos'].clip(lower=0),
                     color=C_LONG, alpha=0.3, label='Long')
    ax1.fill_between(fi_rec.index, fi_rec['net_pos'].clip(upper=0), 0,
                     color=C_SHORT, alpha=0.3, label='Short')
    ax1.plot(fi_rec.index, fi_rec['net_pos'], color=C_FI, linewidth=0.8)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Net Position')
    ax1.set_title(f'FI-PPO Position Over Time — {name} ({code})', fontsize=12)
    ax1.legend(frameon=False, fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Middle panel: Bare net position
    bare_rec = results["Bare"]
    ax2.fill_between(bare_rec.index, 0, bare_rec['net_pos'].clip(lower=0),
                     color=C_LONG, alpha=0.3)
    ax2.fill_between(bare_rec.index, bare_rec['net_pos'].clip(upper=0), 0,
                     color=C_SHORT, alpha=0.3)
    ax2.plot(bare_rec.index, bare_rec['net_pos'], color=C_BARE, linewidth=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Net Position')
    ax2.set_title('Vanilla PPO Position', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Bottom panel: Price (background context)
    ax3.plot(test_df.index, test_df['close'] / test_df['close'].iloc[0] * 100,
             color='#333', linewidth=0.8)
    ax3.set_ylabel('Price (Base=100)')
    ax3.set_title('Moutai Price (Normalized)', fontsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle('Figure 10: Position Dynamics — FI-PPO vs Vanilla PPO', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT}/fig10_positions.pdf', format='pdf')
    fig.savefig(f'{OUTPUT}/fig10_positions.png', format='png')
    plt.close()
    print('[OK] Figure 10: Position Curves')


if __name__ == '__main__':
    print('Generating equity & position curves...\n')
    print('Figure 9 (this takes ~35min for 4 stocks)...')
    fig9_equity_curves()
    print('\nFigure 10 (this takes ~6min for Moutai)...')
    fig10_positions()
    print(f'\nDone! Figures saved to: {OUTPUT}/')
