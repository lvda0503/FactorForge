import sys; sys.path.insert(0, 'd:/JoinQuant/quant_env')
import numpy as np
from factor_informed_rl.preprocessing.factor_engine import FactorEngine
from factor_informed_rl.env.state_builder import StateBuilder

strategies = {
    "Value-Defensive": ["pb_ratio","pe_percentile","rsv_14","std_60","corr_20"],
    "Quality-Offensive": ["roc_60","beta_20","rsqr_20","vma_20","std_20"],
}

for name, factors in strategies.items():
    e = FactorEngine(factors)
    p = np.random.randn(70,4).cumsum(axis=0)+100; p[:,3]=p[:,3].clip(10)
    v = np.abs(np.random.randn(70)*5000+10000)
    r = e.compute_factors(p, v, pb_value=3.5, pe_percentile=0.5)
    sb = StateBuilder(window_size=60, factor_names=factors, market_dim=11)
    print(f"{name}: {len(r)}/{len(factors)} factors, state={sb.state_dim}")
    for k,v in sorted(r.items()):
        print(f"  {k}: {v:.4f}")
    print()
print("SMOKE PASSED")
