"""
多因子加权打分 + Top-K 选取
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


class FactorScorer:
    """IC_IR 加权多因子打分器

    用法:
        fs = FactorScorer(weights=None)  # None = 自动从IC_IR计算
        scores = fs.score(factor_values, ic_history)
        selected = fs.select_top_k(scores, k=10)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 ic_window: int = 120, method: str = "ic_ir_weighted"):
        """
        Args:
            weights: 手动指定因子权重, None = 自动 IC_IR 加权
            ic_window: IC 计算窗口
            method: "ic_ir_weighted" | "rank_equal" | "ic_weighted"
        """
        self.weights = weights
        self.ic_window = ic_window
        self.method = method

    def compute_ic_weights(self, factor_df: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """从历史数据计算每个因子的 IC_IR 权重"""
        weights = {}
        for factor_name in factor_df.columns:
            ics = []
            dates = sorted(set(factor_df.index.get_level_values(0)) &
                          set(returns.index.get_level_values(0)))
            for d in dates[-self.ic_window:]:
                f = factor_df.loc[d, factor_name].dropna()
                r = returns.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) >= 30:
                    ics.append(stats.spearmanr(f[common], r[common])[0])

            if ics:
                ics = np.array(ics)
                ic_mean = ics.mean()
                ic_ir = abs(ic_mean / (ics.std() + 1e-10))
                weights[factor_name] = ic_ir if ic_mean > 0 else 0.0
            else:
                weights[factor_name] = 0.0

        # 归一化
        total = sum(weights.values()) or 1.0
        for k in weights:
            weights[k] /= total
        return weights

    def score(self, factor_values: Dict[str, float]) -> Dict[str, float]:
        """对股票打分: score = &Sigma; w_i &times; rank(factor_i)"""
        if self.weights is None:
            # 回退到等权
            w = {k: 1.0/len(factor_values) for k in factor_values}
        else:
            w = self.weights

        # 收集所有股票代码
        all_codes = set()
        for vals in factor_values.values():
            if isinstance(vals, dict):
                all_codes.update(vals.keys())

        # 对每个因子做截面排名
        ranks = {}
        for fname, vals in factor_values.items():
            if not isinstance(vals, dict):
                continue
            codes_sorted = sorted(vals.keys(), key=lambda c: vals.get(c, 0))
            n = len(codes_sorted)
            for i, code in enumerate(codes_sorted):
                ranks.setdefault(code, {})[fname] = (i + 1) / (n + 1e-10)  # [0, 1]

        # 加权求和
        scores = {}
        for code in all_codes:
            s = 0.0
            for fname, fw in w.items():
                s += fw * ranks.get(code, {}).get(fname, 0.5)
            scores[code] = s

        return scores

    def select_top_k(self, scores: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
