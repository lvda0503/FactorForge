"""
因子非线性变换 — 四种方法

1. SplineTransformer: 惩罚B样条自动学习平滑非线性函数
2. PolynomialTransformer: 多项式扩展 + Lasso 自动选有效项
3. QuantileTransformer:  分位数归一化 + GAM 独立学习
4. TreeBinTransformer:   LightGBM 自动最优分箱离散化

统一接口: fit(因子值, 收益) → transform(因子值) → 变换后因子
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class SplineTransformer:
    """惩罚B样条: 自动学习平滑非线性映射 f(x) &rarr; predicted_alpha

    数学: f(x) = &Sigma;&beta;&#x1d62; &middot; B&#x1d62;(x)
    其中 B&#x1d62; 是三次B样条基函数, &beta;&#x1d62; 由 OLS 估计

    Args:
        df: 自由度 (基函数数量 = df + degree)
        degree: 样条阶数 (3 = 三次样条)
    """

    def __init__(self, df: int = 6, degree: int = 3):
        self.df = df; self.degree = degree
        self.knots_ = {}; self.betas_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """学习样条系数"""
        from scipy.interpolate import splrep
        x_clean = X[~np.isnan(X) & ~np.isnan(y)]
        y_clean = y[~np.isnan(X) & ~np.isnan(y)]
        if len(x_clean) < 50: return self
        try:
            tck = splrep(x_clean, y_clean, k=self.degree, s=len(x_clean)*0.5)
            self.knots_['knots'] = tck[0]
            self.betas_['coeff'] = tck[1]
            self.fitted_ = True
        except:
            self.fitted_ = False
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用样条变换"""
        from scipy.interpolate import splev
        if not getattr(self, 'fitted_', False):
            return np.zeros_like(X)
        try:
            return splev(X, (self.knots_['knots'], self.betas_['coeff'], self.degree))
        except:
            return np.zeros_like(X)


class PolynomialTransformer:
    """多项式扩展 + Lasso 特征选择

    候选特征池: [x, x², x³, &radic;x, log(|x|+1), 1/x]
    LassoCV 自动选出有增量信息的项
    """

    def __init__(self, max_degree: int = 3, include_inverse: bool = True):
        self.max_degree = max_degree
        self.include_inverse = include_inverse
        self.selected_features_ = []

    def _build_candidates(self, X: np.ndarray) -> np.ndarray:
        X = X.flatten()
        features = [X]
        for d in range(2, self.max_degree + 1):
            features.append(np.power(X, d))
        features.append(np.sqrt(np.abs(X)) * np.sign(X))
        features.append(np.log(np.abs(X) + 1) * np.sign(X))
        if self.include_inverse:
            features.append(1.0 / (X + 1e-8 * np.sign(X)))
        return np.column_stack(features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LassoCV
        mask = ~np.isnan(X) & ~np.isnan(y)
        X_clean, y_clean = X[mask], y[mask]
        if len(X_clean) < 50: return self
        candidates = self._build_candidates(X_clean)
        lasso = LassoCV(cv=5, max_iter=5000).fit(candidates, y_clean)
        self.selected_features_ = list(np.where(lasso.coef_ != 0)[0])
        self.coef_ = lasso.coef_
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, 'fitted_', False) or not self.selected_features_:
            return np.zeros_like(X)
        candidates = self._build_candidates(X)
        return candidates[:, self.selected_features_] @ self.coef_[self.selected_features_]


class QuantileTransformer:
    """分位数归一化: 把因子值映射到 [0,1] 均匀分布

    消除极端值影响, 不改变排序。
    fit 阶段: 学习 CDF
    transform: 应用 CDF 映射
    """

    def __init__(self):
        self.cdf_vals_ = {}; self.cdf_probs_ = {}

    def fit(self, X: np.ndarray, y=None):
        x_sorted = np.sort(X[~np.isnan(X)])
        n = len(x_sorted)
        self.cdf_vals_ = x_sorted[::max(1, n//500)]
        self.cdf_probs_ = np.linspace(0, 1, len(self.cdf_vals_))
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            return np.zeros_like(X)
        return np.interp(X, self.cdf_vals_, self.cdf_probs_)


class TreeBinTransformer:
    """LightGBM 最优分箱: 用树模型自动发现因子最优离散化边界

    Args:
        max_leaves: 最多分箱数
        max_depth: 树深度
    """

    def __init__(self, max_leaves: int = 15, max_depth: int = 4):
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.bin_encoder_ = {}  # {bin_id: average_return}

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.tree import DecisionTreeRegressor
        X_2d = X.reshape(-1, 1)
        mask = ~np.isnan(X) & ~np.isnan(y)
        X_clean, y_clean = X_2d[mask], y[mask]
        if len(X_clean) < 100: return self
        tree = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaves, max_depth=self.max_depth
        ).fit(X_clean, y_clean)
        self.tree_ = tree
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            return np.zeros_like(X)
        X_2d = X.reshape(-1, 1)
        leaf_ids = self.tree_.apply(X_2d)
        # 每个叶节点的预测值 = 该分箱的平均超额收益
        result = np.zeros(len(X))
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            result[mask] = self.tree_.tree_.value[leaf][0, 0]
        return result


# ============================================================
class FactorTransformer:
    """统一因子变换器

    对每个因子:
      1. 用训练集 fit 非线性形状
      2. 对比原始IC vs 变换后IC
      3. 只对IC提升的因子采用变换
      4. transform 时应用变换

    用法:
        ft = FactorTransformer(method="spline")
        ft.fit(factor_df, returns, dates)
        transformed = ft.transform(factor_df, date)
    """

    def __init__(self, method: str = "spline", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.transformers: Dict[str, object] = {}
        self.adopted: Dict[str, bool] = {}
        self.ic_before: Dict[str, float] = {}
        self.ic_after: Dict[str, float] = {}

    def fit(self, factor_df: pd.DataFrame, returns: pd.Series,
            dates: list = None, min_ic_improve: float = 0.005):
        """
        Args:
            factor_df: MultiIndex (date, code), columns = factor names
            returns:   MultiIndex (date, code), 前向收益
            min_ic_improve: IC至少提升此阈值才采用变换
        """
        common_dates = sorted(set(factor_df.index.get_level_values(0)) &
                              set(returns.index.get_level_values(0)))
        for factor_name in factor_df.columns:
            # 收集训练集所有日期的因子值+收益
            X_all, y_all = [], []
            for d in common_dates:
                f = factor_df.loc[d, factor_name].dropna()
                r = returns.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) >= 30:
                    X_all.extend(f[common].values)
                    y_all.extend(r[common].values)
            X_all = np.array(X_all); y_all = np.array(y_all)
            if len(X_all) < 100:
                continue

            # 原始 IC
            self.ic_before[factor_name] = stats.spearmanr(X_all, y_all)[0]

            # 创建 & 拟合变换器
            t = self._create_transformer()
            t.fit(X_all, y_all)
            X_transformed = t.transform(X_all)

            # 变换后 IC
            self.ic_after[factor_name] = stats.spearmanr(X_transformed, y_all)[0]

            # 判断是否采用
            ic_delta = abs(self.ic_after[factor_name]) - abs(self.ic_before[factor_name])
            if ic_delta > min_ic_improve:
                self.adopted[factor_name] = True
                self.transformers[factor_name] = t
            else:
                self.adopted[factor_name] = False

    def _create_transformer(self):
        if self.method == "spline":
            return SplineTransformer(**self.kwargs)
        elif self.method == "polynomial":
            return PolynomialTransformer(**self.kwargs)
        elif self.method == "quantile":
            return QuantileTransformer(**self.kwargs)
        elif self.method == "treebin":
            return TreeBinTransformer(**self.kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def transform(self, factor_df: pd.DataFrame, date) -> Dict[str, float]:
        """对单个日期的因子值应用变换"""
        result = {}
        try:
            row = factor_df.loc[date]
        except:
            return result

        for factor_name in factor_df.columns:
            vals = row[factor_name].dropna().values if hasattr(row[factor_name], 'values') else np.array([row[factor_name]])
            if len(vals) == 0:
                continue
            if self.adopted.get(factor_name, False):
                transformed = self.transformers[factor_name].transform(vals)
                for i, code in enumerate(row[factor_name].index if hasattr(row[factor_name], 'index') else [0]):
                    key = code if hasattr(row[factor_name], 'index') else factor_name
                    val = transformed[i] if i < len(transformed) else 0.0
                    result[key] = val
            else:
                # 不采用变换，返回原始值
                if hasattr(row[factor_name], 'index'):
                    for code, val in row[factor_name].items():
                        result[code] = float(val) if not np.isnan(val) else 0.0
        return result

    def summary(self) -> pd.DataFrame:
        """报告每个因子的变换效果"""
        rows = []
        for fn in self.ic_before:
            rows.append({
                'factor': fn,
                'IC_raw': round(self.ic_before[fn], 5),
                'IC_transformed': round(self.ic_after.get(fn, 0), 5),
                'adopted': self.adopted.get(fn, False),
            })
        return pd.DataFrame(rows)
