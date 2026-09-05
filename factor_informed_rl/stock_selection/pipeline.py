"""
选股管线 — 一键: 过滤 &rarr; 中性化 &rarr; 非线性变换 &rarr; 打分
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os, pickle

from .config import SelectionConfig, cfg as default_cfg
from .universe import UniverseFilter
from .neutralizer import BarraNeutralizer
from .factor_transform import FactorTransformer
from .scorer import FactorScorer


class StockSelector:
    """选股管线

    用法:
        selector = StockSelector(config)
        selected = selector.select(date="2025-06-28")
        # &rarr; [("600519", 0.92), ("000858", 0.87), ...]

    管线:
        硬过滤 &rarr; Barra中性化 &rarr; 非线性变换 &rarr; IC_IR加权打分 &rarr; Top-K
    """

    def __init__(self, config: SelectionConfig = None, data_cache: str = None):
        self.config = config or default_cfg

        # 硬过滤器
        self.universe_filter = UniverseFilter(
            min_amount=self.config.universe_cfg.min_daily_amount,
            exclude_st=self.config.universe_cfg.exclude_st,
            exclude_suspended=self.config.universe_cfg.exclude_suspended,
            min_listed_days=self.config.universe_cfg.exclude_new_listed,
        )

        # Barra 中性化器 (延迟加载行业映射)
        self.neutralizer: Optional[BarraNeutralizer] = None
        self._industry_map: Dict[str, str] = {}
        self._init_industry(data_cache)

        # 非线性变换器
        self.transformer = FactorTransformer(
            method=self.config.transform.method,
            df=self.config.transform.spline_df,
            degree=self.config.transform.spline_degree,
            max_degree=self.config.transform.poly_max_degree,
            include_inverse=self.config.transform.poly_include_inverse,
            max_leaves=self.config.transform.tree_max_leaves,
            max_depth=self.config.transform.tree_max_depth,
        )

        # 打分器
        self.scorer = FactorScorer(
            method=self.config.scorer.method,
            ic_window=self.config.scorer.ic_window,
        )

        self.fitted = False

    def _init_industry(self, cache_dir):
        """加载行业分类数据"""
        if cache_dir is None:
            cache_dir = "d:/JoinQuant/quant_env/data_cache"
        path = os.path.join(cache_dir, "baostock_industry.pkl")
        if os.path.exists(path):
            df = pd.read_pickle(path)
            if isinstance(df, pd.DataFrame) and 'code' in df.columns:
                self._industry_map = dict(zip(df['code'], df['industry']))
            elif isinstance(df, dict):
                self._industry_map = df
        if self._industry_map:
            self.neutralizer = BarraNeutralizer(self._industry_map)

    def fit(self, factor_df: pd.DataFrame, returns: pd.Series,
            market_caps: pd.Series = None):
        """在历史数据上拟合变换器

        这一步需要在训练集上运行，之后可以复用。
        fit 完成后会打日志: 哪些因子采用了非线性变换，IC提升了多少。
        """
        # 拟合非线性变换
        self.transformer.fit(factor_df, returns,
                            min_ic_improve=self.config.transform.min_ic_improvement)

        # 计算 IC_IR 权重
        self.scorer.weights = self.scorer.compute_ic_weights(factor_df, returns)

        self.fitted = True

        # 打印变换效果
        summary = self.transformer.summary()
        if len(summary) > 0:
            n_adopted = summary['adopted'].sum()
            print(f"[FactorTransform] {n_adopted}/{len(summary)} factors improved by nonlinear transform")
            for _, row in summary.iterrows():
                if row['adopted']:
                    delta = abs(row['IC_transformed']) - abs(row['IC_raw'])
                    print(f"  {row['factor']}: IC {row['IC_raw']:+.4f} &rarr; {row['IC_transformed']:+.4f} (&Delta;={delta:+.4f})")

        return self

    def select(self, date, factor_values: Dict[str, Dict[str, float]],
               market_caps: Dict[str, float] = None) -> List:
        """
        对某个交易日执行完整的选股流程

        Args:
            date: 日期
            factor_values: {factor_name: {code: value}}
            market_caps:    {code: market_cap}

        Returns:
            [(code, score), ...] 按分数降序
        """
        # Step 1: 硬过滤
        stock_data = UniverseFilter.from_baostock_data(factor_values, date)
        valid = self.universe_filter.filter(stock_data)

        # Step 2: 过滤因子值
        filtered_factors = {}
        for fname, vals in factor_values.items():
            filtered_factors[fname] = {c: v for c, v in vals.items() if c in valid}

        # Step 3: Barra 中性化
        if self.neutralizer is not None and self.config.barra.enable:
            for fname in list(filtered_factors.keys()):
                filtered_factors[fname] = self.neutralizer.neutralize(
                    filtered_factors[fname], market_caps or {})

        # Step 4: 非线性变换 (if fitted)
        if self.fitted:
            # 构造临时 DataFrame
            rows = []
            for fname, vals in filtered_factors.items():
                for code, v in vals.items():
                    rows.append({'date': date, 'code': code, 'factor': fname, 'value': v})
            if rows:
                tmp = pd.DataFrame(rows).pivot_table(
                    index=['date','code'], columns='factor', values='value')
                transformed = self.transformer.transform(tmp, date)
                # merge back
                for code in filtered_factors.get(list(filtered_factors.keys())[0], {}):
                    if code in transformed:
                        for fname in filtered_factors:
                            filtered_factors[fname][code] = transformed.get(code, 0.0)

        # Step 5: IC_IR 加权打分
        scores = self.scorer.score(filtered_factors)

        # Step 6: Top-K
        return self.scorer.select_top_k(scores, k=self.config.scorer.top_k)
