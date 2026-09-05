"""
硬过滤器 — 确定可交易的股票池
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Set


class UniverseFilter:
    """一级硬过滤: ST/停牌/流动性/新股

    用法:
        uf = UniverseFilter(min_amount=1e8)
        valid = uf.filter(stock_data, date)
        # stock_data: {code: {amount, is_st, is_suspended, listed_days}}
    """

    def __init__(
        self,
        min_amount: float = 1e8,        # 最小日成交额
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        min_listed_days: int = 60,       # 上市不足60天排除
    ):
        self.min_amount = min_amount
        self.exclude_st = exclude_st
        self.exclude_suspended = exclude_suspended
        self.min_listed_days = min_listed_days

    def filter(self, stock_data: Dict[str, Dict]) -> Set[str]:
        """过滤，返回有效股票代码集合"""
        valid = set()
        for code, info in stock_data.items():
            if info.get('amount', 0) < self.min_amount:
                continue
            if self.exclude_st and info.get('is_st', False):
                continue
            if self.exclude_suspended and info.get('is_suspended', False):
                continue
            if info.get('listed_days', 9999) < self.min_listed_days:
                continue
            valid.add(code)
        return valid

    @staticmethod
    def from_baostock_data(df_prices: dict, date):
        """从 baostock 价格 DataFrame 构建过滤数据"""
        stock_data = {}
        for code, pdf in df_prices.items():
            if date not in pdf.index:
                continue
            row = pdf.loc[date]
            stock_data[code] = {
                'amount': row.get('amount', row.get('volume', 0) * row.get('close', 0)),
                'is_st': False,  # baostock 不直接提供，需额外判断
                'is_suspended': (row.get('volume', 1) == 0),
                'listed_days': max(0, (date - pdf.index[0]).days) if len(pdf) > 0 else 9999,
            }
        return stock_data
