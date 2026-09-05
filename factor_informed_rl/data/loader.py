"""
统一数据加载层
==============
支持 yfinance (美股) 和 baostock (A股)，返回统一格式。
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import os
import pickle

class DataLoader:
    """统一数据加载器

    用法:
        loader = DataLoader(source="yfinance", ticker="AAPL")
        df = loader.load(start="2015-01-01", end="2025-12-31")
        # df columns: [open, high, low, close, volume, pe, pb]
        # df index: DatetimeIndex
    """

    def __init__(self, source: str = "yfinance", cache_dir: str = "./data_cache"):
        self.source = source
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """加载数据 (带缓存)"""
        cache_file = os.path.join(
            self.cache_dir, f"{self.source}_{ticker}_{start}_{end}.pkl"
        )
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        if self.source == "yfinance":
            df = self._load_yfinance(ticker, start, end)
        elif self.source == "baostock":
            df = self._load_baostock(ticker, start, end)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

        # 统一列名
        df = self._standardize_columns(df)

        # 缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        return df

    def _load_yfinance(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """从 yfinance 加载美股数据"""
        import yfinance as yf

        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)

        # yfinance 返回的列: Open, High, Low, Close, Volume
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = 'date'

        # PE/PB 从 info 获取 (yfinance 不提供历史PE/PB)
        # 用最近的 PE/PB 填充 (近似)
        try:
            info = stock.info
            trailing_pe = info.get('trailingPE', None)
            price_to_book = info.get('priceToBook', None)
        except:
            trailing_pe = None
            price_to_book = None

        if trailing_pe is not None:
            df['pe'] = trailing_pe
        else:
            df['pe'] = np.nan

        if price_to_book is not None:
            df['pb'] = price_to_book
        else:
            df['pb'] = np.nan

        return df

    def _load_baostock(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """从 baostock 加载A股数据"""
        import baostock as bs

        bs.login()

        # baostock ticker格式: sh.600519 或 sz.000001
        start_fmt = start.replace('-', '')
        end_fmt = end.replace('-', '')

        rs = bs.query_history_k_data_plus(
            ticker,
            "date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn",
            start_date=start_fmt, end_date=end_fmt,
            frequency="d", adjustflag="2"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        bs.logout()

        if not data_list:
            raise RuntimeError(f"Baostock returned no data for {ticker}")

        df = pd.DataFrame(data_list, columns=[
            'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
            'pe', 'pb', 'turnover'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount',
                     'pe', 'pb', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列名和数据类型"""
        # 确保需要的列都存在
        required = ['open', 'high', 'low', 'close', 'volume', 'pe', 'pb']
        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        # 只保留需要的列
        df = df[required].copy()

        # 前向填充 PE/PB (财务数据更新频率低)
        df['pe'] = df['pe'].replace(0, np.nan).ffill()
        df['pb'] = df['pb'].replace(0, np.nan).ffill()

        # 删除 PE/PB 为负值的行 (亏损公司无意义)
        # 但保留 NaN (yfinance 可能没有)

        return df

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.70,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间顺序划分数据集"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df
