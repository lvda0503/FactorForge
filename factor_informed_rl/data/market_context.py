"""
市场环境上下文
==============
从 baostock 下载沪深300指数，提取6个维度的市场环境特征。

特征:
  大盘:   CSI300_ret_5d, CSI300_ret_20d, CSI300_vol_20d
  资金:   turnover_ratio, turnover_ratio_vs_ma20
  日历:   earnings_season, month_sin, month_cos
  波动:   vol_regime
  估值:   pe_percentile, pb_percentile
"""
import baostock as bs
import pandas as pd
import numpy as np
import os

from factor_informed_rl import paths


class MarketContext:
    """市场环境特征提取器

    用法:
        ctx = MarketContext()
        ctx.download_index()          # 只需一次
        features = ctx.compute(date, df_stock)  # 每步调用
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or paths.DATA_DIR
        self.csi300 = None
        self._load_or_download()

    def _load_or_download(self):
        path = f"{self.cache_dir}/baostock_csi300.pkl"
        if os.path.exists(path):
            self.csi300 = pd.read_pickle(path)
            return

        bs.login()
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,open,high,low,close,volume,amount",
            start_date="2010-01-01", end_date="2025-12-31",
            frequency="d", adjustflag="2")
        data = []
        while (rs.error_code == "0") and rs.next():
            data.append(rs.get_row_data())
        bs.logout()

        df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount"])
        for c in ["open","high","low","close","volume","amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return_1d"] = df["close"].pct_change()
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["vol_20d"] = df["return_1d"].rolling(20).std()

        df.to_pickle(path)
        self.csi300 = df

    def compute(self, date, df_stock, pe_percentile=0.5, pb_percentile=0.5,
                turnover_ratio=None):
        """提取当天市场环境特征 → numpy array

        Args:
            date: 当前日期 (pd.Timestamp)
            df_stock: 单只股票的历史数据 (用于计算PE/PB分位)
            pe_percentile: 当前PE在历史分位 (0-1)
            pb_percentile: 当前PB在历史分位
            turnover_ratio: 当前换手率

        Returns:
            np.array, shape (11,)
        """
        features = np.zeros(11, dtype=np.float32)

        # 1. 大盘环境 (CSI300)
        if self.csi300 is not None:
            idx = self.csi300.index.get_indexer([date], method="ffill")[0]
            if idx >= 0 and idx < len(self.csi300):
                row = self.csi300.iloc[idx]
                features[0] = float(np.nan_to_num(np.asarray(row.get("return_5d", 0)), nan=0.0))
                features[1] = float(np.nan_to_num(np.asarray(row.get("return_20d", 0)), nan=0.0))
                features[2] = float(np.nan_to_num(np.asarray(row.get("vol_20d", 0)), nan=0.0))

        # 2. 资金面 — 换手率
        if turnover_ratio is not None and not np.isnan(turnover_ratio):
            features[3] = float(np.clip(np.nan_to_num(np.asarray(turnover_ratio), nan=0.0), 0, 20) / 100)
        features[4] = 0.0  # turnover_vs_ma20 (需历史窗口，暂为0)

        # 3. 日历效应
        m = date.month
        features[5] = 1.0 if m in (1, 4, 7, 10) else 0.0  # 财报季
        features[6] = float(np.sin(2 * np.pi * m / 12))    # 月份正弦
        features[7] = float(np.cos(2 * np.pi * m / 12))    # 月份余弦

        # 4. 波动率体制 (从CSI300计算)
        vol_20d = features[2]
        if vol_20d > 0.02:
            features[8] = 1.0   # 高波动
        elif vol_20d > 0.01:
            features[8] = 0.5   # 中波动
        else:
            features[8] = 0.0   # 低波动

        # 5. 估值分位
        features[9] = float(np.clip(pe_percentile, 0, 1))
        features[10] = float(np.clip(pb_percentile, 0, 1))

        return features.astype(np.float32)

    @property
    def feature_dim(self):
        return 11
