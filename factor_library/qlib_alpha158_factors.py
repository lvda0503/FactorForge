"""
Qlib Alpha158 因子库 — 聚宽 Factor 类实现
===========================================
将 Microsoft Qlib 的 158 个手工因子完整翻译为聚宽平台可用的 Factor 类。

因子构成 (默认配置):
  K-bar 形态因子:   9 个  (基于 OHLC 的 K 线形态)
  价格因子:         4 个  (OPEN0/HIGH0/LOW0/VWAP0)
  滚动算子因子:   145 个  (29 种算子 × 5 个窗口 [5,10,20,30,60])
  ─────────────────────
  Alpha158 标准:  158 个
  额外附赠:         1 个  (VOLUME0)
  本文件总计:     159 个

使用方法 (聚宽策略中):
    from qlib_alpha158_factors import *

    # 在策略中直接使用
    factor_values = get_factor_values(context, [KMID(), KLEN(), ROC5(), ROC10(), ...], universe)

    # 或在单因子分析中使用
    from jqfactor import Factor, calc_factors
    values = calc_factors(universe, [KMID(), MA20(), STD10(), ...], start_date, end_date)

依赖:
    聚宽 jqfactor 模块 (Factor 基类, calc_factors)
    numpy, pandas

参考:
    Qlib Alpha158: https://github.com/microsoft/qlib
    原始定义: qlib/contrib/data/loader.py (Alpha158DL.get_feature_config)

作者: 基于 Qlib Alpha158 翻译
日期: 2025-06-24
"""

from jqfactor import Factor
import numpy as np
import pandas as pd


# ============================================================================
# 工具函数
# ============================================================================

def _safe_div(a, b):
    """安全除法，避免除零"""
    return a / (b + 1e-12)


# ============================================================================
# 第一部分：K-bar 形态因子 (9个)
# ============================================================================
# 说明：基于当日 OHLC 数据计算的 K 线形态特征。全部 max_window=1。

class KMID(Factor):
    """K线实体涨幅: (收盘-开盘)/开盘"""
    name = 'kmid'
    max_window = 1
    dependencies = ['open', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        return (c - o) / o


class KLEN(Factor):
    """K线振幅: (最高-最低)/开盘"""
    name = 'klen'
    max_window = 1
    dependencies = ['open', 'high', 'low']

    def calc(self, data):
        o = data['open'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        return (h - l) / o


class KMID2(Factor):
    """标准化K线实体: (收盘-开盘)/(最高-最低)"""
    name = 'kmid2'
    max_window = 1
    dependencies = ['open', 'high', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        return (c - o) / (h - l + 1e-12)


class KUP(Factor):
    """上影线: (最高 - max(开盘,收盘)) / 开盘"""
    name = 'kup'
    max_window = 1
    dependencies = ['open', 'high', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        h = data['high'].iloc[-1]
        c = data['close'].iloc[-1]
        greater = np.maximum(o, c)
        return (h - greater) / o


class KUP2(Factor):
    """标准化上影线: (最高 - max(开盘,收盘)) / (最高-最低)"""
    name = 'kup2'
    max_window = 1
    dependencies = ['open', 'high', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        greater = np.maximum(o, c)
        return (h - greater) / (h - l + 1e-12)


class KLOW(Factor):
    """下影线: (min(开盘,收盘) - 最低) / 开盘"""
    name = 'klow'
    max_window = 1
    dependencies = ['open', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        l = data['low'].iloc[-1]
        lesser = np.minimum(o, c)
        return (lesser - l) / o


class KLOW2(Factor):
    """标准化下影线: (min(开盘,收盘) - 最低) / (最高-最低)"""
    name = 'klow2'
    max_window = 1
    dependencies = ['open', 'high', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        lesser = np.minimum(o, c)
        return (lesser - l) / (h - l + 1e-12)


class KSFT(Factor):
    """价格位移: (2*收盘-最高-最低)/开盘"""
    name = 'ksft'
    max_window = 1
    dependencies = ['open', 'high', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        return (2 * c - h - l) / o


class KSFT2(Factor):
    """标准化价格位移: (2*收盘-最高-最低)/(最高-最低)"""
    name = 'ksft2'
    max_window = 1
    dependencies = ['open', 'high', 'low', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        h = data['high'].iloc[-1]
        l = data['low'].iloc[-1]
        return (2 * c - h - l) / (h - l + 1e-12)


# ============================================================================
# 第二部分：价格因子 (4个)
# ============================================================================
# 说明：当日价格除以当日收盘价进行标准化

class OPEN0(Factor):
    """标准化开盘价: 开盘/收盘"""
    name = 'open0'
    max_window = 1
    dependencies = ['open', 'close']

    def calc(self, data):
        o = data['open'].iloc[-1]
        c = data['close'].iloc[-1]
        return o / c


class HIGH0(Factor):
    """标准化最高价: 最高/收盘"""
    name = 'high0'
    max_window = 1
    dependencies = ['high', 'close']

    def calc(self, data):
        h = data['high'].iloc[-1]
        c = data['close'].iloc[-1]
        return h / c


class LOW0(Factor):
    """标准化最低价: 最低/收盘"""
    name = 'low0'
    max_window = 1
    dependencies = ['low', 'close']

    def calc(self, data):
        l = data['low'].iloc[-1]
        c = data['close'].iloc[-1]
        return l / c


class VWAP0(Factor):
    """标准化均价: 均价(VWAP)/收盘"""
    name = 'vwap0'
    max_window = 1
    dependencies = ['money', 'volume', 'close']

    def calc(self, data):
        vwap = data['money'].iloc[-1] / data['volume'].iloc[-1]
        c = data['close'].iloc[-1]
        return vwap / c


# ============================================================================
# 第三部分：成交量因子 (5个)
# ============================================================================
# 说明：当日成交量除以成交量自身进行标准化（带滞后的因子）

class VOLUME0(Factor):
    """标准化成交量(当日): volume/volume = 1"""
    name = 'volume0'
    max_window = 1
    dependencies = ['volume']

    def calc(self, data):
        v = data['volume'].iloc[-1]
        return v / (v + 1e-12)


# ============================================================================
# 第四部分：滚动算子因子 — 工厂函数
# ============================================================================
# 说明：28 种算子 × 5 个窗口 = 140 个因子
# 使用工厂函数批量生成 Factor 子类，避免手动写 140 个类

def _make_rolling_factor(name, max_window, dependencies, calc_fn):
    """动态创建 Factor 子类的工厂函数"""
    return type(
        name.upper(),
        (Factor,),
        {
            'name': name,
            'max_window': max_window,
            'dependencies': dependencies,
            'calc': calc_fn,
        }
    )


# ---- 滚动算子实现函数 ------------------------------------------------

# 窗口列表
_ROLLING_WINDOWS = [5, 10, 20, 30, 60]

# ROC — 价格动量 (Rate of Change)
# 公式: close[t-d] / close[t]
# 在Qlib中: Ref($close, d) / $close

def _make_roc_calc(d):
    def calc(self, data):
        close = data['close']
        return close.iloc[-(d+1)] / close.iloc[-1]
    return calc


# MA — 简单移动平均
# 公式: Mean(close, d) / close[t]

def _make_ma_calc(d):
    def calc(self, data):
        close = data['close']
        return close.iloc[-d:].mean() / close.iloc[-1]
    return calc


# STD — 收盘价波动率
# 公式: Std(close, d) / close[t]

def _make_std_calc(d):
    def calc(self, data):
        close = data['close']
        return close.iloc[-d:].std() / close.iloc[-1]
    return calc


# BETA — 价格斜率 (线性回归斜率)
# 公式: Slope(close, d) / close[t]
# Beta = Cov(x, y) / Var(x)，x = [0,1,...,d-1], y = 过去d天收盘价

def _make_beta_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-d:]
        x = np.arange(d)
        x_mean = x.mean()
        y_mean = close.mean(axis=0)
        beta = ((x - x_mean).reshape(-1, 1) * (close.values - y_mean.values)).sum(axis=0) / ((x - x_mean) ** 2).sum()
        return pd.Series(beta / close.iloc[-1].values, index=close.columns)
    return calc


# RSQR — 线性回归 R²
# 公式: Rsquare(close, d)
# R² = 1 - SS_res / SS_tot

def _make_rsqr_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-d:]
        x = np.arange(d)
        x_mean = x.mean()
        y_mean = close.mean(axis=0)
        ss_tot = ((close.values - y_mean.values) ** 2).sum(axis=0)
        # 简单线性回归
        beta = ((x - x_mean).reshape(-1, 1) * (close.values - y_mean.values)).sum(axis=0) / ((x - x_mean) ** 2).sum()
        y_pred = y_mean.values + beta * (x.reshape(-1, 1) - x_mean)
        ss_res = ((close.values - y_pred) ** 2).sum(axis=0)
        rsqr = 1 - ss_res / (ss_tot + 1e-12)
        return pd.Series(rsqr, index=close.columns)
    return calc


# RESI — 线性回归残差
# 公式: Resi(close, d) / close[t]

def _make_resi_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-d:]
        x = np.arange(d)
        x_mean = x.mean()
        y_mean = close.mean(axis=0)
        beta = ((x - x_mean).reshape(-1, 1) * (close.values - y_mean.values)).sum(axis=0) / ((x - x_mean) ** 2).sum()
        y_pred_last = y_mean.values + beta * ((d - 1) - x_mean)
        resid = close.iloc[-1].values - y_pred_last
        return pd.Series(resid / close.iloc[-1].values, index=close.columns)
    return calc


# MAX — d日最高价的最大值
# 公式: Max(high, d) / close[t]

def _make_max_calc(d):
    def calc(self, data):
        high = data['high']
        close = data['close']
        return high.iloc[-d:].max() / close.iloc[-1]
    return calc


# MIN — d日最低价的最小值
# 公式: Min(low, d) / close[t]

def _make_min_calc(d):
    def calc(self, data):
        low = data['low']
        close = data['close']
        return low.iloc[-d:].min() / close.iloc[-1]
    return calc


# QTLU — d日收盘价的 80% 分位数
# 公式: Quantile(close, d, 0.8) / close[t]

def _make_qtlu_calc(d):
    def calc(self, data):
        close = data['close']
        return close.iloc[-d:].quantile(0.8) / close.iloc[-1]
    return calc


# QTLD — d日收盘价的 20% 分位数
# 公式: Quantile(close, d, 0.2) / close[t]

def _make_qtld_calc(d):
    def calc(self, data):
        close = data['close']
        return close.iloc[-d:].quantile(0.2) / close.iloc[-1]
    return calc


# RANK — 当前收盘价在d日内的百分位排名
# 公式: Rank(close, d)
# 返回 0~1 之间的值，越大表示当前价格在过去d天中越高

def _make_rank_calc(d):
    def calc(self, data):
        close = data['close']
        window_data = close.iloc[-d:]
        # 对每只股票，计算最新价在窗口中的排名百分比
        return window_data.rank().iloc[-1] / d
    return calc


# RSV — 价格在d日最高最低之间的位置 (类似KDJ的RSV)
# 公式: (close[t] - Min(low, d)) / (Max(high, d) - Min(low, d) + 1e-12)

def _make_rsv_calc(d):
    def calc(self, data):
        high = data['high']
        low = data['low']
        close = data['close']
        h_max = high.iloc[-d:].max()
        l_min = low.iloc[-d:].min()
        return (close.iloc[-1] - l_min) / (h_max - l_min + 1e-12)
    return calc


# IMAX — 距上次最高价的天数比例 (Aroon Up)
# 公式: IdxMax(high, d) / d
# 返回 0~1，越小表示最近创新高

def _make_imax_calc(d):
    def calc(self, data):
        high = data['high']
        window_high = high.iloc[-d:]
        # 找到最高价出现的位置索引 (0 = 最早, d-1 = 最晚/今天)
        max_idx = window_high.values.argmax(axis=0)
        # 距今天的天数 = d - 1 - max_idx
        days_ago = (d - 1) - max_idx
        return pd.Series(days_ago / d, index=high.columns)
    return calc


# IMIN — 距上次最低价的天数比例 (Aroon Down)
# 公式: IdxMin(low, d) / d

def _make_imin_calc(d):
    def calc(self, data):
        low = data['low']
        window_low = low.iloc[-d:]
        min_idx = window_low.values.argmin(axis=0)
        days_ago = (d - 1) - min_idx
        return pd.Series(days_ago / d, index=low.columns)
    return calc


# IMXD — 最高价出现日与最低价出现日的时间差
# 公式: (IdxMax(high, d) - IdxMin(low, d)) / d
# 正值表示高点先于低点出现（潜在下跌趋势）

def _make_imxd_calc(d):
    def calc(self, data):
        high = data['high']
        low = data['low']
        window_high = high.iloc[-d:]
        window_low = low.iloc[-d:]
        max_idx = window_high.values.argmax(axis=0)
        min_idx = window_low.values.argmin(axis=0)
        return pd.Series((max_idx - min_idx) / d, index=high.columns)
    return calc


# CORR — 收盘价与对数成交量的相关性
# 公式: Corr(close, Log(volume+1), d)

def _make_corr_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-d:]
        volume = data['volume'].iloc[-d:]
        log_vol = np.log(volume + 1)
        return close.corrwith(log_vol)
    return calc


# CORD — 价格变化率与成交量变化率的相关性
# 公式: Corr(close/Ref(close,1), Log(volume/Ref(volume,1)+1), d)

def _make_cord_calc(d):
    def calc(self, data):
        close = data['close']
        volume = data['volume']
        # 需要 d+1 个数据点（d个变化率需要d+1个原始值）
        ret = close.iloc[-d-1:].pct_change().iloc[-d:]
        vol_change = volume.iloc[-d-1:].pct_change().iloc[-d:]
        log_vol_change = np.log(vol_change + 1)
        return ret.corrwith(log_vol_change)
    return calc


# CNTP — d日内上涨天数比例
# 公式: Mean(close > Ref(close, 1), d)

def _make_cntp_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-(d+1):]
        up = close.diff().iloc[-d:] > 0
        return up.mean()
    return calc


# CNTN — d日内下跌天数比例
# 公式: Mean(close < Ref(close, 1), d)

def _make_cntn_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-(d+1):]
        down = close.diff().iloc[-d:] < 0
        return down.mean()
    return calc


# CNTD — d日内上涨天数比例与下跌天数比例之差
# 公式: Mean(up, d) - Mean(down, d)

def _make_cntd_calc(d):
    def calc(self, data):
        close = data['close'].iloc[-(d+1):]
        diff = close.diff().iloc[-d:]
        up = diff > 0
        down = diff < 0
        return up.mean() - down.mean()
    return calc


# SUMP — d日总涨幅 / 总绝对变化 (类似RSI)
# 公式: Sum(max(close-Ref(close,1), 0), d) / Sum(abs(close-Ref(close,1)), d)

def _make_sump_calc(d):
    def calc(self, data):
        close = data['close']
        # 需要 d+1 天数据
        diff = close.iloc[-d-1:].diff().iloc[-d:]
        gain = diff.clip(lower=0).sum()
        total = diff.abs().sum()
        return gain / (total + 1e-12)
    return calc


# SUMN — d日总跌幅 / 总绝对变化
# 公式: Sum(max(Ref(close,1)-close, 0), d) / Sum(abs(close-Ref(close,1)), d)

def _make_sumn_calc(d):
    def calc(self, data):
        close = data['close']
        diff = close.iloc[-d-1:].diff().iloc[-d:]
        loss = (-diff).clip(lower=0).sum()
        total = diff.abs().sum()
        return loss / (total + 1e-12)
    return calc


# SUMD — d日净涨跌比例 (RSI类指标)
# 公式: (总涨 - 总跌) / 总绝对变化

def _make_sumd_calc(d):
    def calc(self, data):
        close = data['close']
        diff = close.iloc[-d-1:].diff().iloc[-d:]
        net = diff.sum()
        total = diff.abs().sum()
        return net / (total + 1e-12)
    return calc


# VMA — 成交量移动平均
# 公式: Mean(volume, d) / (volume[t] + 1e-12)

def _make_vma_calc(d):
    def calc(self, data):
        volume = data['volume']
        return volume.iloc[-d:].mean() / (volume.iloc[-1] + 1e-12)
    return calc


# VSTD — 成交量波动率
# 公式: Std(volume, d) / (volume[t] + 1e-12)

def _make_vstd_calc(d):
    def calc(self, data):
        volume = data['volume']
        return volume.iloc[-d:].std() / (volume.iloc[-1] + 1e-12)
    return calc


# WVMA — 成交量加权的价格波动率
# 公式: Std(abs(close/Ref(close,1)-1)*volume, d) / (Mean(abs(ret)*volume, d) + 1e-12)

def _make_wvma_calc(d):
    def calc(self, data):
        close = data['close']
        volume = data['volume']
        # 需要 d+1 天数据
        ret = close.iloc[-d-1:].pct_change().iloc[-d:]
        weighted = ret.abs() * volume.iloc[-d:]
        return weighted.std() / (weighted.mean() + 1e-12)
    return calc


# VSUMP — 成交量增加天数比例
# 公式: Sum(volume > Ref(volume,1), d) / Sum(abs(volume-Ref(volume,1)), d)

def _make_vsump_calc(d):
    def calc(self, data):
        volume = data['volume']
        diff = volume.iloc[-d-1:].diff().iloc[-d:]
        gain = diff.clip(lower=0).sum()
        total = diff.abs().sum()
        return gain / (total + 1e-12)
    return calc


# VSUMN — 成交量减少天数比例
# 公式: Sum(volume < Ref(volume,1), d) / Sum(abs(volume-Ref(volume,1)), d)

def _make_vsumn_calc(d):
    def calc(self, data):
        volume = data['volume']
        diff = volume.iloc[-d-1:].diff().iloc[-d:]
        loss = (-diff).clip(lower=0).sum()
        total = diff.abs().sum()
        return loss / (total + 1e-12)
    return calc


# VSUMD — 成交量净增加比例
# 公式: (Sum(vol_up, d) - Sum(vol_down, d)) / Sum(abs(vol_diff), d)

def _make_vsumd_calc(d):
    def calc(self, data):
        volume = data['volume']
        diff = volume.iloc[-d-1:].diff().iloc[-d:]
        net = diff.sum()
        total = diff.abs().sum()
        return net / (total + 1e-12)
    return calc


# ============================================================================
# 批量生成 140 个滚动算子因子 (28操作符 × 5窗口)
# ============================================================================

# 操作符定义：(因子名前缀, 工厂函数, 额外依赖)
_ROLLING_OPERATORS = [
    ('roc',  _make_roc_calc,  ['close']),
    ('ma',   _make_ma_calc,   ['close']),
    ('std',  _make_std_calc,  ['close']),
    ('beta', _make_beta_calc, ['close']),
    ('rsqr', _make_rsqr_calc, ['close']),
    ('resi', _make_resi_calc, ['close']),
    ('max',  _make_max_calc,  ['close', 'high']),
    ('min',  _make_min_calc,  ['close', 'low']),
    ('qtlu', _make_qtlu_calc, ['close']),
    ('qtld', _make_qtld_calc, ['close']),
    ('rank', _make_rank_calc, ['close']),
    ('rsv',  _make_rsv_calc,  ['close', 'high', 'low']),
    ('imax', _make_imax_calc, ['high']),
    ('imin', _make_imin_calc, ['low']),
    ('imxd', _make_imxd_calc, ['high', 'low']),
    ('corr', _make_corr_calc, ['close', 'volume']),
    ('cord', _make_cord_calc, ['close', 'volume']),
    ('cntp', _make_cntp_calc, ['close']),
    ('cntn', _make_cntn_calc, ['close']),
    ('cntd', _make_cntd_calc, ['close']),
    ('sump', _make_sump_calc, ['close']),
    ('sumn', _make_sumn_calc, ['close']),
    ('sumd', _make_sumd_calc, ['close']),
    ('vma',  _make_vma_calc,  ['volume']),
    ('vstd', _make_vstd_calc, ['volume']),
    ('wvma', _make_wvma_calc, ['close', 'volume']),
    ('vsump',_make_vsump_calc,['volume']),
    ('vsumn',_make_vsumn_calc,['volume']),
    ('vsumd',_make_vsumd_calc,['volume']),
]

# 动态生成并注册到当前模块的全局命名空间
for _prefix, _maker_fn, _deps in _ROLLING_OPERATORS:
    for _w in _ROLLING_WINDOWS:
        _name = f'{_prefix}{_w}'
        _cls = _make_rolling_factor(
            name=_name,
            max_window=_w + 1,  # +1 是为了计算变化率（如CNTP需要d+1天数据）
            dependencies=_deps,
            calc_fn=_maker_fn(_w)
        )
        globals()[_name.upper()] = _cls

del _prefix, _maker_fn, _deps, _w, _name, _cls


# ============================================================================
# 第五部分：完整因子列表 (159个)
# ============================================================================

# K-bar 形态因子 (9个)
KBAR_FACTORS = [KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2]

# 价格因子 (4个)
PRICE_FACTORS = [OPEN0, HIGH0, LOW0, VWAP0]

# 成交量因子 (1个 — 附赠，非Alpha158标配)
VOLUME_FACTORS = [VOLUME0]

# 滚动算子因子 (29 × 5 = 145个)
ROLLING_FACTORS = []
for prefix, _, _ in _ROLLING_OPERATORS:
    for w in _ROLLING_WINDOWS:
        cls_name = f'{prefix}{w}'.upper()
        ROLLING_FACTORS.append(globals()[cls_name])

# 全部因子: 9 K-bar + 4 价格 + 1 成交量 + 145 滚动 = 159
ALL_FACTORS = KBAR_FACTORS + PRICE_FACTORS + VOLUME_FACTORS + ROLLING_FACTORS

# 分类字典 (便于筛选)
FACTOR_CATEGORIES = {
    'kbar':     [f.__name__ for f in KBAR_FACTORS],
    'price':    [f.__name__ for f in PRICE_FACTORS],
    'volume':   [f.__name__ for f in VOLUME_FACTORS],
    'momentum': [f.__name__ for f in ROLLING_FACTORS if f.name.startswith('roc')],
    'ma':       [f.__name__ for f in ROLLING_FACTORS if f.name.startswith('ma') and not f.name.startswith('max')],
    'volatility':[f.__name__ for f in ROLLING_FACTORS if f.name.startswith('std')],
    'trend':    [f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('beta', 'rsqr', 'resi'))],
    'price_range':[f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('max', 'min', 'qtlu', 'qtld', 'rank', 'rsv'))],
    'aroon':    [f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('imax', 'imin', 'imxd'))],
    'correlation':[f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('corr', 'cord'))],
    'price_strength':[f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('cntp', 'cntn', 'cntd', 'sump', 'sumn', 'sumd'))],
    'volume_pattern':[f.__name__ for f in ROLLING_FACTORS if f.name.startswith(('vma', 'vstd', 'wvma', 'vsump', 'vsumn', 'vsumd'))],
}

# 因子的 max_window 信息 (用于确定 calc_factors 的窗口参数)
FACTOR_WINDOWS = {f.name: f.max_window for f in ALL_FACTORS}


def print_factor_summary():
    """打印因子库概览"""
    print("=" * 60)
    print("Qlib Alpha158 因子库 — 聚宽 Factor 实现")
    print("=" * 60)
    print(f"  K-bar 形态:     {len(KBAR_FACTORS):3d} 个  (KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2)")
    print(f"  标准化价格:     {len(PRICE_FACTORS):3d} 个  (OPEN0, HIGH0, LOW0, VWAP0)")
    print(f"  成交量:         {len(VOLUME_FACTORS):3d} 个  (VOLUME0)")
    print(f"  滚动算子:       {len(ROLLING_FACTORS):3d} 个  (29种算子 × 5窗口 [5,10,20,30,60])")
    print(f"  {'─' * 50}")
    print(f"  总计:           {len(ALL_FACTORS):3d} 个")
    print("=" * 60)

    print("\n滚动算子分类:")
    for cat, names in FACTOR_CATEGORIES.items():
        if cat in ('kbar', 'price', 'volume'):
            continue
        print(f"  {cat:20s}: {names[0]} ~ {names[-1]}  ({len(names)}个)")

    print("\n使用示例 (聚宽策略中):")
    print("  from qlib_alpha158_factors import KMID, ROC5, MA20, STD10")
    print("  factor_values = get_factor_values(context, [KMID(), ROC5(), MA20()], universe)")

    print("\n最大回溯窗口: 61天 (部分因子需要60+1天数据)")
    print("最轻量因子: KMID(max_window=1)")
    print("最重量因子: WVMA60(max_window=61)")
