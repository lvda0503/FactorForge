"""
硬过滤层 — 打分前砍掉不合格股票
=================================
Value策略: 防价值陷阱 (净资产负、长期亏损、流动性差)
Quality策略: 防技术反弹陷阱 (长期趋势向下、假突破)
"""
import numpy as np


def value_filter(stock_data):
    """Value-Defensive 硬门槛

    Args:
        stock_data: {code: {pb, pe, roe_ttm, amount, close, ma_200, is_st, ...}}
    Returns:
        set of valid stock codes
    """
    valid = set()
    for code, d in stock_data.items():
        # PB > 0 (净资产不为负)
        if d.get('pb', 0) <= 0:
            continue
        # PB < 50 (排除极端值/数据错误)
        if d.get('pb', 0) > 50:
            continue
        # PE无亏损或非极端 (PE < 0 = 亏损, > 200 = 微利噪点)
        pe = d.get('pe', 20)
        if pe < 0 or pe > 200:
            continue
        # 日成交额 > 1亿 (流动性)
        if d.get('amount', 0) < 1e8:
            continue
        # Close > 0 (数据完整性)
        if d.get('close', 0) <= 0:
            continue
        valid.add(code)
    return valid


def quality_filter(stock_data):
    """Quality-Offensive 硬门槛

    防技术反弹陷阱: 长期趋势必须向上, 短中期信号才可信
    """
    valid = set()
    for code, d in stock_data.items():
        # ROC_250 > 5% (年动量必须为正)
        if d.get('roc_250', -1) < 0.05:
            continue
        # Close > MA_200 (站上牛熊线)
        if d.get('close', 0) <= d.get('ma_200', 1e10):
            continue
        # 日成交额 > 2亿 (流动性, 比Value更严)
        if d.get('amount', 0) < 2e8:
            continue
        # PE > 0 (不亏损)
        if d.get('pe', 20) < 0:
            continue
        # Close > 0
        if d.get('close', 0) <= 0:
            continue
        valid.add(code)
    return valid
