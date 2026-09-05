"""
信号预处理层
============
接口预留：小波降噪 / 卡尔曼滤波 / 简单MA。

MVE阶段使用简单MA作为基准，
后续可替换为 wavelet (pywt) 或 Kalman (pykalman)。
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseDenoiser(ABC):
    """降噪器基类"""
    @abstractmethod
    def denoise(self, series: np.ndarray) -> np.ndarray:
        """输入一维序列，输出去噪后序列 (等长)"""
        pass


class MADenoiser(BaseDenoiser):
    """简单移动平均降噪 (MVE基准)"""
    def __init__(self, window: int = 5):
        self.window = window

    def denoise(self, series: np.ndarray) -> np.ndarray:
        if len(series) < self.window:
            return series
        # 保持边缘: 用更小的窗口处理边界
        from numpy.lib.stride_tricks import sliding_window_view
        result = np.zeros_like(series)
        for i in range(len(series)):
            start = max(0, i - self.window + 1)
            result[i] = np.mean(series[start:i+1])
        return result


class NoDenoiser(BaseDenoiser):
    """无降噪 (直通)"""
    def denoise(self, series: np.ndarray) -> np.ndarray:
        return series


class Denoiser:
    """降噪器工厂 (MVE阶段用简单MA，后续可升级)"""

    def __init__(self, method: str = "ma", **kwargs):
        if method == "ma":
            self.denoiser = MADenoiser(**kwargs)
        elif method == "none":
            self.denoiser = NoDenoiser()
        elif method == "wavelet":
            raise NotImplementedError("小波降噪将在Phase 2实现")
        elif method == "kalman":
            raise NotImplementedError("卡尔曼滤波将在Phase 2实现")
        else:
            raise ValueError(f"Unknown denoising method: {method}")

    def denoise(self, series: np.ndarray) -> np.ndarray:
        return self.denoiser.denoise(series)
