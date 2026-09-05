"""
集中路径配置 — 所有数据/模型路径在此定义，支持环境变量覆盖。

用法:
    from factor_informed_rl import paths
    df = pd.read_pickle(paths.data_cache("600519"))

环境变量覆盖（可选）:
    export FACTORRL_DATA_DIR=/path/to/data
"""
import os
from pathlib import Path


def _env_or_default(env_name: str, default: str) -> str:
    """读环境变量，未设置则用默认值。"""
    return os.environ.get(env_name, default)


# 数据缓存目录（默认在用户主目录下，可用环境变量覆盖）
DATA_DIR = _env_or_default(
    "FACTORRL_DATA_DIR",
    str(Path.home() / ".factorrl" / "data"),
)

# 模型保存目录
MODEL_DIR = _env_or_default(
    "FACTORRL_MODEL_DIR",
    str(Path.home() / ".factorrl" / "models"),
)


def ensure_dirs():
    """确保数据/模型目录存在。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def data_cache(code: str = None) -> str:
    """返回数据缓存路径。

    Args:
        code: 股票代码（如 "600519"），None 则返回缓存根目录
    """
    if code is None:
        return DATA_DIR
    return os.path.join(DATA_DIR, f"{code}.pkl")


def model_path(name: str) -> str:
    """返回模型保存路径。"""
    return os.path.join(MODEL_DIR, f"{name}.pt")
