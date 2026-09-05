# 贡献指南

感谢你对 FactorRL 的关注！以下是贡献前需要了解的事项。

---

## 环境配置

FactorRL 使用集中路径配置（`factor_informed_rl/paths.py`），默认数据目录在用户主目录下：

```
~/.factorrl/
├── data/       # 数据缓存（pkl）
└── models/     # 训练好的模型（pt）
```

如需自定义，设置环境变量：

```bash
export FACTORRL_DATA_DIR=/path/to/your/data
export FACTORRL_MODEL_DIR=/path/to/your/models
```

> ⚠️ **不要提交硬编码的本地路径**（如 `D:\JoinQuant\...`）。使用 `paths.py` 提供的 `paths.data_cache()` 等函数。

---

## 自定义策略

继承 `Strategy` 基类 + 注册即可（详见 README「自定义策略」一节）：

```python
from factor_informed_rl.core import Strategy, StrategyConfig, register_strategy

@register_strategy("my_strategy")
class MyStrategy(Strategy):
    config = StrategyConfig(name="my_strategy", factors=["roc_20", "std_60"])
```

---

## 提交规范

1. **敏感信息检查**：提交前运行 `git diff --cached | grep -iE "your_account|account_id|资金账号|password"`，确保无账号泄露
2. **路径检查**：确认无硬编码本地路径（`D:\JoinQuant` 等）
3. **代码风格**：遵循现有代码风格（英文注释优先，核心创新处可用中文）
4. **测试**：`python -m factor_informed_rl list` 确认策略注册正常

---

## 目录约定

| 目录 | 内容 | 是否入库 |
|------|------|---------|
| `core/` `strategies/` `env/` `models/` `training/` | 核心库 | ✅ |
| `preprocessing/` `stock_selection/` `portfolio/` `sentiment/` | 功能模块 | ✅ |
| `examples/` | 用户示例 | ✅ |
| `experiments/archive/` | 研究脚本（含本地路径） | ❌ .gitignore |
| `qmt/` | QMT 部署适配（需自行改账号/路径） | ⚠️ 需审查 |
| `data_cache/` `models/` | 数据/模型权重 | ❌ .gitignore |

---

## 免责声明

本项目仅用于研究目的，不构成投资建议。使用本代码进行实盘交易需自行承担风险。
