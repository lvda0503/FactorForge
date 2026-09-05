# 个人量化因子库 — 总索引

> 本地因子库 + 5个开源因子库 + 聚宽因子体系的完整索引

---

## 目录结构

```
d:\JoinQuant\quant_env\
├── factor_library/          # 因子库根目录
│   ├── technical/           # 自建技术面因子（待开发）
│   ├── fundamental/         # 自建基本面因子（待开发）
│   ├── alternative/         # 自建另类因子（待开发）
│   ├── composite/           # 自建复合因子（待开发）
│   ├── utils/               # 因子工具（预处理/评价函数）
│   └── open_source/         # 开源因子库（已下载）
│       ├── finhack/         # FinHack — 中文全流程量化框架
│       ├── panda_factor/    # PandaFactor — 高性能因子库
│       ├── factor-research/ # ML因子研究项目
│       ├── alphalens/       # alphalens-reloaded 源码参考
│       └── qlib/            # 微软 Qlib AI量化平台
├── research/                # 因子研究 notebook
├── data/                    # 因子数据存储
└── backtests/               # 回测结果
```

---

## 一、已安装的 Python 量化工具链

### 因子分析
| 包名 | 版本 | 用途 |
|------|------|------|
| `alphalens-reloaded` | 0.4.6 | 单因子绩效分析 (IC/分层收益/换手率 tear sheet) |
| `jqfactor_analyzer` | 1.1.0 | 聚宽单因子分析工具 |

### 回测与绩效
| 包名 | 版本 | 用途 |
|------|------|------|
| `pyfolio-reloaded` | 0.9.9 | 策略绩效 tear sheet |
| `empyrical-reloaded` | 0.5.12 | 量化绩效指标计算 |
| `vectorbt` | 1.0.0 | 极速向量化回测 |

### 组合优化
| 包名 | 版本 | 用途 |
|------|------|------|
| `riskfolio-lib` | 7.3.0 | 投资组合优化 (Markowitz/HRP/风险平价) |
| `cvxpy` | 1.9.2 | 凸优化框架 |
| `cvxopt` | 1.3.3 | 二次规划求解器 |

### 数据获取
| 包名 | 版本 | 用途 |
|------|------|------|
| `jqdatasdk` | 1.9.8 | 聚宽数据本地SDK（需登录账号） |
| `yfinance` | 1.4.1 | Yahoo Finance 全球市场数据 |
| `pandas-datareader` | 0.11.0 | 多源金融数据读取 |

### 技术分析
| 包名 | 版本 | 用途 |
|------|------|------|
| `ta` | 0.11.0 | 技术分析指标库 |
| `numba` | 0.65.1 | 高性能数值计算加速 |

### 机器学习
| 包名 | 版本 | 用途 |
|------|------|------|
| `scikit-learn` | 1.9.0 | 经典机器学习 |
| `statsmodels` | 0.14.6 | 统计建模 |

### 数据处理
| 包名 | 版本 | 用途 |
|------|------|------|
| `pandas` | 2.3.3 | 数据框架 |
| `numpy` | 2.4.6 | 数值计算 |
| `scipy` | 1.17.1 | 科学计算 |
| `pyarrow` | 24.0.0 | 列式数据加速 |

---

## 二、5个开源因子库详解

### 2.1 FinHack (`open_source/finhack/`)

**定位**: 中文A股全流程量化框架

**因子相关功能**:
- **alphaEngine**: 公式引擎，支持 ~70个 numpy/pandas 运算函数
- **内置 Alpha101/191 公式**: 各50个因子公式文本文件
  - `finhack/widgets/templates/empty_project/data/config/factorlist/alphalist/alpha101`
  - `finhack/widgets/templates/empty_project/data/config/factorlist/alphalist/alpha191`
- **因子挖掘**: 支持 LLM (GPT/Kimi) 和 gplearn 遗传算法的自动因子发现
- **因子分析**: 集成 Alphalens, 计算 IC/rank IC/分组收益
- **技术指标**: 130+ 个 (TA-Lib 70+ 自定义 60+)

**待提取的因子**:
- 50个 Alpha101 公式
- 50个 Alpha191 公式
- 120+ 财务指标因子（from `finhack/examples/demo-project/indicators/`）
- 130+ 技术指标

---

### 2.2 PandaFactor (`open_source/panda_factor/`)

**定位**: 高性能因子计算与分析平台（2.7k ⭐）

**因子相关功能**:
- **双模式因子创建**: 
  - 公式模式: `create_factor_from_formula("rank(ts_mean($close, 20))")`
  - 类模式: 继承 `Factor` 基类实现 `calculate()` 方法
- **内置运算函数**: 40+ 个 `FactorUtils` 方法 (RANK/DELAY/STDDEV/TS_RANK/DECAY_LINEAR/...)
- **因子回测引擎**: IC分析/分组收益/换手率/夏普比率/信息比率
- **LLM 集成**: PandaAI 服务自动生成因子代码
- **多数据源**: RiceQuant/Tushare/XTQuant 因子数据清洗器

**学习价值**:
- `panda_factor/panda_factor/generate/factor_utils.py` — 完整的因子运算函数实现
- `panda_factor/panda_factor/generate/macro_factor.py` — 公式解析和安全沙箱

---

### 2.3 Factor-Research (`open_source/factor-research/`)

**定位**: ML驱动的因子研究项目（IC提升72%）

**因子相关功能**:
- **经典 Alpha101 因子**: 10个完整实现 (alpha_001 到 alpha_010)
- **流动性因子**: 4个 (alpha_liq_01~04)
- **波动率和动量因子**: 4个 (vol/mom)
- **高频因子**: 60+ 个
- **特征选择**: IC过滤 → Spearman相关 → LASSO/Ridge → LightGBM 四步筛选
- **ML模型**: Linear/DNN/DeepNet/AlexNet/LSTM+ResNet/Transformer

**学习价值**:
- `notebooks/Alpha_Factor_Generation.ipynb` — 因子生成的完整代码
- `notebooks/Alpha_Factor_Selection.ipynb` — 因子筛选方法论

---

### 2.4 Alphalens (`open_source/alphalens/`)

**定位**: 因子绩效分析标准库（源码参考）

**核心价值**:
- `performance.py` — IC/IC_IR/因子收益/换手率/因子自相关 完整实现
- `plotting.py` — 因子分析可视化
- `tears.py` — 一键生成完整因子报告

> 该库已通过 pip 安装，此处仅为源码参考

---

### 2.5 Microsoft Qlib (`open_source/qlib/`)

**定位**: 微软AI量化平台（7.8 MB）

**因子相关功能**:
- **Alpha158**: 158个手工因子（9个K-bar + 25个价格 + 5个成交量 + 119个滚动算子）
- **Alpha360**: 360个原始行情特征（6字段 × 60天）
- **表达式引擎**: `Ref($close, 5)` / `Mean($close, 20)` / `Corr($close, Log($volume+1), 10)`
- **因子评估**: IC / 多空收益 / 显著性检验
- **全流程ML**: LightGBM → GRU → LSTM → Transformer → 集成

**学习价值**:
- `qlib/contrib/data/loader.py` — Alpha158 因子定义
- `qlib/contrib/eva/alpha.py` — IC计算实现

---

## 三、因子库汇总

### 可直接使用的因子来源

| 来源 | 因子数量 | 类型 | 使用方式 |
|------|----------|------|----------|
| **聚宽平台** | 191 (Alpha191) + 300+ 财务因子 | 价量+基本面 | 在线策略中 `Factor` 基类 / `jqlib.alpha191` |
| **聚宽本地** | jqfactor_analyzer 全部 | 价量+基本面 | `jqdatasdk` 拉取数据 → `jqfactor_analyzer` 分析 |
| **FinHack** | 100 (Alpha101+191) + 250 (技术+财务) | 价量+技术+基本面 | 提取公式 → 在聚宽中实现 |
| **PandaFactor** | 公式引擎+工具函数 | 通用框架 | 学习因子构造方法 |
| **Factor-Research** | 80+ (Alpha101+高频) | 价量+高频 | 参考Jupyter notebook实现 |
| **Qlib Alpha158** | 158 | 价量 | 研究因子定义 → 移植到聚宽 |

### 可移植到聚宽的因子

聚宽 `Factor` 基类用 `dependencies` 声明依赖，以下因子的 `dependencies` 都可用：

```python
# 示例：参考 Qlib Alpha158 的 ROC 因子移植到聚宽
class ROC_20(Factor):
    name = 'roc_20'
    max_window = 20
    dependencies = ['close']
    def calc(self, data):
        close = data['close']
        return close.iloc[-1] / close.iloc[0] - 1

# 示例：参考 Factor-Research 的流动性因子移植到聚宽
class VOLUME_MOMENTUM(Factor):
    name = 'volume_momentum'
    max_window = 21
    dependencies = ['volume']
    def calc(self, data):
        vol = data['volume']
        return vol.iloc[-5:].mean() / vol.iloc[:-5].mean() - 1
```

---

## 四、下一步：开始使用

### 本地分析流程

```python
# 1. 登录聚宽
import jqdatasdk
jqdatasdk.auth('你的手机号', '你的密码')

# 2. 获取数据
from jqdatasdk import get_index_stocks, get_factor_values
stocks = get_index_stocks('000300.XSHG')
factor_data = get_factor_values(stocks, ['market_cap', 'roa_ttm'], 
                                 end_date='2025-06-20', count=1)

# 3. 导入 alphalens 做因子分析
import alphalens as al
# ... 格式化数据 → create_full_tear_sheet()

# 4. 或用 jqfactor_analyzer（聚宽原生）
import jqfactor_analyzer as ja
ja.analyze_factor(factor_data, quantiles=10, periods=(1, 5, 10))
```

### 聚宽在线使用流程

1. 在聚宽平台打开研究环境
2. 用 `Factor` 基类实现因子 → 用单因子分析验证
3. 将验证通过的因子加入 `Multifactor.py` 策略
4. 回测 → 优化 → 实盘
