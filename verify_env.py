"""验证量化环境所有依赖包"""
import sys
print(f"Python: {sys.version}")

# Core libraries
import numpy; print(f"numpy: {numpy.__version__}")
import pandas; print(f"pandas: {pandas.__version__}")
import scipy; print(f"scipy: {scipy.__version__}")
import matplotlib; print(f"matplotlib: {matplotlib.__version__}")
import seaborn; print(f"seaborn: {seaborn.__version__}")
import sklearn; print(f"sklearn: {sklearn.__version__}")
import statsmodels; print(f"statsmodels: {statsmodels.__version__}")

# Quant core
import alphalens; print(f"alphalens-reloaded: {alphalens.__version__}")
import pyfolio; print("pyfolio-reloaded: OK")
import empyrical; print("empyrical-reloaded: OK")

# Data
import yfinance; print(f"yfinance: {yfinance.__version__}")
import jqdatasdk; print(f"jqdatasdk: {jqdatasdk.__version__}")
try:
    import jqfactor_analyzer; print("jqfactor_analyzer: OK")
except Exception as e:
    print(f"jqfactor_analyzer: {e} (may need fastcache)")

# Technical
import ta; print("ta: OK")
import numba; print(f"numba: {numba.__version__}")

# Optimization
import cvxpy; print(f"cvxpy: {cvxpy.__version__}")
import riskfolio; print("riskfolio-lib: OK")

# Jupyter
import jupyterlab; print("jupyterlab: OK")

print("\n=== ALL PACKAGES VERIFIED ===")
