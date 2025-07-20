"""
MMSBVI: Multi-Marginal Schrödinger Bridge Variational Inference
多边际薛定谔桥变分推断

A JAX-based implementation for numerical validation of theoretical results.
基于JAX的数值验证实现
"""

__version__ = "0.1.0"
__author__ = "MMSBVI Team"

# 核心模块导入 / Core module imports
from . import core
from . import solvers
from . import algorithms
from . import visualization
from . import utils

# ---------------------------------------------------------------------------
# Enable double precision by default / 默认启用 64 位精度
# Must be set *before* heavy JAX computations to提高数值精度
import jax, os as _os
# 环境变量优先，运行时亦显式启用
_os.environ.setdefault("JAX_ENABLE_X64", "True")
jax.config.update("jax_enable_x64", True)
# ---------------------------------------------------------------------------

__all__ = [
    "core",
    "solvers", 
    "algorithms",
    "visualization",
    "utils",
    # New modules (placeholders)
    # 新模块（占位符）
    # "integrators",
    # "nets", 
    # "cli",
]