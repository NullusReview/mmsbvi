"""
Constants for MMSBVI
MMSBVI常量定义

This module contains all numerical constants and configuration parameters.
本模块包含所有数值常量和配置参数。
"""

import jax.numpy as jnp

# ============================================================================
# Numerical Constants / 数值常量
# ============================================================================

# Precision and tolerance / 精度和容差
DEFAULT_TOLERANCE = 1e-8  # 默认收敛容差 / Default convergence tolerance
PDE_SOLVER_TOLERANCE = 1e-10  # PDE求解器容差 / PDE solver tolerance
MACHINE_EPSILON = jnp.finfo(jnp.float64).eps  # 机器精度 / Machine epsilon

# Numerical stability / 数值稳定性
MIN_EIGENVALUE = 1e-12  # 最小特征值截断 / Minimum eigenvalue cutoff
MIN_DENSITY = 1e-8  # 最小密度值 / Minimum density value (raised for stability)
LOG_STABILITY = 1e-100  # 对数稳定性参数 / Log stability parameter (further lowered)

# ============================================================================
# Algorithm Parameters / 算法参数
# ============================================================================

# IPFP Algorithm / IPFP算法
MAX_IPFP_ITERATIONS = 1000  # 最大IPFP迭代次数 / Maximum IPFP iterations
IPFP_CONVERGENCE_CHECK_INTERVAL = 10  # 收敛检查间隔 / Convergence check interval

# Anderson Acceleration / Anderson加速
ANDERSON_MEMORY_SIZE = 5  # Anderson加速历史大小 / Anderson acceleration memory
ANDERSON_MIXING_PARAMETER = 0.5  # Anderson混合参数 / Anderson mixing parameter
ANDERSON_REGULARIZATION = 1e-4  # Anderson正则化 / Anderson regularization

# Epsilon Scaling / Epsilon缩放
EPSILON_SCALING_FACTOR = 0.7  # Epsilon缩放因子 / Epsilon scaling factor
INITIAL_EPSILON = 1.0  # 初始epsilon值 / Initial epsilon value
MIN_EPSILON = 1e-4  # 最小epsilon值 / Minimum epsilon value

# ============================================================================
# Grid Parameters for 1D / 1D网格参数
# ============================================================================

# Grid configuration / 网格配置
DEFAULT_GRID_SIZE_1D = 200  # 默认1D网格大小 / Default 1D grid size
DEFAULT_GRID_BOUNDS_1D = (-5.0, 5.0)  # 默认1D网格边界 / Default 1D grid bounds
DEFAULT_GRID_SPACING = 0.05  # 默认网格间距 / Default grid spacing

# FFT parameters / FFT参数
FFT_PADDING_FACTOR = 2  # FFT填充因子 / FFT padding factor

# ============================================================================
# Ornstein-Uhlenbeck Process / OU过程参数
# ============================================================================

# OU process parameters / OU过程参数
DEFAULT_MEAN_REVERSION = 1.0  # 默认均值回归率 / Default mean reversion rate
DEFAULT_DIFFUSION_COEFFICIENT = 1.0  # 默认扩散系数 / Default diffusion coefficient
DEFAULT_EQUILIBRIUM_MEAN = 0.0  # 默认平衡均值 / Default equilibrium mean

# ============================================================================
# Visualization Parameters / 可视化参数
# ============================================================================

# Plot settings / 绘图设置
FIGURE_DPI = 300  # 图像DPI / Figure DPI
DEFAULT_FIGSIZE = (10, 6)  # 默认图像大小 / Default figure size
COLORMAP = "viridis"  # 默认色图 / Default colormap
ANIMATION_FPS = 30  # 动画帧率 / Animation FPS
ANIMATION_INTERVAL = 50  # 动画间隔(ms) / Animation interval (ms)

# ============================================================================
# Computational Resources / 计算资源
# ============================================================================

# Memory management / 内存管理
MEMORY_FRACTION_CPU = 0.85  # CPU内存使用比例 / CPU memory fraction
MEMORY_FRACTION_GPU = 0.90  # GPU内存使用比例 / GPU memory fraction

# Parallelization / 并行化
DEFAULT_NUM_WORKERS = 4  # 默认工作进程数 / Default number of workers
CHUNK_SIZE = 1000  # 批处理块大小 / Batch chunk size

# ============================================================================
# File I/O / 文件输入输出
# ============================================================================

# Data formats / 数据格式
DATA_FORMAT = "npz"  # 数据保存格式 / Data save format
COMPRESSION = True  # 是否压缩 / Whether to compress
CHECKPOINT_INTERVAL = 100  # 检查点间隔 / Checkpoint interval