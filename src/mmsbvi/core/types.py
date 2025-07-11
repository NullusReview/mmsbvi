"""
Core type definitions for MMSBVI
MMSBVI核心类型定义

This module defines the core data types and type annotations used throughout the project.
本模块定义项目中使用的核心数据类型和类型注解。
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union
from jax import Array
import jax.numpy as jnp
from jaxtyping import Array as JArray, Float, Int
import chex

# Type aliases for clarity / 类型别名
Scalar = Union[float, Float[Array, ""]]
Vector = Float[Array, "n"]
Matrix = Float[Array, "n m"]
Tensor3D = Float[Array, "n m k"]

# 1D specific types / 1D专用类型
Density1D = Float[Array, "n"]
Potential1D = Float[Array, "n"]
Velocity1D = Float[Array, "n"]
Grid1D = Float[Array, "n"]

# Time-indexed types / 时间索引类型
TimeIndex = Int[Array, ""]
TimeSteps = Int[Array, "k"]

# Function types / 函数类型
DensityFunction = Callable[[Float[Array, "n"]], Float[Array, "n"]]
PotentialFunction = Callable[[Float[Array, "n"]], Float[Array, "n"]]
KernelFunction = Callable[[Float[Array, "n"], Float[Array, "n"], Scalar], Scalar]

# ============================================================================
# Data Structures / 数据结构
# ============================================================================

@chex.dataclass
class GridConfig1D:
    """
    Configuration for 1D spatial grid
    1D空间网格配置
    """
    n_points: int  # 网格点数 / Number of grid points
    bounds: Tuple[float, float]  # 网格边界 / Grid boundaries
    spacing: float  # 网格间距 / Grid spacing
    points: Grid1D  # 网格点坐标 / Grid point coordinates
    
    @staticmethod
    def create(n_points: int, bounds: Tuple[float, float]) -> "GridConfig1D":
        """Create grid configuration / 创建网格配置"""
        spacing = (bounds[1] - bounds[0]) / (n_points - 1)
        points = jnp.linspace(bounds[0], bounds[1], n_points)
        return GridConfig1D(
            n_points=n_points,
            bounds=bounds,
            spacing=spacing,
            points=points
        )


@chex.dataclass
class OUProcessParams:
    """
    Ornstein-Uhlenbeck process parameters
    Ornstein-Uhlenbeck过程参数
    """
    mean_reversion: Scalar  # 均值回归率 θ / Mean reversion rate θ
    diffusion: Scalar  # 扩散系数 σ / Diffusion coefficient σ
    equilibrium_mean: Scalar  # 平衡均值 μ / Equilibrium mean μ


@chex.dataclass
class IPFPState:
    """
    State of the IPFP algorithm
    IPFP算法状态
    """
    potentials: List[Potential1D]  # 势函数列表 φ_k / List of potentials φ_k
    marginals: List[Density1D]  # 边际密度 ρ_k / Marginal densities ρ_k
    iteration: int  # 当前迭代次数 / Current iteration number
    error: Scalar  # 当前误差 / Current error
    converged: bool  # 是否收敛 / Whether converged
    
    def update(self, **kwargs) -> "IPFPState":
        """Update state with new values / 用新值更新状态"""
        return self.replace(**kwargs)


@chex.dataclass
class MMSBProblem:
    """Multi-marginal Schrödinger bridge problem specification"""
    observation_times: TimeSteps
    ou_params: OUProcessParams
    grid: GridConfig1D
    # 可选硬边际
    observed_marginals: Optional[List[Density1D]] = None
    # 可选观测数据
    y_observations: Optional[jnp.ndarray] = None
    C: float = 1.0
    R: float = 0.05
    
    @property
    def n_marginals(self) -> int:
        """Number of time points (K)"""
        return len(self.observation_times)
    
    @property
    def time_intervals(self) -> List[Scalar]:
        """Time intervals between observations / 观测之间的时间间隔"""
        return [self.observation_times[i+1] - self.observation_times[i]
                for i in range(len(self.observation_times) - 1)]


@chex.dataclass
class MMSBSolution:
    """
    Solution to the multi-marginal Schrödinger bridge problem
    多边际薛定谔桥问题的解
    """
    potentials: List[Potential1D]  # 最优势函数 / Optimal potentials
    path_densities: List[Density1D]  # 路径密度 / Path densities
    velocities: Optional[List[Velocity1D]]  # 速度场 / Velocity fields
    convergence_history: List[Scalar]  # 收敛历史 / Convergence history
    final_error: Scalar  # 最终误差 / Final error
    n_iterations: int  # 迭代次数 / Number of iterations


# ============================================================================
# Algorithm Configuration / 算法配置
# ============================================================================

@chex.dataclass
class IPFPConfig:
    """
    Configuration for IPFP algorithm
    IPFP算法配置
    """
    max_iterations: int = 1000  # 最大迭代次数 / Maximum iterations
    tolerance: float = 1e-8  # 收敛容差 / Convergence tolerance
    check_interval: int = 10  # 检查间隔 / Check interval
    use_anderson: bool = True  # 是否使用Anderson加速 / Use Anderson acceleration
    anderson_memory: int = 5  # Anderson记忆大小 / Anderson memory size
    epsilon_scaling: bool = True  # 是否使用epsilon缩放 / Use epsilon scaling
    initial_epsilon: float = 1.0  # 初始epsilon / Initial epsilon
    epsilon_decay: float = 0.8  # epsilon衰减 / epsilon decay factor per check_interval (balanced decay)
    min_epsilon: float = 1e-4  # 最小epsilon (allow tighter likelihood)
    adaptive_epsilon: bool = True  # 自适应ε
    eps_decay_high: float = 0.9  # 未满足阈值时衰减
    eps_decay_low: float = 0.4   # 满足阈值后更快衰减
    error_threshold: float = 5e-4  # 阈值
    verbose: bool = True  # 详细输出 / Verbose output
    # 新增: 是否更新每个潜能的布尔掩码; True=固定, False=可更新
    fixed_potential_mask: Optional[List[bool]] = None


@chex.dataclass
class PDESolverConfig:
    """
    Configuration for PDE solver
    PDE求解器配置
    """
    method: str = "dense"  # 求解方法 / Solution method (dense, tridiag, pcg)
    tolerance: float = 1e-10  # 求解容差 / Solver tolerance
    max_iterations: int = 1000  # 最大迭代次数 / Maximum iterations
    preconditioner: str = "jacobi"  # 预条件器 / Preconditioner
    boundary_condition: str = "neumann"  # 边界条件 / Boundary condition


# ============================================================================
# Protocols for Extensibility / 扩展性协议
# ============================================================================

class Solver(Protocol):
    """
    Protocol for PDE solvers
    PDE求解器协议
    """
    def solve(self, rho: Density1D, sigma: Density1D) -> Potential1D:
        """Solve the PDE / 求解PDE"""
        ...


class Kernel(Protocol):
    """
    Protocol for transition kernels
    转移核协议
    """
    def apply(self, density: Density1D, dt: Scalar) -> Density1D:
        """Apply transition kernel / 应用转移核"""
        ...
    
    def log_kernel(self, x: Grid1D, y: Grid1D, dt: Scalar) -> Matrix:
        """Log of transition kernel / 转移核的对数"""
        ...