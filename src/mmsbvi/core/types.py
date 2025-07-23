"""
Core type definitions for MMSBVI
MMSBVI核心类型定义

This module defines the core data types and type annotations used throughout the project.
本模块定义项目中使用的核心数据类型和类型注解。
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union
from jax import Array
import jax.numpy as jnp
import jax.random
from jaxtyping import Array as JArray, Float, Int
import chex
import flax

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

# SDE Function types / SDE函数类型
SDEState = Float[Array, "d"]  # SDE状态向量 / SDE state vector
DriftFunction = Callable[[SDEState, float], SDEState]  # 漂移函数 μ(x,t) / drift function μ(x,t)
DiffusionFunction = Callable[[SDEState, float], SDEState]  # 扩散函数 σ(x,t) / diffusion function σ(x,t)
DiffusionDerivative = Callable[[SDEState, float], SDEState]  # 扩散导数 ∂σ/∂x / diffusion derivative ∂σ/∂x
NoiseTerms = Float[Array, "d"]  # 布朗运动噪声项 / Brownian motion noise terms

# Neural Network types / 神经网络类型
NetworkParams = Dict[str, Any]  # Flax parameter tree / Flax参数树
DriftNetworkFunction = Callable[[SDEState, float, NetworkParams], SDEState]  # 神经网络漂移函数
TimeEncoding = Float[Array, "encoding_dim"]  # 时间编码 / Time encoding
BatchStates = Float[Array, "batch_size d"]  # 批量状态 / Batch states
BatchTimes = Float[Array, "batch_size"]  # 批量时间 / Batch times

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


# Forward declaration placeholder for SDEProblem
# Will be defined after SDEIntegratorConfig


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


@chex.dataclass
class SDEIntegratorConfig:
    """
    Configuration for SDE integrators
    SDE积分器配置
    """
    method: str = "euler_maruyama"  # 积分方法 / Integration method
    adaptive: bool = False  # 自适应步长 / Adaptive step size
    rtol: float = 1e-3  # 相对容差 / Relative tolerance
    atol: float = 1e-6  # 绝对容差 / Absolute tolerance
    max_steps: int = 10000  # 最大步数 / Maximum steps
    dt_min: float = 1e-8  # 最小步长 / Minimum step size
    dt_max: float = 1e-1  # 最大步长 / Maximum step size
    save_at: Optional[Float[Array, "k"]] = None  # 保存时间点 / Save time points


@chex.dataclass
class NetworkConfig:
    """
    Configuration for neural networks
    神经网络配置
    """
    hidden_dims: List[int] = (256, 256, 256)  # 隐藏层维度 / Hidden layer dimensions
    n_layers: int = 4  # 网络层数 / Number of layers
    activation: str = "silu"  # 激活函数 / Activation function
    use_attention: bool = True  # 是否使用注意力 / Use attention mechanism
    dropout_rate: float = 0.1  # Dropout率 / Dropout rate
    time_encoding_dim: int = 128  # 时间编码维度 / Time encoding dimension
    use_spectral_norm: bool = True  # 谱归一化 / Spectral normalization
    use_residual: bool = True  # 残差连接 / Residual connections
    use_layer_norm: bool = True  # 层归一化 / Layer normalization
    precision: str = "float32"  # 计算精度 / Computation precision
    

@chex.dataclass
class TrainingConfig:
    """
    Configuration for training
    训练配置
    """
    batch_size: int = 256  # 批量大小 / Batch size
    learning_rate: float = 3e-4  # 学习率 / Learning rate
    num_epochs: int = 1000  # 训练轮数 / Number of epochs
    gradient_clip_norm: float = 0.2  # 梯度裁剪（Neural Control Variational优化） / Gradient clipping (optimized for Neural Control Variational)
    use_mixed_precision: bool = True  # 混合精度训练 / Mixed precision training
    accumulate_gradients: int = 1  # 梯度累积步数 / Gradient accumulation steps
    warmup_steps: int = 1000  # 预热步数 / Warmup steps
    decay_schedule: str = "cosine"  # 学习率衰减 / Learning rate decay
    checkpoint_every: int = 1000  # 检查点保存间隔 / Checkpoint saving interval


@chex.dataclass  
class SDEProblem:
    """
    SDE problem specification
    SDE问题规范
    """
    initial_state: SDEState  # 初始状态 / Initial state
    drift_fn: DriftFunction  # 漂移函数 / Drift function
    diffusion_fn: DiffusionFunction  # 扩散函数 / Diffusion function
    time_span: Tuple[float, float]  # 时间区间 / Time span
    config: SDEIntegratorConfig  # 积分器配置 / Integrator configuration
    
    # 可选的高级特性 / Optional advanced features
    diffusion_derivative: Optional[DiffusionDerivative] = None  # 扩散导数 / Diffusion derivative
    save_at: Optional[Float[Array, "k"]] = None  # 保存时间点 / Save time points


@chex.dataclass
class NetworkTrainingState:
    """
    Training state for neural networks
    神经网络训练状态
    """
    params: NetworkParams  # 网络参数 / Network parameters
    optimizer_state: Any  # 优化器状态 / Optimizer state
    step: int  # 训练步数 / Training step
    best_loss: float  # 最佳损失 / Best loss
    metrics: Dict[str, float]  # 训练指标 / Training metrics
    optimizer: Any = None  # 优化器实例 / Optimizer instance
    
    def update(self, **kwargs) -> "NetworkTrainingState":
        """Update training state / 更新训练状态"""
        return self.replace(**kwargs)
    
    def apply_gradients(self, *, grads, **kwargs) -> "NetworkTrainingState":
        """Apply gradients to parameters / 将梯度应用到参数"""
        import optax
        if self.optimizer is None:
            raise ValueError("Optimizer not set in training state")
        
        updates, new_opt_state = self.optimizer.update(
            grads, self.optimizer_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)
        
        return self.update(
            params=new_params,
            optimizer_state=new_opt_state,
            step=self.step + 1,
            **kwargs
        )


# ============================================================================ 
# Performance and GPU Optimization Types / 性能和GPU优化类型
# ============================================================================

@chex.dataclass
class PerformanceConfig:
    """
    Configuration for performance optimization
    性能优化配置
    """
    use_jit: bool = True  # JIT编译 / JIT compilation
    use_vmap: bool = True  # 向量化 / Vectorization
    use_pmap: bool = False  # 多设备并行 / Multi-device parallelism
    use_scan: bool = True  # 高效循环 / Efficient loops
    use_checkpointing: bool = True  # 梯度检查点 / Gradient checkpointing
    memory_efficient: bool = True  # 内存效率 / Memory efficiency
    cache_time_encoding: bool = True  # 缓存时间编码 / Cache time encoding
    max_batch_size: int = 1024  # 最大批量大小 / Maximum batch size
    num_devices: int = 1  # 设备数量 / Number of devices
    device_batch_size: Optional[int] = None  # 每设备批量大小 / Per-device batch size
    
    def __post_init__(self):
        if self.device_batch_size is None:
            self.device_batch_size = self.max_batch_size // self.num_devices


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


class SDEIntegrator(Protocol):
    """
    Protocol for SDE numerical integrators
    SDE数值积分器协议
    
    Provides unified interface for different SDE integration schemes.
    为不同的SDE积分格式提供统一接口。
    """
    
    def step(
        self,
        t: float,
        state: SDEState,
        drift_fn: DriftFunction,
        diffusion_fn: DiffusionFunction,
        dt: float,
        key: jax.random.PRNGKey
    ) -> SDEState:
        """
        Single integration step
        单步积分
        
        Args:
            t: Current time / 当前时间
            state: Current state / 当前状态
            drift_fn: Drift function μ(x,t) / 漂移函数 μ(x,t)
            diffusion_fn: Diffusion function σ(x,t) / 扩散函数 σ(x,t)
            dt: Time step / 时间步长
            key: Random key for noise / 噪声随机密钥
            
        Returns:
            new_state: Updated state / 更新后的状态
        """
        ...
    
    def integrate(
        self,
        initial_state: SDEState,
        drift_fn: DriftFunction,
        diffusion_fn: DiffusionFunction,
        time_grid: Float[Array, "n"],
        key: jax.random.PRNGKey
    ) -> Float[Array, "n d"]:
        """
        Multi-step integration
        多步积分
        
        Args:
            initial_state: Initial condition / 初始条件
            drift_fn: Drift function / 漂移函数
            diffusion_fn: Diffusion function / 扩散函数
            time_grid: Time points / 时间网格
            key: Random key / 随机密钥
            
        Returns:
            trajectory: State trajectory / 状态轨迹
        """
        ...


class DriftNetwork(Protocol):
    """
    Protocol for neural drift networks
    神经网络漂移协议
    
    Provides interface for Föllmer drift neural networks.
    为Föllmer漂移神经网络提供接口。
    """
    
    def __call__(
        self,
        x: SDEState,
        t: float,
        train: bool = False
    ) -> SDEState:
        """
        Compute drift μ(x,t)
        计算漂移 μ(x,t)
        
        Args:
            x: State vector / 状态向量
            t: Time / 时间
            train: Training mode / 训练模式
            
        Returns:
            drift: Drift vector μ(x,t) / 漂移向量
        """
        ...
    
    def batch_call(
        self,
        x_batch: BatchStates,
        t_batch: BatchTimes,
        train: bool = False
    ) -> BatchStates:
        """
        Batch computation of drift
        批量计算漂移
        
        Args:
            x_batch: Batch of states / 状态批量
            t_batch: Batch of times / 时间批量
            train: Training mode / 训练模式
            
        Returns:
            drift_batch: Batch of drifts / 漂移批量
        """
        ...


# ============================================================================
# Neural Control Variational Types / 神经控制变分类型
# ============================================================================

# Path and control types / 路径和控制类型
PathSamples = Float[Array, "batch_size num_steps state_dim"]  # 路径样本 / Path samples
ControlObjective = Callable[[PathSamples, jnp.ndarray], Tuple[float, Dict[str, float]]]  # 控制目标函数 / Control objective function
DensityLogPdf = Callable[[BatchStates], Float[Array, "batch_size"]]  # 密度对数概率函数 / Density log-pdf function
BoundaryPenalty = float  # 边界惩罚值 / Boundary penalty value
ControlCost = float  # 控制代价值 / Control cost value

# Sampling and integration types / 采样和积分类型
InitialSampler = Callable[[int, jax.random.PRNGKey], BatchStates]  # 初始状态采样器 / Initial state sampler
PathIntegrator = Callable[[BatchStates, NetworkParams, jax.random.PRNGKey], PathSamples]  # 路径积分器 / Path integrator
VarianceReducer = Callable[[jnp.ndarray], jnp.ndarray]  # 方差减少器 / Variance reducer


@chex.dataclass
class ControlGradConfig:
    """
    Configuration for Neural Control Variational method
    神经控制变分方法配置
    """
    # Problem specification / 问题规范
    state_dim: int = 2  # 状态维度 / State dimension
    time_horizon: float = 1.0  # 时间域长度 / Time horizon length
    num_time_steps: int = 100  # 时间步数 / Number of time steps
    diffusion_coeff: float = 0.1  # 扩散系数 / Diffusion coefficient
    
    # Training parameters / 训练参数
    batch_size: int = 1024  # 批量大小 / Batch size
    num_epochs: int = 5000  # 训练轮数 / Number of epochs
    learning_rate: float = 5e-5  # 学习率（Neural Control Variational稳定优化） / Learning rate (stable optimization for Neural Control Variational)
    gradient_clip_norm: float = 0.2  # 梯度裁剪范数（强化稳定性） / Gradient clipping norm (enhanced stability)
    
    # Sampling parameters / 采样参数
    num_samples: int = 10000  # 采样数量 / Number of samples
    importance_sampling: bool = True  # 重要性采样 / Importance sampling
    variance_reduction: bool = True  # 方差减少 / Variance reduction
    
    # Boundary conditions / 边界条件
    initial_distribution: str = "gaussian"  # 初始分布类型 / Initial distribution type
    target_distribution: str = "gaussian"  # 目标分布类型 / Target distribution type
    initial_params: Optional[Dict[str, float]] = None  # 初始分布参数 / Initial distribution parameters
    target_params: Optional[Dict[str, float]] = None  # 目标分布参数 / Target distribution parameters
    
    # Optimization settings / 优化设置
    optimizer: str = "adamw"  # 优化器类型 / Optimizer type
    schedule: str = "cosine"  # 学习率调度 / Learning rate schedule
    warmup_steps: int = 1000  # 预热步数 / Warmup steps
    
    # Performance optimizations / 性能优化
    use_mixed_precision: bool = True  # 混合精度训练 / Mixed precision training
    use_gradient_checkpointing: bool = True  # 梯度检查点 / Gradient checkpointing
    parallel_devices: int = 1  # 并行设备数 / Number of parallel devices
    
    # Loss function weighting (based on Neural Control Variational theory)
    # 损失函数权重（基于神经控制变分理论）
    control_weight: float = 1.0  # 控制代价权重 / Control cost weight
    boundary_weight: float = 1.0  # 边界惩罚权重 / Boundary penalty weight (changed from hardcoded 10.0)
    adaptive_weighting: bool = False  # 自适应权重调整 / Adaptive weight adjustment
    
    # Numerical stability / 数值稳定性
    log_stability_eps: float = 1e-8  # 对数稳定性小量 / Log stability epsilon
    density_estimation_method: str = "kde"  # 密度估计方法 / Density estimation method
    bandwidth_selection: str = "scott"  # 带宽选择方法 / Bandwidth selection method
    
    # Validation and monitoring / 验证和监控
    validation_freq: int = 100  # 验证频率 / Validation frequency
    checkpoint_freq: int = 1000  # 检查点频率 / Checkpoint frequency
    log_freq: int = 10  # 日志频率 / Logging frequency
    
    def __post_init__(self):
        """Initialize default parameters / 初始化默认参数"""
        if self.initial_params is None:
            self.initial_params = {"mean": 0.0, "std": 1.0}
        if self.target_params is None:
            self.target_params = {"mean": 0.0, "std": 1.0}


@chex.dataclass
class ControlGradState:
    """
    Training state for Neural Control Variational method with JAX-optimized history tracking
    神经控制变分方法训练状态，使用JAX优化的历史记录跟踪
    
    IMPORTANT: All history arrays are pre-allocated JAX arrays to avoid JIT performance issues
    重要提示：所有历史数组都是预分配的JAX数组，以避免JIT性能问题
    """
    training_state: NetworkTrainingState  # 网络训练状态 / Network training state
    config: ControlGradConfig  # 配置信息 / Configuration
    step: int  # 训练步数 / Training step
    epoch: int  # 训练轮数 / Training epoch
    best_loss: float  # 最佳损失 / Best loss
    
    # JAX-optimized history arrays (pre-allocated for performance)
    # JAX优化的历史数组（预分配以提高性能）
    loss_history: Float[Array, "max_epochs"]  # 损失历史 / Loss history
    gradient_norm_history: Float[Array, "max_epochs"]  # 梯度范数历史 / Gradient norm history
    time_per_epoch: Float[Array, "max_epochs"]  # 每轮时间 / Time per epoch
    control_cost_history: Float[Array, "max_epochs"]  # 控制代价历史 / Control cost history
    boundary_penalty_history: Float[Array, "max_epochs"]  # 边界惩罚历史 / Boundary penalty history
    
    # Tracking for current position in history arrays
    # 历史数组中当前位置的跟踪
    history_index: int = 0  # 当前历史索引 / Current history index
    
    path_samples: Optional[PathSamples] = None  # 路径样本 / Path samples
    
    def update(self, **kwargs) -> "ControlGradState":
        """Update state with new values / 用新值更新状态"""
        return self.replace(**kwargs)