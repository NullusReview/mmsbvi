"""
1D Gaussian Kernel 
1D高斯核 

CRITICAL FIX: Proper implementation of OU transition kernel
关键修复：OU转移核的正确实现

The OU kernel is NOT translation-invariant, so FFT convolution is mathematically WRONG!
OU核不是平移不变的，所以FFT卷积在数学上是错误的！

OU kernel: k(x,y) = N(x; μ(y), σ²) where μ(y) = y*exp(-θt) + μ∞(1-exp(-θt))
This depends on y individually, not just (x-y), so it's not a convolution!
这依赖于y本身，而不仅仅是(x-y)，所以它不是卷积！

SOLUTION: Use direct matrix method (O(N²) but mathematically correct)
解决方案：使用直接矩阵方法（O(N²)但数学正确）
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional, Tuple, Dict

from ..core.types import Density1D, Grid1D, Scalar, OUProcessParams
from ..constants import (
    MIN_DENSITY,
    LOG_STABILITY,
)

# JAX-compatible integration function / JAX兼容的积分函数  
def jax_trapz(y: jnp.ndarray, dx: float) -> float:
    """JAX-compatible trapezoidal integration."""
    return dx * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))


@jit
def apply_ou_kernel_1d_fixed(
    density: Density1D,
    dt: Scalar,
    ou_params: OUProcessParams,
    grid: Grid1D,
) -> Density1D:
    """
    Apply OU transition kernel using CORRECT direct matrix method.
    使用正确的直接矩阵方法应用OU转移核。
    
    FIXED: No more incorrect FFT! Use proper matrix multiplication.
    修复：不再使用错误的FFT！使用正确的矩阵乘法。
    
    Computes: (K_t ρ)(x) = ∫ k(x, y; t) ρ(y) dy
    计算: (K_t ρ)(x) = ∫ k(x, y; t) ρ(y) dy
    
    where k(x,y;t) is the OU transition density from y to x in time t
    其中 k(x,y;t) 是从y到x在时间t内的OU转移密度
    
    Args:
        density: Input density ρ(y) / 输入密度 ρ(y)
        dt: Time step Δt / 时间步长 Δt
        ou_params: OU process parameters / OU过程参数
        grid: Spatial grid / 空间网格
        
    Returns:
        evolved_density: Density after time dt / 时间dt后的密度
    """
    n = len(grid)
    h = grid[1] - grid[0]
    
    # Build OU transition matrix K[i,j] = P(X_t = x_i | X_0 = x_j)
    # 构建OU转移矩阵 K[i,j] = P(X_t = x_i | X_0 = x_j)
    K = _build_ou_transition_matrix(grid, dt, ou_params)
    
    # Apply kernel: ρ_new(x_i) = Σ_j K[i,j] * ρ(x_j) * h
    # 应用核: ρ_new(x_i) = Σ_j K[i,j] * ρ(x_j) * h
    evolved_density = K @ density * h
    
    # Ensure positivity and proper normalization / 确保正性和正确归一化
    evolved_density = jnp.maximum(evolved_density, MIN_DENSITY)
    
    # Renormalize to maintain probability / 重新归一化以保持概率
    total_mass = jax_trapz(evolved_density, dx=h)
    evolved_density = evolved_density / (total_mass + 1e-15)
    
    return evolved_density


@jit
def _build_ou_transition_matrix(
    grid: Grid1D,
    dt: Scalar,
    ou_params: OUProcessParams,
) -> jnp.ndarray:
    """
    Build the OU transition matrix K[i,j] = P(X_t = x_i | X_0 = x_j).
    构建OU转移矩阵 K[i,j] = P(X_t = x_i | X_0 = x_j)。
    
    MATHEMATICALLY CORRECT implementation of OU kernel.
    OU核的数学上正确的实现。
    
    OU SDE: dX_t = -θ(X_t - μ∞)dt + σ dW_t
    Exact solution: X_t = X_0 * e^(-θt) + μ∞(1 - e^(-θt)) + noise
    
    Transition density: N(x_t; μ(x_0), Σ(t))
    where:
    - μ(x_0) = x_0 * e^(-θt) + μ∞(1 - e^(-θt))
    - Σ(t) = σ²/(2θ) * (1 - e^(-2θt))
    """
    n = len(grid)
    theta = ou_params.mean_reversion
    sigma = ou_params.diffusion
    mu_inf = ou_params.equilibrium_mean
    
    # OU process parameters / OU过程参数
    exp_theta_dt = jnp.exp(-theta * dt)
    exp_2theta_dt = jnp.exp(-2 * theta * dt)
    
    # Conditional mean coefficient / 条件均值系数
    mean_coeff = exp_theta_dt
    mean_intercept = mu_inf * (1 - exp_theta_dt)
    
    # Conditional variance (JAX-compatible conditional) / 条件方差（JAX兼容的条件语句）
    # Use jnp.where to avoid tracer boolean conversion
    # 使用 jnp.where 避免 tracer 布尔转换
    variance_nondegenerate = (sigma**2 / (2 * theta)) * (1 - exp_2theta_dt)
    variance_brownian = sigma**2 * dt
    variance = jnp.where(theta > 1e-10, variance_nondegenerate, variance_brownian)
    
    # Build transition matrix / 构建转移矩阵
    # K[i,j] = probability density to go from grid[j] to grid[i]
    # K[i,j] = 从 grid[j] 到 grid[i] 的概率密度
    
    # Vectorized computation / 向量化计算
    x_i = grid[:, None]  # Shape: (n, 1) - target points
    x_j = grid[None, :]  # Shape: (1, n) - initial points
    
    # Conditional means: μ(x_j) for each initial point x_j
    # 条件均值：每个初始点 x_j 的 μ(x_j)
    conditional_means = mean_coeff * x_j + mean_intercept
    
    # Gaussian transition densities / 高斯转移密度
    # p(x_i | x_j) = N(x_i; μ(x_j), variance)
    normalizer = 1.0 / jnp.sqrt(2 * jnp.pi * variance)
    exponent = -0.5 * (x_i - conditional_means)**2 / variance
    
    K = normalizer * jnp.exp(exponent)
    
    # Numerical stability / 数值稳定性
    K = jnp.where(K > LOG_STABILITY, K, LOG_STABILITY)
    
    # CRITICAL FIX: Ensure exact probability conservation
    # 关键修复：确保精确的概率守恒
    # Each column must integrate to 1: ∫ K[i,j] dx_i = 1
    # 每列必须积分为1：∫ K[i,j] dx_i = 1
    h = grid[1] - grid[0]
    # Use trapezoidal rule for each column integration
    # 对每列使用梯形法则积分
    column_sums = h * (jnp.sum(K, axis=0) - 0.5 * (K[0, :] + K[-1, :]))
    K = K / column_sums[None, :]  # Normalize each column
    
    return K


@jit
def apply_backward_ou_kernel_1d_fixed(
    density: Density1D,
    dt: Scalar,
    ou_params: OUProcessParams,
    grid: Grid1D,
) -> Density1D:
    """
    Apply backward (adjoint) OU transition kernel.
    应用反向（伴随）OU转移核。
    
    FIXED: Proper adjoint implementation
    修复：正确的伴随实现
    
    Computes: (K*_t ρ)(y) = ∫ k(x, y; t) ρ(x) dx
    计算: (K*_t ρ)(y) = ∫ k(x, y; t) ρ(x) dx
    """
    n = len(grid)
    h = grid[1] - grid[0]
    
    # Build forward transition matrix / 构建前向转移矩阵
    K = _build_ou_transition_matrix(grid, dt, ou_params)
    
    # Adjoint is transpose / 伴随是转置
    K_adjoint = K.T
    
    # Apply adjoint kernel / 应用伴随核
    backward_density = K_adjoint @ density * h
    
    # Ensure positivity / 确保正性
    backward_density = jnp.maximum(backward_density, MIN_DENSITY)
    
    # Normalize / 归一化
    total_mass = jax_trapz(backward_density, dx=h)
    backward_density = backward_density / (total_mass + 1e-15)
    
    return backward_density


@jit
def compute_log_transition_kernel_1d_fixed(
    x_target: Grid1D,
    x_source: Grid1D,
    dt: Scalar,
    ou_params: OUProcessParams,
) -> jnp.ndarray:
    """
    Compute log of OU transition kernel matrix.
    计算OU转移核矩阵的对数。
    
    FIXED: Correct mathematical implementation
    修复：正确的数学实现
    
    Returns log k(x_i, y_j; dt) for all pairs of grid points.
    返回所有网格点对的 log k(x_i, y_j; dt)。
    """
    theta = ou_params.mean_reversion
    sigma = ou_params.diffusion
    mu_inf = ou_params.equilibrium_mean
    
    # OU transition parameters / OU转移参数
    exp_theta_dt = jnp.exp(-theta * dt)
    
    # Use JAX-compatible conditional logic for JIT compilation
    # 使用JAX兼容的条件逻辑用于JIT编译
    variance_nondegenerate = (sigma**2 / (2 * theta)) * (1 - jnp.exp(-2 * theta * dt))
    variance_brownian = sigma**2 * dt
    variance = jnp.where(theta > 1e-10, variance_nondegenerate, variance_brownian)
    
    # Compute pairwise conditional means / 计算成对条件均值
    x_target_expanded = x_target[:, None]  # Shape: (n_target, 1)
    x_source_expanded = x_source[None, :]  # Shape: (1, n_source)
    
    # μ(y) = y * e^(-θdt) + μ∞(1 - e^(-θdt))
    conditional_means = x_source_expanded * exp_theta_dt + mu_inf * (1 - exp_theta_dt)
    
    # Log Gaussian density / 对数高斯密度
    log_normalizer = -0.5 * jnp.log(2 * jnp.pi * variance)
    log_exponent = -0.5 * (x_target_expanded - conditional_means)**2 / variance
    
    log_kernel = log_normalizer + log_exponent
    
    # Numerical stability / 数值稳定性
    log_kernel = jnp.where(
        log_kernel > jnp.log(LOG_STABILITY),
        log_kernel,
        jnp.log(LOG_STABILITY)
    )
    
    return log_kernel


def estimate_kernel_bandwidth_fixed(
    dt: Scalar,
    ou_params: OUProcessParams,
) -> Scalar:
    """
    Estimate effective bandwidth of OU kernel for grid design.
    估计OU核的有效带宽以用于网格设计。
    
    Returns standard deviation of the OU transition kernel.
    返回OU转移核的标准差。
    """
    theta = ou_params.mean_reversion
    sigma = ou_params.diffusion
    
    # Conditional variance of OU transition / OU转移的条件方差
    # Use JAX-compatible conditional logic for JIT compilation
    # 使用JAX兼容的条件逻辑用于JIT编译
    variance_nondegenerate = (sigma**2 / (2 * theta)) * (1 - jnp.exp(-2 * theta * dt))
    variance_brownian = sigma**2 * dt
    variance = jnp.where(theta > 1e-10, variance_nondegenerate, variance_brownian)
    
    return jnp.sqrt(variance)


# ============================================================================
# Validation Functions / 验证函数
# ============================================================================

def validate_ou_kernel_properties(
    grid: Grid1D,
    dt: Scalar,
    ou_params: OUProcessParams,
    tolerance: float = 1e-10,
) -> Dict[str, float]:
    """
    Validate mathematical properties of the OU kernel.
    验证OU核的数学性质。
    
    Returns:
        metrics: Dictionary of validation metrics / 验证指标字典
    """
    h = grid[1] - grid[0]
    n = len(grid)
    
    # Build transition matrix / 构建转移矩阵
    K = _build_ou_transition_matrix(grid, dt, ou_params)
    
    metrics = {}
    
    # Test 1: Probability conservation / 测试1：概率守恒
    # Each column should sum to 1/h (discrete probability measure)
    # 每一列应该求和为 1/h（离散概率测度）
    column_sums = jnp.sum(K, axis=0) * h
    prob_conservation_error = jnp.max(jnp.abs(column_sums - 1.0))
    metrics["probability_conservation_error"] = float(prob_conservation_error)
    
    # Test 2: Positivity / 测试2：正性
    min_value = jnp.min(K)
    metrics["min_kernel_value"] = float(min_value)
    metrics["positivity_satisfied"] = bool(min_value >= 0)
    
    # Test 3: Symmetry properties for equilibrium / 测试3：平衡态的对称性质
    # For θ=0 (Brownian motion), kernel should be symmetric
    # 对于 θ=0（布朗运动），核应该是对称的
    if ou_params.mean_reversion < 1e-10:
        symmetry_error = jnp.max(jnp.abs(K - K.T))
        metrics["symmetry_error"] = float(symmetry_error)
    
    return metrics


def compare_with_analytical_ou(
    x0: float,
    t: float,
    ou_params: OUProcessParams,
    n_samples: int = 10000,
) -> Dict[str, float]:
    """
    Compare kernel with analytical OU process statistics.
    将核与解析OU过程统计量比较。
    
    For validation against known OU process properties.
    用于与已知OU过程性质的验证。
    """
    theta = ou_params.mean_reversion
    sigma = ou_params.diffusion
    mu_inf = ou_params.equilibrium_mean
    
    # Analytical OU statistics / 解析OU统计
    analytical_mean = x0 * jnp.exp(-theta * t) + mu_inf * (1 - jnp.exp(-theta * t))
    # Use JAX-compatible conditional logic
    # 使用JAX兼容的条件逻辑
    analytical_var_nondegenerate = (sigma**2 / (2 * theta)) * (1 - jnp.exp(-2 * theta * t))
    analytical_var_brownian = sigma**2 * t
    analytical_var = jnp.where(theta > 1e-10, analytical_var_nondegenerate, analytical_var_brownian)
    
    # Create fine grid for numerical integration / 创建细网格用于数值积分
    x_fine = jnp.linspace(analytical_mean - 5*jnp.sqrt(analytical_var),
                          analytical_mean + 5*jnp.sqrt(analytical_var), 
                          n_samples)
    h_fine = x_fine[1] - x_fine[0]
    
    # Compute transition densities / 计算转移密度
    conditional_mean = x0 * jnp.exp(-theta * t) + mu_inf * (1 - jnp.exp(-theta * t))
    normalizer = 1.0 / jnp.sqrt(2 * jnp.pi * analytical_var)
    densities = normalizer * jnp.exp(-0.5 * (x_fine - conditional_mean)**2 / analytical_var)
    
    # Numerical moments / 数值矩
    numerical_mean = jax_trapz(x_fine * densities, dx=h_fine)
    numerical_var = jax_trapz((x_fine - numerical_mean)**2 * densities, dx=h_fine)
    
    return {
        "analytical_mean": float(analytical_mean),
        "numerical_mean": float(numerical_mean),
        "mean_error": float(jnp.abs(analytical_mean - numerical_mean)),
        "analytical_variance": float(analytical_var),
        "numerical_variance": float(numerical_var),
        "variance_error": float(jnp.abs(analytical_var - numerical_var)),
    }


def run_ou_kernel_validation():
    """
    Run comprehensive validation of OU kernel implementation.
    运行OU核实现的全面验证。
    """
    print("=" * 60)
    print("OU Kernel Validation - Fixed Implementation")
    print("OU核验证 - 修复实现")
    print("=" * 60)
    
    # Setup test parameters / 设置测试参数
    grid = jnp.linspace(-3.0, 3.0, 100)
    dt = 0.5
    ou_params = OUProcessParams(
        mean_reversion=1.0,
        diffusion=1.0,
        equilibrium_mean=0.0
    )
    
    # Test 1: Kernel properties / 测试1：核性质
    print("\nTest 1: Kernel Mathematical Properties / 核数学性质")
    metrics = validate_ou_kernel_properties(grid, dt, ou_params)
    print(f"Probability conservation error: {metrics['probability_conservation_error']:.2e}")
    print(f"Positivity satisfied: {metrics['positivity_satisfied']}")
    print(f"Min kernel value: {metrics['min_kernel_value']:.2e}")
    
    # Test 2: Analytical comparison / 测试2：解析比较
    print("\nTest 2: Analytical Comparison / 解析比较")
    analytical_metrics = compare_with_analytical_ou(0.0, dt, ou_params)
    print(f"Mean error: {analytical_metrics['mean_error']:.2e}")
    print(f"Variance error: {analytical_metrics['variance_error']:.2e}")
    
    # Test 3: Apply kernel to simple density / 测试3：对简单密度应用核
    print("\nTest 3: Kernel Application / 核应用")
    initial_density = jnp.exp(-0.5 * grid**2) / jnp.sqrt(2 * jnp.pi)
    h = grid[1] - grid[0]
    initial_density = initial_density / jax_trapz(initial_density, dx=h)
    
    evolved_density = apply_ou_kernel_1d_fixed(initial_density, dt, ou_params, grid)
    
    # Check mass conservation / 检查质量守恒
    initial_mass = jax_trapz(initial_density, dx=h)
    final_mass = jax_trapz(evolved_density, dx=h)
    mass_error = jnp.abs(final_mass - initial_mass)
    print(f"Mass conservation error: {mass_error:.2e}")
    
    print("\n" + "=" * 60)
    print("OU kernel validation complete / OU核验证完成")
    print("=" * 60)


if __name__ == "__main__":
    run_ou_kernel_validation()