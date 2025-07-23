"""
Gaussian Process Regression (GPR) Core Components / 高斯过程回归核心组件
======================================================================

This module provides the core functionalities for sparse Gaussian Process
regression using the Variational Free Energy (VFE) framework. It is designed
to be self-contained and reusable.

此模块提供基于变分自由能（VFE）框架的稀疏高斯过程回归核心功能。
它的设计是自包含且可复用的。

Key Functions:
- `rbf_kernel`: Computes the RBF kernel matrix.
- `predict_f`: Computes the predictive mean and variance of the GP.
- `kl_divergence`: Computes the KL divergence between the variational and prior distributions.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular
from functools import partial
import chex

from .types import KernelParams, InducingPoints, GPSSMConfig
from .stable_numerics import (
    robust_cholesky_decomposition, robust_solve_triangular, 
    stable_log_determinant, compute_adaptive_jitter
)

# ============================================================================
# Kernel Function / 核函数
# ============================================================================

@partial(jit, static_argnames=())
def rbf_kernel(x1: chex.Array, x2: chex.Array, params: KernelParams) -> chex.Array:
    """
    Computes the Radial Basis Function (RBF) kernel matrix between two sets of points.
    计算两组点之间的径向基函数（RBF）核矩阵。

    k(x, x') = σ_f^2 * exp(-0.5 * ||(x - x') / l||^2)

    Args:
        x1: First set of points [N, D].
        x2: Second set of points [M, D].
        params: Kernel parameters (lengthscale and variance).

    Returns:
        Kernel matrix [N, M].
    """
    # Use broadcasting to compute squared distances efficiently
    x1_scaled = x1 / params.lengthscale
    x2_scaled = x2 / params.lengthscale
    
    sq_dist = jnp.sum(x1_scaled**2, axis=1, keepdims=True) + \
              jnp.sum(x2_scaled**2, axis=1, keepdims=True).T - \
              2 * jnp.dot(x1_scaled, x2_scaled.T)
              
    return params.variance * jnp.exp(-0.5 * sq_dist)


@partial(jit, static_argnames=())
def rbf_kernel_diag(x: chex.Array, params: KernelParams) -> chex.Array:
    """
    Computes the diagonal of the RBF kernel matrix K(x, x).
    计算RBF核矩阵 K(x, x) 的对角线。

    Args:
        x: Input points [N, D].
        params: Kernel parameters.

    Returns:
        Diagonal of the kernel matrix [N].
    """
    return jnp.full(x.shape[0], params.variance)


# ============================================================================
# Sparse GP Core Logic / 稀疏GP核心逻辑
# ============================================================================

@partial(jit, static_argnames=('config', 'state_dim'))
def predict_f(
    x_test: chex.Array,
    inducing_vars: InducingPoints,
    kernel_params: KernelParams,
    config: GPSSMConfig,
    state_dim: int
) -> tuple[chex.Array, chex.Array]:
    """
    Computes the predictive mean and variance of the GP at test points x_test.
    计算在高斯过程在测试点 x_test 处的预测均值和方差。

    Follows the VFE framework by Titsias.
    μ* = K*z Kzz⁻¹ m
    Σ* = K** - K*z Kzz⁻¹ Kz* + K*z Kzz⁻¹ S Kzz⁻¹ Kz*

    Args:
        x_test: Test points [N, D].
        inducing_vars: Inducing point variables (z, m, L).
        kernel_params: Kernel parameters.
        config: GPSSM configuration for jitter value.

    Returns:
        A tuple containing:
        - Predictive mean [N, D].
        - Predictive variance [N, D].
    """
    z, m, L_s = inducing_vars.z, inducing_vars.m, inducing_vars.L
    M = z.shape[0]
    D = state_dim

    # Compute required kernel matrices
    Kzz = rbf_kernel(z, z, kernel_params)
    Kzx = rbf_kernel(z, x_test, kernel_params)
    Kxx_diag = rbf_kernel_diag(x_test, kernel_params)

    # Robust decomposition of Kzz for numerical stability
    Lzz, jitter_used, is_chol = robust_cholesky_decomposition(Kzz, config.jitter)

    # Predictive mean: μ* = Kxz Kzz⁻¹ m
    # Kxz = Kzx.T
    # A = Lzz⁻¹ Kzx
    A = robust_solve_triangular(Lzz, Kzx, is_chol, jitter_used)
    # B = Lzz⁻ᵀ A = Kzz⁻¹ Kzx (for Cholesky case)
    # For SVD case, this is handled differently in robust_solve_triangular
    
    # temp = Kzz⁻¹ @ m
    temp_1 = robust_solve_triangular(Lzz, m, is_chol, jitter_used)
    temp = robust_solve_triangular(Lzz.T, temp_1, is_chol, jitter_used)
    mean = Kzx.T @ temp

    # Predictive variance: Σ*
    # Term 1: Kxx_diag
    # Term 2: - tr(Kzz⁻¹ Kzx Kxz) = -sum(A * A)
    var_diag = Kxx_diag - jnp.sum(A * A, axis=0)
    
    # Term 3: tr(Kzz⁻¹ S Kzz⁻¹ Kzx Kxz)
    # S = L_s @ L_s.T
    # C = Lzz⁻¹ L_s
    C = robust_solve_triangular(Lzz, L_s, is_chol, jitter_used)
    # D_matrix = A.T @ C (need to be careful about matrix dimensions)
    C_temp = robust_solve_triangular(Lzz.T, C, is_chol, jitter_used)
    D_matrix = Kzx.T @ C_temp
    
    # The trace term is the sum of squares of each element in D_matrix, summed over output dimensions
    var_trace = jnp.sum(D_matrix**2, axis=1)
    
    # Total variance, broadcasted over output dimensions
    # The variance is independent for each output dimension in this model
    total_var = var_diag + var_trace
    # Reshape and broadcast to match the output dimension D
    total_var = jnp.broadcast_to(total_var[:, None], (total_var.shape[0], D))

    return mean, total_var


@partial(jit, static_argnames=('config',))
def kl_divergence(
    inducing_vars: InducingPoints,
    kernel_params: KernelParams,
    config: GPSSMConfig
) -> chex.Scalar:
    """
    Computes the KL divergence between q(u) and the prior p(u).
    计算变分分布 q(u) 与先验 p(u) 之间的KL散度。

    KL[q(u) || p(u)] = 0.5 * [Tr(Kzz⁻¹ S) + mᵀ Kzz⁻¹ m - M + log|Kzz| - log|S|]

    Args:
        inducing_vars: Inducing point variables (z, m, L).
        kernel_params: Kernel parameters.
        config: GPSSM configuration for jitter value.

    Returns:
        The KL divergence, a scalar value.
    """
    z, m, L_s = inducing_vars.z, inducing_vars.m, inducing_vars.L
    M, D_out = m.shape

    # Prior covariance - use robust computation
    Kzz = rbf_kernel(z, z, kernel_params)
    
    # Use stable log determinant for Kzz
    log_det_Kzz = stable_log_determinant(Kzz, config.jitter)
    
    # Robust decomposition for solve operations
    Lzz, jitter_used, is_chol = robust_cholesky_decomposition(Kzz, config.jitter)

    # Variational covariance S = L_s @ L_s.T
    # log|S| = 2 * sum(log(diag(L_s)))
    log_det_S = 2 * jnp.sum(jnp.log(jnp.diag(L_s)))

    # Term 1: log|Kzz| - log|S|
    log_det_term = log_det_Kzz - log_det_S

    # Term 2: Tr(Kzz⁻¹ S)
    # A = Lzz⁻¹ L_s
    A = robust_solve_triangular(Lzz, L_s, is_chol, jitter_used)
    # trace_term = Tr(A.T @ A)
    trace_term = jnp.sum(A**2)

    # Term 3: mᵀ Kzz⁻¹ m
    # B = Lzz⁻¹ m
    B = robust_solve_triangular(Lzz, m, is_chol, jitter_used)
    quad_term = jnp.sum(B**2)

    # The KL is summed over all output dimensions
    kl = 0.5 * (D_out * (trace_term - M + log_det_term) + quad_term)
    return kl