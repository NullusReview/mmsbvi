"""
Test suite for the forward-backward algorithm in log-space.
对数空间中前向-后向算法的测试套件。

This suite validates the correctness, generalization, and numerical
stability of the refactored `_compute_current_marginal` function.
该套件验证了重构后的 `_compute_current_marginal` 函数的正确性、
泛化能力和数值稳定性。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import jax
import jax.numpy as jnp
from functools import partial

# Enable 64-bit precision for all tests in this module
# 为本模块中的所有测试启用64位精度
jax.config.update("jax_enable_x64", True)

from src.mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem, IPFPConfig
from src.mmsbvi.algorithms.ipfp_1d import (
    _compute_current_marginal,
    _precompute_log_transition_matrices
)

# Helper for numerical integration / 数值积分辅助函数
def jax_trapz(y: jnp.ndarray, dx: float) -> float:
    """JAX-compatible trapezoidal integration."""
    return dx * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))

# ============================================================================
# Test Fixtures and Helper Functions / 测试夹具和辅助函数
# ============================================================================

def _create_test_problem(K: int, n_points: int = 100, bounds=(-5., 5.), diffusion: float = 1.0):
    """Helper to create a standard test problem for K marginals."""
    grid_config = GridConfig1D.create(n_points=n_points, bounds=bounds)
    h = grid_config.spacing
    grid = grid_config.points
    
    # Create K Gaussians as marginals
    # 创建 K 个高斯分布作为边际
    centers = jnp.linspace(-2.5, 2.5, K)
    marginals = []
    for i in range(K):
        rho = jnp.exp(-0.5 * (grid - centers[i])**2 / 0.1)
        rho_normalized = rho / jax_trapz(rho, dx=h)
        marginals.append(rho_normalized.astype(jnp.float64))
        
    ou_params = OUProcessParams(
        mean_reversion=jnp.array(0.5, dtype=jnp.float64),
        diffusion=jnp.array(diffusion, dtype=jnp.float64),
        equilibrium_mean=jnp.array(0.0, dtype=jnp.float64)
    )
    
    observation_times = jnp.linspace(0., 1., K, dtype=jnp.float64)
    
    problem = MMSBProblem(
        observation_times=observation_times,
        observed_marginals=marginals,
        ou_params=ou_params,
        grid=grid_config
    )
    
    # Initialize potentials (e.g., all zeros)
    # 初始化势函数（例如，全零）
    potentials = [jnp.zeros(n_points, dtype=jnp.float64) for _ in range(K)]
    
    log_transition_matrices = _precompute_log_transition_matrices(problem)
    
    return potentials, log_transition_matrices, problem

# ============================================================================
# "Golden Standard" Direct Probability Space Implementation
# “黄金标准”直接概率空间实现
# ============================================================================

def _compute_current_marginal_direct(
    k: int,
    potentials: list[jnp.ndarray],
    log_transition_matrices: list[jnp.ndarray],
    grid_spacing: float
) -> jnp.ndarray:
    """
    Naive forward-backward in direct probability space.
    在直接概率空间中的朴素前向-后向算法。
    
    This version is for validation. It is numerically unstable for long chains
    or small probabilities but serves as a "golden standard" for correctness.
    此版本用于验证。对于长链或小概率，它在数值上不稳定，但可作为正确性的“黄金标准”。
    """
    K = len(potentials)
    n_points = potentials[0].shape[0]
    h = grid_spacing

    # Convert log transition matrices to direct probability space
    # 将对数转移矩阵转换为直接概率空间
    transition_matrices = [jnp.exp(log_K) for log_K in log_transition_matrices]

    # Forward messages (alpha)
    # 前向消息 (alpha)
    # alpha_fwd[i] is the message from time 0 to i
    # alpha_fwd[i] 是从时间 0 到 i 的消息
    alpha_fwd = [jnp.zeros((n_points,), dtype=jnp.float64) for _ in range(K)]
    alpha_fwd[0] = jnp.exp(potentials[0])

    for i in range(1, K):
        # Message passes from i-1 to i
        # 消息从 i-1 传递到 i
        # prev_msg is alpha_fwd[i-1]
        # prev_msg 是 alpha_fwd[i-1]
        # K_mat is transition from i-1 to i
        # K_mat 是从 i-1 到 i 的转移
        K_mat = transition_matrices[i-1]
        
        # Propagate forward: integral(K(x_i, x_{i-1}) * alpha_{i-1}(x_{i-1}) dx_{i-1})
        # 前向传播: integral(K(x_i, x_{i-1}) * alpha_{i-1}(x_{i-1}) dx_{i-1})
        propagated_msg = jnp.dot(K_mat, alpha_fwd[i-1]) * h
        
        # Include potential at time i
        # 包含时间 i 的势函数
        alpha_fwd[i] = propagated_msg * jnp.exp(potentials[i])

    # Backward messages (beta)
    # 后向消息 (beta)
    # beta_bwd[i] is the message from time K-1 to i
    # beta_bwd[i] 是从时间 K-1 到 i 的消息
    beta_bwd = [jnp.zeros((n_points,), dtype=jnp.float64) for _ in range(K)]
    beta_bwd[K-1] = jnp.exp(potentials[K-1])

    for i in range(K - 2, -1, -1):
        # Message passes from i+1 to i
        # 消息从 i+1 传递到 i
        # next_msg is beta_bwd[i+1]
        # next_msg 是 beta_bwd[i+1]
        # K_mat is transition from i to i+1
        # K_mat 是从 i 到 i+1 的转移
        K_mat = transition_matrices[i]
        
        # Propagate backward: integral(K(x_{i+1}, x_i) * beta_{i+1}(x_{i+1}) dx_{i+1})
        # 后向传播: integral(K(x_{i+1}, x_i) * beta_{i+1}(x_{i+1}) dx_{i+1})
        # Note: K is symmetric, so K.T = K
        # 注意: K 是对称的, 所以 K.T = K
        propagated_msg = jnp.dot(K_mat.T, beta_bwd[i+1]) * h
        
        # Include potential at time i
        # 包含时间 i 的势函数
        beta_bwd[i] = propagated_msg * jnp.exp(potentials[i])

    # Compute marginal at time k
    # 计算时间 k 的边际
    # The marginal at k is the product of the forward message to k and the backward message to k,
    # divided by the potential at k (since it's counted in both).
    # k 处的边际是(到 k 的前向消息)与(到 k 的后向消息)的乘积，再除以 k 处的势函数（因为它在两者中都被计算了）。
    marginal_k = alpha_fwd[k] * beta_bwd[k] / jnp.exp(potentials[k])
    
    # Normalize the marginal
    # 归一化边际
    mass = jax_trapz(marginal_k, dx=h)
    # Add a small epsilon to avoid division by zero if mass is zero
    # 添加一个小的 epsilon 以避免在质量为零时除以零
    marginal_k_normalized = marginal_k / (mass + 1e-15)
    
    return marginal_k_normalized

# ============================================================================
# Test Cases / 测试用例
# ============================================================================

class TestForwardBackward:
    """Test suite for the forward-backward algorithm."""

    def test_consistency_with_direct_method(self):
        """
        Tests that the log-space implementation is numerically equivalent
        to a direct probability-space implementation for a simple case (K=2).
        测试对数空间实现在简单情况（K=2）下与直接概率空间实现的数值等价性。
        """
        # Setup a simple K=2 problem
        # 设置一个简单的 K=2 问题
        potentials, log_trans_mats, problem = _create_test_problem(K=2, n_points=50)
        h = problem.grid.spacing

        # --- Compute using the new log-space method ---
        # --- 使用新的对数空间方法计算 ---
        # We need to wrap the function to match the JIT signature if we want to test it directly
        # 如果我们想直接测试它，需要包装函数以匹配JIT签名
        jitted_log_space_marginal = jax.jit(partial(_compute_current_marginal, grid_spacing=h))
        
        log_marginal_0_new = jitted_log_space_marginal(k=0, potentials=potentials, log_transition_matrices=log_trans_mats)
        log_marginal_1_new = jitted_log_space_marginal(k=1, potentials=potentials, log_transition_matrices=log_trans_mats)
        
        marginal_0_new = jnp.exp(log_marginal_0_new)
        marginal_1_new = jnp.exp(log_marginal_1_new)

        # --- Compute using the "golden standard" direct method ---
        # --- 使用“黄金标准”直接方法计算 ---
        marginal_0_direct = _compute_current_marginal_direct(k=0, potentials=potentials, log_transition_matrices=log_trans_mats, grid_spacing=h)
        marginal_1_direct = _compute_current_marginal_direct(k=1, potentials=potentials, log_transition_matrices=log_trans_mats, grid_spacing=h)

        # --- Compare the results ---
        # --- 比较结果 ---
        # Use a tight tolerance to ensure high fidelity
        # 使用严格的容差以确保高保真度
        tolerance = 1e-9
        
        l1_error_0 = jax_trapz(jnp.abs(marginal_0_new - marginal_0_direct), dx=h)
        l1_error_1 = jax_trapz(jnp.abs(marginal_1_new - marginal_1_direct), dx=h)

        assert l1_error_0 < tolerance, f"L1 error for k=0 is too high: {l1_error_0:.2e}"
        assert l1_error_1 < tolerance, f"L1 error for k=1 is too high: {l1_error_1:.2e}"

        # Also check L-infinity norm (max absolute difference)
        # 同时检查 L-无穷范数（最大绝对差）
        linf_error_0 = jnp.max(jnp.abs(marginal_0_new - marginal_0_direct))
        linf_error_1 = jnp.max(jnp.abs(marginal_1_new - marginal_1_direct))

        assert linf_error_0 < tolerance, f"L-inf error for k=0 is too high: {linf_error_0:.2e}"
        assert linf_error_1 < tolerance, f"L-inf error for k=1 is too high: {linf_error_1:.2e}"

    def test_generalization_for_k4(self):
        """
        Tests that the algorithm runs correctly and produces valid probability
        distributions for a more complex case (K=4).
        测试算法在更复杂的情况下（K=4）能正确运行并产生有效的概率分布。
        """
        # Setup a K=4 problem
        # 设置一个 K=4 的问题
        potentials, log_trans_mats, problem = _create_test_problem(K=4, n_points=60)
        h = problem.grid.spacing

        # Compute all four marginals using the log-space method
        # 使用对数空间方法计算所有四个边际
        jitted_log_space_marginal = jax.jit(partial(_compute_current_marginal, grid_spacing=h))

        for k in range(4):
            log_marginal_k = jitted_log_space_marginal(k=k, potentials=potentials, log_transition_matrices=log_trans_mats)
            marginal_k = jnp.exp(log_marginal_k)

            # --- Validation Checks ---
            # --- 验证检查 ---

            # 1. Check for NaNs or Infs, which indicate numerical instability
            #    检查是否存在 NaN 或 Inf，这表明存在数值不稳定性
            assert not jnp.any(jnp.isnan(marginal_k)), f"NaNs found in marginal {k} for K=4"
            assert not jnp.any(jnp.isinf(marginal_k)), f"Infs found in marginal {k} for K=4"

            # 2. Check that the marginal is a valid probability distribution (integrates to 1)
            #    检查边际是否为有效的概率分布（积分为1）
            mass = jax_trapz(marginal_k, dx=h)
            assert jnp.allclose(mass, 1.0, atol=1e-9), f"Marginal {k} for K=4 not normalized. Mass={mass:.6f}"

            # 3. Check that all values are non-negative
            #    检查所有值是否为非负
            assert jnp.all(marginal_k >= 0), f"Negative probabilities found in marginal {k} for K=4"

    def test_numerical_stability_stress_test(self):
        """
        Tests that the log-space implementation succeeds where the direct
        method fails due to numerical underflow.
        测试对数空间实现在直接方法因数值下溢失败时仍能成功。
        """
        # Create a problem designed to cause underflow: long chain, very small diffusion
        # 创建一个旨在导致下溢的问题：长链、非常小的扩散
        K = 15  # Even longer chain of observations / 更长的观测链
        potentials, log_trans_mats, problem = _create_test_problem(
            K=K, n_points=50, diffusion=0.05
        )
        h = problem.grid.spacing

        # --- 1. Run the numerically unstable direct method and assert failure ---
        # --- 1. 运行数值不稳定的直接方法并断言其失败 ---
        # We expect this to produce all zeros due to underflow
        # 我们预计由于下溢，这将产生全零结果
        direct_marginal = _compute_current_marginal_direct(
            k=K//2,
            potentials=potentials,
            log_transition_matrices=log_trans_mats,
            grid_spacing=h
        )
        
        direct_mass = jax_trapz(direct_marginal, dx=h)
        # The direct method should fail completely (mass becomes zero)
        # 直接方法应该完全失败（质量变为零）
        assert jnp.allclose(direct_mass, 0.0, atol=1e-9), \
            f"Direct method did not fail as expected. Mass={direct_mass:.2e}"

        # --- 2. Run the new log-space method and assert success ---
        # --- 2. 运行新的对数空间方法并断言其成功 ---
        # We need to create a new jitted function because K has changed, which affects the shapes
        # 我们需要创建一个新的JIT编译函数，因为K的改变会影响形状
        jitted_log_space_marginal = jax.jit(
            partial(_compute_current_marginal, grid_spacing=h)
        )
        
        log_marginal_new = jitted_log_space_marginal(
            k=K//2,
            potentials=potentials,
            log_transition_matrices=log_trans_mats
        )
        marginal_new = jnp.exp(log_marginal_new)
        
        # The new method should produce a valid, non-zero probability distribution
        # 新方法应该产生一个有效的、非零的概率分布
        assert not jnp.any(jnp.isnan(marginal_new)), "Log-space method produced NaNs"
        
        new_mass = jax_trapz(marginal_new, dx=h)
        assert jnp.allclose(new_mass, 1.0, atol=1e-9), \
            f"Log-space method failed to produce a normalized distribution. Mass={new_mass:.6f}"
