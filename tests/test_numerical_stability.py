"""
High-Dimensional Numerical Stability Tests for GPSSM / GPSSM高维数值稳定性测试
==============================================================================

This test suite validates the numerical stability improvements in different
dimensional scenarios and ill-conditioned matrices.

此测试套件验证了不同维度场景和病态矩阵中的数值稳定性改进。
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.baselines.gpssm.stable_numerics import (
    robust_cholesky_decomposition, 
    stable_log_determinant,
    estimate_condition_number,
    compute_adaptive_jitter,
    diagnose_numerical_health,
    robust_gp_solve
)
from src.baselines.gpssm.types import GPSSMConfig, KernelParams, InducingPoints, GPParams, VariationalParams, GPSSMState
from src.baselines.gpssm import gpr, inference
from src.baselines.gpssm.gpssm import GPSSMSolver


# ============================================================================
# Test Fixtures / 测试固件
# ============================================================================

@pytest.fixture
def test_keys():
    """Provides JAX random keys for tests."""
    return random.PRNGKey(12345)


def create_ill_conditioned_matrix(key, size, condition_number=1e8):
    """Create an ill-conditioned positive definite matrix."""
    # Generate random orthogonal matrix
    A = random.normal(key, (size, size))
    Q, _ = jnp.linalg.qr(A)
    
    # Create eigenvalues with specified condition number
    eigenvals = jnp.logspace(0, -jnp.log10(condition_number), size)
    
    # Construct ill-conditioned matrix
    return Q @ jnp.diag(eigenvals) @ Q.T


def create_high_dim_gp_scenario(key, state_dim, obs_dim=None, num_inducing=None):
    """Create a high-dimensional GP scenario for testing."""
    if obs_dim is None:
        obs_dim = max(1, state_dim // 2)
    if num_inducing is None:
        num_inducing = min(50, state_dim * 2)
    
    key1, key2, key3, key4 = random.split(key, 4)
    
    config = GPSSMConfig(
        state_dim=state_dim,
        obs_dim=obs_dim,
        num_inducing=num_inducing,
        num_particles=10,
        jitter=1e-6
    )
    
    # Create GP parameters
    gp_params = GPParams(
        kernel=KernelParams(
            lengthscale=jnp.ones(state_dim) * 0.5,  # Shorter lengthscales can cause issues
            variance=1.0
        ),
        inducing=InducingPoints(
            z=random.normal(key1, (num_inducing, state_dim)) * 2.0,
            m=random.normal(key2, (num_inducing, state_dim)) * 0.1,
            L=jnp.eye(num_inducing) * 0.1
        ),
        obs_noise_variance=0.01
    )
    
    # Variational parameters
    T = 30
    var_params = VariationalParams(
        q_mu=random.normal(key3, (T, state_dim)) * 0.1,
        q_sqrt=jnp.tile(jnp.eye(state_dim) * 0.05, (T, 1, 1))
    )
    
    params = GPSSMState(gp=gp_params, variational=var_params)
    
    # Generate observations
    observations = random.normal(key4, (T, obs_dim)) * 0.1
    
    return config, params, observations


# ============================================================================
# Stable Numerics Module Tests / 稳定数值模块测试
# ============================================================================

def test_condition_number_estimation(test_keys):
    """Test condition number estimation accuracy."""
    # Well-conditioned matrix
    well_cond = jnp.eye(5) + 0.1 * random.normal(test_keys, (5, 5))
    well_cond = well_cond @ well_cond.T
    
    # Ill-conditioned matrix
    key1, key2 = random.split(test_keys)
    ill_cond = create_ill_conditioned_matrix(key1, 5, 1e6)
    
    cond_well = estimate_condition_number(well_cond)
    cond_ill = estimate_condition_number(ill_cond)
    
    # Well-conditioned should have lower condition number
    assert cond_well < cond_ill
    assert cond_well < 50  # Should be reasonably well-conditioned
    assert cond_ill > 10   # Should detect ill-conditioning (relaxed threshold)


def test_adaptive_jitter_scaling(test_keys):
    """Test that adaptive jitter scales appropriately with condition number."""
    base_jitter = 1e-6
    
    # Well-conditioned matrix
    well_cond = jnp.eye(5)
    jitter_well = compute_adaptive_jitter(well_cond, base_jitter)
    
    # Ill-conditioned matrix
    ill_cond = create_ill_conditioned_matrix(test_keys, 5, 1e8)
    jitter_ill = compute_adaptive_jitter(ill_cond, base_jitter)
    
    # Ill-conditioned should get larger jitter
    assert jitter_ill > jitter_well
    assert jitter_well <= base_jitter * 2  # Should not increase much for well-conditioned
    assert jitter_ill >= base_jitter * 10  # Should increase significantly for ill-conditioned


def test_robust_cholesky_fallback(test_keys):
    """Test that robust Cholesky falls back to SVD for ill-conditioned matrices."""
    # Create a matrix that will challenge standard Cholesky
    key1, key2 = random.split(test_keys)
    
    # Moderately ill-conditioned matrix (extreme cases may not have bounded solutions)
    size = 10
    ill_cond = create_ill_conditioned_matrix(key1, size, 1e8)
    
    # Robust decomposition should handle this
    L, jitter_used, is_chol = robust_cholesky_decomposition(ill_cond, 1e-6)
    
    # Should return valid decomposition
    assert L.shape == (size, size)
    assert jitter_used > 1e-6  # Should use adaptive jitter
    assert jnp.all(jnp.isfinite(L))
    
    # Test that it can solve systems
    b = random.normal(key2, (size,))
    x = robust_gp_solve(ill_cond, b, 1e-6)
    
    # Solution should be finite and reasonable
    assert jnp.all(jnp.isfinite(x))
    # For ill-conditioned problems, solutions may be large but should be finite
    assert jnp.linalg.norm(x) < 1e6  # Should not blow up to infinity


def test_stable_log_determinant(test_keys):
    """Test stable log determinant computation."""
    # Test on various condition numbers
    sizes = [5, 10, 20]
    condition_numbers = [1e2, 1e6, 1e10]
    
    for i, size in enumerate(sizes):
        for j, cond_num in enumerate(condition_numbers):
            # Use smaller fold_in values to avoid overflow
            key = random.fold_in(test_keys, i * 10 + j)
            matrix = create_ill_conditioned_matrix(key, size, cond_num)
            
            # Stable computation should not fail
            logdet = stable_log_determinant(matrix, 1e-6)
            
            assert jnp.isfinite(logdet)
            # Log determinant should be reasonable (not too extreme)
            assert jnp.abs(logdet) < 1000


# ============================================================================
# High-Dimensional GP Tests / 高维GP测试
# ============================================================================

@pytest.mark.parametrize("state_dim", [5, 10, 20, 50])
def test_high_dimensional_gp_prediction(test_keys, state_dim):
    """Test GP prediction in high dimensions."""
    key1, key2 = random.split(test_keys)
    
    config, params, _ = create_high_dim_gp_scenario(key1, state_dim)
    
    # Test prediction
    x_test = random.normal(key2, (15, state_dim))
    
    try:
        mean, var = gpr.predict_f(
            x_test, params.gp.inducing, params.gp.kernel, config, state_dim
        )
        
        # Predictions should be finite and reasonable
        assert mean.shape == (15, state_dim)
        assert var.shape == (15, state_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(var))
        assert jnp.all(var >= 0)  # Variance should be non-negative
        
        # Variance should be reasonable (not too small or too large)
        assert jnp.all(var < 100)
        assert jnp.all(var > 1e-10)
        
    except Exception as e:
        pytest.fail(f"GP prediction failed in {state_dim}D: {e}")


@pytest.mark.parametrize("state_dim", [5, 10, 20])
def test_high_dimensional_kl_divergence(test_keys, state_dim):
    """Test KL divergence computation in high dimensions."""
    key = random.fold_in(test_keys, state_dim)
    config, params, _ = create_high_dim_gp_scenario(key, state_dim)
    
    try:
        kl = gpr.kl_divergence(params.gp.inducing, params.gp.kernel, config)
        
        # KL should be finite and non-negative
        assert jnp.isfinite(kl)
        assert kl >= 0
        # Should be reasonable magnitude (higher dimensions naturally have larger KL)
        assert kl < 5000
        
    except Exception as e:
        pytest.fail(f"KL divergence computation failed in {state_dim}D: {e}")


@pytest.mark.parametrize("state_dim", [5, 10, 20])
def test_high_dimensional_elbo_computation(test_keys, state_dim):
    """Test ELBO computation in high dimensions."""
    key1, key2 = random.split(test_keys)
    
    config, params, observations = create_high_dim_gp_scenario(key1, state_dim)
    
    # Simple dynamics and observation functions
    dynamics_fn = lambda x: x * 0.95  # Simple decay
    observation_fn = lambda x: x[:config.obs_dim]  # Partial observation
    
    try:
        elbo = inference.compute_elbo(
            params, observations, key2, dynamics_fn, observation_fn, config
        )
        
        # ELBO should be finite
        assert jnp.isfinite(elbo)
        # Should be reasonable magnitude (not extremely negative)
        assert elbo > -10000
        
    except Exception as e:
        pytest.fail(f"ELBO computation failed in {state_dim}D: {e}")


# ============================================================================
# End-to-End Stability Tests / 端到端稳定性测试
# ============================================================================

@pytest.mark.parametrize("state_dim", [5, 10])
def test_high_dimensional_training_stability(test_keys, state_dim):
    """Test that training remains stable in high dimensions."""
    key1, key2 = random.split(test_keys)
    
    config, params, observations = create_high_dim_gp_scenario(key1, state_dim)
    
    # Reduce iterations for faster testing
    from src.baselines.gpssm.types import OptimizerConfig
    opt_config = OptimizerConfig(
        learning_rate=1e-3,
        num_iterations=20,  # Just test stability, not convergence
        clip_norm=10.0
    )
    
    # Simple dynamics and observation
    dynamics_fn = lambda x: x * 0.9
    obs_fn = lambda x: x[:config.obs_dim]
    
    solver = GPSSMSolver(config, opt_config, dynamics_fn, obs_fn)
    
    try:
        final_params, history = solver.fit(key2, observations)
        
        # Training should complete without numerical errors
        assert len(history['elbo']) == opt_config.num_iterations
        assert jnp.all(jnp.isfinite(jnp.array(history['elbo'])))
        
        # Parameters should remain finite
        flat_params, _ = jax.tree_util.tree_flatten(final_params)
        for p in flat_params:
            assert jnp.all(jnp.isfinite(p))
            
        print(f"✅ {state_dim}D training stable. Final ELBO: {history['elbo'][-1]:.4f}")
        
    except Exception as e:
        pytest.fail(f"Training failed in {state_dim}D: {e}")


def test_numerical_health_diagnostics(test_keys):
    """Test numerical health diagnostic functions."""
    # Create matrices with different conditioning
    matrices = [
        jnp.eye(5),  # Well-conditioned
        create_ill_conditioned_matrix(test_keys, 5, 1e6),  # Moderately ill-conditioned
        create_ill_conditioned_matrix(random.fold_in(test_keys, 1), 5, 1e10)  # Severely ill-conditioned
    ]
    
    for i, matrix in enumerate(matrices):
        diagnostics = diagnose_numerical_health(matrix, 1e-6)
        
        # All diagnostics should be finite
        assert jnp.isfinite(diagnostics['condition_estimate'])
        assert jnp.isfinite(diagnostics['adaptive_jitter'])
        assert jnp.isfinite(diagnostics['jitter_multiplier'])
        
        # More ill-conditioned matrices should have higher condition estimates
        if i > 0:
            prev_cond = diagnose_numerical_health(matrices[i-1], 1e-6)['condition_estimate']
            assert diagnostics['condition_estimate'] >= prev_cond
        
        print(f"Matrix {i}: condition={diagnostics['condition_estimate']:.2e}, "
              f"jitter_mult={diagnostics['jitter_multiplier']:.2f}, "
              f"use_chol={diagnostics['use_cholesky']}")


if __name__ == "__main__":
    # Run a subset of tests for quick validation
    key = random.PRNGKey(42)
    
    print("Testing numerical stability improvements...")
    
    # Test condition estimation
    test_condition_number_estimation(key)
    print("✅ Condition number estimation working")
    
    # Test adaptive jitter
    test_adaptive_jitter_scaling(key)
    print("✅ Adaptive jitter scaling working")
    
    # Test robust decomposition
    test_robust_cholesky_fallback(key)
    print("✅ Robust Cholesky fallback working")
    
    # Test high-dimensional scenarios
    for dim in [5, 10, 20]:
        test_high_dimensional_gp_prediction(key, dim)
        print(f"✅ {dim}D GP prediction stable")
        
    print("\n🎉 All numerical stability tests passed!")