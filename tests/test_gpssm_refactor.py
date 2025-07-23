"""
Unit and Integration Tests for the Refactored GPSSM Implementation
====================================================================

This test suite validates the correctness of the refactored GPSSM modules,
ensuring that the mathematical implementations are sound and the components
integrate correctly.

此测试套件用于验证重构后的GPSSM模块的正确性，确保数学实现的可靠性
以及各组件能正确集成。
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components from the refactored gpssm baseline
from src.baselines.gpssm.types import (
    GPSSMConfig, OptimizerConfig, KernelParams, InducingPoints, GPParams, VariationalParams, GPSSMState
)
from src.baselines.gpssm import gpr, inference, models
from src.baselines.gpssm.gpssm import GPSSMSolver

# ============================================================================
# Test Fixtures / 测试固件
# ============================================================================

@pytest.fixture
def test_keys():
    """Provides a JAX random key for tests."""
    return random.PRNGKey(42)

@pytest.fixture
def basic_config():
    """Provides a basic GPSSM configuration for testing."""
    return GPSSMConfig(
        state_dim=2,
        obs_dim=1,
        num_inducing=10,
        num_particles=5,
        jitter=1e-6
    )

@pytest.fixture
def sample_params(basic_config, test_keys):
    """Provides a sample set of GPSSM parameters."""
    D, M = basic_config.state_dim, basic_config.num_inducing
    key1, key2, key3 = random.split(test_keys, 3)
    
    gp_params = GPParams(
        kernel=KernelParams(lengthscale=jnp.ones(D), variance=1.0),
        inducing=InducingPoints(
            z=random.normal(key1, (M, D)),
            m=random.normal(key2, (M, D)),
            L=jnp.eye(M)
        ),
        obs_noise_variance=0.1
    )
    
    T = 20 # Sample time series length
    var_params = VariationalParams(
        q_mu=random.normal(key3, (T, D)),
        q_sqrt=jnp.tile(jnp.eye(D) * 0.1, (T, 1, 1))
    )
    
    return GPSSMState(gp=gp_params, variational=var_params)

# ============================================================================
# Unit Tests for GPR Module / GPR模块单元测试
# ============================================================================

def test_rbf_kernel(basic_config):
    """Tests the RBF kernel for shape and basic properties."""
    D = basic_config.state_dim
    params = KernelParams(lengthscale=jnp.ones(D), variance=1.5)
    x1 = jnp.zeros((5, D))
    x2 = jnp.ones((3, D))
    
    K = gpr.rbf_kernel(x1, x2, params)
    assert K.shape == (5, 3)
    assert jnp.all(K > 0)

    K_diag = gpr.rbf_kernel_diag(x1, params)
    assert K_diag.shape == (5,)
    assert jnp.allclose(K_diag, 1.5)

def test_gpr_predict_f(basic_config, sample_params, test_keys):
    """Tests the GP prediction function for output shapes."""
    x_test = random.normal(test_keys, (15, basic_config.state_dim))
    mean, var = gpr.predict_f(
        x_test, sample_params.gp.inducing, sample_params.gp.kernel, basic_config, basic_config.state_dim
    )
    assert mean.shape == (15, basic_config.state_dim)
    assert var.shape == (15, basic_config.state_dim)
    assert jnp.all(var >= 0)

def test_kl_divergence(basic_config, sample_params):
    """Tests the KL divergence calculation."""
    kl = gpr.kl_divergence(sample_params.gp.inducing, sample_params.gp.kernel, basic_config)
    assert isinstance(kl, jax.Array)
    assert kl.shape == ()
    assert kl >= 0

# ============================================================================
# Unit Tests for Inference Module / Inference模块单元测试
# ============================================================================

def test_sample_states(basic_config, sample_params, test_keys):
    """Tests the state sampling function."""
    samples = inference.sample_states(sample_params.variational, test_keys, basic_config.num_particles)
    T, D = sample_params.variational.q_mu.shape
    K = basic_config.num_particles
    assert samples.shape == (K, T, D)

def test_gaussian_log_pdf():
    """Tests the Gaussian log PDF with a known case."""
    mean = jnp.zeros(2)
    cov = jnp.eye(2)
    x = jnp.zeros(2)
    # For standard normal at origin, log_pdf is -0.5 * (2*log(2pi))
    expected = -0.5 * (2 * jnp.log(2 * jnp.pi))
    log_prob = inference.gaussian_log_pdf(x, mean, cov)
    assert jnp.allclose(log_prob, expected)

def test_compute_elbo(basic_config, sample_params, test_keys):
    """Tests that the ELBO computation runs and returns a scalar."""
    T, D = sample_params.variational.q_mu.shape
    _, P = basic_config.state_dim, basic_config.obs_dim
    
    observations = jnp.zeros((T, P))
    dynamics_fn = lambda x: x
    observation_fn = lambda x: x[:P]
    
    elbo = inference.compute_elbo(
        sample_params, observations, test_keys, dynamics_fn, observation_fn, basic_config
    )
    assert isinstance(elbo, jax.Array)
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)

# ============================================================================
# Integration Test for GPSSMSolver / GPSSMSolver集成测试
# ============================================================================

def test_gpssm_solver_full_cycle():
    """
    Tests the full initialize -> fit -> predict cycle on a simple linear system.
    在一个简单的线性系统上测试完整的 initialize -> fit -> predict 流程。
    """
    # 1. Setup a simple linear system
    key = random.PRNGKey(0)
    T, D, P = 50, 2, 1
    
    A = jnp.array([[0.9, 0.1], [-0.1, 0.9]])
    H = jnp.array([[1.0, 0.0]])
    
    true_states = [jnp.array([1.0, 0.0])]
    for t in range(1, T):
        true_states.append(A @ true_states[-1])
    true_states = jnp.array(true_states)
    observations = true_states @ H.T + random.normal(key, (T, P)) * 0.1

    # 2. Configure and initialize the solver
    gpssm_config = GPSSMConfig(state_dim=D, obs_dim=P, num_inducing=15, num_particles=10)
    opt_config = OptimizerConfig(learning_rate=1e-2, num_iterations=100, clip_norm=10.0)
    
    # The GP will learn the residual dynamics, so we provide a simple identity as f_det
    dynamics_fn = lambda x: x
    obs_fn = lambda x: H @ x
    
    solver = GPSSMSolver(gpssm_config, opt_config, dynamics_fn, obs_fn)
    
    # 3. Run the fitting process
    fit_key, pred_key = random.split(key)
    final_params, history = solver.fit(fit_key, observations)

    # 4. Assertions
    # Check if ELBO improved
    assert history['elbo'][-1] > history['elbo'][0]
    
    # Check for valid parameter values
    flat_params, _ = jax.tree_util.tree_flatten(final_params)
    for p in flat_params:
        assert jnp.all(jnp.isfinite(p))

    # Check prediction
    initial_state_for_pred = final_params.variational.q_mu[0]
    predictions = solver.predict(final_params, num_steps=10, initial_state=initial_state_for_pred)
    assert predictions.shape == (10, D)
    assert jnp.all(jnp.isfinite(predictions))
