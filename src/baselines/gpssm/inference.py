"""
Variational Inference Core Logic for GPSSM / GPSSM的变分推断核心逻辑
===================================================================

This module implements the core mathematical components for variational
inference in Gaussian Process State-Space Models, focusing on the calculation
of the Evidence Lower Bound (ELBO).

此模块实现了高斯过程状态空间模型中变分推断的核心数学组件，
专注于证据下界（ELBO）的计算。

The ELBO is composed of:
ELBO 由以下部分组成：
- Expected Log-Likelihood of Observations / 观测的期望对数似然
- Expected Log-Likelihood of State Transitions / 状态转移的期望对数似然
- KL Divergence between q(u) and p(u) / q(u)与p(u)的KL散度
- Entropy of the state variational distribution q(x) / 状态变分分布q(x)的熵
"""

import jax
import chex
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cholesky, solve_triangular
from functools import partial
from typing import Callable

from .types import GPSSMConfig, GPSSMState, VariationalParams
from . import gpr
from .stable_numerics import robust_cholesky_decomposition, robust_solve_triangular

# ============================================================================
# Helper Functions / 辅助函数
# ============================================================================

@partial(jit, static_argnames=('num_particles',))
def sample_states(
    params: VariationalParams,
    key: chex.PRNGKey,
    num_particles: int
) -> chex.Array:
    """
    Sample state trajectories from the variational distribution q(x_{1:T}).
    从变分分布 q(x_{1:T}) 中采样状态轨迹。

    Uses the reparameterization trick: x = μ + L * ε, where ε ~ N(0, I).
    使用重参数化技巧：x = μ + L * ε, 其中 ε ~ N(0, I)。

    Args:
        params: Variational parameters (q_mu, q_sqrt).
        key: JAX random key.
        num_particles: The number of sample trajectories to draw.

    Returns:
        Sampled state trajectories [num_particles, T, D].
    """
    T, D = params.q_mu.shape
    noise = jax.random.normal(key, (num_particles, T, D))
    # x_k = μ + L @ ε_k
    # Einsum is efficient for batch matrix-vector products.
    # 'tij,tkj->tki' would be for (T,D,D) and (T,K,D), giving (T,K,D)
    # We have (T,D,D) and (K,T,D), giving (K,T,D)
    # 'tdj,ktd->ktj'
    samples = params.q_mu[None, ...] + jnp.einsum('tde,kte->ktd', params.q_sqrt, noise)
    return samples


@partial(jit, static_argnums=())
def gaussian_log_pdf(x: chex.Array, mean: chex.Array, cov: chex.Array) -> chex.Scalar:
    """
    Computes the log probability density of a multivariate Gaussian.
    计算多元高斯分布的对数概率密度。

    log p(x) = -0.5 * [ (x-μ)ᵀ Σ⁻¹ (x-μ) + log|2πΣ| ]

    Args:
        x: The point to evaluate [D].
        mean: The mean of the Gaussian [D].
        cov: The covariance matrix of the Gaussian [D, D].

    Returns:
        The log probability density, a scalar value.
    """
    D = x.shape[-1]
    
    # Use robust decomposition for numerical stability
    L, jitter_used, is_chol = robust_cholesky_decomposition(cov, 1e-6)
    diff = x - mean
    
    # alpha = L⁻¹ * diff (or equivalent for SVD case)
    alpha = robust_solve_triangular(L, diff, is_chol, jitter_used)
    
    # quad_form = diffᵀ Σ⁻¹ diff = alphaᵀ alpha
    quad_form = jnp.sum(alpha**2, axis=-1)
    
    # Use stable log determinant computation
    # Since we always use Cholesky now, this is simplified
    log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    
    log_prob = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det + quad_form)
    return log_prob


# ============================================================================
# ELBO Components / ELBO组成部分
# ============================================================================

@partial(jit, static_argnames=('observation_fn',))
def expected_log_likelihood(
    params: GPSSMState,
    observations: chex.Array,
    state_samples: chex.Array,
    observation_fn: Callable[[chex.Array], chex.Array]
) -> chex.Scalar:
    """
    Computes the expected log-likelihood of observations (the first term in ELBO).
    计算观测的期望对数似然（ELBO第一项）。

    Term: E_q(x) [ Σ_t log p(y_t | x_t) ]

    Args:
        params: All GPSSM parameters.
        observations: The sequence of observations [T, P].
        state_samples: Sampled state trajectories [K, T, D].
        observation_fn: The observation function h(x).

    Returns:
        The expected log-likelihood, averaged over particles.
    """
    obs_cov = params.gp.obs_noise_variance * jnp.eye(observations.shape[-1])
    
    # Vectorize over particles and time steps
    # Input shapes: state_samples (K, T, D), observations (T, P)
    # We want to compute log_pdf for each particle and time step.
    # vmap over particles (axis 0), then over time (axis 0 of inner arrays)
    def log_lik_per_particle(single_state_traj):
        # single_state_traj: [T, D]
        predicted_obs = vmap(observation_fn)(single_state_traj) # [T, P]
        # Use in_axes=(0, 0, None) to broadcast the same covariance matrix
        log_probs = vmap(gaussian_log_pdf, in_axes=(0, 0, None))(
            observations, predicted_obs, obs_cov
        )
        return jnp.sum(log_probs)

    total_log_likelihood = jnp.mean(vmap(log_lik_per_particle)(state_samples))
    return total_log_likelihood


@partial(jit, static_argnames=('dynamics_fn', 'config'))
def expected_log_transition(
    params: GPSSMState,
    state_samples: chex.Array,
    dynamics_fn: Callable[[chex.Array], chex.Array],
    config: GPSSMConfig
) -> chex.Scalar:
    """
    Computes the expected log-likelihood of state transitions (the second term in ELBO).
    计算状态转移的期望对数似然（ELBO第二项）。

    Term: E_q(x) [ Σ_t log p(x_{t+1} | x_t) ]
    where p(x_{t+1} | x_t) = ∫ p(x_{t+1} | f(x_t)) p(f(x_t) | u) df(x_t)

    Args:
        params: All GPSSM parameters.
        state_samples: Sampled state trajectories [K, T, D].
        dynamics_fn: The deterministic part of the dynamics f_det(x).
        config: GPSSM configuration.

    Returns:
        The expected log transition probability, averaged over particles.
    """
    K, T, D = state_samples.shape
    
    # We need to compute the transition probability for each particle from t=0 to T-2
    # state_samples_t: [K, D], state_samples_t_plus_1: [K, D]
    
    def transition_log_prob_per_step(states_t, states_t_plus_1):
        # states_t: [K, D], states_t_plus_1: [K, D]
        
        # Get predictive distribution for f(x_t) from the GP
        # The prediction is done for all K particles at once
        f_mean, f_var = gpr.predict_f(
            states_t, params.gp.inducing, params.gp.kernel, config, config.state_dim
        )
        
        # The mean of x_{t+1} is f_det(x_t) + E[f(x_t)]
        x_next_mean = vmap(dynamics_fn)(states_t) + f_mean
        
        # The covariance of x_{t+1} is Var[f(x_t)] (as process noise is separate)
        # Here we assume process noise is part of the GP, which is a common choice.
        # If it were separate, we would add Q here.
        x_next_cov = f_var # [K, D]
        
        # We need a covariance matrix [K, D, D]
        x_next_cov_diag = vmap(jnp.diag)(x_next_cov)
        
        # Compute log p(x_{t+1} | x_t) for each particle
        log_probs = vmap(gaussian_log_pdf)(states_t_plus_1, x_next_mean, x_next_cov_diag)
        return jnp.sum(log_probs)

    # Loop over time steps, applying the vmapped function
    total_log_prob = 0.0
    for t in range(T - 1):
        total_log_prob += transition_log_prob_per_step(
            state_samples[:, t, :], state_samples[:, t + 1, :]
        )
        
    return total_log_prob / K # Average over particles


@partial(jit, static_argnames=())
def state_entropy(params: VariationalParams) -> chex.Scalar:
    """
    Computes the entropy of the Gaussian variational distribution over states.
    计算状态的高斯变分分布的熵。

    H(q(x)) = 0.5 * Σ_t [ D * log(2πe) + log|Σ_t| ]

    Args:
        params: Variational parameters (q_mu, q_sqrt).

    Returns:
        The total entropy of the state variational distribution.
    """
    # log|Σ_t| = log|L_t L_tᵀ| = 2 * log|L_t| = 2 * Σ_i log(L_t_ii)
    log_det_cov = 2 * jnp.sum(jnp.log(jnp.diagonal(params.q_sqrt, axis1=-2, axis2=-1)), axis=1)
    
    D = params.q_mu.shape[-1]
    entropy_per_step = 0.5 * (D * (1 + jnp.log(2 * jnp.pi)) + log_det_cov)
    
    return jnp.sum(entropy_per_step)


@partial(jit, static_argnames=('dynamics_fn', 'observation_fn', 'config'))
def compute_elbo(
    params: GPSSMState,
    observations: chex.Array,
    key: chex.PRNGKey,
    dynamics_fn: Callable[[chex.Array], chex.Array],
    observation_fn: Callable[[chex.Array], chex.Array],
    config: GPSSMConfig
) -> chex.Scalar:
    """
    Computes the Evidence Lower Bound (ELBO) for the GPSSM.
    计算GPSSM的证据下界（ELBO）。

    ELBO = E_q[log p(y|x)] + E_q[log p(x|z,u)] - KL[q(u)||p(u)] + H[q(x)]

    Args:
        params: All GPSSM parameters.
        observations: The sequence of observations [T, P].
        key: JAX random key for sampling.
        dynamics_fn: The deterministic part of the dynamics f_det(x).
        observation_fn: The observation function h(x).
        config: GPSSM configuration.

    Returns:
        The ELBO, a scalar value.
    """
    # 1. Sample from q(x)
    state_samples = sample_states(params.variational, key, config.num_particles)
    
    # 2. Compute E_q[log p(y|x)]
    log_lik = expected_log_likelihood(params, observations, state_samples, observation_fn)
    
    # 3. Compute E_q[log p(x_{t+1}|x_t)]
    log_trans = expected_log_transition(params, state_samples, dynamics_fn, config)
    
    # 4. Compute KL[q(u)||p(u)]
    kl_u = gpr.kl_divergence(params.gp.inducing, params.gp.kernel, config)
    
    # 5. Compute H[q(x)]
    entropy_x = state_entropy(params.variational)
    
    # Combine terms
    elbo = log_lik + log_trans - kl_u + entropy_x
    
    return elbo