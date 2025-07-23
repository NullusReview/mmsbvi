"""
GPSSM Usage Examples / GPSSM使用示例
=========================================================

This script demonstrates how to use the GPSSM framework.
此脚本演示如何使用GPSSM框架。

Example Systems / 示例系统:
1. Pendulum System / 单摆系统
2. Lorenz Chaotic System / 洛伦兹混沌系统
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any
import time
from jax import vmap

# Import from the new refactored modules
from .gpssm import GPSSMSolver
from .types import GPSSMConfig, OptimizerConfig
from .models import create_pendulum_system, create_lorenz_system

# ============================================================================
# Data Generation Functions (largely unchanged) / 数据生成函数（基本不变）
# ============================================================================

def generate_pendulum_data(
    T: int = 100,
    dt: float = 0.05,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates data from the pendulum system."""
    initial_state = jnp.array([jnp.pi / 4, 0.0])
    dynamics_model, obs_model = create_pendulum_system(dt=dt)
    dynamics_fn = dynamics_model.get_mean_function()
    
    # Note: In a real scenario, process noise would be added.
    # Here we simulate a deterministic trajectory for clarity.
    
    states = [initial_state]
    current_state = initial_state
    for _ in range(1, T):
        current_state = dynamics_fn(current_state)
        states.append(current_state)
    
    true_states = jnp.array(states)
    
    # Generate noisy observations
    obs_fn = obs_model.get_observation_function()
    obs_noise_std = 0.05
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, 1)) * obs_noise_std
    observations = vmap(obs_fn)(true_states) + noise
    
    return true_states, observations


def generate_lorenz_data(
    T: int = 1000,
    dt: float = 0.01,
    key: jax.random.PRNGKey = jax.random.PRNGKey(123)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates data from the Lorenz system."""
    initial_state = jnp.array([1.0, 1.0, 1.0])
    dynamics_model, obs_model = create_lorenz_system(dt=dt)
    dynamics_fn = dynamics_model.get_mean_function()

    states = [initial_state]
    current_state = initial_state
    for _ in range(1, T):
        current_state = dynamics_fn(current_state)
        states.append(current_state)
        
    true_states = jnp.array(states)
    
    obs_fn = obs_model.get_observation_function()
    obs_noise_std = 0.1
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, 2)) * obs_noise_std
    observations = vmap(obs_fn)(true_states) + noise
    
    return true_states, observations


# ============================================================================
# Refactored GPSSM Example Functions / 重构后的GPSSM示例函数
# ============================================================================

def example_pendulum_gpssm():
    """Pendulum system GPSSM example using the new framework."""
    print("🔄 [Refactored] Pendulum System GPSSM Example")
    print("=" * 60)
    
    # 1. Generate Data
    key = random.PRNGKey(42)
    data_key, fit_key = random.split(key)
    true_states, observations = generate_pendulum_data(T=100, key=data_key)
    print(f"Generated data: {len(observations)} time steps")

    # 2. Configure Models
    gpssm_config = GPSSMConfig(
        state_dim=2,
        obs_dim=1,
        num_inducing=30,
        num_particles=20,
        jitter=1e-5
    )
    opt_config = OptimizerConfig(
        learning_rate=1e-2,
        num_iterations=1000,
        clip_norm=10.0
    )

    # 3. Get Dynamics and Observation Functions
    dynamics_model, obs_model = create_pendulum_system()
    dynamics_fn = dynamics_model.get_mean_function()
    obs_fn = obs_model.get_observation_function()

    # 4. Create and Run Solver
    solver = GPSSMSolver(gpssm_config, opt_config, dynamics_fn, obs_fn)
    
    final_params, history = solver.fit(fit_key, observations)
    
    # 5. Predict and Visualize
    smoothed_states = final_params.variational.q_mu
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['elbo'])
    plt.title('ELBO Training History')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(true_states[:, 0], 'k--', label='True Angle', alpha=0.6)
    plt.plot(observations, 'rx', label='Observations', markersize=4, alpha=0.5)
    plt.plot(smoothed_states[:, 0], 'b-', label='Smoothed Angle (q_mu)')
    plt.title('Pendulum State Estimation')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return final_params, history


def example_lorenz_gpssm():
    """Lorenz system GPSSM example using the new framework."""
    print("\n🌪️ [Refactored] Lorenz Chaotic System GPSSM Example")
    print("=" * 60)
    
    # 1. Generate Data
    key = random.PRNGKey(123)
    data_key, fit_key = random.split(key)
    true_states, observations = generate_lorenz_data(T=200, key=data_key)
    print(f"Generated Lorenz data: {len(observations)} time steps")

    # 2. Configure Models
    gpssm_config = GPSSMConfig(
        state_dim=3,
        obs_dim=2,
        num_inducing=50,
        num_particles=25,
        jitter=1e-5
    )
    opt_config = OptimizerConfig(
        learning_rate=5e-3,
        num_iterations=1500,
        clip_norm=15.0
    )

    # 3. Get Dynamics and Observation Functions
    dynamics_model, obs_model = create_lorenz_system()
    dynamics_fn = dynamics_model.get_mean_function()
    obs_fn = obs_model.get_observation_function()

    # 4. Create and Run Solver
    solver = GPSSMSolver(gpssm_config, opt_config, dynamics_fn, obs_fn)
    final_params, history = solver.fit(fit_key, observations)

    # 5. Visualize
    smoothed_states = final_params.variational.q_mu
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history['elbo'])
    ax1.set_title('ELBO Training History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO')
    ax1.grid(True)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], 'k--', alpha=0.5, label='True Trajectory')
    ax2.plot(smoothed_states[:, 0], smoothed_states[:, 1], smoothed_states[:, 2], 'b-', label='Smoothed Trajectory')
    ax2.set_title('Lorenz Attractor Estimation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    return final_params, history


# ============================================================================
# Main Function
# ============================================================================

def run_all_examples():
    """Runs all refactored GPSSM examples."""
    print("🚀 [Refactored] GPSSM Framework Complete Examples")
    print("=" * 80)
    
    try:
        pendulum_params, pendulum_history = example_pendulum_gpssm()
        lorenz_params, lorenz_history = example_lorenz_gpssm()
        
        print("\n✅ All refactored examples completed successfully!")
        return {
            'pendulum': (pendulum_params, pendulum_history),
            'lorenz': (lorenz_params, lorenz_history)
        }
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # To avoid issues with matplotlib in some environments
    # jax.config.update("jax_platform_name", "cpu")
    run_all_examples()