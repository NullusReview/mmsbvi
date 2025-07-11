#!/usr/bin/env python3
"""
Test log-space marginal computation
测试对数空间边际计算
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from src.mmsbvi.algorithms.ipfp_1d import _compute_current_marginal, jax_trapz
from src.mmsbvi.solvers.gaussian_kernel_1d import compute_log_transition_kernel_1d_fixed
from src.mmsbvi.core.types import OUProcessParams

jax.config.update('jax_enable_x64', True)

def test_two_marginal_log_computation():
    """Test log-space computation for two-marginal case"""
    
    # Simple setup
    n = 20
    grid = jnp.linspace(-1.0, 1.0, n)
    h = grid[1] - grid[0]
    
    # Simple potentials
    phi_0 = jnp.zeros(n)
    phi_1 = jnp.zeros(n)
    potentials = [phi_0, phi_1]
    
    # OU parameters and log transition matrix
    ou_params = OUProcessParams(mean_reversion=1.0, diffusion=1.0, equilibrium_mean=0.0)
    dt = 1.0
    log_K = compute_log_transition_kernel_1d_fixed(grid, grid, dt, ou_params)
    log_transition_matrices = [log_K]
    
    print(f"Grid spacing: {h}")
    print(f"Log transition matrix shape: {log_K.shape}")
    
    # Test marginal computation
    for k in [0, 1]:
        log_marginal = _compute_current_marginal(k, potentials, log_transition_matrices, h)
        marginal = jnp.exp(log_marginal)
        
        print(f"\nMarginal {k}:")
        print(f"  Log marginal range: [{jnp.min(log_marginal):.3f}, {jnp.max(log_marginal):.3f}]")
        print(f"  Marginal range: [{jnp.min(marginal):.6f}, {jnp.max(marginal):.6f}]")
        print(f"  Mass (trapz): {jax_trapz(marginal, dx=h):.6f}")
        print(f"  Mass (sum*h): {jnp.sum(marginal)*h:.6f}")
        
        # Check if it's reasonable (should be roughly uniform for zero potentials)
        expected_uniform = 1.0 / (grid[-1] - grid[0])  # uniform density
        print(f"  Expected uniform density: {expected_uniform:.6f}")
        print(f"  Mean density: {jnp.mean(marginal):.6f}")

def compare_with_direct_computation():
    """Compare with direct coupling computation for K=2"""
    
    n = 15  # Smaller for direct computation
    grid = jnp.linspace(-1.0, 1.0, n)
    h = grid[1] - grid[0]
    
    # Simple potentials
    phi_0 = 0.1 * jnp.sin(grid)  # Add some variation
    phi_1 = 0.1 * jnp.cos(grid)
    potentials = [phi_0, phi_1]
    
    # OU parameters
    ou_params = OUProcessParams(mean_reversion=0.5, diffusion=1.0, equilibrium_mean=0.0)
    dt = 0.5
    log_K = compute_log_transition_kernel_1d_fixed(grid, grid, dt, ou_params)
    K = jnp.exp(log_K)  # Convert to linear space for direct computation
    
    print("\n=== Comparing Log-space vs Direct Computation ===")
    
    # Direct computation (for K=2 case)
    # coupling[i,j] = exp(phi_0[i] + phi_1[j]) * K[i,j]
    phi_sum = phi_0[:, None] + phi_1[None, :]  # Broadcasting
    coupling = jnp.exp(phi_sum) * K
    
    # Direct marginals
    marginal_0_direct = jnp.sum(coupling, axis=1) * h  # sum over j
    marginal_1_direct = jnp.sum(coupling, axis=0) * h  # sum over i
    
    # Log-space computation
    log_transition_matrices = [log_K]
    log_marginal_0 = _compute_current_marginal(0, potentials, log_transition_matrices, h)
    log_marginal_1 = _compute_current_marginal(1, potentials, log_transition_matrices, h)
    
    marginal_0_log = jnp.exp(log_marginal_0)
    marginal_1_log = jnp.exp(log_marginal_1)
    
    # Compare
    for k, (direct, log_space) in enumerate([(marginal_0_direct, marginal_0_log), 
                                             (marginal_1_direct, marginal_1_log)]):
        mass_direct = jax_trapz(direct, dx=h)
        mass_log = jax_trapz(log_space, dx=h)
        max_diff = jnp.max(jnp.abs(direct - log_space))
        
        print(f"\nMarginal {k}:")
        print(f"  Direct mass: {mass_direct:.6f}")
        print(f"  Log-space mass: {mass_log:.6f}")
        print(f"  Mass difference: {jnp.abs(mass_direct - mass_log):.2e}")
        print(f"  Max pointwise difference: {max_diff:.2e}")
        print(f"  Relative max difference: {max_diff / jnp.max(direct):.2e}")

if __name__ == "__main__":
    print("Testing log-space marginal computation...")
    test_two_marginal_log_computation()
    compare_with_direct_computation()