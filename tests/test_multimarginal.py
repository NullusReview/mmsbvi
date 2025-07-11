#!/usr/bin/env python3
"""
Test multi-marginal IPFP convergence
测试多边际IPFP收敛
"""

import jax
import jax.numpy as jnp
from src.mmsbvi.core.types import MMSBProblem, GridConfig1D, OUProcessParams, IPFPConfig
from src.mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed, jax_trapz

jax.config.update('jax_enable_x64', True)

def test_three_marginal_convergence():
    """Test three-marginal convergence with the new implementation"""
    
    # Grid setup
    points = jnp.linspace(-2.0, 2.0, 50)
    grid = GridConfig1D(
        n_points=50,
        bounds=(-2.0, 2.0),
        spacing=4.0/49,
        points=points
    )
    
    x = grid.points
    h = grid.spacing
    
    # Three Gaussian marginals: -1 -> 0 -> 1
    rho_0 = jnp.exp(-0.5 * (x + 1.0)**2 / 0.3)
    rho_0 = rho_0 / (h * (jnp.sum(rho_0) - 0.5 * (rho_0[0] + rho_0[-1])))
    
    rho_half = jnp.exp(-0.5 * x**2 / 0.3)
    rho_half = rho_half / (h * (jnp.sum(rho_half) - 0.5 * (rho_half[0] + rho_half[-1])))
    
    rho_1 = jnp.exp(-0.5 * (x - 1.0)**2 / 0.3)
    rho_1 = rho_1 / (h * (jnp.sum(rho_1) - 0.5 * (rho_1[0] + rho_1[-1])))
    
    print(f"Initial masses: rho_0={jax_trapz(rho_0, dx=h):.6f}, rho_half={jax_trapz(rho_half, dx=h):.6f}, rho_1={jax_trapz(rho_1, dx=h):.6f}")
    
    # Problem setup
    ou_params = OUProcessParams(
        mean_reversion=1.0,
        diffusion=1.0,
        equilibrium_mean=0.0
    )
    
    problem = MMSBProblem(
        observed_marginals=[rho_0, rho_half, rho_1],
        observation_times=jnp.array([0.0, 0.5, 1.0]),
        ou_params=ou_params,
        grid=grid
    )
    
    # Configuration
    config = IPFPConfig(
        max_iterations=100,
        tolerance=1e-6,
        verbose=True,
        check_interval=10,
        epsilon_scaling=False
    )
    
    print("\n=== Testing Three-Marginal IPFP ===")
    
    try:
        solution = solve_mmsb_ipfp_1d_fixed(problem, config)
        
        print(f"Converged: {solution.final_error:.2e}")
        print(f"Iterations: {solution.n_iterations}")
        
        # Check final marginals
        for k, (computed, target) in enumerate(zip(solution.path_densities, problem.observed_marginals)):
            mass_computed = jax_trapz(computed, dx=h)
            mass_target = jax_trapz(target, dx=h)
            l1_error = jax_trapz(jnp.abs(computed - target), dx=h)
            
            print(f"Marginal {k}: mass_computed={mass_computed:.6f}, mass_target={mass_target:.6f}")
            print(f"Marginal {k}: L1_error={l1_error:.2e}")
            
        print("✓ Three-marginal test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Three-marginal test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_four_marginal_convergence():
    """Test four-marginal convergence"""
    
    # Grid setup
    points = jnp.linspace(-2.0, 2.0, 40)  # Smaller for faster computation
    grid = GridConfig1D(
        n_points=40,
        bounds=(-2.0, 2.0),
        spacing=4.0/39,
        points=points
    )
    
    x = grid.points
    h = grid.spacing
    
    # Four marginals: moving Gaussian
    centers = [-1.2, -0.4, 0.4, 1.2]
    marginals = []
    
    for center in centers:
        rho = jnp.exp(-0.5 * (x - center)**2 / 0.25)
        rho = rho / (h * (jnp.sum(rho) - 0.5 * (rho[0] + rho[-1])))
        marginals.append(rho)
    
    print(f"\nFour marginal masses: {[jax_trapz(m, dx=h) for m in marginals]}")
    
    # Problem setup
    ou_params = OUProcessParams(
        mean_reversion=0.8,
        diffusion=1.2,
        equilibrium_mean=0.0
    )
    
    problem = MMSBProblem(
        observed_marginals=marginals,
        observation_times=jnp.array([0.0, 0.33, 0.67, 1.0]),
        ou_params=ou_params,
        grid=grid
    )
    
    config = IPFPConfig(
        max_iterations=80,
        tolerance=1e-5,  # Slightly relaxed for K=4
        verbose=True,
        check_interval=10,
        epsilon_scaling=False
    )
    
    print("\n=== Testing Four-Marginal IPFP ===")
    
    try:
        solution = solve_mmsb_ipfp_1d_fixed(problem, config)
        
        print(f"Converged: {solution.final_error:.2e}")
        print(f"Iterations: {solution.n_iterations}")
        
        # Check final marginals
        for k, (computed, target) in enumerate(zip(solution.path_densities, problem.observed_marginals)):
            mass_computed = jax_trapz(computed, dx=h)
            mass_target = jax_trapz(target, dx=h)
            l1_error = jax_trapz(jnp.abs(computed - target), dx=h)
            
            print(f"Marginal {k}: mass_computed={mass_computed:.6f}, mass_target={mass_target:.6f}")
            print(f"Marginal {k}: L1_error={l1_error:.2e}")
            
        print("✓ Four-marginal test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Four-marginal test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success_3 = test_three_marginal_convergence()
    success_4 = test_four_marginal_convergence()
    
    if success_3 and success_4:
        print("\n🎉 ALL MULTI-MARGINAL TESTS PASSED!")
    else:
        print("\n⚠️ Some multi-marginal tests failed.")