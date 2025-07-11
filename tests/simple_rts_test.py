#!/usr/bin/env python3
"""
Simple RTS equivalence test with direct output
简单的RTS等价性测试，直接输出
"""

import jax
import jax.numpy as jnp
import math
from src.mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem
from src.mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed, jax_trapz, IPFPConfig

jax.config.update('jax_enable_x64', True)

def gauss(x, mu, sigma2):
    return (1.0 / jnp.sqrt(2 * jnp.pi * sigma2)) * jnp.exp(-0.5 * (x - mu) ** 2 / sigma2)

def kalman_filter(y, A, Q, C, R, mu0, P0):
    n = len(y)
    mu_f, P_f = [], []
    mu_pred, P_pred = mu0, P0
    for k in range(n):
        # update
        S = C * P_pred * C + R
        K_gain = P_pred * C / S
        mu_upd = mu_pred + K_gain * (y[k] - C * mu_pred)
        P_upd = (1 - K_gain * C) * P_pred
        mu_f.append(mu_upd)
        P_f.append(P_upd)
        # predict
        mu_pred = A * mu_upd
        P_pred = A * P_upd * A + Q
    return jnp.array(mu_f), jnp.array(P_f)

def rts_smoother(mu_f, P_f, A, Q):
    n = len(mu_f)
    mu_s = [None] * n
    P_s = [None] * n
    mu_s[-1], P_s[-1] = mu_f[-1], P_f[-1]
    for k in range(n - 2, -1, -1):
        P_pred = A * P_f[k] * A + Q
        G = P_f[k] * A / P_pred
        mu_s[k] = mu_f[k] + G * (mu_s[k + 1] - A * mu_f[k])
        P_s[k] = P_f[k] + G * (P_s[k + 1] - P_pred) * G
    return jnp.array(mu_s), jnp.array(P_s)

def main():
    print("=== Simple RTS-MMSB Equivalence Test ===")
    
    # System parameters
    A, Q, C, R = 0.8, 0.1, 1.0, 0.05
    mu0, P0 = -1.0, 0.3
    
    # Synthetic data
    true_x = jnp.array([-1.0, -0.2, 0.8])
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (3,)) * jnp.sqrt(R)
    y = C * true_x + noise
    
    print(f"Observations: {y}")
    
    # Kalman filter and RTS smoother
    mu_f, P_f = kalman_filter(y, A, Q, C, R, mu0, P0)
    mu_s, P_s = rts_smoother(mu_f, P_f, A, Q)
    
    print(f"Filter means: {mu_f}")
    print(f"Filter vars: {P_f}")
    print(f"Smoother means: {mu_s}")
    print(f"Smoother vars: {P_s}")
    
    # OU parameters
    dt = 1.0
    theta = -math.log(A) / dt
    sigma = math.sqrt(2 * theta * Q / (1 - A**2))
    ou_params = OUProcessParams(
        mean_reversion=theta,
        diffusion=sigma,
        equilibrium_mean=0.0
    )
    
    print(f"OU params: theta={theta:.6f}, sigma={sigma:.6f}")
    
    # Grid and problem setup
    L = 6.0
    mu_min = float(jnp.min(mu_f))
    mu_max = float(jnp.max(mu_f))
    sigma_max = float(jnp.max(jnp.sqrt(P_f)))
    bounds = (mu_min - L * sigma_max, mu_max + L * sigma_max)
    grid = GridConfig1D.create(200, bounds)  # Smaller grid for faster computation
    
    print(f"Grid bounds: {bounds}, spacing: {grid.spacing:.6f}")
    
    # Create observed densities (correctly normalized)
    obs_densities = []
    h = grid.spacing
    for k in range(3):
        d = gauss(grid.points, mu_f[k], P_f[k])
        mass = jax_trapz(d, dx=h)
        d = d / mass
        obs_densities.append(d)
        print(f"Observed density {k}: mass = {jax_trapz(d, dx=h):.6f}")
    
    problem = MMSBProblem(
        observation_times=jnp.array([0., 1., 2.]),
        observed_marginals=obs_densities,
        ou_params=ou_params,
        grid=grid,
    )
    
    # Solve MMSB
    config = IPFPConfig(max_iterations=200, tolerance=1e-7, verbose=False)
    solution = solve_mmsb_ipfp_1d_fixed(problem, config)
    
    print(f"MMSB converged: {solution.final_error:.2e} in {solution.n_iterations} iterations")
    
    # Compute MMSB statistics
    mu_mmsb = []
    P_mmsb = []
    for k in range(3):
        density = solution.path_densities[k]
        mass = jax_trapz(density, dx=h)
        mu_k = jax_trapz(density * grid.points, dx=h) / mass
        P_k = jax_trapz(density * (grid.points - mu_k)**2, dx=h) / mass
        mu_mmsb.append(mu_k)
        P_mmsb.append(P_k)
        print(f"MMSB density {k}: mass = {mass:.6f}, mean = {mu_k:.6f}, var = {P_k:.6f}")
    
    mu_mmsb = jnp.array(mu_mmsb)
    P_mmsb = jnp.array(P_mmsb)
    
    # Compare
    mean_error = jnp.max(jnp.abs(mu_s - mu_mmsb))
    var_error = jnp.max(jnp.abs(P_s - P_mmsb))
    
    print(f"\n=== Equivalence Results ===")
    print(f"RTS means: {mu_s}")
    print(f"MMSB means: {mu_mmsb}")
    print(f"Mean differences: {mu_s - mu_mmsb}")
    print(f"Max mean error: {mean_error:.2e}")
    
    print(f"RTS vars: {P_s}")
    print(f"MMSB vars: {P_mmsb}")  
    print(f"Var differences: {P_s - P_mmsb}")
    print(f"Max var error: {var_error:.2e}")
    
    # Success criteria
    mean_threshold = 1e-3  # Relaxed threshold
    var_threshold = 1e-3
    
    success = mean_error < mean_threshold and var_error < var_threshold
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Mean error < {mean_threshold}, Var error < {var_threshold}")
    
    return success

if __name__ == "__main__":
    main()