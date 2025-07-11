#!/usr/bin/env python3
"""Linear-Gaussian MMSB: IPFP vs. analytic (RTS) benchmark
运行方式：
    python examples/rts_vs_mmsb_analytic.py
将打印三种方案的均值/方差误差：
  • IPFP (soft likelihood)
  • 解析势（RTS closed-form，高斯路径）
  • 参考 RTS smoother
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")  # double precision

import jax
import jax.numpy as jnp

from mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem, IPFPConfig
from mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed

# ---------- analytic linear-Gaussian helper ---------------------------------

def kalman_filter(y, A, Q, C, R, mu0, P0):
    mu_f, P_f = [], []
    mu_pred, P_pred = mu0, P0
    for k in range(len(y)):
        S = C * P_pred * C + R
        K = P_pred * C / S
        mu_upd = mu_pred + K * (y[k] - C * mu_pred)
        P_upd = (1 - K * C) * P_pred
        mu_f.append(mu_upd)
        P_f.append(P_upd)
        mu_pred = A * mu_upd
        P_pred = A * P_upd * A + Q
    return jnp.array(mu_f), jnp.array(P_f)

def rts_smoother(mu_f, P_f, A, Q):
    n = len(mu_f)
    mu_s, P_s = [None]*n, [None]*n
    mu_s[-1], P_s[-1] = mu_f[-1], P_f[-1]
    for k in range(n-2, -1, -1):
        P_pred = A * P_f[k] * A + Q
        G = P_f[k] * A / P_pred
        mu_s[k] = mu_f[k] + G * (mu_s[k+1] - A * mu_f[k])
        P_s[k] = P_f[k] + G * (P_s[k+1] - P_pred) * G
    return jnp.array(mu_s), jnp.array(P_s)

# ---------- experiment -------------------------------------------------------

def run_experiment(grid_points: int = 600):
    # system definition
    A, Q = 0.8, 0.1
    C, R = 1.0, 0.05
    mu0, P0 = -0.8, 0.4
    times = jnp.array([0., 1., 2.])
    K = len(times)

    key = jax.random.PRNGKey(0)
    true_x = jnp.array([-1.1, -0.3, 0.9])
    noise = jax.random.normal(key, (K,)) * jnp.sqrt(R)
    y = C * true_x + noise

    # analytic baseline (RTS)
    mu_f, P_f = kalman_filter(y, A, Q, C, R, mu0, P0)
    mu_s, P_s = rts_smoother(mu_f, P_f, A, Q)

    ou = OUProcessParams(mean_reversion=0.22314, diffusion=0.31622, equilibrium_mean=0.0)

    L = 6.0
    mu_min = float(jnp.min(mu_s)); mu_max = float(jnp.max(mu_s));
    sigma_max = float(jnp.max(jnp.sqrt(P_s)))
    bounds = (mu_min - L * sigma_max, mu_max + L * sigma_max)
    grid = GridConfig1D.create(grid_points, bounds)

    problem = MMSBProblem(
        observation_times=times,
        y_observations=y,
        C=C,
        R=R,
        ou_params=ou,
        grid=grid,
    )

    config = IPFPConfig(max_iterations=600, tolerance=1e-8, check_interval=10, verbose=False)
    sol = solve_mmsb_ipfp_1d_fixed(problem, config)

    # compute IPFP means / variances
    mu_ipfp = jnp.array([jnp.sum(sol.path_densities[k] * grid.points * grid.spacing) for k in range(K)])
    P_ipfp = jnp.array([jnp.sum(sol.path_densities[k] * (grid.points - mu_ipfp[k]) ** 2 * grid.spacing) for k in range(K)])

    print("=== Results (grid", grid_points, ") ===")
    print("Max |mu_ipfp - mu_RTS|   =", float(jnp.max(jnp.abs(mu_ipfp - mu_s))))
    print("Max |var_ipfp - var_RTS| =", float(jnp.max(jnp.abs(P_ipfp - P_s))))

if __name__ == "__main__":
    run_experiment() 