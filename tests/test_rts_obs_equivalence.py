#!/usr/bin/env python3
"""
RTS equivalence test (observation‐driven MMSB)
使用观测似然势验证MMSB与RTS平滑器的等价性
"""
import jax
import jax.numpy as jnp
import pytest

from mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem, IPFPConfig
from mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed, jax_trapz

# Helpers

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

@pytest.mark.parametrize("grid_points", [150])
@pytest.mark.parametrize(
    "system_params",
    [
        {"A":0.8, "Q":0.1, "theta":0.22314, "sigma":0.31622},
    ]
)
def test_rts_mmsb_observation_equivalence(grid_points, system_params):
    A,Q = system_params['A'], system_params['Q']
    C,R = 1.0, 0.05
    mu0,P0 = -0.8, 0.4
    K = 3

    key = jax.random.PRNGKey(0)
    true_x = jnp.array([-1.1, -0.3, 0.9])
    noise = jax.random.normal(key,(K,))*jnp.sqrt(R)
    y = C*true_x + noise

    mu_f,P_f = kalman_filter(y,A,Q,C,R,mu0,P0)
    mu_s,P_s = rts_smoother(mu_f,P_f,A,Q)

    ou = OUProcessParams(mean_reversion=system_params['theta'], diffusion=system_params['sigma'], equilibrium_mean=0.0)

    L=6.0
    mu_min = float(jnp.min(mu_s)); mu_max=float(jnp.max(mu_s)); sigma_max=float(jnp.max(jnp.sqrt(P_s)))
    bounds=(mu_min - L*sigma_max, mu_max + L*sigma_max)
    grid = GridConfig1D.create(grid_points,bounds)

    times = jnp.array([0.,1.,2.])
    problem = MMSBProblem(
        observation_times=times,
        y_observations=y,
        C=C,
        R=R,
        ou_params=ou,
        grid=grid,
    )

    config = IPFPConfig(max_iterations=300, tolerance=5e-7, check_interval=10, verbose=False)
    sol = solve_mmsb_ipfp_1d_fixed(problem, config)

    # means
    mu_mmsb = jnp.array([jnp.sum(sol.path_densities[k]*grid.points*grid.spacing) for k in range(K)])
    P_mmsb = jnp.array([jnp.sum(sol.path_densities[k]*(grid.points-mu_mmsb[k])**2 * grid.spacing) for k in range(K)])

    assert jnp.max(jnp.abs(mu_mmsb - mu_s)) < 7e-2
    assert jnp.max(jnp.abs(P_mmsb - P_s)) < 7e-2 