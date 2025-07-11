import math
import pytest
import jax
import jax.numpy as jnp

from mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem
from mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed, jax_trapz

# -------------------- Helpers --------------------

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


@pytest.mark.parametrize("grid_points", [200])
@pytest.mark.parametrize(
    "system_params",
    [
        # Note: A, Q are derived from mean_reversion and diffusion for consistency
        {"mean_reversion": 0.22314, "diffusion": 0.31622, "A": 0.8, "Q": 0.1},
        {"mean_reversion": 0.5, "diffusion": 0.4, "A": 0.6065, "Q": 0.095}
    ]
)
def test_rts_ssm_equivalence(grid_points, system_params):
    # Model parameters
    A = system_params['A']
    Q = system_params['Q']
    C, R = 1.0, 0.05
    mu0, P0 = -1.0, 0.3
    K = 3

    # Synthetic data
    key = jax.random.PRNGKey(0)
    true_x = jnp.array([-1.0, -0.2, 0.8])
    noise = jax.random.normal(key, (K,)) * jnp.sqrt(R)
    y = C * true_x + noise

    # Filter & smooth
    mu_f, P_f = kalman_filter(y, A, Q, C, R, mu0, P0)
    mu_s, P_s = rts_smoother(mu_f, P_f, A, Q)

    # OU params corresponding to A, Q
    ou = OUProcessParams(
        mean_reversion=system_params['mean_reversion'],
        diffusion=system_params['diffusion'],
        equilibrium_mean=0.0
    )

    # Build MMSB problem
    L = 6.0
    mu_min = float(jnp.min(mu_s))
    mu_max = float(jnp.max(mu_s))
    sigma_max = float(jnp.max(jnp.sqrt(P_s)))
    bounds = (mu_min - L * sigma_max, mu_max + L * sigma_max)
    grid = GridConfig1D.create(grid_points, bounds)
    obs_densities = []
    h = grid.spacing
    for k in range(K):
        d = gauss(grid.points, mu_s[k], P_s[k])
        # Use consistent trapezoidal normalization
        mass = jax_trapz(d, dx=h)
        d = d / mass
        obs_densities.append(d)
    times = jnp.array([0., 1., 2.])
    problem = MMSBProblem(
        observation_times=times,
        observed_marginals=obs_densities,
        ou_params=ou,
        grid=grid,
    )

    sol = solve_mmsb_ipfp_1d_fixed(problem)

    # Compute MMSB means & variances
    mu_mmsb = jnp.array([jnp.sum(sol.path_densities[k] * grid.points * grid.spacing) for k in range(K)])
    P_mmsb = jnp.array([
        jnp.sum(sol.path_densities[k] * (grid.points - mu_mmsb[k]) ** 2 * grid.spacing) for k in range(K)
    ])

    # Assertions
    assert jnp.max(jnp.abs(mu_s - mu_mmsb)) < 5e-3
    assert jnp.max(jnp.abs(P_s - P_mmsb)) < 5e-3
    assert sol.final_error < 1e-6 