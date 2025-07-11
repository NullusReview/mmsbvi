import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mmsbvi.core.types import GridConfig1D, OUProcessParams, MMSBProblem
from mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed
from mmsbvi.utils.logger import get_logger
import os

logger = get_logger(__name__)

jax.config.update("jax_enable_x64", True)

# -------------------- SSM parameters --------------------
A = 0.8  # state transition
Q = 0.1  # process noise var
C = 1.0  # observe matrix
R = 0.05  # obs noise var
K_STEPS = 3

# Observation times k=0,1,2  mapped to t = 0,1,2 in discrete units
obs_times = jnp.array([0., 1., 2.])

# --------------------------------------------------------
# Helper: 1D Gaussian density

def gauss(x, mu, sigma2):
    return (1.0 / jnp.sqrt(2 * jnp.pi * sigma2)) * jnp.exp(-0.5 * (x - mu) ** 2 / sigma2)

# Simple Kalman filter (scalar)

def kalman_filter(y, A, Q, C, R, mu0, P0):
    n = len(y)
    mu_f = []
    P_f = []
    mu_pred = mu0
    P_pred = P0
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

# RTS smoother (scalar)

def rts_smoother(mu_f, P_f, A, Q):
    n = len(mu_f)
    mu_s = [None] * n
    P_s = [None] * n
    mu_s[-1] = mu_f[-1]
    P_s[-1] = P_f[-1]
    for k in range(n - 2, -1, -1):
        P_pred = A * P_f[k] * A + Q
        G = P_f[k] * A / P_pred
        mu_s[k] = mu_f[k] + G * (mu_s[k + 1] - A * mu_f[k])
        P_s[k] = P_f[k] + G * (P_s[k + 1] - P_pred) * G
    return jnp.array(mu_s), jnp.array(P_s)

# -------------------- generate synthetic data --------------------
mu0, P0 = -1.0, 0.3
true_x = jnp.array([-1.0, -0.2, 0.8])
key = jax.random.PRNGKey(0)
noise = jax.random.normal(key, (K_STEPS,)) * jnp.sqrt(R)
y = C * true_x + noise

# filtering & smoothing
mu_f, P_f = kalman_filter(y, A, Q, C, R, mu0, P0)
mu_s, P_s = rts_smoother(mu_f, P_f, A, Q)

# -------------------- map to OU kernel --------------------
# Choose dt=1 between discrete steps. theta from A
import math

dt = 1.0
theta = -math.log(A) / dt
sigma = math.sqrt(2 * theta * Q / (1 - A ** 2))

ou_params = OUProcessParams(mean_reversion=theta, diffusion=sigma, equilibrium_mean=0.0)

# -------------------- build MMSB problem --------------------
# adaptive grid bounds
L = 6.0  # cover ±L sigma
mu_min = float(jnp.min(mu_f))
mu_max = float(jnp.max(mu_f))
sigma_max = float(jnp.max(jnp.sqrt(P_f)))
bounds = (mu_min - L * sigma_max, mu_max + L * sigma_max)
GRID = GridConfig1D.create(401, bounds)

# Create properly normalized observed densities using trapezoidal integration
# 使用梯形积分创建正确归一化的观测密度
obs_densities = []
from mmsbvi.algorithms.ipfp_1d import jax_trapz
h = GRID.spacing
for k in range(K_STEPS):
    d = gauss(GRID.points, mu_f[k], P_f[k])
    # Normalize using trapezoidal integration
    mass = jax_trapz(d, dx=h)
    d = d / mass
    obs_densities.append(d)
problem = MMSBProblem(
    observation_times=obs_times,
    observed_marginals=obs_densities,
    ou_params=ou_params,
    grid=GRID,
)

solution = solve_mmsb_ipfp_1d_fixed(problem)

# -------------------- metrics --------------------
mu_mmsb = jnp.array([jnp.sum(solution.path_densities[k] * GRID.points * GRID.spacing) for k in range(K_STEPS)])
P_mmsb = jnp.array([
    jnp.sum(solution.path_densities[k] * (GRID.points - mu_mmsb[k]) ** 2 * GRID.spacing) for k in range(K_STEPS)
])

mean_err = jnp.max(jnp.abs(mu_s - mu_mmsb))
cov_err = jnp.max(jnp.abs(P_s - P_mmsb))

# KL divergence (approx)

def kl_gauss_disc(p_density, mu, var):
    q = gauss(GRID.points, mu, var)
    ratio = jnp.maximum(p_density / q, 1e-16)
    return jnp.sum(p_density * jnp.log(ratio) * GRID.spacing)

kl_vals = jnp.array([kl_gauss_disc(solution.path_densities[k], mu_s[k], P_s[k]) for k in range(K_STEPS)])
max_kl = jnp.max(kl_vals)

logger.info(
    f"RTS-MMSB equivalence test\n"
    f"max mean error      : {mean_err:.2e}\n"
    f"max covariance error: {cov_err:.2e}\n"
    f"max KL divergence   : {max_kl:.2e}\n"
    f"IPFP iters          : {solution.n_iterations}"
)

# ensure results directory
os.makedirs("results", exist_ok=True)

# -------------------- plot --------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for k in range(K_STEPS):
    ax[k].plot(GRID.points, solution.path_densities[k], label="MMSB")
    ax[k].plot(GRID.points, gauss(GRID.points, mu_s[k], P_s[k]), ls="--", label="RTS")
    ax[k].plot(GRID.points, obs_densities[k], ls=":", label="Filter post")
    ax[k].set_title(f"k={k}")
    ax[k].legend(fontsize=6)
plt.tight_layout()
plt.savefig("results/rts_equivalence_density.png", dpi=150)
plt.close()

plt.semilogy(solution.convergence_history)
plt.xlabel("iteration x check_interval")
plt.ylabel("potential change")
plt.tight_layout()
plt.savefig("results/rts_ipfp_convergence.png", dpi=150)
plt.close() 