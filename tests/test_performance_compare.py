# tests/test_performance_compare.py
# -*- coding: utf-8 -*-
"""
快速性能对比测试 / Quick performance comparison test

目的：确保 PendulumMMSBSolver、PendulumEKFSmoother、PendulumUKFSmoother
均能在小规模示例上运行且返回合理输出。
只检查接口与数值稳定性，不做严格精度断言。
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.experiments.large_angle_pendulum.pendulum_mmsb_solver import (
    PendulumMMSBSolver, PendulumMMSBConfig
)
from src.baselines.ekf.ekf_smoother import PendulumEKFSmoother
from src.baselines.ukf.ukf_smoother import PendulumUKFSmoother
from src.experiments.large_angle_pendulum.data_generator import PendulumTrajectory, ObservationConfig


def _dummy_trajectory(T: int = 5):
    """生成极小样本轨迹用于单元测试 / Generate tiny trajectory for unit test"""
    key = jax.random.PRNGKey(0)
    dt = 0.1
    times = jnp.linspace(0.0, (T - 1) * dt, T)
    theta = 0.1 * jax.random.normal(key, (T,))
    obs_cfg = ObservationConfig(obs_times=times, obs_noise_std=0.05, sparse_strategy="test")
    from src.experiments.large_angle_pendulum.data_generator import PendulumParams
    traj = PendulumTrajectory(
        times=times,
        states=jnp.zeros((T, 2)),
        observations=theta,
        obs_times=times,
        true_obs_values=theta,
        params=PendulumParams(),
        obs_config=obs_cfg,
    )
    return traj


def test_three_methods_run():
    """确保三种方法在5步数据上均可运行且输出形状一致"""
    traj = _dummy_trajectory(5)

    # MMSB-VI（小网格加速）
    mmsb_cfg = PendulumMMSBConfig(theta_grid_points=32, omega_grid_points=16, ipfp_max_iterations=50)
    solver = PendulumMMSBSolver(mmsb_cfg)
    mmsb_res = solver.solve(traj, verbose=False)

    assert len(mmsb_res.density_estimates) == 5

    # EKF
    dt = float(traj.times[1] - traj.times[0])
    ekf = PendulumEKFSmoother(dt=dt, g=9.81, L=1.0, gamma=0.2, sigma=0.3)
    ekf_res = ekf.smooth(traj.observations, initial_mean=None, initial_cov=None)
    est_ekf = ekf.extract_estimates(ekf_res)
    assert est_ekf["theta_mean"].shape[0] == 5

    # UKF
    ukf = PendulumUKFSmoother(dt=dt, g=9.81, L=1.0, gamma=0.2, sigma=0.3)
    ukf_res = ukf.smooth(traj.observations, initial_mean=None, initial_cov=None)
    est_ukf = ukf.extract_estimates(ukf_res)
    assert est_ukf["theta_mean"].shape[0] == 5

    # 简单一致性检查：三方法第一时刻角度均值应相近（差<0.2rad）
    theta0 = traj.observations[0]
    assert abs(est_ekf["theta_mean"][0] - theta0) < 0.2
    assert abs(est_ukf["theta_mean"][0] - theta0) < 0.2 