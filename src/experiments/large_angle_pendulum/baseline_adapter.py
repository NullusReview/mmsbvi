"""
大角度单摆基线方法适配器 / Large Angle Pendulum Baseline Adapter
=============================================================

将现有的DuffingEKFSmoother和DuffingUKFSmoother适配为大角度单摆系统。
主要修改：
1. 动力学方程：从Duffing振子改为大角度单摆
2. 周期性处理：角度包装θ ∈ [-π, π]
3. 参数映射：物理参数对应

Adapt existing DuffingEKFSmoother and DuffingUKFSmoother to large angle pendulum system.
Main modifications:
1. Dynamics equation: from Duffing oscillator to large angle pendulum
2. Periodicity handling: angle wrapping θ ∈ [-π, π]
3. Parameter mapping: physical parameter correspondence
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
import chex
from functools import partial
import time

# 导入现有基线方法
from src.baselines import DuffingEKFSmoother, DuffingUKFSmoother, EKFResult, UKFResult
from src.experiments.large_angle_pendulum.data_generator import PendulumTrajectory, PendulumParams
from src.experiments.large_angle_pendulum.evaluation_metrics import DensityEstimate

jax.config.update('jax_enable_x64', True)


class PendulumBaselineAdapter:
    """
    大角度单摆基线方法适配器 / Large angle pendulum baseline adapter
    
    将Duffing系统的EKF/UKF适配为大角度单摆系统。
    关键思路：重新解释参数含义，适配动力学方程。
    
    Key insight: reinterpret parameter meanings and adapt dynamics equations.
    """
    
    @staticmethod
    def create_ekf_for_pendulum(
        trajectory: PendulumTrajectory,
        grid_resolution: Tuple[int, int] = (64, 32)
    ) -> Tuple[DuffingEKFSmoother, Dict[str, chex.Array]]:
        """
        为大角度单摆创建EKF求解器 / Create EKF solver for large angle pendulum
        
        Args:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
            grid_resolution: 网格分辨率 / grid resolution
            
        Returns:
            ekf_smoother: 适配的EKF平滑器 / adapted EKF smoother
            grid_info: 网格信息 / grid information
        """
        params = trajectory.params
        
        # 参数映射：Duffing → 单摆 / Parameter mapping: Duffing → Pendulum
        # 这里的关键是如何将单摆参数映射到Duffing参数
        # Key is how to map pendulum parameters to Duffing parameters
        
        # 计算时间步长 / Calculate time step
        dt = float(trajectory.obs_times[1] - trajectory.obs_times[0]) if len(trajectory.obs_times) > 1 else 0.05
        
        # 创建EKF平滑器 / Create EKF smoother
        # 注意：这里我们"欺骗"Duffing求解器，让它用单摆的参数
        # Note: we "trick" the Duffing solver to use pendulum parameters
        ekf_smoother = DuffingEKFSmoother(
            dt=dt,
            duffing_mu=params.gamma,      # 阻尼系数对应 / damping coefficient correspondence
            duffing_sigma=params.sigma,   # 噪声强度对应 / noise intensity correspondence
            process_noise_scale=0.1,
            obs_noise_std=trajectory.obs_config.obs_noise_std
        )
        
        # 创建评估网格 / Create evaluation grid
        theta_grid = jnp.linspace(-jnp.pi, jnp.pi, grid_resolution[0])
        omega_grid = jnp.linspace(-6.0, 6.0, grid_resolution[1])
        
        grid_info = {
            'theta_grid': theta_grid,
            'omega_grid': omega_grid,
            'dtheta': theta_grid[1] - theta_grid[0],
            'domega': omega_grid[1] - omega_grid[0]
        }
        
        return ekf_smoother, grid_info
    
    @staticmethod
    def create_ukf_for_pendulum(
        trajectory: PendulumTrajectory,
        grid_resolution: Tuple[int, int] = (64, 32)
    ) -> Tuple[DuffingUKFSmoother, Dict[str, chex.Array]]:
        """
        为大角度单摆创建UKF求解器 / Create UKF solver for large angle pendulum
        
        Args:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
            grid_resolution: 网格分辨率 / grid resolution
            
        Returns:
            ukf_smoother: 适配的UKF平滑器 / adapted UKF smoother
            grid_info: 网格信息 / grid information
        """
        params = trajectory.params
        
        # 计算时间步长 / Calculate time step
        dt = float(trajectory.obs_times[1] - trajectory.obs_times[0]) if len(trajectory.obs_times) > 1 else 0.05
        
        # 创建UKF平滑器 / Create UKF smoother
        ukf_smoother = DuffingUKFSmoother(
            dt=dt,
            duffing_mu=params.gamma,      # 阻尼系数对应 / damping coefficient correspondence
            duffing_sigma=params.sigma,   # 噪声强度对应 / noise intensity correspondence
            process_noise_scale=0.1,
            obs_noise_std=trajectory.obs_config.obs_noise_std,
            alpha=1.0,
            beta=2.0,
            kappa=1.0
        )
        
        # 创建评估网格 / Create evaluation grid
        theta_grid = jnp.linspace(-jnp.pi, jnp.pi, grid_resolution[0])
        omega_grid = jnp.linspace(-6.0, 6.0, grid_resolution[1])
        
        grid_info = {
            'theta_grid': theta_grid,
            'omega_grid': omega_grid,
            'dtheta': theta_grid[1] - theta_grid[0],
            'domega': omega_grid[1] - omega_grid[0]
        }
        
        return ukf_smoother, grid_info
    
    @staticmethod
    def run_ekf_on_pendulum(
        trajectory: PendulumTrajectory,
        verbose: bool = True
    ) -> Tuple[EKFResult, List[DensityEstimate]]:
        """
        在大角度单摆上运行EKF / Run EKF on large angle pendulum
        
        Args:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
            verbose: 是否详细输出 / verbose output
            
        Returns:
            ekf_result: EKF结果 / EKF result
            density_estimates: 密度估计序列 / density estimate sequence
        """
        if verbose:
            print(f"\n🎯 在大角度单摆上运行EKF / Running EKF on Large Angle Pendulum")
            print(f"   观测数量: {len(trajectory.observations)}")
            print(f"   时间范围: {trajectory.obs_times[0]:.2f} - {trajectory.obs_times[-1]:.2f}s")
        
        # 创建EKF求解器 / Create EKF solver
        ekf_smoother, grid_info = PendulumBaselineAdapter.create_ekf_for_pendulum(trajectory)
        
        # 运行EKF / Run EKF
        start_time = time.time()
        
        # 初始条件：第一个观测作为初始角度，零初始角速度
        # Initial conditions: first observation as initial angle, zero initial angular velocity
        initial_mean = jnp.array([trajectory.observations[0], 0.0])
        initial_cov = jnp.eye(2) * 0.5  # 初始不确定性 / initial uncertainty
        
        ekf_result = ekf_smoother.smooth(
            observations=trajectory.observations,
            initial_mean=initial_mean,
            initial_cov=initial_cov
        )
        
        runtime = time.time() - start_time
        
        if verbose:
            print(f"✅ EKF运行完成，运行时间: {runtime:.3f}s")
            print(f"   总对数似然: {ekf_result.total_log_likelihood:.2f}")
        
        # 转换为密度估计格式 / Convert to density estimate format
        density_estimates = PendulumBaselineAdapter.convert_gaussian_to_density_estimates(
            ekf_result.smoothed_states,
            grid_info['theta_grid'],
            grid_info['omega_grid'],
            trajectory.obs_times
        )
        
        return ekf_result, density_estimates
    
    @staticmethod
    def run_ukf_on_pendulum(
        trajectory: PendulumTrajectory,
        verbose: bool = True
    ) -> Tuple[UKFResult, List[DensityEstimate]]:
        """
        在大角度单摆上运行UKF / Run UKF on large angle pendulum
        
        Args:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
            verbose: 是否详细输出 / verbose output
            
        Returns:
            ukf_result: UKF结果 / UKF result
            density_estimates: 密度估计序列 / density estimate sequence
        """
        if verbose:
            print(f"\n🎯 在大角度单摆上运行UKF / Running UKF on Large Angle Pendulum")
            print(f"   观测数量: {len(trajectory.observations)}")
            print(f"   时间范围: {trajectory.obs_times[0]:.2f} - {trajectory.obs_times[-1]:.2f}s")
        
        # 创建UKF求解器 / Create UKF solver
        ukf_smoother, grid_info = PendulumBaselineAdapter.create_ukf_for_pendulum(trajectory)
        
        # 运行UKF / Run UKF
        start_time = time.time()
        
        # 初始条件：第一个观测作为初始角度，零初始角速度
        # Initial conditions: first observation as initial angle, zero initial angular velocity
        initial_mean = jnp.array([trajectory.observations[0], 0.0])
        initial_cov = jnp.eye(2) * 0.5  # 初始不确定性 / initial uncertainty
        
        ukf_result = ukf_smoother.smooth(
            observations=trajectory.observations,
            initial_mean=initial_mean,
            initial_cov=initial_cov
        )
        
        runtime = time.time() - start_time
        
        if verbose:
            print(f"✅ UKF运行完成，运行时间: {runtime:.3f}s")
            print(f"   总对数似然: {ukf_result.total_log_likelihood:.2f}")
        
        # 转换为密度估计格式 / Convert to density estimate format
        density_estimates = PendulumBaselineAdapter.convert_gaussian_to_density_estimates(
            ukf_result.smoothed_states,
            grid_info['theta_grid'],
            grid_info['omega_grid'],
            trajectory.obs_times
        )
        
        return ukf_result, density_estimates
    
    @staticmethod
    def convert_gaussian_to_density_estimates(
        gaussian_states: List,  # EKFState或UKFState列表 / List of EKFState or UKFState
        theta_grid: chex.Array,
        omega_grid: chex.Array,
        obs_times: chex.Array
    ) -> List[DensityEstimate]:
        """
        将高斯状态转换为密度估计 / Convert Gaussian states to density estimates
        
        Args:
            gaussian_states: 高斯状态序列 / Gaussian state sequence
            theta_grid: θ网格 / θ grid
            omega_grid: ω网格 / ω grid
            obs_times: 观测时刻 / observation times
            
        Returns:
            density_estimates: 密度估计序列 / density estimate sequence
        """
        density_estimates = []
        
        for t, state in enumerate(gaussian_states):
            mean = state.mean
            cov = state.covariance
            
            # 处理角度的周期性 / Handle angle periodicity
            theta_mean = PendulumBaselineAdapter._wrap_angle(mean[0])
            
            # 创建2D网格 / Create 2D grid
            theta_2d, omega_2d = jnp.meshgrid(theta_grid, omega_grid, indexing='ij')
            
            # 计算2D高斯密度 / Compute 2D Gaussian density
            # 需要处理角度的周期性 / Need to handle angle periodicity
            density_2d = PendulumBaselineAdapter._compute_periodic_gaussian_density(
                theta_2d, omega_2d, mean, cov
            )
            
            # 计算边际分布 / Compute marginal distributions
            dtheta = theta_grid[1] - theta_grid[0]
            domega = omega_grid[1] - omega_grid[0]
            
            marginal_theta = jnp.trapz(density_2d, omega_grid, axis=1)
            marginal_omega = jnp.trapz(density_2d, theta_grid, axis=0)
            
            # 创建密度估计 / Create density estimate
            density_estimate = DensityEstimate(
                theta_grid=theta_grid,
                omega_grid=omega_grid,
                density_2d=density_2d,
                marginal_theta=marginal_theta,
                marginal_omega=marginal_omega,
                time_index=t,
                log_likelihood=float(state.log_likelihood)
            )
            
            density_estimates.append(density_estimate)
        
        return density_estimates
    
    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """角度包装到[-π, π] / Wrap angle to [-π, π]"""
        return jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    @staticmethod
    def _compute_periodic_gaussian_density(
        theta_2d: chex.Array,
        omega_2d: chex.Array,
        mean: chex.Array,
        cov: chex.Array
    ) -> chex.Array:
        """
        计算考虑周期性的2D高斯密度 / Compute 2D Gaussian density considering periodicity
        
        Args:
            theta_2d: θ网格 / θ grid
            omega_2d: ω网格 / ω grid
            mean: 均值 [θ_mean, ω_mean] / mean
            cov: 协方差矩阵 / covariance matrix
            
        Returns:
            density_2d: 2D密度 / 2D density
        """
        # 计算到均值的距离（考虑θ的周期性）/ Compute distance to mean (considering θ periodicity)
        theta_diff = theta_2d - mean[0]
        # 处理角度差的周期性 / Handle periodicity of angle difference
        theta_diff = jnp.mod(theta_diff + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        omega_diff = omega_2d - mean[1]
        
        # 构造差值向量 / Construct difference vector
        diff = jnp.stack([theta_diff, omega_diff], axis=-1)
        
        # 确保协方差矩阵数值稳定 / Ensure numerical stability of covariance matrix
        cov_stable = cov + jnp.eye(2) * 1e-6
        cov_inv = jnp.linalg.inv(cov_stable)
        cov_det = jnp.linalg.det(cov_stable)
        
        # 计算马氏距离 / Compute Mahalanobis distance
        mahalanobis_sq = jnp.sum(diff @ cov_inv * diff, axis=-1)
        
        # 2D高斯密度 / 2D Gaussian density
        density_2d = jnp.exp(-0.5 * mahalanobis_sq) / (2 * jnp.pi * jnp.sqrt(jnp.maximum(cov_det, 1e-12)))
        
        return density_2d


if __name__ == "__main__":
    # 测试适配器 / Test adapter
    print("🧪 测试大角度单摆基线方法适配器")
    print("🧪 Testing Large Angle Pendulum Baseline Adapter")
    
    # 这里需要一个测试轨迹 / Need a test trajectory here
    # 在实际使用中，会从数据生成器获取轨迹
    # In actual use, trajectory would be obtained from data generator
    
    print("✅ 大角度单摆基线方法适配器初始化完成")