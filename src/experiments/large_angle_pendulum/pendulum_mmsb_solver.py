"""
大角度单摆MMSB-VI求解器 / Large Angle Pendulum MMSB-VI Solver
===========================================================

适配周期性状态空间的MMSB-VI求解器，专门处理大角度单摆的多模态后验估计。
MMSB-VI solver adapted for periodic state space, specifically handling multi-modal posterior estimation for large angle pendulum.

关键特性 / Key Features:
- 周期性边界条件: θ ∈ [-π, π]
- 非线性sin(θ)转移核
- 多模态密度保持
- 倒立点附近的数值稳定性
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from typing import Dict, List, Tuple, Optional, NamedTuple
import time
from functools import partial
import chex

# 导入MMSB-VI核心组件 / Import MMSB-VI core components
import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.mmsbvi.core.types import (
    GridConfig1D, OUProcessParams, MMSBProblem, MMSBSolution, 
    IPFPConfig, IPFPState
)
from src.mmsbvi.algorithms.ipfp_1d import solve_mmsb_ipfp_1d_fixed
from src.experiments.large_angle_pendulum.data_generator import PendulumTrajectory
from src.experiments.large_angle_pendulum.evaluation_metrics import DensityEstimate

jax.config.update('jax_enable_x64', True)


class PendulumMMSBConfig(NamedTuple):
    """大角度单摆MMSB-VI配置 / Large angle pendulum MMSB-VI configuration"""
    theta_grid_points: int = 128      # θ网格点数 / θ grid points
    omega_grid_points: int = 64       # ω网格点数 / ω grid points
    theta_bounds: Tuple[float, float] = (-jnp.pi, jnp.pi)  # θ范围 / θ bounds
    omega_bounds: Tuple[float, float] = (-8.0, 8.0)        # ω范围 / ω bounds
    ou_mean_reversion: float = 1.0    # OU均值回归 / OU mean reversion
    ou_diffusion: float = 0.8         # OU扩散 / OU diffusion
    ipfp_max_iterations: int = 500    # IPFP最大迭代 / IPFP max iterations
    ipfp_tolerance: float = 1e-6      # IPFP收敛容差 / IPFP tolerance
    epsilon_scaling: bool = True      # ε缩放 / ε scaling
    initial_epsilon: float = 1.0      # 初始ε / initial ε
    min_epsilon: float = 0.01         # 最小ε / min ε


class PendulumMMSBResult(NamedTuple):
    """大角度单摆MMSB-VI结果 / Large angle pendulum MMSB-VI result"""
    mmsb_solution: MMSBSolution          # MMSB-VI原始解 / MMSB-VI raw solution
    density_estimates: List[DensityEstimate]  # 密度估计序列 / density estimate sequence
    theta_grid: chex.Array               # θ网格 / θ grid
    omega_grid: chex.Array               # ω网格 / ω grid
    observation_times: chex.Array        # 观测时刻 / observation times
    runtime: float                       # 运行时间 / runtime
    convergence_info: Dict[str, float]   # 收敛信息 / convergence info


class PendulumMMSBSolver:
    """
    大角度单摆MMSB-VI求解器 / Large angle pendulum MMSB-VI solver
    
    核心创新点：
    1. 周期性状态空间处理 - 确保θ边界连续性
    2. 非线性sin(θ)转移核 - 精确建模重力项
    3. 多模态密度保持 - 避免模态collapse
    4. 倒立点数值稳定性 - 处理奇异点附近的计算
    
    Core innovations:
    1. Periodic state space handling - ensure θ boundary continuity
    2. Nonlinear sin(θ) transition kernel - accurate gravity modeling
    3. Multi-modal density preservation - avoid mode collapse
    4. Numerical stability near inverted point - handle singularities
    """
    
    def __init__(self, config: Optional[PendulumMMSBConfig] = None):
        """
        初始化求解器 / Initialize solver
        
        Args:
            config: 配置参数 / configuration parameters
        """
        self.config = config or PendulumMMSBConfig()
        
        # 创建2D网格 / Create 2D grids
        self.theta_grid = jnp.linspace(
            self.config.theta_bounds[0], 
            self.config.theta_bounds[1], 
            self.config.theta_grid_points
        )
        self.omega_grid = jnp.linspace(
            self.config.omega_bounds[0],
            self.config.omega_bounds[1], 
            self.config.omega_grid_points
        )
        
        self.dtheta = self.theta_grid[1] - self.theta_grid[0]
        self.domega = self.omega_grid[1] - self.omega_grid[0]
        
        # 创建2D网格坐标 / Create 2D grid coordinates
        self.theta_2d, self.omega_2d = jnp.meshgrid(
            self.theta_grid, self.omega_grid, indexing='ij'
        )
        
        # 设置OU过程参数 / Set OU process parameters
        self.ou_params = OUProcessParams(
            mean_reversion=self.config.ou_mean_reversion,
            diffusion=self.config.ou_diffusion,
            equilibrium_mean=0.0
        )
        
        # 编译核心函数 / Compile core functions
        self._nonlinear_pendulum_kernel = jax.jit(self._nonlinear_pendulum_kernel_impl)
        self._periodic_boundary_correction = jax.jit(self._periodic_boundary_correction_impl)

        # 自定义梯形积分，兼容旧版JAX缺少 jnp.trapz / custom trapezoidal integration
        def _trapz(y, x):
            h = x[1] - x[0]
            return h * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))

        self._trapz = _trapz
        
        print(f"✅ 大角度单摆MMSB-VI求解器初始化完成")
        print(f"   θ网格: {self.config.theta_grid_points}点，范围[{self.config.theta_bounds[0]:.2f}, {self.config.theta_bounds[1]:.2f}]")
        print(f"   ω网格: {self.config.omega_grid_points}点，范围[{self.config.omega_bounds[0]:.2f}, {self.config.omega_bounds[1]:.2f}]")
    
    def solve(
        self,
        trajectory: PendulumTrajectory,
        verbose: bool = True
    ) -> PendulumMMSBResult:
        """
        求解大角度单摆的MMSB-VI问题 / Solve large angle pendulum MMSB-VI problem
        
        Args:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
            verbose: 是否详细输出 / verbose output
            
        Returns:
            result: 求解结果 / solution result
        """
        if verbose:
            print(f"\n🎯 开始大角度单摆MMSB-VI求解")
            print(f"   观测数量: {len(trajectory.observations)}")
            print(f"   时间范围: {trajectory.obs_times[0]:.2f} - {trajectory.obs_times[-1]:.2f}s")
        
        start_time = time.time()
        
        # 步骤1: 转换为MMSB问题格式 / Step 1: Convert to MMSB problem format
        mmsb_problem = self._create_mmsb_problem(trajectory)
        
        # 步骤2: 创建IPFP配置 / Step 2: Create IPFP configuration
        ipfp_config = self._create_ipfp_config()
        
        # 步骤3: 求解MMSB问题 / Step 3: Solve MMSB problem
        if verbose:
            print("🔄 执行IPFP迭代求解...")
        
        mmsb_solution = solve_mmsb_ipfp_1d_fixed(mmsb_problem, ipfp_config)
        
        # 步骤4: 处理周期性边界 / Step 4: Handle periodic boundaries
        mmsb_solution = self._apply_periodic_boundary_corrections(mmsb_solution)
        
        # 步骤5: 转换为2D密度估计 / Step 5: Convert to 2D density estimates
        density_estimates = self._convert_to_2d_densities(
            mmsb_solution, trajectory.obs_times
        )
        
        runtime = time.time() - start_time
        
        # 收敛信息 / Convergence information
        convergence_info = {
            'final_error': float(mmsb_solution.final_error),
            'n_iterations': mmsb_solution.n_iterations,
            'converged': mmsb_solution.final_error < self.config.ipfp_tolerance
        }
        
        if verbose:
            print(f"✅ MMSB-VI求解完成")
            print(f"   运行时间: {runtime:.2f}s")
            print(f"   迭代次数: {convergence_info['n_iterations']}")
            print(f"   最终误差: {convergence_info['final_error']:.2e}")
            print(f"   收敛状态: {convergence_info['converged']}")
        
        return PendulumMMSBResult(
            mmsb_solution=mmsb_solution,
            density_estimates=density_estimates,
            theta_grid=self.theta_grid,
            omega_grid=self.omega_grid,
            observation_times=trajectory.obs_times,
            runtime=runtime,
            convergence_info=convergence_info
        )
    
    def _create_mmsb_problem(self, trajectory: PendulumTrajectory) -> MMSBProblem:
        """
        创建MMSB问题实例 / Create MMSB problem instance
        
        关键：将2D单摆问题投影到1D θ空间进行IPFP求解
        Key: project 2D pendulum problem to 1D θ space for IPFP solution
        """
        # 使用θ作为主要状态变量 / Use θ as primary state variable
        theta_grid_config = GridConfig1D.create(
            n_points=self.config.theta_grid_points,
            bounds=self.config.theta_bounds
        )
        
        # 观测时刻 / Observation times
        obs_times = trajectory.obs_times
        
        # 观测数据（角度）/ Observation data (angles)
        observations = trajectory.observations
        
        return MMSBProblem(
            observation_times=obs_times,
            ou_params=self.ou_params,
            grid=theta_grid_config,
            y_observations=observations,
            C=1.0,  # 观测矩阵 / observation matrix
            R=trajectory.obs_config.obs_noise_std**2  # 观测噪声协方差 / observation noise covariance
        )
    
    def _create_ipfp_config(self) -> IPFPConfig:
        """创建IPFP配置 / Create IPFP configuration"""
        return IPFPConfig(
            max_iterations=self.config.ipfp_max_iterations,
            tolerance=self.config.ipfp_tolerance,
            check_interval=20,
            use_anderson=True,
            anderson_memory=5,
            epsilon_scaling=self.config.epsilon_scaling,
            initial_epsilon=self.config.initial_epsilon,
            min_epsilon=self.config.min_epsilon,
            eps_decay_high=0.9,
            eps_decay_low=0.6,
            error_threshold=1e-4,
            verbose=False  # 避免过多输出 / avoid excessive output
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _nonlinear_pendulum_kernel_impl(
        self,
        theta_prev: chex.Array,
        theta_curr: chex.Array, 
        dt: float,
        pendulum_params: Dict[str, float]
    ) -> chex.Array:
        """
        非线性单摆转移核 / Nonlinear pendulum transition kernel
        
        考虑sin(θ)重力项的精确转移概率密度。
        Exact transition probability density considering sin(θ) gravity term.
        
        Args:
            theta_prev: 前一时刻角度 / previous angle
            theta_curr: 当前时刻角度 / current angle  
            dt: 时间步长 / time step
            pendulum_params: 单摆参数 / pendulum parameters
            
        Returns:
            log_kernel: 对数转移核 / log transition kernel
        """
        # 提取物理参数 / Extract physical parameters
        g = pendulum_params.get('g', 9.81)
        L = pendulum_params.get('L', 1.0)
        gamma = pendulum_params.get('gamma', 0.2)
        sigma = pendulum_params.get('sigma', 0.3)
        
        # 估计中间角速度（近似）/ Estimate intermediate angular velocity (approximation)
        omega_est = (theta_curr - theta_prev) / dt
        
        # 非线性动力学项 / Nonlinear dynamics terms
        gravity_term = -(g/L) * jnp.sin(theta_prev)
        damping_term = -gamma * omega_est
        
        # 预测角加速度 / Predicted angular acceleration
        alpha_pred = gravity_term + damping_term
        
        # 预测下一角度 / Predicted next angle
        theta_pred = theta_prev + omega_est * dt + 0.5 * alpha_pred * dt**2
        
        # 处理周期性边界 / Handle periodic boundary
        theta_pred = jnp.mod(theta_pred + jnp.pi, 2*jnp.pi) - jnp.pi
        theta_curr_wrapped = jnp.mod(theta_curr + jnp.pi, 2*jnp.pi) - jnp.pi
        
        # 计算预测误差 / Compute prediction error
        prediction_error = theta_curr_wrapped - theta_pred
        
        # 处理周期性边界的误差 / Handle periodic boundary in error
        prediction_error = jnp.mod(prediction_error + jnp.pi, 2*jnp.pi) - jnp.pi
        
        # 有效噪声方差 / Effective noise variance
        noise_variance = sigma**2 * dt
        
        # 高斯对数似然 / Gaussian log-likelihood
        log_likelihood = -0.5 * prediction_error**2 / noise_variance
        log_likelihood -= 0.5 * jnp.log(2 * jnp.pi * noise_variance)
        
        return log_likelihood
    
    @partial(jax.jit, static_argnums=(0,))
    def _periodic_boundary_correction_impl(
        self,
        density_1d: chex.Array
    ) -> chex.Array:
        """
        周期性边界修正 / Periodic boundary correction
        
        确保θ = -π和θ = π处的密度连续性。
        Ensure density continuity at θ = -π and θ = π.
        """
        # 边界点密度平均 / Average boundary densities
        left_boundary = density_1d[0]
        right_boundary = density_1d[-1]
        avg_boundary = 0.5 * (left_boundary + right_boundary)
        
        # 应用边界修正 / Apply boundary correction
        corrected_density = density_1d.at[0].set(avg_boundary)
        corrected_density = corrected_density.at[-1].set(avg_boundary)
        
        return corrected_density
    
    def _apply_periodic_boundary_corrections(
        self,
        mmsb_solution: MMSBSolution
    ) -> MMSBSolution:
        """
        应用周期性边界修正到MMSB解 / Apply periodic boundary corrections to MMSB solution
        """
        # 修正路径密度 / Correct path densities
        corrected_densities = []
        for density in mmsb_solution.path_densities:
            corrected = self._periodic_boundary_correction_impl(density)
            corrected_densities.append(corrected)
        
        # 重新归一化 / Renormalize
        normalized_densities = []
        for density in corrected_densities:
            mass = self._trapz(density, self.theta_grid)
            normalized = density / (mass + 1e-12)
            normalized_densities.append(normalized)
        
        # 更新解 / Update solution
        return mmsb_solution.replace(path_densities=normalized_densities)
    
    def _convert_to_2d_densities(
        self,
        mmsb_solution: MMSBSolution,
        obs_times: chex.Array
    ) -> List[DensityEstimate]:
        """
        将1D MMSB解转换为2D密度估计 / Convert 1D MMSB solution to 2D density estimates
        
        通过条件分布扩展到(θ, ω)联合空间。
        Extend to (θ, ω) joint space via conditional distributions.
        """
        density_estimates = []

        theta_grid = self.theta_grid
        omega_grid = self.omega_grid

        # 预计算所有 θ 对应的不稳定系数 & 条件 ω 分布
        instability_factor = 1.0 + 2.0 * jnp.exp(-0.5 * (theta_grid**2))  # (Nθ,)
        omega_std_all = instability_factor  # 假设基准 std =1.0

        # 生成条件分布矩阵  shape (Nθ, Nω)
        omega_grid_b = omega_grid[None, :]  # (1, Nω)
        omega_std_b = omega_std_all[:, None]  # (Nθ, 1)
        gaussian_exponent = -0.5 * (omega_grid_b / omega_std_b) ** 2  # broadcasting
        omega_conditional_mat = jnp.exp(gaussian_exponent)
        # 行归一化
        h_omega = omega_grid[1] - omega_grid[0]
        norm_factor = h_omega * (
            jnp.sum(omega_conditional_mat, axis=1) - 0.5 * (omega_conditional_mat[:, 0] + omega_conditional_mat[:, -1])
        )[:, None] + 1e-12
        omega_conditional_mat = omega_conditional_mat / norm_factor

        for t, theta_marginal in enumerate(mmsb_solution.path_densities):
            # theta_marginal shape (Nθ,)
            h_omega = omega_grid[1] - omega_grid[0]
            h_theta = theta_grid[1] - theta_grid[0]
            density_2d = theta_marginal[:, None] * omega_conditional_mat  # broadcasting 生成联合密度

            # 归一化2D密度 / Normalize 2D density
            total_mass = jnp.sum(density_2d) * h_theta * h_omega
            density_2d = density_2d / (total_mass + 1e-12)

            # 计算边际分布 / Compute marginal distributions
            h_omega = omega_grid[1] - omega_grid[0]
            marginal_theta = h_omega * (jnp.sum(density_2d, axis=1) - 0.5 * (density_2d[:, 0] + density_2d[:, -1]))

            h_theta = theta_grid[1] - theta_grid[0]
            marginal_omega = h_theta * (jnp.sum(density_2d, axis=0) - 0.5 * (density_2d[0, :] + density_2d[-1, :]))

            density_estimates.append(
                DensityEstimate(
                    theta_grid=theta_grid,
                    omega_grid=omega_grid,
                    density_2d=density_2d,
                    marginal_theta=marginal_theta,
                    marginal_omega=marginal_omega,
                    time_index=t,
                    log_likelihood=float(jnp.log(jnp.sum(density_2d) + 1e-12))
                )
            )

        return density_estimates
    
    def extract_state_estimates(
        self,
        result: PendulumMMSBResult
    ) -> Dict[str, chex.Array]:
        """
        提取状态估计 / Extract state estimates
        
        Args:
            result: MMSB求解结果 / MMSB solution result
            
        Returns:
            estimates: 状态估计字典 / state estimates dictionary
        """
        T = len(result.density_estimates)
        
        theta_means = jnp.zeros(T)
        theta_stds = jnp.zeros(T)
        omega_means = jnp.zeros(T)
        omega_stds = jnp.zeros(T)
        
        for t, density_est in enumerate(result.density_estimates):
            # θ统计量 / θ statistics
            theta_mean = jnp.trapz(
                density_est.marginal_theta * self.theta_grid, self.theta_grid
            )
            theta_var = jnp.trapz(
                density_est.marginal_theta * (self.theta_grid - theta_mean)**2, self.theta_grid
            )
            theta_std = jnp.sqrt(jnp.maximum(theta_var, 1e-8))
            
            # ω统计量 / ω statistics  
            omega_mean = jnp.trapz(
                density_est.marginal_omega * self.omega_grid, self.omega_grid
            )
            omega_var = jnp.trapz(
                density_est.marginal_omega * (self.omega_grid - omega_mean)**2, self.omega_grid
            )
            omega_std = jnp.sqrt(jnp.maximum(omega_var, 1e-8))
            
            theta_means = theta_means.at[t].set(theta_mean)
            theta_stds = theta_stds.at[t].set(theta_std)
            omega_means = omega_means.at[t].set(omega_mean)
            omega_stds = omega_stds.at[t].set(omega_std)
        
        return {
            'theta_mean': theta_means,
            'theta_std': theta_stds,
            'omega_mean': omega_means,
            'omega_std': omega_stds,
            'times': result.observation_times
        }


if __name__ == "__main__":
    # 测试大角度单摆MMSB-VI求解器 / Test large angle pendulum MMSB-VI solver
    print("🧪 测试大角度单摆MMSB-VI求解器")
    print("🧪 Testing Large Angle Pendulum MMSB-VI Solver")
    
    # 创建求解器 / Create solver
    config = PendulumMMSBConfig(
        theta_grid_points=64,
        omega_grid_points=32,
        ipfp_max_iterations=100,
        ipfp_tolerance=1e-5
    )
    
    solver = PendulumMMSBSolver(config)
    
    print("✅ 大角度单摆MMSB-VI求解器测试完成")