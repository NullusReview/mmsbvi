#!/usr/bin/env python3
"""
大角度单摆方法综合对比 / Comprehensive Large Angle Pendulum Method Comparison
=============================================================================

对比四种方法在大角度单摆系统上的性能：
1. MMSB-VI (Multi-Marginal Schrödinger Bridge Variational Inference)
2. EKF (Extended Kalman Filter)  
3. UKF (Unscented Kalman Filter)
4. SVI (Stochastic Variational Inference)

使用概率密度质量评估指标：
- Negative Log-Likelihood (NLL)
- 95% Credible Coverage
- Bimodality Detection

Compare four methods on large angle pendulum system:
1. MMSB-VI 
2. EKF
3. UKF
4. SVI

Using probability density quality metrics:
- Negative Log-Likelihood (NLL)
- 95% Credible Coverage  
- Bimodality Detection
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import pathlib
from typing import Dict, List, Tuple, Any
import pandas as pd

# 添加项目根目录到路径
root_dir = pathlib.Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

jax.config.update('jax_enable_x64', True)

# 导入我们的组件
from data_generator import (
    LargeAnglePendulumGenerator, PendulumParams, ObservationConfig
)
from pendulum_mmsb_solver import (
    PendulumMMSBSolver, PendulumMMSBConfig
)
from evaluation_metrics import (
    DensityQualityMetrics, DensityEstimate
)
from src.baselines import (
    PendulumEKFSmoother, PendulumUKFSmoother, PendulumSVISmoother
)


class ComprehensiveComparison:
    """
    大角度单摆方法综合对比器 / Comprehensive pendulum method comparator
    """
    
    def __init__(self, 
                 scenario_name: str = "inverted_equilibrium",
                 random_seed: int = 42):
        """
        初始化对比器 / Initialize comparator
        
        Args:
            scenario_name: 测试场景名称 / test scenario name
            random_seed: 随机种子 / random seed
        """
        self.scenario_name = scenario_name
        self.seed = random_seed
        self.key = jax.random.PRNGKey(random_seed)
        
        # 创建数据生成器 / Create data generator
        self.pendulum_params = PendulumParams(
            g=9.81,
            L=1.0, 
            gamma=0.2,
            sigma=0.3
        )
        
        # 创建观测时刻 / Create observation times
        obs_times = jnp.arange(0.0, 3.0, 0.1)  # 每0.1s观测一次，总共3s
        
        self.obs_config = ObservationConfig(
            obs_times=obs_times,
            obs_noise_std=0.1,
            sparse_strategy="skip_unstable"
        )
        
        self.data_generator = LargeAnglePendulumGenerator(
            params=self.pendulum_params,
            dt=0.05,
            total_time=3.0
        )
        
        # 创建求解器 / Create solvers
        self._setup_solvers()
        
        # 创建评估器 / Create evaluator
        self.evaluator = DensityQualityMetrics()
        
        print(f"🎯 大角度单摆综合对比初始化完成")
        print(f"   场景: {scenario_name}")
        print(f"   随机种子: {random_seed}")
        print(f"   物理参数: g={self.pendulum_params.g}, L={self.pendulum_params.L}")
        print(f"   噪声参数: γ={self.pendulum_params.gamma}, σ={self.pendulum_params.sigma}")
    
    def _setup_solvers(self):
        """设置所有求解器 / Setup all solvers"""
        
        # MMSB-VI配置 / MMSB-VI configuration
        self.mmsb_config = PendulumMMSBConfig(
            theta_grid_points=64,
            omega_grid_points=32,
            ipfp_max_iterations=300,
            ipfp_tolerance=1e-6
        )
        self.mmsb_solver = PendulumMMSBSolver(self.mmsb_config)
        
        # EKF配置 / EKF configuration
        self.ekf_smoother = PendulumEKFSmoother(
            dt=self.data_generator.dt,
            g=self.pendulum_params.g,
            L=self.pendulum_params.L,
            gamma=self.pendulum_params.gamma,
            sigma=self.pendulum_params.sigma,
            obs_noise_std=self.obs_config.obs_noise_std
        )
        
        # UKF配置 / UKF configuration
        self.ukf_smoother = PendulumUKFSmoother(
            dt=self.data_generator.dt,
            g=self.pendulum_params.g,
            L=self.pendulum_params.L,
            gamma=self.pendulum_params.gamma,
            sigma=self.pendulum_params.sigma,
            obs_noise_std=self.obs_config.obs_noise_std,
            alpha=1.0,
            beta=2.0,
            kappa=1.0
        )
        
        # SVI配置 / SVI configuration
        self.svi_smoother = PendulumSVISmoother(
            dt=self.data_generator.dt,
            g=self.pendulum_params.g,
            L=self.pendulum_params.L,
            gamma=self.pendulum_params.gamma,
            sigma=self.pendulum_params.sigma,
            obs_noise_std=self.obs_config.obs_noise_std,
            learning_rate=0.01,
            n_samples=20,
            max_iterations=1000,
            convergence_tol=1e-6
        )
    
    def generate_test_scenario(self) -> Any:
        """
        生成测试场景数据 / Generate test scenario data
        
        Returns:
            trajectory: 单摆轨迹数据 / pendulum trajectory data
        """
        if self.scenario_name == "inverted_equilibrium":
            # 倒立点附近的关键场景 / Critical scenario near inverted point
            initial_theta = jnp.pi - 0.1 + 0.05 * jax.random.normal(self.key)
            initial_omega = 0.0 + 0.1 * jax.random.normal(self.key)
            initial_state = jnp.array([initial_theta, initial_omega])
            
        elif self.scenario_name == "large_swing":
            # 大幅摆动场景 / Large swing scenario
            initial_theta = jnp.pi/3 + 0.1 * jax.random.normal(self.key)
            initial_omega = 2.0 + 0.2 * jax.random.normal(self.key)
            initial_state = jnp.array([initial_theta, initial_omega])
            
        else:
            # 默认小角度场景 / Default small angle scenario
            initial_theta = 0.2 + 0.05 * jax.random.normal(self.key)
            initial_omega = 0.0 + 0.1 * jax.random.normal(self.key)
            initial_state = jnp.array([initial_theta, initial_omega])
        
        print(f"\n📊 生成测试数据...")
        print(f"   初始状态: θ₀={initial_state[0]:.3f}, ω₀={initial_state[1]:.3f}")
        
        # 生成轨迹 / Generate trajectory
        if self.scenario_name == "inverted_equilibrium":
            # 使用内置的倒立点场景生成器
            trajectory = self.data_generator.generate_unstable_scenario(
                key=self.key,
                theta_perturbation=0.05,
                omega_perturbation=0.02
            )
        else:
            # 使用自定义初始状态生成轨迹
            trajectory = self.data_generator._generate_trajectory_impl(
                initial_state, self.key, self.obs_config
            )
        
        print(f"   观测数量: {len(trajectory.observations)}")
        print(f"   时间范围: {trajectory.obs_times[0]:.2f} - {trajectory.obs_times[-1]:.2f}s")
        
        return trajectory
    
    def run_mmsb_vi(self, trajectory) -> Tuple[Any, Dict[str, float]]:
        """运行MMSB-VI方法 / Run MMSB-VI method"""
        print(f"\n🔬 运行MMSB-VI...")
        
        start_time = time.time()
        try:
            result = self.mmsb_solver.solve(trajectory, verbose=False)
            runtime = time.time() - start_time
            
            # 评估性能 / Evaluate performance
            # 创建时刻索引 / Create time indices
            time_indices = jnp.arange(len(result.density_estimates))
            
            quality_metrics = self.evaluator.evaluate_density_sequence(
                result.density_estimates, 
                trajectory.states,
                time_indices
            )
            
            # 转换为字典格式 / Convert to dictionary format
            metrics = {
                'nll_mean': float(jnp.mean(quality_metrics.nll_per_time)),
                'nll_std': float(jnp.std(quality_metrics.nll_per_time)),
                'coverage_95': float(quality_metrics.coverage_95),
                'bimodality_score': float(quality_metrics.theta_ks_statistic),
                'runtime': runtime,
                'converged': result.convergence_info['converged'],
                'n_iterations': result.convergence_info['n_iterations']
            }
            
            print(f"✅ MMSB-VI完成: {runtime:.2f}s, 收敛: {result.convergence_info['converged']}")
            
            return result, metrics
            
        except Exception as e:
            print(f"❌ MMSB-VI失败: {e}")
            return None, {'runtime': time.time() - start_time, 'error': str(e)}
    
    def run_ekf(self, trajectory) -> Tuple[Any, Dict[str, float]]:
        """运行EKF方法 / Run EKF method"""
        print(f"\n🔬 运行EKF...")
        
        start_time = time.time()
        try:
            # 设置初始条件 / Set initial conditions
            initial_mean = jnp.array([trajectory.observations[0], 0.0])
            initial_cov = jnp.eye(2) * 0.5
            
            result = self.ekf_smoother.smooth(
                trajectory.observations,
                initial_mean=initial_mean,
                initial_cov=initial_cov
            )
            runtime = time.time() - start_time
            
            # 转换为密度估计格式 / Convert to density estimate format
            density_estimates = self._convert_gaussian_to_densities(
                result.smoothed_states,
                trajectory.obs_times,
                "EKF"
            )
            
            # 评估性能 / Evaluate performance
            # 创建时刻索引 / Create time indices
            time_indices = jnp.arange(len(density_estimates))
            
            quality_metrics = self.evaluator.evaluate_density_sequence(
                density_estimates,
                trajectory.states,
                time_indices
            )
            
            # 转换为字典格式 / Convert to dictionary format
            metrics = {
                'nll_mean': float(jnp.mean(quality_metrics.nll_per_time)),
                'nll_std': float(jnp.std(quality_metrics.nll_per_time)),
                'coverage_95': float(quality_metrics.coverage_95),
                'bimodality_score': float(quality_metrics.theta_ks_statistic),
                'runtime': runtime,
                'log_likelihood': result.total_log_likelihood
            }
            
            print(f"✅ EKF完成: {runtime:.2f}s")
            
            return result, metrics
            
        except Exception as e:
            print(f"❌ EKF失败: {e}")
            return None, {'runtime': time.time() - start_time, 'error': str(e)}
    
    def run_ukf(self, trajectory) -> Tuple[Any, Dict[str, float]]:
        """运行UKF方法 / Run UKF method"""
        print(f"\n🔬 运行UKF...")
        
        start_time = time.time()
        try:
            # 设置初始条件 / Set initial conditions
            initial_mean = jnp.array([trajectory.observations[0], 0.0])
            initial_cov = jnp.eye(2) * 0.5
            
            result = self.ukf_smoother.smooth(
                trajectory.observations,
                initial_mean=initial_mean,
                initial_cov=initial_cov
            )
            runtime = time.time() - start_time
            
            # 转换为密度估计格式 / Convert to density estimate format
            density_estimates = self._convert_gaussian_to_densities(
                result.smoothed_states,
                trajectory.obs_times,
                "UKF"
            )
            
            # 评估性能 / Evaluate performance
            # 创建时刻索引 / Create time indices
            time_indices = jnp.arange(len(density_estimates))
            
            quality_metrics = self.evaluator.evaluate_density_sequence(
                density_estimates,
                trajectory.states,
                time_indices
            )
            
            # 转换为字典格式 / Convert to dictionary format
            metrics = {
                'nll_mean': float(jnp.mean(quality_metrics.nll_per_time)),
                'nll_std': float(jnp.std(quality_metrics.nll_per_time)),
                'coverage_95': float(quality_metrics.coverage_95),
                'bimodality_score': float(quality_metrics.theta_ks_statistic),
                'runtime': runtime,
                'log_likelihood': result.total_log_likelihood
            }
            
            print(f"✅ UKF完成: {runtime:.2f}s")
            
            return result, metrics
            
        except Exception as e:
            print(f"❌ UKF失败: {e}")
            return None, {'runtime': time.time() - start_time, 'error': str(e)}
    
    def run_svi(self, trajectory) -> Tuple[Any, Dict[str, float]]:
        """运行SVI方法 / Run SVI method"""
        print(f"\n🔬 运行SVI...")
        
        start_time = time.time()
        try:
            # 设置初始条件 / Set initial conditions
            initial_mean = jnp.array([trajectory.observations[0], 0.0])
            initial_cov = jnp.eye(2) * 0.5
            
            result = self.svi_smoother.smooth(
                trajectory.observations,
                initial_mean=initial_mean,
                initial_cov=initial_cov
            )
            runtime = time.time() - start_time
            
            # 转换为密度估计格式 / Convert to density estimate format
            density_estimates = self._convert_svi_to_densities(
                result,
                trajectory.obs_times,
                "SVI"
            )
            
            # 评估性能 / Evaluate performance
            # 创建时刻索引 / Create time indices
            time_indices = jnp.arange(len(density_estimates))
            
            quality_metrics = self.evaluator.evaluate_density_sequence(
                density_estimates,
                trajectory.states,
                time_indices
            )
            
            # 转换为字典格式 / Convert to dictionary format
            metrics = {
                'nll_mean': float(jnp.mean(quality_metrics.nll_per_time)),
                'nll_std': float(jnp.std(quality_metrics.nll_per_time)),
                'coverage_95': float(quality_metrics.coverage_95),
                'bimodality_score': float(quality_metrics.theta_ks_statistic),
                'runtime': runtime,
                'elbo': float(result.elbo),
                'log_likelihood': float(result.total_log_likelihood)
            }
            
            print(f"✅ SVI完成: {runtime:.2f}s, ELBO: {result.elbo:.3f}")
            
            return result, metrics
            
        except Exception as e:
            print(f"❌ SVI失败: {e}")
            return None, {'runtime': time.time() - start_time, 'error': str(e)}
    
    def _convert_svi_to_densities(self, svi_result, obs_times, method_name):
        """将SVI结果转换为密度估计格式 / Convert SVI result to density estimates"""
        
        # 使用与MMSB-VI相同的网格 / Use same grid as MMSB-VI
        theta_grid = self.mmsb_solver.theta_grid
        omega_grid = self.mmsb_solver.omega_grid
        
        density_estimates = []
        
        for t in range(len(svi_result.means)):
            # 获取变分参数 / Get variational parameters
            mean = svi_result.means[t]
            std = jnp.exp(svi_result.log_stds[t])
            cov = jnp.diag(std**2)
            
            # 创建2D网格 / Create 2D grid
            theta_2d, omega_2d = jnp.meshgrid(theta_grid, omega_grid, indexing='ij')
            
            # 计算2D高斯密度（考虑周期性）/ Compute 2D Gaussian density (considering periodicity)
            density_2d = self._compute_periodic_gaussian_density(
                theta_2d, omega_2d, mean, cov
            )
            
            # 计算边际分布 / Compute marginal distributions
            dtheta = theta_grid[1] - theta_grid[0]
            domega = omega_grid[1] - omega_grid[0]
            
            # 使用自定义梯形积分替代jnp.trapz
            domega = omega_grid[1] - omega_grid[0]
            marginal_theta = domega * (jnp.sum(density_2d, axis=1) - 0.5 * (density_2d[:, 0] + density_2d[:, -1]))
            
            dtheta = theta_grid[1] - theta_grid[0]
            marginal_omega = dtheta * (jnp.sum(density_2d, axis=0) - 0.5 * (density_2d[0, :] + density_2d[-1, :]))
            
            # 创建密度估计 / Create density estimate
            density_estimate = DensityEstimate(
                theta_grid=theta_grid,
                omega_grid=omega_grid,
                density_2d=density_2d,
                marginal_theta=marginal_theta,
                marginal_omega=marginal_omega,
                time_index=t,
                log_likelihood=float(svi_result.total_log_likelihood / len(svi_result.means))  # 平均每时刻的对数似然
            )
            
            density_estimates.append(density_estimate)
        
        return density_estimates
    
    def _convert_gaussian_to_densities(self, gaussian_states, obs_times, method_name):
        """将高斯状态转换为密度估计格式 / Convert Gaussian states to density estimates"""
        
        # 使用与MMSB-VI相同的网格 / Use same grid as MMSB-VI
        theta_grid = self.mmsb_solver.theta_grid
        omega_grid = self.mmsb_solver.omega_grid
        
        density_estimates = []
        
        for t, state in enumerate(gaussian_states):
            mean = state.mean
            cov = state.covariance
            
            # 创建2D网格 / Create 2D grid
            theta_2d, omega_2d = jnp.meshgrid(theta_grid, omega_grid, indexing='ij')
            
            # 计算2D高斯密度（考虑周期性）/ Compute 2D Gaussian density (considering periodicity)
            density_2d = self._compute_periodic_gaussian_density(
                theta_2d, omega_2d, mean, cov
            )
            
            # 计算边际分布 / Compute marginal distributions
            dtheta = theta_grid[1] - theta_grid[0]
            domega = omega_grid[1] - omega_grid[0]
            
            # 使用自定义梯形积分替代jnp.trapz
            domega = omega_grid[1] - omega_grid[0]
            marginal_theta = domega * (jnp.sum(density_2d, axis=1) - 0.5 * (density_2d[:, 0] + density_2d[:, -1]))
            
            dtheta = theta_grid[1] - theta_grid[0]
            marginal_omega = dtheta * (jnp.sum(density_2d, axis=0) - 0.5 * (density_2d[0, :] + density_2d[-1, :]))
            
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
    
    def _compute_periodic_gaussian_density(self, theta_2d, omega_2d, mean, cov):
        """计算考虑周期性的2D高斯密度 / Compute 2D Gaussian density considering periodicity"""
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
    
    def run_comprehensive_comparison(self) -> pd.DataFrame:
        """
        运行综合对比 / Run comprehensive comparison
        
        Returns:
            results_df: 结果数据框 / results dataframe
        """
        print("="*60)
        print("🎯 开始大角度单摆方法综合对比")
        print("🎯 Starting Comprehensive Large Angle Pendulum Comparison")
        print("="*60)
        
        # 生成测试数据 / Generate test data
        trajectory = self.generate_test_scenario()
        
        # 运行所有方法 / Run all methods
        results = {}
        
        # MMSB-VI
        mmsb_result, mmsb_metrics = self.run_mmsb_vi(trajectory)
        results['MMSB-VI'] = mmsb_metrics
        
        # EKF
        ekf_result, ekf_metrics = self.run_ekf(trajectory)
        results['EKF'] = ekf_metrics
        
        # UKF
        ukf_result, ukf_metrics = self.run_ukf(trajectory)
        results['UKF'] = ukf_metrics
        
        # SVI
        svi_result, svi_metrics = self.run_svi(trajectory)
        results['SVI'] = svi_metrics
        
        # 整理结果 / Organize results
        results_df = self._create_results_dataframe(results)
        
        # 显示结果 / Display results
        self._display_results(results_df)
        
        return results_df
    
    def _create_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """创建结果数据框 / Create results dataframe"""
        
        # 定义关键指标 / Define key metrics
        key_metrics = [
            'nll_mean', 'nll_std',
            'coverage_95', 
            'bimodality_score',
            'runtime',
        ]
        
        # 创建数据框 / Create dataframe
        df_data = []
        for method, metrics in results.items():
            row = {'Method': method}
            for metric in key_metrics:
                if metric in metrics:
                    row[metric] = metrics[metric]
                else:
                    row[metric] = np.nan
            
            # 添加额外信息 / Add extra info
            if 'error' in metrics:
                row['Status'] = f"Error: {metrics['error']}"
            else:
                row['Status'] = "Success"
                
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def _display_results(self, results_df: pd.DataFrame):
        """显示对比结果 / Display comparison results"""
        
        print("\n" + "="*80)
        print("📊 综合对比结果 / Comprehensive Comparison Results")
        print("="*80)
        
        # 显示基本性能表 / Display basic performance table
        print("\n🏆 性能指标对比 / Performance Metrics Comparison:")
        print("-" * 80)
        
        # 格式化显示 / Formatted display
        for _, row in results_df.iterrows():
            method = row['Method']
            status = row['Status']
            
            if status == "Success":
                nll_mean = row['nll_mean']
                coverage = row['coverage_95']
                bimodality = row['bimodality_score']
                runtime = row['runtime']
                
                print(f"{method:>10} | NLL: {nll_mean:6.3f} | Coverage: {coverage:5.1%} | Bimodality: {bimodality:5.3f} | Time: {runtime:6.2f}s")
            else:
                print(f"{method:>10} | {status}")
        
        print("-" * 80)
        
        # 分析最佳方法 / Analyze best methods
        success_df = results_df[results_df['Status'] == 'Success']
        
        if len(success_df) > 0:
            print("\n🎖️  最佳性能分析 / Best Performance Analysis:")
            
            # NLL最低（密度质量最好）/ Lowest NLL (best density quality)
            best_nll_idx = success_df['nll_mean'].idxmin()
            best_nll_method = success_df.loc[best_nll_idx, 'Method']
            print(f"   最佳密度质量 (最低NLL): {best_nll_method} ({success_df.loc[best_nll_idx, 'nll_mean']:.3f})")
            
            # Coverage最高（校准最好）/ Highest coverage (best calibration)
            best_coverage_idx = success_df['coverage_95'].idxmax()
            best_coverage_method = success_df.loc[best_coverage_idx, 'Method']
            print(f"   最佳校准质量 (最高Coverage): {best_coverage_method} ({success_df.loc[best_coverage_idx, 'coverage_95']:.1%})")
            
            # Bimodality最高（多模态检测最好）/ Highest bimodality (best multimodal detection)
            best_bimodal_idx = success_df['bimodality_score'].idxmax()
            best_bimodal_method = success_df.loc[best_bimodal_idx, 'Method']
            print(f"   最佳多模态检测: {best_bimodal_method} ({success_df.loc[best_bimodal_idx, 'bimodality_score']:.3f})")
            
            # 最快速度 / Fastest runtime
            fastest_idx = success_df['runtime'].idxmin()
            fastest_method = success_df.loc[fastest_idx, 'Method']
            print(f"   最快运行速度: {fastest_method} ({success_df.loc[fastest_idx, 'runtime']:.2f}s)")
        
        print("\n" + "="*80)
        print("💡 分析要点 / Key Analysis Points:")
        print("   - NLL越低表示密度估计质量越好")
        print("   - Coverage越接近95%表示不确定性校准越准确")
        print("   - Bimodality Score越高表示多模态检测能力越强")
        print("   - 这些指标更公平地评估概率建模质量，而非仅点估计精度")
        print("="*80)


def main():
    """主函数 / Main function"""
    
    # 创建对比器 / Create comparator
    comparator = ComprehensiveComparison(
        scenario_name="inverted_equilibrium",  # 关键的倒立点场景
        random_seed=42
    )
    
    # 运行对比 / Run comparison
    results_df = comparator.run_comprehensive_comparison()
    
    # 保存结果 / Save results
    output_file = root_dir / "results" / "pendulum_comparison_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存到: {output_file}")
    
    return results_df


if __name__ == "__main__":
    results = main()