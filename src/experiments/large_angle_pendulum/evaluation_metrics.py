"""
概率密度质量评估指标 / Probability Density Quality Assessment Metrics
================================================================

实现公正的概率建模质量评估，专注于密度估计而非点估计：
Implement fair probabilistic modeling quality assessment focused on density estimation rather than point estimation:

1. Negative Log-Likelihood (NLL) - 真实轨迹在估计密度下的对数似然
2. 95% Credible Coverage - 非参数密度积分的置信区间覆盖
3. Bimodality Significance - 多模态显著性统计检验

这些指标能公正评估MMSB-VI的多模态建模优势。
These metrics fairly assess MMSB-VI's multi-modal modeling advantages.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Dict, List, Tuple, Optional, NamedTuple
import chex
from functools import partial
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

jax.config.update('jax_enable_x64', True)


class DensityEstimate(NamedTuple):
    """密度估计结果 / Density estimation result"""
    theta_grid: chex.Array      # 角度网格 / angle grid
    omega_grid: chex.Array      # 角速度网格 / angular velocity grid  
    density_2d: chex.Array      # 2D密度 (n_theta, n_omega) / 2D density
    marginal_theta: chex.Array  # θ边际密度 / θ marginal density
    marginal_omega: chex.Array  # ω边际密度 / ω marginal density
    time_index: int             # 时刻索引 / time index
    log_likelihood: float       # 对数似然 / log likelihood


class QualityMetrics(NamedTuple):
    """质量评估指标 / Quality assessment metrics"""
    nll_total: float           # 总负对数似然 / total negative log-likelihood
    nll_per_time: chex.Array   # 每时刻NLL / per-time NLL
    coverage_95: float         # 95%覆盖率 / 95% coverage rate
    coverage_per_time: chex.Array  # 每时刻覆盖率 / per-time coverage
    bimodality_p_value: float  # 双模态p值 / bimodality p-value
    bimodality_detected: bool  # 是否检测到双模态 / bimodality detected
    theta_ks_statistic: float  # θ边际KS统计量 / θ marginal KS statistic
    effective_sample_size: float  # 有效样本数 / effective sample size


class DensityQualityMetrics:
    """
    概率密度质量评估器 / Probability density quality assessor
    
    专为MMSB-VI vs 经典方法的公正比较设计。
    Designed for fair comparison between MMSB-VI and classical methods.
    
    核心理念：评估概率密度建模质量，而非简单的点估计精度。
    Core philosophy: assess probabilistic modeling quality, not just point estimation accuracy.
    """
    
    def __init__(
        self,
        theta_bounds: Tuple[float, float] = (-jnp.pi, jnp.pi),
        omega_bounds: Tuple[float, float] = (-6.0, 6.0),
        grid_resolution: Tuple[int, int] = (64, 32),
        confidence_level: float = 0.95
    ):
        """
        初始化评估器 / Initialize assessor
        
        Args:
            theta_bounds: 角度范围 / angle bounds
            omega_bounds: 角速度范围 / angular velocity bounds
            grid_resolution: 网格分辨率 (n_theta, n_omega) / grid resolution
            confidence_level: 置信水平 / confidence level
        """
        self.theta_bounds = theta_bounds
        self.omega_bounds = omega_bounds
        self.grid_resolution = grid_resolution
        self.confidence_level = confidence_level
        
        # 创建评估网格 / Create evaluation grids
        self.theta_grid = jnp.linspace(theta_bounds[0], theta_bounds[1], grid_resolution[0])
        self.omega_grid = jnp.linspace(omega_bounds[0], omega_bounds[1], grid_resolution[1])
        self.dtheta = self.theta_grid[1] - self.theta_grid[0]
        self.domega = self.omega_grid[1] - self.omega_grid[0]
        
        # 编译核心函数 / Compile core functions
        self._compute_nll_at_time = jax.jit(self._compute_nll_at_time_impl)
        self._compute_coverage_at_time = jax.jit(self._compute_coverage_at_time_impl)
    
    def evaluate_density_sequence(
        self,
        density_estimates: List[DensityEstimate],
        true_trajectory: chex.Array,  # (T, 2) [θ, ω]
        time_indices: chex.Array      # 对应的时刻索引 / corresponding time indices
    ) -> QualityMetrics:
        """
        评估密度序列的质量 / Evaluate quality of density sequence
        
        Args:
            density_estimates: 密度估计列表 / list of density estimates
            true_trajectory: 真实轨迹 / true trajectory  
            time_indices: 时刻索引 / time indices
            
        Returns:
            metrics: 质量评估指标 / quality assessment metrics
        """
        T = len(density_estimates)
        
        # 1. 计算每时刻的NLL / Compute per-time NLL
        nll_per_time = []
        for i, density_est in enumerate(density_estimates):
            true_state = true_trajectory[time_indices[i]]
            nll = self._compute_nll_at_time_impl(density_est, true_state)
            nll_per_time.append(nll)
        
        nll_per_time = jnp.array(nll_per_time)
        nll_total = float(jnp.sum(nll_per_time))
        
        # 2. 计算每时刻的覆盖率 / Compute per-time coverage
        coverage_per_time = []
        for i, density_est in enumerate(density_estimates):
            true_state = true_trajectory[time_indices[i]]
            coverage = self._compute_coverage_at_time_impl(
                density_est, true_state, self.confidence_level
            )
            coverage_per_time.append(coverage)
        
        coverage_per_time = jnp.array(coverage_per_time)
        coverage_95 = float(jnp.mean(coverage_per_time))
        
        # 3. 双模态显著性检验 / Bimodality significance test
        bimodality_metrics = self._test_bimodality_significance(density_estimates)
        
        # 4. 计算有效样本数（近似）/ Compute effective sample size (approximation)
        ess = self._estimate_effective_sample_size(density_estimates)
        
        return QualityMetrics(
            nll_total=nll_total,
            nll_per_time=nll_per_time,
            coverage_95=coverage_95,
            coverage_per_time=coverage_per_time,
            bimodality_p_value=bimodality_metrics['p_value'],
            bimodality_detected=bimodality_metrics['detected'],
            theta_ks_statistic=bimodality_metrics['ks_statistic'],
            effective_sample_size=ess
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_nll_at_time_impl(
        self,
        density_est: DensityEstimate,
        true_state: chex.Array  # [θ_true, ω_true]
    ) -> float:
        """
        计算单时刻的负对数似然 / Compute negative log-likelihood at single time
        
        通过双线性插值计算真实状态在估计密度下的概率。
        Compute probability of true state under estimated density via bilinear interpolation.
        """
        theta_true, omega_true = true_state[0], true_state[1]
        
        # 处理周期性边界 / Handle periodic boundaries
        theta_wrapped = jnp.mod(theta_true + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # 检查是否在网格范围内 / Check if within grid bounds
        theta_in_bounds = jnp.logical_and(
            theta_wrapped >= self.theta_bounds[0],
            theta_wrapped <= self.theta_bounds[1]
        )
        omega_in_bounds = jnp.logical_and(
            omega_true >= self.omega_bounds[0],
            omega_true <= self.omega_bounds[1]
        )
        
        in_bounds = jnp.logical_and(theta_in_bounds, omega_in_bounds)
        
        # 双线性插值计算密度值 / Bilinear interpolation for density value
        def interpolate_density():
            # 找到网格索引 / Find grid indices
            theta_idx = (theta_wrapped - self.theta_bounds[0]) / self.dtheta
            omega_idx = (omega_true - self.omega_bounds[0]) / self.domega
            
            # 限制到有效范围 / Clamp to valid range
            theta_idx = jnp.clip(theta_idx, 0, self.grid_resolution[0] - 1.001)
            omega_idx = jnp.clip(omega_idx, 0, self.grid_resolution[1] - 1.001)
            
            # 整数部分和小数部分 / Integer and fractional parts
            i0, i1 = jnp.floor(theta_idx).astype(int), jnp.ceil(theta_idx).astype(int)
            j0, j1 = jnp.floor(omega_idx).astype(int), jnp.ceil(omega_idx).astype(int)
            
            # 插值权重 / Interpolation weights
            w_theta = theta_idx - i0
            w_omega = omega_idx - j0
            
            # 双线性插值 / Bilinear interpolation
            density_val = (
                (1 - w_theta) * (1 - w_omega) * density_est.density_2d[i0, j0] +
                w_theta * (1 - w_omega) * density_est.density_2d[i1, j0] +
                (1 - w_theta) * w_omega * density_est.density_2d[i0, j1] +
                w_theta * w_omega * density_est.density_2d[i1, j1]
            )
            
            return density_val
        
        # 如果在边界内则插值，否则返回极小值 / Interpolate if in bounds, else return tiny value
        density_value = jax.lax.cond(
            in_bounds,
            interpolate_density,
            lambda: 1e-12
        )
        
        # 确保密度为正 / Ensure positive density
        density_value = jnp.maximum(density_value, 1e-12)
        
        # 返回负对数似然 / Return negative log-likelihood
        return -jnp.log(density_value)
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_coverage_at_time_impl(
        self,
        density_est: DensityEstimate,
        true_state: chex.Array,
        confidence_level: float
    ) -> float:
        """
        计算单时刻的覆盖率 / Compute coverage at single time
        
        通过非参数密度积分计算置信区间覆盖。
        Compute confidence interval coverage via non-parametric density integration.
        """
        density_2d = density_est.density_2d
        
        # 展平密度并排序 / Flatten and sort density
        flat_density = density_2d.flatten()
        sorted_density = jnp.sort(flat_density)[::-1]  # 降序 / descending
        
        # 累积概率质量 / Cumulative probability mass
        grid_area = self.dtheta * self.domega
        cumulative_mass = jnp.cumsum(sorted_density) * grid_area
        
        # 找到置信区间阈值 / Find confidence interval threshold
        threshold_idx = jnp.searchsorted(cumulative_mass, confidence_level)
        threshold_idx = jnp.clip(threshold_idx, 0, len(sorted_density) - 1)
        density_threshold = sorted_density[threshold_idx]
        
        # 计算真实状态的密度值 / Compute density at true state
        true_density = self._interpolate_density_value(density_est, true_state)
        
        # 检查是否在置信区间内 / Check if within confidence interval
        is_covered = true_density >= density_threshold
        
        return jnp.where(is_covered, 1.0, 0.0)
    
    def _interpolate_density_value(
        self, 
        density_est: DensityEstimate,
        state: chex.Array
    ) -> float:
        """插值计算状态点的密度值 / Interpolate density value at state point"""
        # 重复NLL计算中的插值逻辑，但返回原始密度值
        # Repeat interpolation logic from NLL computation but return raw density value
        theta, omega = state[0], state[1]
        theta_wrapped = jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # 检查边界 / Check bounds
        theta_in_bounds = jnp.logical_and(
            theta_wrapped >= self.theta_bounds[0],
            theta_wrapped <= self.theta_bounds[1]
        )
        omega_in_bounds = jnp.logical_and(
            omega >= self.omega_bounds[0],
            omega <= self.omega_bounds[1]
        )
        
        in_bounds = jnp.logical_and(theta_in_bounds, omega_in_bounds)
        
        def interpolate():
            theta_idx = (theta_wrapped - self.theta_bounds[0]) / self.dtheta
            omega_idx = (omega - self.omega_bounds[0]) / self.domega
            
            theta_idx = jnp.clip(theta_idx, 0, self.grid_resolution[0] - 1.001)
            omega_idx = jnp.clip(omega_idx, 0, self.grid_resolution[1] - 1.001)
            
            i0, i1 = jnp.floor(theta_idx).astype(int), jnp.ceil(theta_idx).astype(int)
            j0, j1 = jnp.floor(omega_idx).astype(int), jnp.ceil(omega_idx).astype(int)
            
            w_theta = theta_idx - i0
            w_omega = omega_idx - j0
            
            return (
                (1 - w_theta) * (1 - w_omega) * density_est.density_2d[i0, j0] +
                w_theta * (1 - w_omega) * density_est.density_2d[i1, j0] +
                (1 - w_theta) * w_omega * density_est.density_2d[i0, j1] +
                w_theta * w_omega * density_est.density_2d[i1, j1]
            )
        
        return jax.lax.cond(in_bounds, interpolate, lambda: 1e-12)
    
    def _test_bimodality_significance(
        self,
        density_estimates: List[DensityEstimate]
    ) -> Dict[str, float]:
        """
        检验双模态显著性 / Test bimodality significance
        
        使用Kolmogorov-Smirnov检验和Dip检验评估多模态性。
        Use Kolmogorov-Smirnov and Dip tests to assess multi-modality.
        
        Returns:
            metrics: 双模态检验结果 / bimodality test results
        """
        # 选择中间时刻进行分析（预期最多模态的时刻）
        # Select middle time points for analysis (expected most multi-modal moments)
        mid_idx = len(density_estimates) // 2
        density_est = density_estimates[mid_idx]
        
        # 提取θ边际分布 / Extract θ marginal distribution
        theta_marginal = np.array(density_est.marginal_theta)
        theta_grid = np.array(density_est.theta_grid)
        
        # 归一化边际分布 / Normalize marginal distribution
        theta_marginal = theta_marginal / (np.trapz(theta_marginal, theta_grid) + 1e-12)
        
        # 生成从边际分布采样的近似样本 / Generate approximate samples from marginal
        # 这是一个近似方法，用于统计检验 / This is an approximation for statistical testing
        n_samples = 1000
        cumulative = np.cumsum(theta_marginal) * (theta_grid[1] - theta_grid[0])
        cumulative = cumulative / cumulative[-1]
        
        # 逆变换采样 / Inverse transform sampling
        uniform_samples = np.random.uniform(0, 1, n_samples)
        theta_samples = np.interp(uniform_samples, cumulative, theta_grid)
        
        # KS检验对比单模态（正态分布）/ KS test against unimodal (normal distribution)
        # 拟合正态分布 / Fit normal distribution
        mu_hat = np.mean(theta_samples)
        sigma_hat = np.std(theta_samples)
        
        # KS检验 / KS test
        ks_stat, ks_p_value = stats.kstest(
            theta_samples, 
            lambda x: stats.norm.cdf(x, mu_hat, sigma_hat)
        )
        
        # Dip检验（如果可用）/ Dip test (if available)
        try:
            from diptest import diptest
            dip_stat, dip_p_value = diptest(theta_samples)
            # 合并检验结果 / Combine test results
            combined_p_value = min(ks_p_value, dip_p_value)
        except ImportError:
            # 仅使用KS检验 / Use only KS test
            combined_p_value = ks_p_value
            print("Warning: diptest not available, using only KS test")
        
        # 显著性阈值 / Significance threshold
        significance_level = 0.05
        bimodality_detected = combined_p_value < significance_level
        
        return {
            'p_value': float(combined_p_value),
            'detected': bool(bimodality_detected),
            'ks_statistic': float(ks_stat)
        }
    
    def _estimate_effective_sample_size(
        self,
        density_estimates: List[DensityEstimate]
    ) -> float:
        """
        估计有效样本数 / Estimate effective sample size
        
        基于密度的平滑度和尖锐度估计等效的蒙特卡洛样本数。
        Estimate equivalent Monte Carlo sample count based on density smoothness and sharpness.
        """
        if not density_estimates:
            return 0.0
        
        # 选择典型时刻 / Select representative time point
        mid_density = density_estimates[len(density_estimates) // 2]
        density_2d = np.array(mid_density.density_2d)
        
        # 计算信息熵作为有效性度量 / Compute information entropy as effectiveness measure
        density_flat = density_2d.flatten()
        density_normalized = density_flat / (np.sum(density_flat) + 1e-12)
        
        # Shannon熵 / Shannon entropy
        entropy = -np.sum(
            density_normalized * np.log(density_normalized + 1e-12)
        )
        
        # 最大可能熵（均匀分布）/ Maximum possible entropy (uniform distribution)
        max_entropy = np.log(len(density_flat))
        
        # 归一化熵作为效率指标 / Normalized entropy as efficiency measure
        efficiency = entropy / max_entropy
        
        # 估计等效样本数 / Estimate equivalent sample count
        # 这是一个启发式估计 / This is a heuristic estimate
        base_samples = 1000  # 基础参考样本数 / base reference sample count
        effective_samples = base_samples * efficiency
        
        return float(effective_samples)
    
    def compare_methods(
        self,
        mmsb_metrics: QualityMetrics,
        baseline_metrics: Dict[str, QualityMetrics],
        method_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        比较不同方法的性能 / Compare performance of different methods
        
        Args:
            mmsb_metrics: MMSB-VI指标 / MMSB-VI metrics
            baseline_metrics: 基线方法指标字典 / baseline method metrics dict
            method_names: 方法名称列表 / method name list
            
        Returns:
            comparison: 比较结果 / comparison results
        """
        comparison = {}
        
        # MMSB-VI结果 / MMSB-VI results
        comparison['MMSB-VI'] = {
            'NLL': mmsb_metrics.nll_total,
            'Coverage_95': mmsb_metrics.coverage_95,
            'Bimodality_P': mmsb_metrics.bimodality_p_value,
            'Bimodality_Detected': float(mmsb_metrics.bimodality_detected),
            'Effective_Sample_Size': mmsb_metrics.effective_sample_size
        }
        
        # 基线方法结果 / Baseline method results
        for method_name in method_names:
            if method_name in baseline_metrics:
                metrics = baseline_metrics[method_name]
                comparison[method_name] = {
                    'NLL': metrics.nll_total,
                    'Coverage_95': metrics.coverage_95,
                    'Bimodality_P': metrics.bimodality_p_value,
                    'Bimodality_Detected': float(metrics.bimodality_detected),
                    'Effective_Sample_Size': metrics.effective_sample_size
                }
        
        # 计算相对改进 / Compute relative improvements
        for method_name in method_names:
            if method_name in comparison:
                baseline = comparison[method_name]
                mmsb = comparison['MMSB-VI']
                
                # NLL改进（越小越好）/ NLL improvement (lower is better)
                nll_improvement = (baseline['NLL'] - mmsb['NLL']) / baseline['NLL']
                
                # 覆盖率改进（越大越好）/ Coverage improvement (higher is better)
                coverage_improvement = (mmsb['Coverage_95'] - baseline['Coverage_95'])
                
                comparison[f'{method_name}_vs_MMSB'] = {
                    'NLL_improvement_pct': nll_improvement * 100,
                    'Coverage_improvement': coverage_improvement,
                    'Bimodality_advantage': mmsb['Bimodality_Detected'] - baseline['Bimodality_Detected']
                }
        
        return comparison
    
    def print_comparison_summary(self, comparison: Dict[str, Dict[str, float]]):
        """打印比较结果摘要 / Print comparison summary"""
        print("\n" + "="*70)
        print("概率密度质量评估对比结果 / Probability Density Quality Assessment Comparison")
        print("="*70)
        
        # 主要指标对比 / Main metrics comparison
        methods = [k for k in comparison.keys() if not k.endswith('_vs_MMSB')]
        
        print(f"\n{'方法/Method':<15} {'NLL':<12} {'Coverage':<12} {'Bimodal':<10} {'ESS':<10}")
        print("-" * 65)
        
        for method in methods:
            metrics = comparison[method]
            print(f"{method:<15} {metrics['NLL']:<12.2f} {metrics['Coverage_95']:<12.3f} "
                  f"{metrics['Bimodality_Detected']:<10.0f} {metrics['Effective_Sample_Size']:<10.0f}")
        
        # MMSB-VI优势分析 / MMSB-VI advantage analysis
        print(f"\n🎯 MMSB-VI相对改进 / MMSB-VI Relative Improvements:")
        for key in comparison:
            if key.endswith('_vs_MMSB'):
                method_name = key.replace('_vs_MMSB', '')
                improvements = comparison[key]
                print(f"  vs {method_name}:")
                print(f"    NLL改进: {improvements['NLL_improvement_pct']:.1f}%")
                print(f"    覆盖率改进: {improvements['Coverage_improvement']:.3f}")
                print(f"    双模态优势: {improvements['Bimodality_advantage']:.0f}")


if __name__ == "__main__":
    # 测试密度质量评估器 / Test density quality assessor
    print("🧪 测试概率密度质量评估指标")
    print("🧪 Testing Probability Density Quality Assessment Metrics")
    
    # 创建评估器 / Create assessor
    assessor = DensityQualityMetrics(
        grid_resolution=(32, 16),
        confidence_level=0.95
    )
    
    print("✅ 概率密度质量评估器初始化完成")
    print("✅ Probability density quality assessor initialized")