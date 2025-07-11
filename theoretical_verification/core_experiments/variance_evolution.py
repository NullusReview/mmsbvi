#!/usr/bin/env python3
"""
Variance Evolution Constraint Experiment
方差演化约束实验

展示多边际薛定谔桥如何控制密度形状的平滑演化
Demonstrates how multi-marginal Schrödinger bridge controls smooth evolution of density shapes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time

# Enable 64-bit precision / 启用64位精度
jax.config.update('jax_enable_x64', True)

from src.mmsbvi.core import (
    GridConfig1D, OUProcessParams, MMSBProblem, IPFPConfig
)
from src.mmsbvi.algorithms.ipfp_1d import (
    solve_mmsb_ipfp_1d_fixed,
    validate_ipfp_solution_fixed
)

# JAX-compatible integration function / JAX兼容的积分函数  
def jax_trapz(y: jnp.ndarray, dx: float) -> float:
    """JAX-compatible trapezoidal integration."""
    return dx * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))


def create_variance_evolution_marginals(grid_config: GridConfig1D) -> tuple:
    """
    Create three Gaussian marginals with evolving variance
    创建具有演化方差的三个高斯边际分布
    
    Mathematical significance:
    - This tests how multi-marginal constraints control the *shape* evolution
    - Without intermediate constraint: variance could change arbitrarily
    - With intermediate constraint: enforces smooth variance transition
    
    数学意义：
    - 测试多边际约束如何控制*形状*演化
    - 没有中间约束：方差可以任意变化
    - 有中间约束：强制平滑的方差过渡
    """
    grid = grid_config.points
    h = grid_config.spacing
    
    # Variance schedule: narrow → medium → wide
    # 方差计划：窄 → 中等 → 宽
    variances = [0.2, 0.5, 1.0]
    time_points = [0.0, 0.5, 1.0]
    
    marginals = []
    
    for i, (t, var) in enumerate(zip(time_points, variances)):
        # All Gaussians centered at origin but with different variances
        # 所有高斯都以原点为中心，但方差不同
        rho = jnp.exp(-0.5 * grid**2 / var) / jnp.sqrt(2 * jnp.pi * var)
        rho = rho / jax_trapz(rho, dx=h)  # Normalize
        marginals.append(rho)
        
        print(f"t={t:.1f}: N(0, {var:.1f}), peak = {jnp.max(rho):.4f}")
    
    return marginals


def analyze_variance_evolution(solution, problem, grid_config):
    """
    Analyze how variance evolves under multi-marginal constraints
    分析多边际约束下方差如何演化
    """
    grid = grid_config.points
    h = grid_config.spacing
    
    print("\n" + "="*60)
    print("VARIANCE EVOLUTION ANALYSIS / 方差演化分析")
    print("="*60)
    
    # Compute sample variances of computed marginals
    # 计算计算得到的边际分布的样本方差
    computed_variances = []
    target_variances = [0.2, 0.5, 1.0]
    
    for k, density in enumerate(solution.path_densities):
        # Sample variance: Var[X] = E[X²] - E[X]²
        mean = jax_trapz(grid * density, dx=h)
        second_moment = jax_trapz(grid**2 * density, dx=h)
        variance = second_moment - mean**2
        
        computed_variances.append(variance)
        target_var = target_variances[k]
        error = abs(variance - target_var)
        
        print(f"Marginal {k} (t={problem.observation_times[k]:.1f}):")
        print(f"  Target variance: {target_var:.4f}")
        print(f"  Computed variance: {variance:.4f}")
        print(f"  Error: {error:.2e}")
        print(f"  Peak density: {jnp.max(density):.4f}")
    
    # Check smoothness of variance evolution
    # 检查方差演化的平滑性
    variance_diff_1 = computed_variances[1] - computed_variances[0] 
    variance_diff_2 = computed_variances[2] - computed_variances[1]
    
    print(f"\nVariance evolution smoothness:")
    print(f"  Δvar(0→0.5): {variance_diff_1:.4f}")
    print(f"  Δvar(0.5→1): {variance_diff_2:.4f}")
    print(f"  Ratio: {variance_diff_2/variance_diff_1:.4f} (should be ≈ 1.0 for linear)")
    
    return computed_variances


def compare_with_free_transport(problem):
    """
    Compare with what would happen in free (unconstrained) transport
    与自由（无约束）传输进行比较
    """
    print("\n" + "="*50)
    print("COMPARISON WITH FREE TRANSPORT / 与自由传输比较")
    print("="*50)
    
    # For OU process, compute the natural variance evolution without constraints
    # 对于OU过程，计算无约束情况下的自然方差演化
    ou_params = problem.ou_params
    theta = ou_params.mean_reversion
    sigma = ou_params.diffusion
    
    initial_var = 0.2
    final_var = 1.0
    t_mid = 0.5
    
    # OU process variance evolution: var(t) = var₀e^(-2θt) + σ²/(2θ)(1-e^(-2θt))
    # OU过程方差演化
    equilibrium_var = sigma**2 / (2 * theta)  # = 0.4 for θ=1, σ=0.8
    
    def ou_variance(t, var0):
        return var0 * jnp.exp(-2 * theta * t) + equilibrium_var * (1 - jnp.exp(-2 * theta * t))
    
    # What variance would naturally occur at t=0.5 starting from var=0.2?
    # 从var=0.2开始，在t=0.5时自然出现的方差是多少？
    natural_var_mid = ou_variance(t_mid, initial_var)
    
    print(f"OU parameters: θ={theta}, σ={sigma}")
    print(f"Equilibrium variance: {equilibrium_var:.4f}")
    print(f"Natural variance at t=0.5: {natural_var_mid:.4f}")
    print(f"Constrained variance at t=0.5: 0.5000")
    print(f"Deviation from natural evolution: {abs(0.5 - natural_var_mid):.4f}")
    
    if abs(0.5 - natural_var_mid) > 0.1:
        print("✓ Significant constraint effect! Multi-marginal bridge enforces non-natural path.")
        print("✓ 显著的约束效应！多边际桥强制执行非自然路径。")
    else:
        print("~ Constraint effect is mild. Natural OU evolution already close to constraint.")
        print("~ 约束效应轻微。自然OU演化已接近约束。")


def run_variance_evolution_experiment():
    """
    Run the variance evolution constraint experiment
    运行方差演化约束实验
    """
    print("=" * 80)
    print("VARIANCE EVOLUTION CONSTRAINT EXPERIMENT")
    print("方差演化约束实验")
    print("=" * 80)
    print("This demonstrates multi-marginal control of density SHAPE evolution!")
    print("这展示了多边际对密度形状演化的控制！")
    
    # Setup grid / 设置网格
    grid_config = GridConfig1D.create(n_points=80, bounds=(-4.0, 4.0))
    print(f"\nGrid: {grid_config.n_points} points, spacing = {grid_config.spacing:.4f}")
    
    # Create variance evolution marginals / 创建方差演化边际分布
    marginals = create_variance_evolution_marginals(grid_config)
    
    # OU process parameters / OU过程参数
    ou_params = OUProcessParams(
        mean_reversion=1.0,   # Moderate mean reversion
        diffusion=0.8,        # Controls natural variance evolution  
        equilibrium_mean=0.0  # Centered
    )
    print(f"\nOU parameters: θ={ou_params.mean_reversion}, σ={ou_params.diffusion}")
    
    # Create variance evolution MMSB problem / 创建方差演化MMSB问题
    problem = MMSBProblem(
        observation_times=jnp.array([0.0, 0.5, 1.0]),
        observed_marginals=marginals,
        ou_params=ou_params,
        grid=grid_config
    )
    
    # Compare with free transport first / 首先与自由传输比较
    compare_with_free_transport(problem)
    
    # IPFP algorithm configuration / IPFP算法配置
    config = IPFPConfig(
        max_iterations=600,    # May need more for shape constraints
        tolerance=1e-7,        # High precision for variance accuracy
        check_interval=15,     # 
        verbose=True           
    )
    
    print(f"\nIPFP config: max_iter={config.max_iterations}, tol={config.tolerance:.0e}")
    
    # Solve the variance evolution problem / 求解方差演化问题
    print("\nSolving variance evolution MMSB problem...")
    start_time = time.time()
    
    solution = solve_mmsb_ipfp_1d_fixed(problem, config)
    
    solve_time = time.time() - start_time
    print(f"Solution time: {solve_time:.2f} seconds")
    
    # Validate solution / 验证解
    print("\nValidating variance evolution solution...")
    metrics = validate_ipfp_solution_fixed(solution, problem)
    
    print(f"Convergence: {'✓' if solution.final_error < config.tolerance else '✗'}")
    print(f"Final error: {solution.final_error:.2e}")
    print(f"Iterations: {solution.n_iterations}")
    
    # Check marginal constraints / 检查边际约束
    print("\nMarginal constraint satisfaction:")
    for k in range(3):
        l1_error = metrics[f"l1_marginal_{k}"]
        l2_error = metrics[f"l2_marginal_{k}"]
        mass_error = metrics[f"mass_error_{k}"]
        
        print(f"  Marginal {k} (var={[0.2, 0.5, 1.0][k]:.1f}): "
              f"L1={l1_error:.2e}, L2={l2_error:.2e}, mass={mass_error:.2e}")
    
    # Analyze variance evolution / 分析方差演化
    computed_variances = analyze_variance_evolution(solution, problem, grid_config)
    
    # Create visualizations / 创建可视化
    create_variance_visualization(grid_config, problem, solution, metrics, 
                                computed_variances, config)
    
    return solution, metrics, computed_variances


def create_variance_visualization(grid_config, problem, solution, metrics, 
                                computed_variances, config):
    """
    Create visualization for variance evolution experiment
    为方差演化实验创建可视化
    """
    grid = grid_config.points
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Variance Evolution Control / 方差演化控制', fontsize=16, y=0.95)
    
    # Plot 1: Overlaid density evolution / 图1：叠加的密度演化
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'red']
    labels = ['t=0 (σ²=0.2)', 't=0.5 (σ²=0.5)', 't=1 (σ²=1.0)']
    
    for k in range(3):
        # Target densities
        ax1.plot(grid, problem.observed_marginals[k], colors[k], 
                linewidth=3, alpha=0.8, label=f'Target {labels[k]}')
        # Computed densities  
        ax1.plot(grid, solution.path_densities[k], colors[k], 
                linestyle='--', linewidth=2, alpha=0.7, label=f'Computed {labels[k]}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Shape Evolution / 密度形状演化')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Variance evolution tracking / 图2：方差演化跟踪
    ax2 = axes[0, 1]
    
    target_variances = [0.2, 0.5, 1.0]
    times = [0.0, 0.5, 1.0]
    
    ax2.plot(times, target_variances, 'ko-', linewidth=3, markersize=8, 
            label='Target Variance', alpha=0.8)
    ax2.plot(times, computed_variances, 'rs--', linewidth=2, markersize=6,
            label='Computed Variance', alpha=0.8)
    
    # Linear interpolation (what we expect with good control)
    ax2.plot(times, target_variances, 'g:', linewidth=1, alpha=0.6, 
            label='Linear (Ideal)')
    
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Variance σ²')
    ax2.set_title('Variance Evolution Tracking / 方差演化跟踪')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Potentials and their gradients / 图3：势函数及其梯度
    ax3 = axes[0, 2]
    h = grid_config.spacing
    
    for k in range(3):
        phi = solution.potentials[k]
        grad_phi = jnp.gradient(phi, h)
        ax3.plot(grid, grad_phi, colors[k], linewidth=2,
                label=f'∇φ_{k}(x)')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('Potential Gradient')
    ax3.set_title('Transport Velocity Fields / 传输速度场')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence history / 图4：收敛历史
    ax4 = axes[1, 0]
    if solution.convergence_history:
        iterations = np.arange(len(solution.convergence_history)) * config.check_interval
        ax4.semilogy(iterations, solution.convergence_history, 'purple', linewidth=2)
        ax4.axhline(y=config.tolerance, color='red', linestyle='--', alpha=0.7,
                   label=f'Tolerance {config.tolerance:.0e}')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Error')
        ax4.set_title('Convergence History / 收敛历史')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Variance error analysis / 图5：方差误差分析
    ax5 = axes[1, 1]
    
    variance_errors = [abs(comp - target) for comp, target in 
                      zip(computed_variances, target_variances)]
    
    bars = ax5.bar(range(3), variance_errors, alpha=0.7, 
                  color=['blue', 'green', 'red'])
    ax5.set_xlabel('Time Point')
    ax5.set_ylabel('Variance Error |σ²_comp - σ²_target|')
    ax5.set_title('Variance Constraint Accuracy / 方差约束精度')
    ax5.set_xticks(range(3))
    ax5.set_xticklabels(['t=0', 't=0.5', 't=1'])
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Add error values on bars
    for i, (bar, error) in enumerate(zip(bars, variance_errors)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Transport cost by component / 图6：按组分的传输成本
    ax6 = axes[1, 2]
    
    # Compute transport costs
    transport_costs = []
    for k, phi in enumerate(solution.potentials):
        grad_phi = jnp.gradient(phi, h)
        kinetic = jax_trapz(grad_phi**2, dx=h)
        transport_costs.append(kinetic)
    
    bars = ax6.bar(range(3), transport_costs, alpha=0.7,
                  color=['blue', 'green', 'red'])
    ax6.set_xlabel('Potential Index')
    ax6.set_ylabel('Transport Cost (Kinetic Energy)')
    ax6.set_title('Transport Cost by Component / 各组分传输成本')
    ax6.set_xticks(range(3))
    ax6.set_xticklabels(['φ₀', 'φ₁', 'φ₂'])
    ax6.grid(True, alpha=0.3)
    
    # Add cost values on bars
    for bar, cost in zip(bars, transport_costs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot / 保存图片
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'variance_evolution_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVariance evolution visualization saved to: {output_path}")
    
    # Show if running interactively / 如果交互运行则显示
    if __name__ == "__main__":
        plt.show()


def main():
    """Main function to run the variance evolution experiment"""
    try:
        solution, metrics, computed_variances = run_variance_evolution_experiment()
        
        print("\n" + "=" * 80)
        print("VARIANCE EVOLUTION EXPERIMENT SUMMARY / 方差演化实验总结")
        print("=" * 80)
        
        # Check convergence / 检查收敛
        convergence_success = solution.final_error < 1e-6
        
        # Check marginal constraints / 检查边际约束
        constraint_success = all(metrics[f"l1_marginal_{k}"] < 1e-5 for k in range(3))
        
        # Check variance accuracy / 检查方差精度
        target_variances = [0.2, 0.5, 1.0]
        variance_errors = [abs(comp - target) for comp, target in 
                          zip(computed_variances, target_variances)]
        variance_success = all(error < 1e-3 for error in variance_errors)  # 0.1% accuracy
        
        overall_success = convergence_success and constraint_success and variance_success
        
        print(f"Convergence: {'✓' if convergence_success else '✗'} "
              f"(error = {solution.final_error:.2e})")
        print(f"Marginal constraints: {'✓' if constraint_success else '✗'}")
        print(f"Variance accuracy: {'✓' if variance_success else '✗'}")
        
        print(f"\nVariance tracking errors:")
        for k, error in enumerate(variance_errors):
            print(f"  t={k*0.5:.1f}: {error:.4f} ({error/target_variances[k]*100:.2f}%)")
        
        if overall_success:
            print("\n✅ VARIANCE EVOLUTION EXPERIMENT SUCCESSFUL!")
            print("✅ 方差演化实验成功!")
            print("- Multi-marginal bridge successfully controls density shape evolution")
            print("- Intermediate variance constraint enforced accurately")
            print("- This demonstrates shape control beyond just position transport")
        else:
            print("\n⚠️  VARIANCE EVOLUTION EXPERIMENT NEEDS ATTENTION")
            print("⚠️  方差演化实验需要注意")
            if not variance_success:
                print("- Variance constraints not met accurately enough")
            if not convergence_success:
                print(f"- Convergence issue: {solution.final_error:.2e}")
            if not constraint_success:
                print("- Marginal constraint errors too large")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ VARIANCE EVOLUTION EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)