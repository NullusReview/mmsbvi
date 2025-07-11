#!/usr/bin/env python3
"""
Three-marginal Gaussian Transport Experiment
三边际高斯传输实验

真正的多边际薛定谔桥：三个时间点的高斯分布之间的平滑传输
True multi-marginal Schrödinger bridge: smooth transport between Gaussians at three time points
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


def create_three_gaussians(grid_config: GridConfig1D) -> tuple:
    """
    Create three Gaussian marginals representing a transport path
    创建表示传输路径的三个高斯边际分布
    
    Mathematical significance: 
    - t=0: Left Gaussian (initial state)
    - t=0.5: Center Gaussian (intermediate constraint) 
    - t=1: Right Gaussian (final state)
    
    The intermediate constraint is what makes this a true multi-marginal problem!
    中间约束使这成为真正的多边际问题！
    """
    grid = grid_config.points
    h = grid_config.spacing
    
    # t=0: Initial Gaussian N(-1.5, 0.3)
    # t=0: 初始高斯 N(-1.5, 0.3)
    rho_0 = jnp.exp(-0.5 * (grid + 1.5)**2 / 0.3)
    rho_0 = rho_0 / jax_trapz(rho_0, dx=h)
    
    # t=0.5: Intermediate Gaussian N(0, 0.3) - This is the KEY constraint!
    # t=0.5: 中间高斯 N(0, 0.3) - 这是关键约束！
    rho_half = jnp.exp(-0.5 * grid**2 / 0.3)
    rho_half = rho_half / jax_trapz(rho_half, dx=h)
    
    # t=1: Final Gaussian N(1.5, 0.3)
    # t=1: 最终高斯 N(1.5, 0.3)
    rho_1 = jnp.exp(-0.5 * (grid - 1.5)**2 / 0.3)
    rho_1 = rho_1 / jax_trapz(rho_1, dx=h)
    
    return rho_0, rho_half, rho_1


def analyze_transport_smoothness(solution, grid_config):
    """
    Analyze the smoothness properties of the multi-marginal transport
    分析多边际传输的平滑性质
    """
    grid = grid_config.points
    h = grid_config.spacing
    
    print("\n" + "="*60)
    print("TRANSPORT SMOOTHNESS ANALYSIS / 传输平滑性分析")
    print("="*60)
    
    # Compute potential gradients (related to velocity field)
    # 计算势函数梯度（与速度场相关）
    gradients = []
    for i, phi in enumerate(solution.potentials):
        # Simple finite difference gradient
        grad = jnp.gradient(phi, h)
        gradients.append(grad)
        
        # Compute gradient magnitude statistics
        grad_mean = jnp.mean(jnp.abs(grad))
        grad_max = jnp.max(jnp.abs(grad))
        grad_std = jnp.std(grad)
        
        print(f"Potential φ_{i}: |∇φ| mean={grad_mean:.4f}, max={grad_max:.4f}, std={grad_std:.4f}")
    
    # Compute total variation of potentials (smoothness measure)
    # 计算势函数的全变分（平滑性度量）
    total_variations = []
    for i, phi in enumerate(solution.potentials):
        tv = jnp.sum(jnp.abs(jnp.diff(phi))) * h
        total_variations.append(tv)
        print(f"Total variation TV(φ_{i}) = {tv:.6f}")
    
    return gradients, total_variations


def compute_transport_cost(solution, problem):
    """
    Compute the transport cost for the multi-marginal bridge
    计算多边际桥的传输成本
    """
    # Simplified cost based on potential energies
    # 基于势函数能量的简化成本
    h = problem.grid.spacing
    
    total_cost = 0.0
    for i, phi in enumerate(solution.potentials):
        # Kinetic energy contribution: ∫ |∇φ|² dx
        grad_phi = jnp.gradient(phi, h)
        kinetic = jax_trapz(grad_phi**2, dx=h)
        
        # Potential energy contribution: ∫ φ² dx  
        potential = jax_trapz(phi**2, dx=h)
        
        marginal_cost = 0.5 * (kinetic + 0.1 * potential)  # Weighted combination
        total_cost += marginal_cost
        
        print(f"Marginal {i} cost: kinetic={kinetic:.6f}, potential={potential:.6f}")
    
    return total_cost


def run_three_marginal_experiment():
    """
    Run the three-marginal Gaussian transport experiment  
    运行三边际高斯传输实验
    """
    print("=" * 80)
    print("THREE-MARGINAL GAUSSIAN TRANSPORT EXPERIMENT")
    print("三边际高斯传输实验")
    print("=" * 80)
    print("This demonstrates TRUE multi-marginal Schrödinger bridge!")
    print("这演示了真正的多边际薛定谔桥!")
    
    # Setup grid / 设置网格
    grid_config = GridConfig1D.create(n_points=60, bounds=(-3.0, 3.0))
    print(f"\nGrid: {grid_config.n_points} points, spacing = {grid_config.spacing:.4f}")
    
    # Create three marginals / 创建三个边际分布
    rho_0, rho_half, rho_1 = create_three_gaussians(grid_config)
    
    print(f"t=0.0: peak at x = {grid_config.points[jnp.argmax(rho_0)]:.2f}")
    print(f"t=0.5: peak at x = {grid_config.points[jnp.argmax(rho_half)]:.2f} [INTERMEDIATE CONSTRAINT]")
    print(f"t=1.0: peak at x = {grid_config.points[jnp.argmax(rho_1)]:.2f}")
    
    # OU process parameters / OU过程参数
    ou_params = OUProcessParams(
        mean_reversion=1.0,   # Stronger mean reversion for multi-marginal
        diffusion=0.8,        # Moderate diffusion
        equilibrium_mean=0.0  # Centered equilibrium
    )
    print(f"\nOU parameters: θ={ou_params.mean_reversion}, σ={ou_params.diffusion}")
    
    # Create multi-marginal MMSB problem / 创建多边际MMSB问题
    problem = MMSBProblem(
        observation_times=jnp.array([0.0, 0.5, 1.0]),  # Three time points!
        observed_marginals=[rho_0, rho_half, rho_1],   # Three constraints!
        ou_params=ou_params,
        grid=grid_config
    )
    
    print(f"Multi-marginal problem: K = {problem.n_marginals} marginals")
    print(f"Time intervals: {problem.time_intervals}")
    
    # IPFP algorithm configuration / IPFP算法配置
    config = IPFPConfig(
        max_iterations=800,    # More iterations for multi-marginal
        tolerance=1e-6,        # Reasonable precision for 3-marginal
        check_interval=20,     # Check less frequently
        verbose=True           # Show progress
    )
    
    print(f"\nIPFP config: max_iter={config.max_iterations}, tol={config.tolerance:.0e}")
    
    # Solve the multi-marginal problem / 求解多边际问题
    print("\nSolving multi-marginal MMSB problem...")
    start_time = time.time()
    
    solution = solve_mmsb_ipfp_1d_fixed(problem, config)
    
    solve_time = time.time() - start_time
    print(f"Solution time: {solve_time:.2f} seconds")
    
    # Validate solution / 验证解
    print("\nValidating multi-marginal solution...")
    metrics = validate_ipfp_solution_fixed(solution, problem)
    
    print(f"Convergence: {'✓' if solution.final_error < config.tolerance else '✗'}")
    print(f"Final error: {solution.final_error:.2e}")
    print(f"Iterations: {solution.n_iterations}")
    
    # Check all marginal constraints / 检查所有边际约束
    print("\nMarginal constraint satisfaction:")
    for k in range(3):
        l1_error = metrics[f"l1_marginal_{k}"]
        l2_error = metrics[f"l2_marginal_{k}"]
        mass_error = metrics[f"mass_error_{k}"]
        
        print(f"  Marginal {k} (t={problem.observation_times[k]:.1f}): "
              f"L1={l1_error:.2e}, L2={l2_error:.2e}, mass_error={mass_error:.2e}")
    
    # Analyze transport properties / 分析传输性质
    gradients, total_variations = analyze_transport_smoothness(solution, grid_config)
    
    # Compute transport cost / 计算传输成本
    print("\nTransport cost analysis:")
    total_cost = compute_transport_cost(solution, problem)
    print(f"Total transport cost: {total_cost:.6f}")
    
    # Create visualizations / 创建可视化
    create_multi_marginal_visualization(grid_config, problem, solution, metrics, config)
    
    return solution, metrics


def create_multi_marginal_visualization(grid_config, problem, solution, metrics, config=None):
    """
    Create comprehensive visualization for multi-marginal transport
    为多边际传输创建全面的可视化
    """
    grid = grid_config.points
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Three-marginal Gaussian Transport / 三边际高斯传输', fontsize=16)
    
    # Plot 1: All marginal densities / 图1：所有边际密度
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'red']
    time_labels = ['t=0.0', 't=0.5', 't=1.0']
    
    for k in range(3):
        ax1.plot(grid, problem.observed_marginals[k], colors[k], linewidth=3, 
                label=f'Target {time_labels[k]}', alpha=0.8)
        ax1.plot(grid, solution.path_densities[k], colors[k], linestyle='--', 
                linewidth=2, label=f'Computed {time_labels[k]}', alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('Multi-marginal Densities / 多边际密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All potentials / 图2：所有势函数
    ax2 = axes[0, 1]
    for k in range(3):
        ax2.plot(grid, solution.potentials[k], colors[k], linewidth=2, 
                label=f'φ_{k}(x) at {time_labels[k]}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Potential')
    ax2.set_title('All Potentials / 所有势函数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Transport "velocity" field (gradient of potentials)
    # 图3：传输"速度"场（势函数梯度）
    ax3 = axes[0, 2]
    h = grid_config.spacing
    for k in range(3):
        velocity = -jnp.gradient(solution.potentials[k], h)  # v = -∇φ
        ax3.plot(grid, velocity, colors[k], linewidth=2, 
                label=f'v_{k}(x) = -∇φ_{k}')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Transport Velocity Fields / 传输速度场')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence history / 图4：收敛历史
    ax4 = axes[1, 0]
    if solution.convergence_history and config is not None:
        iterations = np.arange(len(solution.convergence_history)) * config.check_interval
        ax4.semilogy(iterations, solution.convergence_history, 'purple', linewidth=2)
        ax4.axhline(y=config.tolerance, color='red', linestyle='--', alpha=0.7, 
                   label=f'Tolerance {config.tolerance:.0e}')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Error')
        ax4.set_title('Convergence History / 收敛历史')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    elif solution.convergence_history:
        # Fallback if config not provided
        ax4.semilogy(solution.convergence_history, 'purple', linewidth=2)
        ax4.set_xlabel('Check Interval')
        ax4.set_ylabel('Error')
        ax4.set_title('Convergence History / 收敛历史')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No convergence history\navailable', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Convergence History / 收敛历史')
    
    # Plot 5: Error comparison across marginals / 图5：各边际的误差比较
    ax5 = axes[1, 1]
    
    marginal_names = ['ρ₀ (t=0)', 'ρ₀.₅ (t=0.5)', 'ρ₁ (t=1)']
    l1_errors = [metrics[f"l1_marginal_{k}"] for k in range(3)]
    l2_errors = [metrics[f"l2_marginal_{k}"] for k in range(3)]
    mass_errors = [metrics[f"mass_error_{k}"] for k in range(3)]
    
    x_pos = np.arange(3)
    width = 0.25
    
    ax5.bar(x_pos - width, l1_errors, width, label='L1 Error', alpha=0.8, color='skyblue')
    ax5.bar(x_pos, l2_errors, width, label='L2 Error', alpha=0.8, color='lightgreen')
    ax5.bar(x_pos + width, mass_errors, width, label='Mass Error', alpha=0.8, color='salmon')
    
    ax5.set_xlabel('Marginal')
    ax5.set_ylabel('Error')
    ax5.set_title('Error Metrics by Marginal / 各边际误差指标')
    ax5.set_yscale('log')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(marginal_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Density evolution animation snapshot / 图6：密度演化动画快照
    ax6 = axes[1, 2]
    
    # Create intermediate densities by interpolation (for visualization)
    # 通过插值创建中间密度（用于可视化）
    n_snapshots = 5
    times = np.linspace(0, 1, n_snapshots)
    colors_interp = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    for i, t in enumerate(times):
        if t <= 0.5:
            # Interpolate between ρ₀ and ρ₀.₅
            alpha = t / 0.5
            density_interp = (1 - alpha) * solution.path_densities[0] + alpha * solution.path_densities[1]
        else:
            # Interpolate between ρ₀.₅ and ρ₁  
            alpha = (t - 0.5) / 0.5
            density_interp = (1 - alpha) * solution.path_densities[1] + alpha * solution.path_densities[2]
        
        ax6.plot(grid, density_interp, color=colors_interp[i], linewidth=2, 
                alpha=0.8, label=f't={t:.1f}')
    
    ax6.set_xlabel('x')
    ax6.set_ylabel('Density')
    ax6.set_title('Transport Evolution / 传输演化')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot / 保存图片
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'three_marginal_gaussian_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive visualization saved to: {output_path}")
    
    # Show if running interactively / 如果交互运行则显示
    if __name__ == "__main__":
        plt.show()


def main():
    """Main function to run the three-marginal experiment"""
    try:
        solution, metrics = run_three_marginal_experiment()
        
        print("\n" + "=" * 80)
        print("THREE-MARGINAL EXPERIMENT SUMMARY / 三边际实验总结")
        print("=" * 80)
        
        # Check if experiment was successful / 检查实验是否成功
        convergence_success = solution.final_error < 1e-5  # Looser for 3-marginal
        constraint_success = all(metrics[f"l1_marginal_{k}"] < 1e-4 for k in range(3))
        
        success = convergence_success and constraint_success
        
        if success:
            print("✅ THREE-MARGINAL EXPERIMENT SUCCESSFUL!")
            print("✅ 三边际实验成功!")
            print("- Multi-marginal IPFP converged successfully")
            print("- All three marginal constraints satisfied")
            print("- Intermediate constraint at t=0.5 enforced")
            print("- This demonstrates true multi-marginal smoothing!")
        else:
            print("⚠️  THREE-MARGINAL EXPERIMENT NEEDS ATTENTION")
            print("⚠️  三边际实验需要注意")
            if not convergence_success:
                print(f"- IPFP convergence issue: final_error = {solution.final_error:.2e}")
            if not constraint_success:
                print("- Some marginal constraints not satisfied")
                for k in range(3):
                    error = metrics[f"l1_marginal_{k}"]
                    if error >= 1e-4:
                        print(f"  Marginal {k}: L1_error = {error:.2e}")
            
        return success
        
    except Exception as e:
        print(f"❌ THREE-MARGINAL EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)