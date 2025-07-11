#!/usr/bin/env python3
"""
Two-marginal Gaussian Transport Experiment
两边际高斯传输实验

最简单的MMSB验证：两个分离的高斯分布之间的传输
Simplest MMSB validation: transport between two separated Gaussians
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


def create_separated_gaussians(grid_config: GridConfig1D) -> tuple:
    """
    Create two well-separated Gaussian marginals
    创建两个分离良好的高斯边际分布
    """
    grid = grid_config.points
    h = grid_config.spacing
    
    # Left Gaussian: N(-1.5, 0.3)
    # 左高斯：N(-1.5, 0.3)
    rho_0 = jnp.exp(-0.5 * (grid + 1.5)**2 / 0.3)
    rho_0 = rho_0 / jax_trapz(rho_0, dx=h)
    
    # Right Gaussian: N(1.5, 0.3)  
    # 右高斯：N(1.5, 0.3)
    rho_1 = jnp.exp(-0.5 * (grid - 1.5)**2 / 0.3)
    rho_1 = rho_1 / jax_trapz(rho_1, dx=h)
    
    return rho_0, rho_1


def run_gaussian_transport_experiment():
    """
    Run the two-marginal Gaussian transport experiment
    运行两边际高斯传输实验
    """
    print("=" * 80)
    print("TWO-MARGINAL GAUSSIAN TRANSPORT EXPERIMENT")
    print("两边际高斯传输实验")
    print("=" * 80)
    
    # Setup grid / 设置网格
    grid_config = GridConfig1D.create(n_points=50, bounds=(-3.0, 3.0))
    print(f"Grid: {grid_config.n_points} points, spacing = {grid_config.spacing:.4f}")
    
    # Create marginals / 创建边际分布
    rho_0, rho_1 = create_separated_gaussians(grid_config)
    
    print(f"Initial marginal (t=0): peak at {grid_config.points[jnp.argmax(rho_0)]:.2f}")
    print(f"Final marginal (t=1):   peak at {grid_config.points[jnp.argmax(rho_1)]:.2f}")
    
    # OU process parameters / OU过程参数
    ou_params = OUProcessParams(
        mean_reversion=0.5,  # Moderate mean reversion / 中等均值回复
        diffusion=1.0,       # Standard diffusion / 标准扩散
        equilibrium_mean=0.0 # Centered equilibrium / 中心平衡态
    )
    print(f"OU parameters: θ={ou_params.mean_reversion}, σ={ou_params.diffusion}")
    
    # Create MMSB problem / 创建MMSB问题
    problem = MMSBProblem(
        observation_times=jnp.array([0.0, 1.0]),
        observed_marginals=[rho_0, rho_1],
        ou_params=ou_params,
        grid=grid_config
    )
    
    # IPFP algorithm configuration / IPFP算法配置
    config = IPFPConfig(
        max_iterations=500,    # More iterations for safety / 更多迭代以确保安全
        tolerance=1e-8,        # High precision / 高精度
        check_interval=10,     # Check convergence frequently / 频繁检查收敛
        verbose=True           # Show progress / 显示进度
    )
    
    print(f"IPFP config: max_iter={config.max_iterations}, tol={config.tolerance:.0e}")
    
    # Solve the problem / 求解问题
    print("\nSolving MMSB problem...")
    start_time = time.time()
    
    solution = solve_mmsb_ipfp_1d_fixed(problem, config)
    
    solve_time = time.time() - start_time
    print(f"Solution time: {solve_time:.2f} seconds")
    
    # Validate solution / 验证解
    print("\nValidating solution...")
    metrics = validate_ipfp_solution_fixed(solution, problem)
    
    print(f"Convergence: {'✓' if solution.final_error < config.tolerance else '✗'}")
    print(f"Final error: {solution.final_error:.2e}")
    print(f"Iterations: {solution.n_iterations}")
    
    # Check marginal constraints / 检查边际约束
    for k in range(2):
        l1_error = metrics[f"l1_marginal_{k}"]
        l2_error = metrics[f"l2_marginal_{k}"]
        mass_error = metrics[f"mass_error_{k}"]
        
        print(f"Marginal {k}: L1={l1_error:.2e}, L2={l2_error:.2e}, mass_error={mass_error:.2e}")
    
    # Analyze transport cost / 分析传输成本
    h = grid_config.spacing
    total_cost = 0.0
    for phi in solution.potentials:
        cost_contribution = jnp.sum(phi**2) * h  # Simplified cost metric
        total_cost += cost_contribution
    
    print(f"Transport cost (simplified): {total_cost:.4f}")
    
    # Create visualizations / 创建可视化
    create_visualization(grid_config, problem, solution, metrics)
    
    return solution, metrics


def create_visualization(grid_config, problem, solution, metrics):
    """
    Create streamlined plots with unified color scheme and Excel export
    创建具有统一配色方案和Excel导出的精简图表
    """
    grid = grid_config.points
    
    # Sophisticated aesthetic color palette / 精致美学配色方案
    colors = {
        'cream': '#fbf1c2',      # 温暖奶油黄 - 高亮和背景装饰
        'teal': '#566970',       # 深蓝绿色 - 主要元素和文本
        'coral': '#e67a7f',      # 珊瑚粉红 - 对比色和重点
        'wine': '#9d3151',       # 深酒红色 - 强调色和标题
        'mint': '#87baa7',       # 薄荷绿 - 辅助元素和填充
        'white': '#ffffff'       # 纯白背景
    }
    
    # Set serif font globally / 全局设置衬线字体
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 200,
        'savefig.dpi': 400,
        'text.usetex': False,
        'axes.unicode_minus': False
    })
    
    # Create streamlined academic layout / 创建精简的学术布局
    fig = plt.figure(figsize=(15, 10), facecolor=colors['white'])
    fig.suptitle('Two-marginal Gaussian Transport', fontsize=20, fontweight='bold', 
                 color=colors['wine'])
    
    # Use 2x2 layout for cleaner design / 使用2x2布局以获得更清洁的设计
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1],
                         hspace=0.3, wspace=0.3)
    
    # Plot 1: Sophisticated marginal densities with artistic styling (spanning top row)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create elegant overlapping design with all 5 colors
    # Base layers with mint for subtle foundation
    ax1.fill_between(grid, 0, problem.observed_marginals[0], alpha=0.2, 
                     color=colors['mint'], zorder=1)
    ax1.fill_between(grid, 0, problem.observed_marginals[1], alpha=0.2, 
                     color=colors['mint'], zorder=1)
    
    # Primary curves with sophisticated color mapping
    ax1.plot(grid, problem.observed_marginals[0], '-', linewidth=4, 
             color=colors['teal'], label='Target ρ₀(x)', alpha=0.9, zorder=5)
    ax1.plot(grid, problem.observed_marginals[1], '-', linewidth=4, 
             color=colors['wine'], label='Target ρ₁(x)', alpha=0.9, zorder=5)
    
    # Computed densities with complementary styling
    ax1.plot(grid, solution.path_densities[0], '--', linewidth=3, 
             color=colors['coral'], alpha=0.8, label='Computed ρ₀', zorder=4)
    ax1.plot(grid, solution.path_densities[1], '--', linewidth=3, 
             color=colors['coral'], alpha=0.8, label='Computed ρ₁', zorder=4)
    
    # Add cream highlights at peak regions for artistic flair
    peak0_idx = jnp.argmax(problem.observed_marginals[0])
    peak1_idx = jnp.argmax(problem.observed_marginals[1])
    
    # Highlight around peaks with cream
    for peak_idx, density in [(peak0_idx, problem.observed_marginals[0]), 
                             (peak1_idx, problem.observed_marginals[1])]:
        peak_range = slice(max(0, peak_idx-5), min(len(grid), peak_idx+6))
        ax1.fill_between(grid[peak_range], 0, density[peak_range], 
                        alpha=0.4, color=colors['cream'], zorder=3)
    
    # Artistic styling
    ax1.set_xlabel('State x', fontsize=12, color=colors['teal'])
    ax1.set_ylabel('Probability Density', fontsize=12, color=colors['teal'])
    ax1.set_title('A. Marginal Densities Evolution', fontsize=14, fontweight='bold', color=colors['wine'])
    
    # Elegant legend with sophisticated styling
    legend = ax1.legend(frameon=True, fancybox=True, shadow=True, 
                       facecolor=colors['cream'], edgecolor=colors['teal'], 
                       framealpha=0.95, loc='upper center', ncol=2)
    legend.get_frame().set_linewidth(2)
    
    ax1.grid(True, alpha=0.2, color=colors['teal'], linestyle=':', linewidth=1)
    ax1.set_facecolor(colors['white'])
    
    # Add subtle spines styling
    for spine in ax1.spines.values():
        spine.set_color(colors['teal'])
        spine.set_linewidth(1.5)
    
    # Plot 2: Sophisticated convergence visualization with artistic gradient
    ax2 = fig.add_subplot(gs[1, 0])
    if solution.convergence_history:
        iterations = range(len(solution.convergence_history))
        
        # Create artistic gradient fill with multiple colors
        ax2.fill_between(iterations, 1e-12, solution.convergence_history, 
                        alpha=0.3, color=colors['mint'], zorder=1)
        ax2.fill_between(iterations, 1e-10, solution.convergence_history, 
                        alpha=0.2, color=colors['cream'], zorder=2)
        
        # Main convergence curve with sophisticated styling
        ax2.semilogy(solution.convergence_history, '-', linewidth=4, 
                    color=colors['wine'], label='IPFP Convergence', alpha=0.9, zorder=5)
        
        # Add convergence markers for visual interest
        if len(solution.convergence_history) > 1:
            marker_points = range(0, len(solution.convergence_history), 
                                max(1, len(solution.convergence_history)//5))
            ax2.scatter([iterations[i] for i in marker_points], 
                       [solution.convergence_history[i] for i in marker_points],
                       s=80, c=colors['coral'], edgecolors=colors['wine'], 
                       linewidth=2, alpha=0.8, zorder=6)
        
        # Elegant target tolerance line
        tolerance = 1e-8
        ax2.axhline(y=tolerance, color=colors['teal'], linestyle='--', 
                   linewidth=3, alpha=0.8, label='Target Tolerance', zorder=4)
        
        # Add subtle success region if converged
        if solution.convergence_history[-1] < tolerance:
            ax2.axhspan(1e-12, tolerance, alpha=0.1, color=colors['mint'], zorder=0)
        
        ax2.set_xlabel('Iteration (×10)', fontsize=12, color=colors['teal'])
        ax2.set_ylabel('Error (log scale)', fontsize=12, color=colors['teal'])
        ax2.set_title('B. Convergence Dynamics', fontsize=14, fontweight='bold', color=colors['wine'])
        
        # Sophisticated legend
        legend = ax2.legend(frameon=True, fancybox=True, shadow=True, 
                           facecolor=colors['cream'], edgecolor=colors['wine'], 
                           framealpha=0.95, loc='best')
        legend.get_frame().set_linewidth(2)
        
        ax2.grid(True, alpha=0.2, color=colors['teal'], linestyle=':', linewidth=1)
        ax2.set_facecolor(colors['white'])
        
        # Artistic spines
        for spine in ax2.spines.values():
            spine.set_color(colors['teal'])
            spine.set_linewidth(1.5)
    
    # Plot 3: Sophisticated validation dashboard with artistic design
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Extract error metrics
    l1_errors = [metrics[f"l1_marginal_{k}"] for k in range(2)]
    l2_errors = [metrics[f"l2_marginal_{k}"] for k in range(2)]
    mass_errors = [metrics[f"mass_error_{k}"] for k in range(2)]
    
    # Create sophisticated error display with artistic grouping
    error_groups = [
        ("L1 Accuracy", [("ρ₀", l1_errors[0]), ("ρ₁", l1_errors[1])]),
        ("L2 Precision", [("ρ₀", l2_errors[0]), ("ρ₁", l2_errors[1])]),
        ("Mass Conservation", [("ρ₀", mass_errors[0]), ("ρ₁", mass_errors[1])])
    ]
    
    # Artistic metric cards with sophisticated design
    y_positions = [0.8, 0.5, 0.2]
    group_colors = [colors['teal'], colors['wine'], colors['coral']]
    
    for i, (group_name, metrics_pair) in enumerate(error_groups):
        # Create elegant group header
        from matplotlib.patches import FancyBboxPatch
        
        # Group header with artistic styling
        header_box = FancyBboxPatch((0.05, y_positions[i]+0.08), 0.9, 0.08,
                                   boxstyle="round,pad=0.02",
                                   facecolor=group_colors[i], alpha=0.15,
                                   edgecolor=group_colors[i], linewidth=2,
                                   transform=ax3.transAxes)
        ax3.add_patch(header_box)
        
        ax3.text(0.5, y_positions[i]+0.12, group_name, fontsize=11, fontweight='bold',
                color=group_colors[i], ha='center', va='center', transform=ax3.transAxes)
        
        # Display individual metrics with sophisticated styling
        for j, (marginal, value) in enumerate(metrics_pair):
            x_offset = 0.15 + j * 0.35
            passed = float(value) < 1e-5
            
            # Status indicator with artistic design
            indicator_color = colors['mint'] if passed else colors['coral']
            status_symbol = '✓' if passed else '✗'
            
            # Metric card
            metric_card = FancyBboxPatch((x_offset-0.1, y_positions[i]-0.04), 0.2, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor=colors['cream'], alpha=0.6,
                                        edgecolor=group_colors[i], linewidth=1,
                                        transform=ax3.transAxes)
            ax3.add_patch(metric_card)
            
            # Status circle with depth
            circle = plt.Circle((x_offset, y_positions[i]), 0.02, 
                              color=indicator_color, transform=ax3.transAxes, zorder=10)
            ax3.add_patch(circle)
            
            ax3.text(x_offset, y_positions[i], status_symbol, fontsize=8, fontweight='bold',
                    color=colors['white'], ha='center', va='center', 
                    transform=ax3.transAxes, zorder=11)
            
            # Metric text with sophisticated typography
            ax3.text(x_offset, y_positions[i]-0.025, f"{marginal}: {value:.1e}", 
                    fontsize=8, color=colors['teal'], ha='center', va='center',
                    transform=ax3.transAxes, style='italic')
    
    # Overall validation status with artistic flair
    all_passed = all(float(v) < 1e-5 for error_list in [l1_errors, l2_errors, mass_errors] 
                    for v in error_list)
    
    # Artistic overall status
    status_color = colors['mint'] if all_passed else colors['wine']
    status_text = "VALIDATION\nSUCCESS" if all_passed else "VALIDATION\nPENDING"
    
    status_box = FancyBboxPatch((0.2, 0.02), 0.6, 0.12,
                               boxstyle="round,pad=0.02",
                               facecolor=status_color, alpha=0.2,
                               edgecolor=status_color, linewidth=2,
                               transform=ax3.transAxes)
    ax3.add_patch(status_box)
    
    ax3.text(0.5, 0.08, status_text, fontsize=10, fontweight='bold',
            color=status_color, ha='center', va='center', 
            transform=ax3.transAxes, linespacing=1.2)
    
    ax3.set_title('C. Validation Dashboard', fontsize=14, fontweight='bold', 
                 color=colors['wine'], pad=25)
    
    plt.tight_layout()
    
    # Save plot with enhanced quality
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'two_marginal_gaussian_results.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor=colors['white'])
    print(f"\nStreamlined visualization saved to: {output_path}")
    
    # Export detailed data to Excel
    excel_path = os.path.join(results_dir, 'two_marginal_gaussian_data.xlsx')
    export_gaussian_transport_data_to_excel(grid_config, problem, solution, metrics, excel_path)
    
    # Show if running interactively
    if __name__ == "__main__":
        plt.show()


def export_gaussian_transport_data_to_excel(grid_config, problem, solution, metrics, excel_path):
    """Export detailed Gaussian transport experiment data to Excel."""
    try:
        import pandas as pd
        
        grid = grid_config.points
        
        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Grid and Densities
            density_data = {
                'Grid_Points': [float(x) for x in grid],
                'Target_Rho0': [float(d) for d in problem.observed_marginals[0]],
                'Target_Rho1': [float(d) for d in problem.observed_marginals[1]],
                'Computed_Rho0': [float(d) for d in solution.path_densities[0]],
                'Computed_Rho1': [float(d) for d in solution.path_densities[1]]
            }
            if len(solution.potentials) >= 2:
                density_data['Potential_Phi0'] = [float(p) for p in solution.potentials[0]]
                density_data['Potential_Phi1'] = [float(p) for p in solution.potentials[1]]
            
            density_df = pd.DataFrame(density_data)
            density_df.to_excel(writer, sheet_name='Grid_and_Densities', index=False)
            
            # Sheet 2: Error Metrics
            l1_errors = [metrics[f"l1_marginal_{k}"] for k in range(2)]
            l2_errors = [metrics[f"l2_marginal_{k}"] for k in range(2)]
            mass_errors = [metrics[f"mass_error_{k}"] for k in range(2)]
            
            error_data = {
                'Marginal': ['Rho0', 'Rho1'],
                'L1_Error': [float(e) for e in l1_errors],
                'L2_Error': [float(e) for e in l2_errors],
                'Mass_Error': [float(e) for e in mass_errors]
            }
            error_df = pd.DataFrame(error_data)
            error_df.to_excel(writer, sheet_name='Error_Metrics', index=False)
            
            # Sheet 3: Convergence History
            if solution.convergence_history:
                conv_data = {
                    'Iteration': list(range(len(solution.convergence_history))),
                    'Error': [float(e) for e in solution.convergence_history]
                }
                conv_df = pd.DataFrame(conv_data)
                conv_df.to_excel(writer, sheet_name='Convergence_History', index=False)
            
            # Sheet 4: Experiment Configuration
            config_data = {
                'Parameter': [
                    'Grid_Points', 'Grid_Lower_Bound', 'Grid_Upper_Bound', 'Grid_Spacing',
                    'OU_Mean_Reversion', 'OU_Diffusion', 'OU_Equilibrium_Mean',
                    'IPFP_Max_Iterations', 'IPFP_Tolerance', 'IPFP_Check_Interval',
                    'Final_Error', 'Total_Iterations', 'Converged'
                ],
                'Value': [
                    grid_config.n_points, float(grid_config.bounds[0]), float(grid_config.bounds[1]), 
                    float(grid_config.spacing), float(problem.ou_params.mean_reversion),
                    float(problem.ou_params.diffusion), float(problem.ou_params.equilibrium_mean),
                    500, 1e-8, 10,  # IPFP config values
                    float(solution.final_error), solution.n_iterations,
                    'Yes' if solution.final_error < 1e-8 else 'No'
                ]
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Experiment_Config', index=False)
            
            # Sheet 5: Summary Statistics
            summary_data = {
                'Metric': [
                    'Max_L1_Error', 'Max_L2_Error', 'Max_Mass_Error',
                    'Mean_L1_Error', 'Mean_L2_Error', 'Mean_Mass_Error',
                    'Rho0_Peak_Position', 'Rho1_Peak_Position',
                    'Rho0_Peak_Value', 'Rho1_Peak_Value'
                ],
                'Value': [
                    float(max(l1_errors)), float(max(l2_errors)), float(max(mass_errors)),
                    float(np.mean(l1_errors)), float(np.mean(l2_errors)), float(np.mean(mass_errors)),
                    float(grid[jnp.argmax(problem.observed_marginals[0])]),
                    float(grid[jnp.argmax(problem.observed_marginals[1])]),
                    float(jnp.max(problem.observed_marginals[0])),
                    float(jnp.max(problem.observed_marginals[1]))
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        print(f"Detailed experiment data exported to: {excel_path}")
        
    except ImportError:
        print("Warning: pandas not available for Excel export. Skipping data export.")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")


def main():
    """Main function to run the experiment"""
    try:
        solution, metrics = run_gaussian_transport_experiment()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY / 实验总结")
        print("=" * 80)
        
        # Check if experiment was successful / 检查实验是否成功
        success = (
            solution.final_error < 1e-6 and
            all(metrics[f"l1_marginal_{k}"] < 1e-5 for k in range(2))
        )
        
        if success:
            print("✅ EXPERIMENT SUCCESSFUL!")
            print("✅ 实验成功!")
            print("- IPFP converged to required tolerance")
            print("- Marginal constraints satisfied")
            print("- Ready for more complex experiments")
        else:
            print("⚠️  EXPERIMENT NEEDS ATTENTION")
            print("⚠️  实验需要注意")
            print("- Check convergence criteria or increase iterations")
            
        return success
        
    except Exception as e:
        print(f"❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)