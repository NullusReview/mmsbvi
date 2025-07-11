#!/usr/bin/env python3
"""
Geometric Limits Visualization for MMSB-VI
MMSB-VI的几何极限可视化
===================================================

Publication-quality visualization for geometric limits validation results - Individual plots.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import logging
import sys

# Setup project paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Import validation result - simplified
try:
    from theoretical_verification.core_experiments.geometric_limits_validation import ValidationResult
except ImportError:
    # Fallback mock
    from dataclasses import dataclass
    from typing import List, Any, Optional
    
    @dataclass
    class ValidationResult:
        sigma_values: List[float]
        distances_mean: List[float]
        confidence_intervals: List[tuple] 
        p_values: List[float]
        effect_sizes: List[float]
        numerical_stability: List[Any]
        convergence_analysis: Optional[dict]
        validation_passed: bool = True

# Set Times New Roman font and publication aesthetics
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'text.usetex': False,
    'axes.unicode_minus': False,
    'figure.constrained_layout.use': True
})

# Simple color scheme for validation experiments
COLORS = {
    'primary': '#9D110E',        # Deep red
    'secondary': '#000000',      # Black
    'background': '#FFFFFF',     # White
    'grid': '#E5E5E5'            # Light gray for grid
}


class GeometricLimitsVisualizer:
    """
    Publication-quality visualizer for geometric limits validation results.
    几何极限验证结果的发表质量可视化器。
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_individual_validation_figures(self, 
                                             sigma_inf_results: ValidationResult,
                                             sigma_zero_results: Optional[ValidationResult] = None,
                                             transition_results: Optional[ValidationResult] = None,
                                             save_dir: str = "../results/geometric_limits/"):
        """
        Generate individual validation figures - one per plot.
        生成单独的验证图表 - 每个图表一个文件。
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        self._plot_individual_sigma_infinity_convergence(sigma_inf_results, save_path / "sigma_infinity_convergence.png")
        self._plot_individual_statistical_significance(sigma_inf_results, save_path / "statistical_significance.png")
        self._plot_individual_effect_sizes(sigma_inf_results, save_path / "effect_sizes.png")
        self._plot_individual_numerical_stability(sigma_inf_results, save_path / "numerical_stability.png")
        self._plot_individual_convergence_rate_fit(sigma_inf_results, save_path / "convergence_rate_fit.png")
        self._plot_individual_validation_summary(sigma_inf_results, sigma_zero_results, transition_results, save_path / "validation_summary.png")
        
        self.logger.info(f"Individual validation figures saved to: {save_dir}")
        
    def _plot_individual_sigma_infinity_convergence(self, results: ValidationResult, save_path: Path):
        """Plot individual σ→∞ convergence rate analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        sigma_vals = results.sigma_values
        means = results.distances_mean
        cis = results.confidence_intervals
        
        # Filter valid data
        valid_mask = ~jnp.isnan(jnp.array(means))
        sigma_valid = sigma_vals[valid_mask]
        means_valid = jnp.array(means)[valid_mask]
        ci_lower = jnp.array([ci[0] for ci in cis])[valid_mask]
        ci_upper = jnp.array([ci[1] for ci in cis])[valid_mask]
        
        # Plot empirical data
        ax.loglog(sigma_valid, means_valid, 'o-', color=COLORS['secondary'], 
                 linewidth=3, markersize=10, label='Empirical Distance')
        ax.fill_between(sigma_valid, ci_lower, ci_upper, 
                       color=COLORS['primary'], alpha=0.3, label='99% Confidence Interval')
        
        # Theoretical O(1/σ) line
        if len(means_valid) > 0:
            theoretical_line = means_valid[0] * sigma_valid[0] / sigma_valid
            ax.loglog(sigma_valid, theoretical_line, '--', color=COLORS['secondary'], 
                     linewidth=2, alpha=0.7, label='Theoretical O(1/σ)')
        
        ax.set_xlabel('σ (Diffusion Parameter)', fontsize=14)
        ax.set_ylabel('Distance to Mixture Geodesic', fontsize=14)
        ax.set_title('σ→∞ Convergence Rate Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_statistical_significance(self, results: ValidationResult, save_path: Path):
        """Plot individual statistical significance analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        sigma_vals = results.sigma_values
        p_vals = jnp.array(results.p_values)
        
        # Filter valid data
        valid_mask = ~jnp.isnan(p_vals) & (p_vals > 0)
        sigma_valid = sigma_vals[valid_mask]
        p_valid = p_vals[valid_mask]
        
        if len(p_valid) > 0:
            ax.semilogy(sigma_valid, p_valid, 'o-', color=COLORS['secondary'], 
                       linewidth=3, markersize=10, label='p-values')
            ax.axhline(y=0.001, color=COLORS['primary'], linestyle='--', 
                      linewidth=2, label='Significance Threshold (α=0.001)')
        
        ax.set_xlabel('σ (Diffusion Parameter)', fontsize=14)
        ax.set_ylabel('p-value', fontsize=14)
        ax.set_title('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_effect_sizes(self, results: ValidationResult, save_path: Path):
        """Plot individual effect size analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        sigma_vals = results.sigma_values
        effect_sizes = jnp.array(results.effect_sizes)
        
        # Filter valid data
        valid_mask = ~jnp.isnan(effect_sizes) & (effect_sizes > 0)
        sigma_valid = sigma_vals[valid_mask]
        effect_valid = effect_sizes[valid_mask]
        
        if len(effect_valid) > 0:
            ax.loglog(sigma_valid, effect_valid, 'o-', color=COLORS['secondary'], 
                     linewidth=3, markersize=10, label='Effect Size (Cohen\'s d)')
            
            # Add effect size interpretation thresholds
            ax.axhline(y=0.2, color=COLORS['primary'], linestyle=':', alpha=0.7, label='Small Effect (0.2)')
            ax.axhline(y=0.5, color=COLORS['primary'], linestyle='--', alpha=0.7, label='Medium Effect (0.5)')
            ax.axhline(y=0.8, color=COLORS['primary'], linestyle='-', alpha=0.7, label='Large Effect (0.8)')
        
        ax.set_xlabel('σ (Diffusion Parameter)', fontsize=14)
        ax.set_ylabel('Effect Size (Cohen\'s d)', fontsize=14)
        ax.set_title('Effect Size Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_numerical_stability(self, results: ValidationResult, save_path: Path):
        """Plot individual numerical stability analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        sigma_vals = results.sigma_values
        stability_data = results.numerical_stability
        
        # Extract stability fractions
        if isinstance(stability_data, list) and len(stability_data) > 0:
            if isinstance(stability_data[0], dict):
                stability_fractions = [s.get('fraction_stable', 1.0) for s in stability_data]
            else:
                stability_fractions = list(stability_data)
        else:
            stability_fractions = [1.0] * len(sigma_vals)
        
        ax.semilogx(sigma_vals, jnp.array(stability_fractions) * 100, 'o-', 
                   color=COLORS['secondary'], linewidth=3, markersize=10, label='Stability Fraction')
        ax.axhline(y=80, color=COLORS['primary'], linestyle='--', linewidth=2,
                  label='Minimum Threshold (80%)')
        
        ax.set_xlabel('σ (Diffusion Parameter)', fontsize=14)
        ax.set_ylabel('Numerical Stability (%)', fontsize=14)
        ax.set_title('Numerical Stability Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_convergence_rate_fit(self, results: ValidationResult, save_path: Path):
        """Plot individual convergence rate fit analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        conv_analysis = results.convergence_analysis
        
        if conv_analysis and conv_analysis.get('success', False):
            sigma_vals = results.sigma_values
            means = jnp.array(results.distances_mean)
            
            # Filter valid data
            valid_mask = ~jnp.isnan(means) & (means > 0) & (sigma_vals > 0)
            sigma_valid = sigma_vals[valid_mask]
            means_valid = means[valid_mask]
            
            if len(means_valid) > 0:
                empirical_rate = conv_analysis['empirical_rate']
                r_squared = conv_analysis['r_squared']
                
                # Show regression fit
                log_sigma = jnp.log(sigma_valid)
                log_means = jnp.log(means_valid)
                fit_line = conv_analysis['intercept'] + empirical_rate * log_sigma
                
                ax.plot(log_sigma, log_means, 'o', color=COLORS['secondary'], 
                       markersize=10, label='Data Points')
                ax.plot(log_sigma, fit_line, '-', color=COLORS['primary'], 
                       linewidth=3, label=f'Fit: slope={empirical_rate:.3f}')
                
                ax.set_xlabel('log(σ)', fontsize=14)
                ax.set_ylabel('log(Distance)', fontsize=14)
                ax.set_title(f'Convergence Rate Fit (R²={r_squared:.3f})', fontsize=16, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, color=COLORS['grid'])
            else:
                ax.text(0.5, 0.5, 'No Valid\nData Available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=16, color=COLORS['secondary'])
        else:
            ax.text(0.5, 0.5, 'Convergence\nAnalysis\nFailed', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16, color=COLORS['secondary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_validation_summary(self, sigma_inf_results, sigma_zero_results, transition_results, save_path: Path):
        """Plot individual validation summary."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        ax.axis('off')
        
        # Summary validation status
        results_status = {
            'σ→∞ Validation': sigma_inf_results.validation_passed if sigma_inf_results else False,
            'σ→0 Validation': sigma_zero_results.validation_passed if sigma_zero_results else False,
            'Transition Validation': transition_results.validation_passed if transition_results else False
        }
        
        # Create status display
        statuses = list(results_status.keys())
        passed = [results_status[s] for s in statuses]
        y_positions = [0.7, 0.5, 0.3]
        
        for i, (status, pass_status) in enumerate(results_status.items()):
            from matplotlib.patches import FancyBboxPatch
            
            # Status indicator
            indicator_color = COLORS['primary'] if pass_status else COLORS['secondary']
            
            # Card
            card = FancyBboxPatch((0.1, y_positions[i]-0.08), 0.8, 0.12,
                                boxstyle="round,pad=0.03",
                                facecolor=COLORS['background'], alpha=0.7,
                                edgecolor=COLORS['secondary'], linewidth=2,
                                transform=ax.transAxes)
            ax.add_patch(card)
            
            # Status circle
            circle = plt.Circle((0.2, y_positions[i]), 0.04, color=indicator_color, 
                              transform=ax.transAxes, zorder=10)
            ax.add_patch(circle)
            
            # Text
            text = 'PASSED' if pass_status else 'FAILED'
            ax.text(0.25, y_positions[i], text, ha='center', va='center', 
                   fontweight='bold', fontsize=12, color=indicator_color,
                   transform=ax.transAxes)
            
            ax.text(0.4, y_positions[i], status, fontsize=12, 
                   color=COLORS['secondary'], va='center', transform=ax.transAxes)
        
        # Add overall status
        overall_passed = all(results_status.values())
        overall_color = COLORS['primary'] if overall_passed else COLORS['secondary']
        overall_text = f"Overall Status: {'ALL PASSED' if overall_passed else 'SOME FAILED'}"
        
        # Overall status box
        overall_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.15,
                                   boxstyle="round,pad=0.03",
                                   facecolor=overall_color, alpha=0.15,
                                   edgecolor=overall_color, linewidth=3,
                                   transform=ax.transAxes)
        ax.add_patch(overall_box)
        
        ax.text(0.5, 0.125, overall_text, ha='center', va='center', 
               transform=ax.transAxes, fontweight='bold', fontsize=14,
               color=overall_color)
        
        ax.set_title('Geometric Limits Validation Summary', fontsize=16, fontweight='bold', 
                    color=COLORS['secondary'], pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()


def visualize_geometric_limits_from_file(results_file: str = "../results/geometric_limits/ultra_rigorous_geometric_validation_results.pkl",
                                       save_dir: str = "../results/geometric_limits/"):
    """
    Load results from file and generate individual visualizations.
    从文件加载结果并生成单独的可视化。
    """
    try:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract results
        sigma_inf_results = data.get('sigma_inf_results')
        sigma_zero_results = data.get('sigma_zero_results')
        transition_results = data.get('transition_results')
        
        # Create visualizer and generate individual figures
        visualizer = GeometricLimitsVisualizer()
        visualizer.generate_individual_validation_figures(
            sigma_inf_results=sigma_inf_results,
            sigma_zero_results=sigma_zero_results,
            transition_results=transition_results,
            save_dir=save_dir
        )
        
        print(f"Geometric limits individual visualizations completed!")
        print(f"几何极限单独可视化完成!")
        print(f"Individual figures saved to: {save_dir}")
        print(f"单独图表已保存至: {save_dir}")
        
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        print(f"结果文件未找到: {results_file}")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print(f"生成可视化时出错: {e}")


if __name__ == "__main__":
    # Generate individual visualizations from saved results
    visualize_geometric_limits_from_file()