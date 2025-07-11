#!/usr/bin/env python3
"""
Parameter Sensitivity Visualization for MMSB-VI
MMSB-VI的参数敏感性可视化
==================================================

Visualization for parameter sensitivity analysis results - Individual plots.
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

# Import sensitivity result - simplified
try:
    from theoretical_verification.core_experiments.parameter_sensitivity_analysis import SensitivityResult
except ImportError:
    # Fallback mock
    from dataclasses import dataclass
    from typing import List, Any, Optional
    
    @dataclass  
    class SensitivityResult:
        parameter_values: List[float]
        sensitivity_measures: Dict[str, Any]
        gradient_analysis: Dict[str, Any]
        hessian_analysis: Dict[str, Any]
        stability_analysis: Dict[str, Any]
        perturbation_analysis: Dict[str, Any]
        convergence_analysis: Dict[str, Any]
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


class ParameterSensitivityVisualizer:
    """
    Publication-quality visualizer for parameter sensitivity analysis results.
    参数敏感性分析结果的发表质量可视化器。
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_individual_sensitivity_figures(self, 
                                              sigma_results: SensitivityResult,
                                              drift_results: SensitivityResult,
                                              save_dir: str = "../results/parameter_sensitivity/"):
        """
        Generate individual sensitivity analysis figures - one per plot.
        生成单独的敏感性分析图表 - 每个图表一个文件。
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        self._plot_individual_sigma_response(sigma_results, save_path / "sigma_response_surface.png")
        self._plot_individual_drift_spectrum(drift_results, save_path / "drift_matrix_spectrum.png")
        self._plot_individual_stability_dashboard(sigma_results, drift_results, save_path / "stability_dashboard.png")
        self._plot_individual_combined_landscape(sigma_results, drift_results, save_path / "combined_parameter_landscape.png")
        
        self.logger.info(f"Individual sensitivity figures saved to: {save_dir}")
        
    def _plot_individual_sigma_response(self, results: SensitivityResult, save_path: Path):
        """Plot individual sigma response surface."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        if hasattr(results, 'parameter_values') and len(results.parameter_values) > 0:
            sigma_vals = results.parameter_values
            objectives = results.sensitivity_measures.get('objective_values', [])
            
            if len(objectives) > 0:
                ax.semilogx(sigma_vals, objectives, 'o-', color=COLORS['secondary'], 
                           linewidth=3, markersize=10, label='Objective Function')
                
                # Mark stable regions
                if 'stable_regime' in results.stability_analysis:
                    stable_indices = results.stability_analysis['stable_regime']
                    if len(stable_indices) > 0:
                        stable_sigma = sigma_vals[stable_indices]
                        stable_obj = jnp.array(objectives)[stable_indices]
                        ax.scatter(stable_sigma, stable_obj, s=100, c=COLORS['primary'], 
                                 marker='s', label='Stable Regime', zorder=5)
        
        ax.set_xlabel('σ (Diffusion Parameter)', fontsize=14)
        ax.set_ylabel('Objective Function Value', fontsize=14)
        ax.set_title('σ Response Surface Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_drift_spectrum(self, results: SensitivityResult, save_path: Path):
        """Plot individual drift matrix spectrum."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['background'])
        
        if 'eigenvalues' in results.hessian_analysis:
            eigenvalue_data = results.hessian_analysis['eigenvalues']
            
            real_parts = []
            param_indices = []
            for i, eigenvals in enumerate(eigenvalue_data):
                if hasattr(eigenvals, '__len__'):
                    for ev in eigenvals:
                        real_parts.append(float(jnp.real(ev)))
                        param_indices.append(i)
                else:
                    real_parts.append(float(jnp.real(eigenvals)))
                    param_indices.append(i)
            
            if real_parts:
                ax.scatter(param_indices, real_parts, s=80, c=COLORS['secondary'], 
                          alpha=0.8, edgecolors=COLORS['primary'], linewidth=1.5)
                
                # Add trend line
                if len(param_indices) > 1:
                    z = jnp.polyfit(param_indices, real_parts, 1)
                    p = jnp.poly1d(z)
                    ax.plot(param_indices, p(param_indices), '--', 
                           linewidth=3, color=COLORS['primary'], alpha=0.8)
                
                # Stability threshold
                ax.axhline(y=0, color=COLORS['secondary'], linestyle='-', 
                          linewidth=2, alpha=0.7, label='Stability Boundary')
        
        ax.set_xlabel('Parameter Index', fontsize=14)
        ax.set_ylabel('Real(Eigenvalue)', fontsize=14)
        ax.set_title('Drift Matrix Spectrum Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_stability_dashboard(self, sigma_results, drift_results, save_path: Path):
        """Plot individual stability dashboard."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=COLORS['background'])
        ax.axis('off')
        
        # Extract stability metrics
        sigma_stable = len(sigma_results.stability_analysis.get('stable_regime', [])) > 0
        drift_stable = 'stability_measures' in drift_results.stability_analysis
        
        # Create metric cards
        metrics = [
            ("σ Stability", "STABLE" if sigma_stable else "UNSTABLE", sigma_stable),
            ("Drift Stability", "STABLE" if drift_stable else "UNSTABLE", drift_stable),
            ("Overall", "VALIDATED" if (sigma_stable and drift_stable) else "PENDING", 
             sigma_stable and drift_stable)
        ]
        
        # Display metrics
        y_positions = [0.7, 0.5, 0.3]
        
        for i, (metric, status, passed) in enumerate(metrics):
            from matplotlib.patches import FancyBboxPatch
            
            # Status indicator
            indicator_color = COLORS['primary'] if passed else COLORS['secondary']
            
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
            ax.text(0.35, y_positions[i]+0.03, metric, fontsize=12, fontweight='bold',
                   color=COLORS['secondary'], va='center', transform=ax.transAxes)
            ax.text(0.35, y_positions[i]-0.03, status, fontsize=10,
                   color=COLORS['secondary'], va='center', transform=ax.transAxes)
        
        ax.set_title('Parameter Sensitivity Stability Dashboard', fontsize=16, fontweight='bold', 
                    color=COLORS['secondary'], pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
    def _plot_individual_combined_landscape(self, sigma_results, drift_results, save_path: Path):
        """Plot individual combined parameter landscape."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor=COLORS['background'])
        
        if (hasattr(sigma_results, 'parameter_values') and 
            hasattr(drift_results, 'parameter_values')):
            
            sigma_vals = sigma_results.parameter_values
            sigma_obj = sigma_results.sensitivity_measures.get('objective_values', [])
            
            if len(sigma_obj) > 0:
                # Create landscape
                ax.fill_between(sigma_vals, 0, sigma_obj, alpha=0.3, color=COLORS['primary'], 
                               zorder=1, label='Sensitivity Landscape')
                
                # Main response curve
                ax.semilogx(sigma_vals, sigma_obj, '-', linewidth=4, color=COLORS['secondary'], 
                           alpha=0.9, zorder=5, label='Combined Response')
                
                # Markers
                n_markers = min(8, len(sigma_vals))
                marker_indices = jnp.linspace(0, len(sigma_vals)-1, n_markers, dtype=int)
                
                for i, idx in enumerate(marker_indices):
                    ax.scatter(sigma_vals[idx], sigma_obj[idx], s=100, 
                              c=COLORS['secondary'], edgecolors=COLORS['primary'], 
                              linewidth=2, alpha=0.9, zorder=6)
        
        ax.set_xlabel('Parameter Value (log scale)', fontsize=14)
        ax.set_ylabel('Sensitivity Magnitude', fontsize=14)
        ax.set_title('Combined Parameter Sensitivity Landscape', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()


def visualize_parameter_sensitivity_from_file(results_file: str = "../results/parameter_sensitivity/ultra_rigorous_parameter_sensitivity_results.pkl",
                                            save_dir: str = "../results/parameter_sensitivity/"):
    """
    Load results from file and generate individual visualizations.
    从文件加载结果并生成单独的可视化。
    """
    try:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract results
        sigma_results = data.get('sigma_sensitivity_results')
        drift_results = data.get('drift_matrix_sensitivity_results')
        
        if sigma_results is None or drift_results is None:
            print(f"❌ Missing sensitivity results in file: {results_file}")
            return
        
        # Create visualizer and generate individual figures
        visualizer = ParameterSensitivityVisualizer()
        visualizer.generate_individual_sensitivity_figures(
            sigma_results=sigma_results,
            drift_results=drift_results,
            save_dir=save_dir
        )
        
        print(f"Parameter sensitivity individual visualizations completed!")
        print(f"参数敏感性单独可视化完成!")
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
    visualize_parameter_sensitivity_from_file()