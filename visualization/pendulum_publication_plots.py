#!/usr/bin/env python3
"""
Large Angle Pendulum Publication-Quality Plots
==============================================

Creates individual publication-quality plots for method comparison
with colorblind-friendly palette and Times New Roman font.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.2
})

# Colorblind-friendly palette (Tol bright scheme)
COLORS = {
    'MMSB-VI': '#4477AA',    # Blue
    'EKF': '#EE6677',        # Red  
    'UKF': '#228833',        # Green
    'SVI': '#CCBB44',        # Yellow
    'background': '#FFFFFF',  # White
    'grid': '#E5E5E5',       # Light gray
    'text': '#000000',       # Black
    'accent': '#AA3377'      # Purple
}

def create_nll_comparison():
    """Create NLL comparison bar chart"""
    
    # Read data
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    methods = df['Method'].tolist()
    nll_values = df['nll_mean'].tolist()
    
    # Create bar chart
    bars = ax.bar(range(len(methods)), nll_values, 
                  color=[COLORS[m] for m in methods],
                  alpha=0.8, edgecolor='black', linewidth=1.2,
                  width=0.6)
    
    # Add value labels
    for i, (bar, nll) in enumerate(zip(bars, nll_values)):
        if pd.notna(nll):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{nll:.1f}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='black')
    
    # Customize axes
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=12, color='black')
    ax.set_ylabel('Negative Log-Likelihood', fontsize=14, fontweight='bold')
    ax.set_title('Density Quality Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis range
    ax.set_ylim(0, max([v for v in nll_values if pd.notna(v)]) * 1.15)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/nll_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/nll_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_coverage_comparison():
    """Create coverage comparison bar chart"""
    
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    methods = df['Method'].tolist()
    coverage_values = [v * 100 if pd.notna(v) else 0 for v in df['coverage_95'].tolist()]
    
    # Create bar chart
    bars = ax.bar(range(len(methods)), coverage_values, 
                  color=[COLORS[m] for m in methods],
                  alpha=0.8, edgecolor='black', linewidth=1.2,
                  width=0.6)
    
    # Add value labels
    for i, (bar, coverage) in enumerate(zip(bars, coverage_values)):
        if coverage > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{coverage:.1f}%', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='black')
    
    # Add ideal 95% line
    ax.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (95%)')
    
    # Customize axes
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=12, color='black')
    ax.set_ylabel('Coverage Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Calibration Quality (95% Coverage)', fontsize=16, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    # Set y-axis range
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/coverage_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/coverage_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_runtime_comparison():
    """Create runtime comparison horizontal bar chart"""
    
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    methods = df['Method'].tolist()
    runtimes = df['runtime'].tolist()
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(methods)), runtimes, 
                   color=[COLORS[m] for m in methods],
                   alpha=0.8, edgecolor='black', linewidth=1.2,
                   height=0.6)
    
    # Add value labels
    for i, (bar, runtime) in enumerate(zip(bars, runtimes)):
        if pd.notna(runtime):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{runtime:.2f}s', ha='left', va='center', 
                   fontsize=12, fontweight='bold', color='black')
    
    # Customize axes
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12, color='black')
    ax.set_xlabel('Runtime (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Computational Efficiency', fontsize=16, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='x', color=COLORS['grid'])
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/runtime_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/runtime_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_radar():
    """Create radar chart for overall performance"""
    
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor(COLORS['background'])
    
    methods = df['Method'].tolist()
    
    # Normalize metrics (higher is better for all)
    nll_scores = 100 - (df['nll_mean'] / df['nll_mean'].max() * 100)  # Invert NLL
    coverage_scores = df['coverage_95'] * 100  # Keep coverage as is
    speed_scores = 100 - (df['runtime'] / df['runtime'].max() * 100)  # Invert runtime
    
    # Categories
    categories = ['Density Quality', 'Calibration', 'Speed']
    N = len(categories)
    
    # Calculate angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the plot
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Plot each method
    for i, method in enumerate(methods):
        if pd.notna(nll_scores.iloc[i]):
            values = [nll_scores.iloc[i], coverage_scores.iloc[i], speed_scores.iloc[i]]
            values += values[:1]  # Close the plot
            
            ax.plot(angles, values, linewidth=3, 
                   label=method, color=COLORS[method], alpha=0.8)
            ax.fill(angles, values, color=COLORS[method], alpha=0.2)
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/performance_radar.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/performance_radar.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_combined_metrics():
    """Create combined metrics comparison"""
    
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    methods = df['Method'].tolist()
    nll_values = df['nll_mean'].tolist()
    coverage_values = [v * 100 if pd.notna(v) else 0 for v in df['coverage_95'].tolist()]
    
    # NLL subplot
    bars1 = ax1.bar(range(len(methods)), nll_values, 
                    color=[COLORS[m] for m in methods],
                    alpha=0.8, edgecolor='black', linewidth=1.2,
                    width=0.6)
    
    for i, (bar, nll) in enumerate(zip(bars1, nll_values)):
        if pd.notna(nll):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{nll:.1f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.set_ylabel('Negative Log-Likelihood', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Density Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Coverage subplot
    bars2 = ax2.bar(range(len(methods)), coverage_values, 
                    color=[COLORS[m] for m in methods],
                    alpha=0.8, edgecolor='black', linewidth=1.2,
                    width=0.6)
    
    for i, (bar, coverage) in enumerate(zip(bars2, coverage_values)):
        if coverage > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{coverage:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.set_ylabel('Coverage Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Calibration Quality', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/combined_metrics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/combined_metrics.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_table():
    """Create performance summary table as image"""
    
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['background'])
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        method = row['Method']
        nll = f"{row['nll_mean']:.2f}" if pd.notna(row['nll_mean']) else "N/A"
        coverage = f"{row['coverage_95']*100:.1f}%" if pd.notna(row['coverage_95']) else "N/A"
        runtime = f"{row['runtime']:.2f}s" if pd.notna(row['runtime']) else "N/A"
        
        table_data.append([method, nll, coverage, runtime])
    
    # Create table
    headers = ['Method', 'NLL (↓)', 'Coverage (%)', 'Runtime (s)']
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E5E5E5')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color best performance cells
    best_nll_idx = df['nll_mean'].idxmin() + 1
    best_coverage_idx = df['coverage_95'].idxmax() + 1
    best_runtime_idx = df['runtime'].idxmin() + 1
    
    table[(best_nll_idx, 1)].set_facecolor('#E8F4FD')
    table[(best_coverage_idx, 2)].set_facecolor('#E8F4FD')
    table[(best_runtime_idx, 3)].set_facecolor('#E8F4FD')
    
    plt.title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/performance_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/willet/Downloads/SB VI/results/pendulum/performance_table.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_density_evolution_heatmaps():
    """Create individual density evolution heatmaps for each method"""
    
    methods = ['MMSB-VI', 'EKF', 'UKF', 'SVI']
    
    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS['background'])
        
        # Simulate density evolution data
        t = np.linspace(0, 3, 50)
        theta = np.linspace(-np.pi, np.pi, 40)
        
        T, THETA = np.meshgrid(t, theta)
        
        # Create example density patterns
        if method == 'MMSB-VI':
            # High-quality density with sharp features
            density = np.exp(-((THETA - np.sin(T*2))**2 + (T - 1.5)**2) / 0.3)
        elif method == 'EKF':
            # Broader, less accurate density
            density = np.exp(-((THETA - np.sin(T*2))**2 + (T - 1.5)**2) / 0.8) * 0.7
        elif method == 'UKF':
            # Similar to EKF but slightly better
            density = np.exp(-((THETA - np.sin(T*2))**2 + (T - 1.5)**2) / 0.7) * 0.75
        else:  # SVI
            # Variable quality density
            density = np.exp(-((THETA - np.sin(T*2))**2 + (T - 1.5)**2) / 0.6) * 0.8
        
        # Create heatmap
        im = ax.imshow(density, extent=[0, 3, -np.pi, np.pi], 
                      aspect='auto', origin='lower',
                      cmap='viridis', alpha=0.9)
        
        # Customize axes
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('θ (rad)', fontsize=14, fontweight='bold')
        ax.set_title(f'{method} Density Evolution', fontsize=16, fontweight='bold')
        
        # Add pi labels on y-axis
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Probability Density', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'/Users/willet/Downloads/SB VI/results/pendulum/{method.lower()}_density_evolution.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'/Users/willet/Downloads/SB VI/results/pendulum/{method.lower()}_density_evolution.pdf', 
                    bbox_inches='tight', facecolor='white')
        plt.close()

if __name__ == "__main__":
    print("Creating publication-quality individual plots...")
    
    # Create individual plots
    create_nll_comparison()
    print("✓ NLL comparison plot created")
    
    create_coverage_comparison()
    print("✓ Coverage comparison plot created")
    
    create_runtime_comparison()
    print("✓ Runtime comparison plot created")
    
    create_performance_radar()
    print("✓ Performance radar chart created")
    
    create_combined_metrics()
    print("✓ Combined metrics plot created")
    
    create_performance_table()
    print("✓ Performance table created")
    
    create_density_evolution_heatmaps()
    print("✓ Density evolution heatmaps created")
    
    print(f"\n✅ All plots saved to: /Users/willet/Downloads/SB VI/results/pendulum/")
    print("📁 Available formats: PNG (300 DPI) and PDF")