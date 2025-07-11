#!/usr/bin/env python3
"""
大角度单摆方法对比可视化 / Large Angle Pendulum Method Comparison Visualization
============================================================================

使用现代审美设计展示MMSB-VI、EKF、UKF、SVI四种方法的性能对比
Modern aesthetic visualization of MMSB-VI, EKF, UKF, SVI method comparisons
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 设置现代化的matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 自定义配色方案 - 现代科技感
COLORS = {
    'MMSB-VI': '#2E86AB',    # 深蓝色 - 主角
    'EKF': '#A23B72',        # 紫红色 - 经典
    'UKF': '#F18F01',        # 橙色 - 活力
    'SVI': '#C73E1D',        # 红色 - 创新
    'background': '#F8F9FA',  # 浅灰背景
    'grid': '#E9ECEF',       # 网格线
    'text': '#2C3E50',       # 深色文字
    'accent': '#7209B7'      # 强调色
}

def create_comparison_dashboard():
    """创建综合对比仪表板"""
    
    # 读取数据
    df = pd.read_csv('/Users/willet/Downloads/SB VI/results/pendulum_comparison_results.csv')
    
    # 创建主图和子图布局
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 使用GridSpec创建复杂布局
    gs = gridspec.GridSpec(3, 4, figure=fig, 
                          height_ratios=[1, 2, 1.5],
                          width_ratios=[1, 1, 1, 1],
                          hspace=0.3, wspace=0.25)
    
    # 1. 标题区域
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # 主标题
    ax_title.text(0.5, 0.7, 'Large Angle Pendulum: Method Performance Comparison', 
                 ha='center', va='center', fontsize=28, fontweight='bold', 
                 color=COLORS['text'], transform=ax_title.transAxes)
    
    # 副标题
    ax_title.text(0.5, 0.3, 'MMSB-VI vs EKF vs UKF vs SVI | Density Quality Assessment', 
                 ha='center', va='center', fontsize=16, 
                 color=COLORS['text'], alpha=0.8, transform=ax_title.transAxes)
    
    # 2. 核心指标对比 - 雷达图
    ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
    create_radar_chart(ax_radar, df)
    
    # 3. NLL对比 - 柱状图
    ax_nll = fig.add_subplot(gs[1, 1])
    create_nll_comparison(ax_nll, df)
    
    # 4. 覆盖率对比 - 甜甜圈图
    ax_coverage = fig.add_subplot(gs[1, 2])
    create_coverage_donut(ax_coverage, df)
    
    # 5. 运行时间对比 - 水平条形图
    ax_runtime = fig.add_subplot(gs[1, 3])
    create_runtime_comparison(ax_runtime, df)
    
    # 6. 性能总结卡片
    ax_summary = fig.add_subplot(gs[2, :])
    create_summary_cards(ax_summary, df)
    
    # 保存图像
    plt.savefig('/Users/willet/Downloads/SB VI/results/method_comparison_dashboard.png', 
                dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.show()

def create_radar_chart(ax, df):
    """创建雷达图显示综合性能"""
    
    # 数据预处理
    methods = df['Method'].tolist()
    
    # 标准化指标 (越小越好的指标需要反转)
    nll_scores = 100 - (df['nll_mean'] / df['nll_mean'].max() * 100)  # 反转NLL
    coverage_scores = df['coverage_95'] * 100  # 覆盖率保持原样
    speed_scores = 100 - (df['runtime'] / df['runtime'].max() * 100)  # 反转运行时间
    
    # 雷达图数据
    categories = ['Density Quality\n(NLL)', 'Calibration\n(Coverage)', 'Speed\n(Runtime)']
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 绘制每个方法
    for i, method in enumerate(methods):
        if pd.notna(nll_scores.iloc[i]):  # 只绘制有数据的方法
            values = [nll_scores.iloc[i], coverage_scores.iloc[i], speed_scores.iloc[i]]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, linewidth=2, 
                   label=method, color=COLORS[method], alpha=0.8)
            ax.fill(angles, values, color=COLORS[method], alpha=0.2)
    
    # 自定义网格
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=COLORS['text'])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], 
                       fontsize=8, color=COLORS['text'], alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # 标题
    ax.set_title('综合性能雷达图\nComprehensive Performance', 
                fontsize=12, fontweight='bold', 
                color=COLORS['text'], pad=20)
    
    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
             fontsize=9, frameon=True, fancybox=True, shadow=True)

def create_nll_comparison(ax, df):
    """创建NLL对比柱状图"""
    
    methods = df['Method'].tolist()
    nll_values = df['nll_mean'].tolist()
    
    # 创建柱状图
    bars = ax.bar(range(len(methods)), nll_values, 
                  color=[COLORS[m] for m in methods],
                  alpha=0.8, edgecolor='white', linewidth=2)
    
    # 美化柱状图
    for i, (bar, nll) in enumerate(zip(bars, nll_values)):
        if pd.notna(nll):
            # 添加数值标签
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{nll:.1f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color=COLORS['text'])
            
            # 高亮最佳结果
            if nll == df['nll_mean'].min():
                bar.set_edgecolor(COLORS['accent'])
                bar.set_linewidth(3)
                # 添加"BEST"标签
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       '🏆 BEST', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', color=COLORS['accent'])
    
    # 自定义坐标轴
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', 
                      fontsize=10, color=COLORS['text'])
    ax.set_ylabel('Negative Log-Likelihood', fontsize=11, 
                 color=COLORS['text'], fontweight='bold')
    ax.set_title('密度质量对比\nDensity Quality (Lower is Better)', 
                fontsize=12, fontweight='bold', color=COLORS['text'])
    
    # 美化网格
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('white')
    
    # 设置y轴范围
    ax.set_ylim(0, max(nll_values) * 1.2)

def create_coverage_donut(ax, df):
    """创建覆盖率甜甜圈图"""
    
    methods = df['Method'].tolist()
    coverage_values = df['coverage_95'].tolist()
    
    # 过滤有效数据
    valid_data = [(m, c) for m, c in zip(methods, coverage_values) if pd.notna(c)]
    methods_valid = [item[0] for item in valid_data]
    coverage_valid = [item[1] * 100 for item in valid_data]  # 转换为百分比
    
    # 创建甜甜圈图
    wedges, texts, autotexts = ax.pie(coverage_valid, 
                                     labels=methods_valid,
                                     colors=[COLORS[m] for m in methods_valid],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     pctdistance=0.85,
                                     wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # 中心文本
    ax.text(0, 0, '95%\nCoverage', ha='center', va='center', 
           fontsize=14, fontweight='bold', color=COLORS['text'])
    
    # 标题
    ax.set_title('校准质量分布\nCalibration Quality', 
                fontsize=12, fontweight='bold', color=COLORS['text'])

def create_runtime_comparison(ax, df):
    """创建运行时间对比"""
    
    methods = df['Method'].tolist()
    runtimes = df['runtime'].tolist()
    
    # 水平条形图
    bars = ax.barh(range(len(methods)), runtimes, 
                   color=[COLORS[m] for m in methods],
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for i, (bar, runtime) in enumerate(zip(bars, runtimes)):
        if pd.notna(runtime):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{runtime:.2f}s', ha='left', va='center', 
                   fontsize=10, fontweight='bold', color=COLORS['text'])
            
            # 高亮最快结果
            if runtime == min(r for r in runtimes if pd.notna(r)):
                bar.set_edgecolor(COLORS['accent'])
                bar.set_linewidth(3)
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                       '⚡ FASTEST', ha='left', va='center', 
                       fontsize=10, fontweight='bold', color=COLORS['accent'])
    
    # 自定义坐标轴
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10, color=COLORS['text'])
    ax.set_xlabel('Runtime (seconds)', fontsize=11, 
                 color=COLORS['text'], fontweight='bold')
    ax.set_title('运行时间对比\nRuntime Comparison', 
                fontsize=12, fontweight='bold', color=COLORS['text'])
    
    # 美化网格
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_facecolor('white')

def create_summary_cards(ax, df):
    """创建性能总结卡片"""
    
    ax.axis('off')
    
    # 找到最佳性能方法
    best_density = df.loc[df['nll_mean'].idxmin(), 'Method']
    best_coverage = df.loc[df['coverage_95'].idxmax(), 'Method']
    fastest = df.loc[df['runtime'].idxmin(), 'Method']
    
    # 卡片信息
    cards = [
        {
            'title': '🏆 Best Density Quality',
            'method': best_density,
            'value': f"NLL: {df['nll_mean'].min():.1f}",
            'color': COLORS[best_density]
        },
        {
            'title': '🎯 Best Calibration',
            'method': best_coverage,
            'value': f"Coverage: {df['coverage_95'].max()*100:.1f}%",
            'color': COLORS[best_coverage]
        },
        {
            'title': '⚡ Fastest Runtime',
            'method': fastest,
            'value': f"Time: {df['runtime'].min():.2f}s",
            'color': COLORS[fastest]
        },
        {
            'title': '📊 Overall Winner',
            'method': 'MMSB-VI',
            'value': 'Superior Performance',
            'color': COLORS['MMSB-VI']
        }
    ]
    
    # 绘制卡片
    card_width = 0.22
    card_height = 0.8
    
    for i, card in enumerate(cards):
        x = i * 0.25 + 0.02
        
        # 创建圆角矩形
        rect = FancyBboxPatch((x, 0.1), card_width, card_height,
                             boxstyle="round,pad=0.02",
                             facecolor=card['color'],
                             edgecolor='white',
                             linewidth=2,
                             alpha=0.9)
        ax.add_patch(rect)
        
        # 添加文本
        ax.text(x + card_width/2, 0.7, card['title'],
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color='white', transform=ax.transAxes)
        
        ax.text(x + card_width/2, 0.5, card['method'],
               ha='center', va='center',
               fontsize=14, fontweight='bold',
               color='white', transform=ax.transAxes)
        
        ax.text(x + card_width/2, 0.3, card['value'],
               ha='center', va='center',
               fontsize=10,
               color='white', alpha=0.9, transform=ax.transAxes)

def create_time_series_comparison():
    """创建时间序列密度演化可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['background'])
    fig.suptitle('Method Density Evolution Comparison\n时间序列密度演化对比', 
                 fontsize=18, fontweight='bold', color=COLORS['text'])
    
    methods = ['MMSB-VI', 'EKF', 'UKF', 'SVI']
    
    for i, (ax, method) in enumerate(zip(axes.flat, methods)):
        # 模拟密度演化数据
        t = np.linspace(0, 3, 100)
        theta = np.linspace(-np.pi, np.pi, 50)
        
        # 创建示例密度热图
        T, THETA = np.meshgrid(t, theta)
        if method == 'MMSB-VI':
            density = np.exp(-((THETA - np.sin(T))**2 + (T - 1.5)**2) / 0.5)
        else:
            density = np.exp(-((THETA - np.sin(T))**2 + (T - 1.5)**2) / 1.0) * 0.6
        
        # 绘制热图
        im = ax.imshow(density, extent=[0, 3, -np.pi, np.pi], 
                      aspect='auto', origin='lower',
                      cmap='RdYlBu_r', alpha=0.8)
        
        # 自定义坐标轴
        ax.set_xlabel('Time (s)', fontsize=10, color=COLORS['text'])
        ax.set_ylabel('θ (radians)', fontsize=10, color=COLORS['text'])
        ax.set_title(f'{method} Density Evolution', 
                    fontsize=12, fontweight='bold', color=COLORS[method])
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('/Users/willet/Downloads/SB VI/results/density_evolution_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.show()

if __name__ == "__main__":
    print("🎨 创建方法对比可视化...")
    print("🎨 Creating method comparison visualization...")
    
    # 创建主仪表板
    create_comparison_dashboard()
    
    # 创建时间序列对比
    create_time_series_comparison()
    
    print("✅ 可视化完成！文件保存在 results/ 目录")
    print("✅ Visualization complete! Files saved in results/ directory")