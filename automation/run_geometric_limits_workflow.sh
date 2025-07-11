#!/bin/bash
#
# MMSB-VI Geometric Limits Validation Workflow
# MMSB-VI几何极限验证工作流
# ==========================================
#
# This script runs the complete geometric limits validation workflow:
# 此脚本运行完整的几何极限验证工作流：
# 1. Execute geometric limits validation experiment
# 2. Generate publication-quality visualization
# 3. Display results summary
#
# Usage: ./run_geometric_limits_workflow.sh
# 用法：./run_geometric_limits_workflow.sh
#

set -e  # Exit on any error | 遇到错误立即退出

# Configuration | 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
RESULTS_DIR="$PROJECT_ROOT/results"  # results output to project root
LOGS_DIR="$PROJECT_ROOT/logs"        # logs output to project root
GL_DIR="$RESULTS_DIR/geometric_limits"

# Colors for output | 输出颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function | 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python environment is ready | 检查Python环境是否就绪
check_environment() {
    log_info "Checking Python environment... | 检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+ | 未找到Python，请安装Python 3.8+"
        exit 1
    fi
    
    # Check required packages | 检查必需包
    python -c "import jax, matplotlib, numpy, scipy" 2>/dev/null || {
        log_error "Required packages missing. Please install: jax, matplotlib, numpy, scipy"
        log_error "缺少必需包。请安装: jax, matplotlib, numpy, scipy"
        exit 1
    }
    
    log_success "Python environment ready | Python环境就绪"
}

# Create necessary directories | 创建必要目录
setup_directories() {
    log_info "Setting up directories... | 设置目录..."
    
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$GL_DIR"
    
    log_success "Directories ready | 目录就绪"
}

# Run geometric limits validation experiment | 运行几何极限验证实验
run_experiment() {
    log_info "Running geometric limits validation experiment..."
    log_info "运行几何极限验证实验..."
    
    cd "$PROJECT_ROOT/theoretical_verification/core_experiments"
    
    # Run the experiment | 运行实验
    python geometric_limits_validation.py || {
        log_error "Geometric limits validation experiment failed | 几何极限验证实验失败"
        exit 1
    }

    # Move generated pkl file to results directory | 将生成的pkl移动到results目录
    if [[ -f "ultra_rigorous_geometric_validation_results.pkl" ]]; then
        mv -f "ultra_rigorous_geometric_validation_results.pkl" "$GL_DIR/" || true
    fi
    # Move log file to logs directory | 将日志文件移动到logs目录
    if [[ -f "geometric_limits_validation.log" ]]; then
        mv -f "geometric_limits_validation.log" "$LOGS_DIR/" || true
    fi
    
    log_success "Geometric limits validation experiment completed | 几何极限验证实验完成"
}

# Generate visualization | 生成可视化
generate_visualization() {
    log_info "Generating publication-quality visualization..."
    log_info "生成发表质量可视化..."
    
    cd "$PROJECT_ROOT/visualization"
    
    # Generate figures | 生成图形
    python geometric_limits_visualization.py || {
        log_error "Visualization generation failed | 可视化生成失败"
        exit 1
    }
    
    log_success "Visualization generated successfully | 可视化生成成功"
}

# Display results summary | 显示结果摘要
display_summary() {
    log_info "Workflow Summary | 工作流摘要"
    echo "=========================="
    
    # Check if result files exist | 检查结果文件是否存在
    if [[ -f "$GL_DIR/ultra_rigorous_geometric_validation_results.pkl" ]]; then
        log_success "Experimental data saved | 实验数据已保存"
        echo "   Location: $GL_DIR/ultra_rigorous_geometric_validation_results.pkl"
    else
        log_warning "Experimental data not found | 未找到实验数据"
    fi
    
    if [[ -f "$GL_DIR/geometric_limits_validation.png" ]]; then
        log_success "Validation figures generated | 验证图形已生成"
        echo "   Location: $GL_DIR/geometric_limits_validation.png"
    else
        log_warning "Validation figures not found | 未找到验证图形"
    fi
    
    # Display log location | 显示日志位置
    if [[ -f "$LOGS_DIR/geometric_limits_validation.log" ]]; then
        log_info "Detailed logs available at: $LOGS_DIR/geometric_limits_validation.log"
        log_info "详细日志位置: $LOGS_DIR/geometric_limits_validation.log"
    fi
    
    echo ""
    log_success "Geometric Limits Validation Workflow Complete!"
    log_success "几何极限验证工作流完成!"
}

# Main workflow | 主工作流
main() {
    echo "========================================================"
    echo "MMSB-VI Geometric Limits Validation Workflow"
    echo "MMSB-VI几何极限验证工作流"
    echo "========================================================"
    echo ""
    
    check_environment
    setup_directories
    run_experiment
    generate_visualization
    display_summary
    
    echo ""
    echo "========================================================"
    log_success "All tasks completed successfully! | 所有任务成功完成!"
    echo "========================================================"
}

# Run main function | 运行主函数
main "$@"