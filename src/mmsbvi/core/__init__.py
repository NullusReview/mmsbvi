"""
Core module for MMSBVI
MMSBVI核心模块

This module provides core data structures and type definitions.
本模块提供核心数据结构和类型定义。
"""

from .types import (
    # Type aliases / 类型别名
    Scalar,
    Vector,
    Matrix,
    Tensor3D,
    Density1D,
    Potential1D,
    Velocity1D,
    Grid1D,
    TimeIndex,
    TimeSteps,
    
    # Data structures / 数据结构
    GridConfig1D,
    OUProcessParams,
    IPFPState,
    MMSBProblem,
    MMSBSolution,
    
    # Configurations / 配置
    IPFPConfig,
    PDESolverConfig,
    
    # Protocols / 协议
    Solver,
    Kernel,
)

__all__ = [
    # Type aliases
    "Scalar",
    "Vector", 
    "Matrix",
    "Tensor3D",
    "Density1D",
    "Potential1D",
    "Velocity1D",
    "Grid1D",
    "TimeIndex",
    "TimeSteps",
    
    # Data structures
    "GridConfig1D",
    "OUProcessParams",
    "IPFPState",
    "MMSBProblem",
    "MMSBSolution",
    
    # Configurations
    "IPFPConfig",
    "PDESolverConfig",
    
    # Protocols
    "Solver",
    "Kernel",
]