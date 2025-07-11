"""
扩展卡尔曼滤波基线模块 / Extended Kalman Filter Baseline Module
==========================================================

实现针对大角度单摆系统的EKF平滑器，包括：
- 非线性sin(θ)状态转移和雅可比矩阵计算
- 前向滤波和后向平滑
- 周期性角度处理
- 数值稳定性保证
- 性能评估接口

Implements EKF smoother for large angle pendulum system, including:
- Nonlinear sin(θ) state transition and Jacobian computation
- Forward filtering and backward smoothing
- Periodic angle handling
- Numerical stability guarantees
- Performance evaluation interface
"""

from .ekf_smoother import PendulumEKFSmoother, EKFState, EKFResult

__all__ = [
    "PendulumEKFSmoother",
    "EKFState", 
    "EKFResult"
]