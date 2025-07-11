"""
无味卡尔曼滤波基线模块 / Unscented Kalman Filter Baseline Module
===========================================================

实现针对大角度单摆系统的UKF平滑器，包括：
- Sigma点生成和权重计算
- 无味变换（Unscented Transform）
- 前向滤波和后向平滑
- 周期性角度处理
- 数值稳定性保证
- 性能评估接口

Implements UKF smoother for large angle pendulum system, including:
- Sigma point generation and weight computation
- Unscented Transform
- Forward filtering and backward smoothing
- Periodic angle handling
- Numerical stability guarantees
- Performance evaluation interface
"""

from .ukf_smoother import PendulumUKFSmoother, UKFState, UKFResult

__all__ = [
    "PendulumUKFSmoother",
    "UKFState", 
    "UKFResult"
]