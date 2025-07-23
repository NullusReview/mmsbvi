"""
基线方法模块 / Baseline Methods Module
=====================================

实现各种基线方法用于与MMSB-VI算法比较，包括：
- EKF平滑器（扩展卡尔曼滤波）
- UKF平滑器（无味卡尔曼滤波）
- SVI（结构化变分推断）[待实现]
- BBVI（黑盒变分推断）[待实现]

Implements various baseline methods for comparison with MMSB-VI algorithm, including:
- EKF Smoother (Extended Kalman Filter)
- UKF Smoother (Unscented Kalman Filter)
- SVI (Structured Variational Inference) [to be implemented]
- BBVI (Black-Box Variational Inference) [to be implemented]
"""

from .ekf import PendulumEKFSmoother, EKFState, EKFResult
from .ukf import PendulumUKFSmoother, UKFState, UKFResult

__all__ = [
    "PendulumEKFSmoother",
    "EKFState", 
    "EKFResult",
    "PendulumUKFSmoother",
    "UKFState",
    "UKFResult",
]