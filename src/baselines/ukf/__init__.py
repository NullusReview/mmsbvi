"""
UKF (Unscented Kalman Filter) Package / 无迹卡尔曼滤波器包
=======================================================

高效、通用的无迹卡尔曼滤波器实现，支持任意非线性系统。
Efficient, generic Unscented Kalman Filter implementation supporting arbitrary nonlinear systems.

Key Features / 主要特性:
- 通用系统接口 / Generic system interface
- GPU并行批处理 / GPU parallel batch processing  
- 数值稳定性保证 / Numerical stability guarantees
- 内存高效实现 / Memory-efficient implementation
- 向后兼容原始API / Backward compatible with original API
"""

# 主要接口 / Main interfaces
from .ukf import GenericUKF, create_pendulum_ukf

# 配置和数据结构 / Configuration and data structures
from .config import UKFConfig, UKFState, UKFResult

# Note: Original PendulumUKFSmoother has been integrated into GenericUKF
# Use create_pendulum_ukf() for pendulum-specific functionality

__all__ = [
    # 主要接口 / Main interfaces
    'GenericUKF',
    'create_pendulum_ukf',
    
    # 配置和数据结构 / Configuration and data structures
    'UKFConfig',
    'UKFState', 
    'UKFResult'
]