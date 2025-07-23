"""
UKF Configuration and Data Structures / UKF配置和数据结构
======================================================

无迹卡尔曼滤波器的配置参数和数据结构定义。
Configuration parameters and data structures for Unscented Kalman Filter.
"""

from typing import NamedTuple, List
import chex
from dataclasses import dataclass


@dataclass
class UKFConfig:
    """
    UKF配置参数 / UKF configuration parameters
    
    Attributes:
        alpha: UKF缩放参数 / UKF scaling parameter (default: 1e-3)
        beta: 分布参数 / Distribution parameter (default: 2.0 for Gaussian)
        kappa: 次要缩放参数 / Secondary scaling parameter (default: None, auto-set)
        regularization_eps: 数值稳定性参数 / Numerical stability parameter
    """
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = None
    regularization_eps: float = 1e-8


class UKFState(NamedTuple):
    """
    UKF状态表示 / UKF state representation
    
    Attributes:
        mean: 状态均值 / State mean
        covariance: 状态协方差 / State covariance  
        log_likelihood: 累积对数似然 / Accumulated log-likelihood
    """
    mean: chex.Array
    covariance: chex.Array
    log_likelihood: chex.Scalar


class UKFResult(NamedTuple):
    """
    UKF处理结果 / UKF processing result
    
    Attributes:
        filtered_states: 滤波状态序列 / Filtered state sequence
        smoothed_states: 平滑状态序列 / Smoothed state sequence  
        total_log_likelihood: 总对数似然 / Total log-likelihood
        runtime: 运行时间 / Runtime in seconds
    """
    filtered_states: List[UKFState]
    smoothed_states: List[UKFState]
    total_log_likelihood: float
    runtime: float