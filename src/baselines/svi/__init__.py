"""
SVI (Stochastic Variational Inference) 基线方法 / SVI Baseline Methods
==================================================================

实现随机变分推断用于大角度单摆系统的状态估计。
Implements stochastic variational inference for large angle pendulum system state estimation.
"""

from .svi_smoother import PendulumSVISmoother, SVIState, SVIParams

__all__ = ['PendulumSVISmoother', 'SVIState', 'SVIParams']