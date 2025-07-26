"""
大角度单摆实验模块 / Large Angle Pendulum Experiment Module
========================================================

实现完整的大角度单摆状态估计实验，包括：
- 非线性动力学建模
- 多模态场景设计  
- 概率密度质量评估
- 周期性状态空间处理

Implements complete large angle pendulum state estimation experiments including:
- Nonlinear dynamics modeling
- Multi-modal scenario design
- Probability density quality assessment
- Periodic state space handling
"""

from .data_generator import LargeAnglePendulumGenerator, PendulumParams

__all__ = [
    "LargeAnglePendulumGenerator",
]