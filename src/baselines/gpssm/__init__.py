"""
GPSSM (Gaussian Process State Space Models) Baseline (Refactored)
=================================================================

This package provides a refactored, modular, and numerically stable
implementation of the Gaussian Process State-Space Model from Frigola et al., 2014.

此包提供了Frigola等人2014年提出的高斯过程状态空间模型的一个重构的、
模块化的、数值稳定的实现。

Key Components:
- GPSSMSolver: The main solver for training and prediction.
- GPSSMConfig, OptimizerConfig: Type-safe configuration objects.
- A library of predefined dynamics and observation models.
"""

# Main solver class
from .gpssm import GPSSMSolver

# Configuration data classes
from .types import GPSSMConfig, OptimizerConfig

# Abstract base classes for custom models
from .base import DynamicsModel, ObservationModel

# Predefined model library and factory functions
from .models import (
    # Dynamics models
    LinearDynamics,
    PendulumDynamics,
    LorenzDynamics,
    DoublePendulumDynamics,
    # Observation models
    LinearObservation,
    PartialObservation,
    NonlinearObservation,
    RangeObservation,
    # Factory functions
    create_pendulum_system,
    create_lorenz_system,
    create_linear_system,
    create_tracking_system
)

__all__ = [
    # Main Solver
    'GPSSMSolver',

    # Configurations
    'GPSSMConfig',
    'OptimizerConfig',

    # Base Classes
    'DynamicsModel',
    'ObservationModel',

    # Model Library
    'LinearDynamics',
    'PendulumDynamics',
    'LorenzDynamics',
    'DoublePendulumDynamics',
    'LinearObservation',
    'PartialObservation',
    'NonlinearObservation',
    'RangeObservation',
    'create_pendulum_system',
    'create_lorenz_system',
    'create_linear_system',
    'create_tracking_system'
]