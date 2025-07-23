"""
Base Abstract Classes for GPSSM Models / GPSSM模型抽象基类
=========================================================

This module defines the abstract base classes (ABCs) for dynamics and
observation models, establishing a clear interface for all model components.

此模块为动态和观测模型定义了抽象基类（ABC），为所有模型组件建立了清晰的接口。
"""

from abc import ABC, abstractmethod
import chex
from typing import Callable

class DynamicsModel(ABC):
    """
    Abstract base class for dynamics models.
    动态模型的抽象基类。

    Defines the deterministic part of the state transition function.
    定义了状态转移函数的确定性部分。
    x_{t+1} = f_det(x_t) + f_gp(x_t) + ε_t
    """
    
    @abstractmethod
    def get_mean_function(self) -> Callable[[chex.Array], chex.Array]:
        """
        Returns the deterministic part of the dynamics function, f_det(x).
        返回动态函数的确定性部分, f_det(x)。
        """
        pass


class ObservationModel(ABC):
    """
    Abstract base class for observation models.
    观测模型的抽象基类。

    Defines the observation function.
    定义了观测函数。
    y_t = h(x_t) + η_t
    """
    
    @abstractmethod
    def get_observation_function(self) -> Callable[[chex.Array], chex.Array]:
        """
        Returns the observation function, h(x).
        返回观测函数, h(x)。
        """
        pass