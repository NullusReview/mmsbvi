"""
GPSSM模型库 / GPSSM Model Library
================================

提供常用的动态模型和观测模型实现。
Provides common dynamics and observation model implementations.

支持的模型 / Supported Models:
- LinearDynamics: 线性动态系统
- PendulumDynamics: 大角度单摆系统  
- LorenzDynamics: 洛伦兹混沌系统
- LinearObservation: 线性观测模型
- NonlinearObservation: 非线性观测模型
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional
import chex

from .base import DynamicsModel, ObservationModel
from typing import Callable


# ============================================================================
# 动态模型实现 / Dynamics Model Implementations
# ============================================================================

class LinearDynamics(DynamicsModel):
    """
    线性动态系统 / Linear dynamics system
    
    x_{t+1} = A x_t + b + ε_t
    其中 ε_t ~ N(0, Q)
    """
    
    def __init__(self, A: chex.Array, b: Optional[chex.Array] = None, Q: Optional[chex.Array] = None):
        """
        Args:
            A: 转移矩阵 [state_dim, state_dim] / transition matrix
            b: 偏置向量 [state_dim] / bias vector
            Q: 过程噪声协方差 [state_dim, state_dim] / process noise covariance
        """
        self.A = A
        self.b = b if b is not None else jnp.zeros(A.shape[0])
        self.Q = Q if Q is not None else 0.1 * jnp.eye(A.shape[0])
    
    def get_mean_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the deterministic part of the dynamics function."""
        @jit
        def mean_fn(state: chex.Array) -> chex.Array:
            return self.A @ state + self.b
        return mean_fn


class PendulumDynamics(DynamicsModel):
    """
    大角度单摆动态系统 / Large angle pendulum dynamics system
    
    dθ/dt = ω
    dω/dt = -(g/L)sin(θ) - γω + σ*noise
    
    这是一个经典的非线性动态系统示例。
    This is a classic nonlinear dynamics system example.
    """
    
    def __init__(self, 
                 dt: float = 0.05,
                 g: float = 9.81,
                 L: float = 1.0, 
                 gamma: float = 0.2,
                 process_noise_std: float = 0.1):
        """
        Args:
            dt: 时间步长 / time step
            g: 重力加速度 / gravitational acceleration
            L: 单摆长度 / pendulum length
            gamma: 阻尼系数 / damping coefficient
            process_noise_std: 过程噪声标准差 / process noise std
        """
        self.dt = dt
        self.g = g
        self.L = L
        self.gamma = gamma
        
        # 过程噪声协方差矩阵 / process noise covariance matrix
        self.Q = jnp.array([
            [0.0, 0.0],  # 角度无直接噪声
            [0.0, (process_noise_std * dt)**2]  # 角速度有噪声
        ])
    
    def get_mean_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the deterministic part of the pendulum dynamics."""
        @jit
        def mean_fn(state: chex.Array) -> chex.Array:
            theta, omega = state[0], state[1]
            dtheta_dt = omega
            domega_dt = -(self.g / self.L) * jnp.sin(theta) - self.gamma * omega
            next_theta = theta + self.dt * dtheta_dt
            next_omega = omega + self.dt * domega_dt
            next_theta = jnp.mod(next_theta + jnp.pi, 2 * jnp.pi) - jnp.pi
            return jnp.array([next_theta, next_omega])
        return mean_fn


class LorenzDynamics(DynamicsModel):
    """
    洛伦兹混沌系统 / Lorenz chaotic system
    
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    
    这是一个经典的混沌动态系统，用于测试GPSSM处理复杂非线性动态的能力。
    This is a classic chaotic dynamical system for testing GPSSM's ability to handle complex nonlinear dynamics.
    """
    
    def __init__(self,
                 dt: float = 0.01,
                 sigma: float = 10.0,
                 rho: float = 28.0,
                 beta: float = 8.0/3.0,
                 process_noise_std: float = 0.1):
        """
        Args:
            dt: 时间步长 / time step
            sigma, rho, beta: 洛伦兹系统参数 / Lorenz system parameters
            process_noise_std: 过程噪声标准差 / process noise std
        """
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
        # 等方差过程噪声 / isotropic process noise
        self.Q = (process_noise_std * dt)**2 * jnp.eye(3)
        
    def get_mean_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the deterministic part of the Lorenz dynamics."""
        @jit
        def mean_fn(state: chex.Array) -> chex.Array:
            x, y, z = state[0], state[1], state[2]
            dx_dt = self.sigma * (y - x)
            dy_dt = x * (self.rho - z) - y
            dz_dt = x * y - self.beta * z
            next_x = x + self.dt * dx_dt
            next_y = y + self.dt * dy_dt
            next_z = z + self.dt * dz_dt
            return jnp.array([next_x, next_y, next_z])
        return mean_fn


class DoublePendulumDynamics(DynamicsModel):
    """
    双摆系统 / Double pendulum system
    
    复杂的4维非线性动态系统，具有丰富的混沌行为。
    Complex 4D nonlinear dynamical system with rich chaotic behavior.
    
    状态: [θ₁, θ₂, ω₁, ω₂]
    其中 θᵢ 是角度，ωᵢ 是角速度
    """
    
    def __init__(self,
                 dt: float = 0.01,
                 m1: float = 1.0,
                 m2: float = 1.0,
                 L1: float = 1.0,
                 L2: float = 1.0,
                 g: float = 9.81,
                 process_noise_std: float = 0.05):
        """
        Args:
            dt: 时间步长 / time step
            m1, m2: 摆球质量 / pendulum masses
            L1, L2: 摆长 / pendulum lengths
            g: 重力加速度 / gravitational acceleration
            process_noise_std: 过程噪声标准差 / process noise std
        """
        self.dt = dt
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        
        # 过程噪声只加在角速度上 / process noise only on angular velocities
        self.Q = jnp.diag(jnp.array([0.0, 0.0, process_noise_std**2, process_noise_std**2])) * dt
    
    def get_mean_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the deterministic part of the double pendulum dynamics."""
        @jit
        def mean_fn(state: chex.Array) -> chex.Array:
            theta1, theta2, omega1, omega2 = state[0], state[1], state[2], state[3]
            delta_theta = theta2 - theta1
            den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * jnp.cos(delta_theta)**2
            den2 = (self.L2 / self.L1) * den1
            num1 = (-self.m2 * self.L1 * omega1**2 * jnp.sin(delta_theta) * jnp.cos(delta_theta) +
                    self.m2 * self.g * jnp.sin(theta2) * jnp.cos(delta_theta) +
                    self.m2 * self.L2 * omega2**2 * jnp.sin(delta_theta) -
                    (self.m1 + self.m2) * self.g * jnp.sin(theta1))
            num2 = (-self.m2 * self.L2 * omega2**2 * jnp.sin(delta_theta) * jnp.cos(delta_theta) +
                    (self.m1 + self.m2) * self.g * jnp.sin(theta1) * jnp.cos(delta_theta) -
                    (self.m1 + self.m2) * self.L1 * omega1**2 * jnp.sin(delta_theta) -
                    (self.m1 + self.m2) * self.g * jnp.sin(theta2))
            domega1_dt = num1 / den1
            domega2_dt = num2 / den2
            next_theta1 = theta1 + self.dt * omega1
            next_theta2 = theta2 + self.dt * omega2
            next_omega1 = omega1 + self.dt * domega1_dt
            next_omega2 = omega2 + self.dt * domega2_dt
            next_theta1 = jnp.mod(next_theta1 + jnp.pi, 2 * jnp.pi) - jnp.pi
            next_theta2 = jnp.mod(next_theta2 + jnp.pi, 2 * jnp.pi) - jnp.pi
            return jnp.array([next_theta1, next_theta2, next_omega1, next_omega2])
        return mean_fn


# ============================================================================
# 观测模型实现 / Observation Model Implementations
# ============================================================================

class LinearObservation(ObservationModel):
    """
    线性观测模型 / Linear observation model
    
    y_t = H x_t + η_t
    其中 η_t ~ N(0, R)
    """
    
    def __init__(self, H: chex.Array, R: chex.Array):
        """
        Args:
            H: 观测矩阵 [obs_dim, state_dim] / observation matrix
            R: 观测噪声协方差 [obs_dim, obs_dim] / observation noise covariance
        """
        self.H = H
        self.R = R
    
    def get_observation_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the linear observation function."""
        @jit
        def obs_fn(state: chex.Array) -> chex.Array:
            return self.H @ state
        return obs_fn


class PartialObservation(ObservationModel):
    """
    部分观测模型 / Partial observation model
    
    只观测状态的某些分量，常用于实际应用中。
    Observe only some components of the state, commonly used in practical applications.
    """
    
    def __init__(self, observed_indices: list, obs_noise_std: float = 0.1):
        """
        Args:
            observed_indices: 被观测的状态分量索引 / indices of observed state components
            obs_noise_std: 观测噪声标准差 / observation noise std
        """
        self.observed_indices = jnp.array(observed_indices)
        self.obs_dim = len(observed_indices)
        self.R = obs_noise_std**2 * jnp.eye(self.obs_dim)
    
    def get_observation_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the partial observation function."""
        @jit
        def obs_fn(state: chex.Array) -> chex.Array:
            return state[self.observed_indices]
        return obs_fn


class NonlinearObservation(ObservationModel):
    """
    非线性观测模型 / Nonlinear observation model
    
    y_t = h(x_t) + η_t
    其中 h(·) 是非线性函数
    """
    
    def __init__(self, observation_fn: callable, R: chex.Array):
        """
        Args:
            observation_fn: 非线性观测函数 / nonlinear observation function
            R: 观测噪声协方差 / observation noise covariance
        """
        self.observation_fn = jit(observation_fn)
        self.R = R
    
    def get_observation_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the nonlinear observation function."""
        return self.observation_fn


class RangeObservation(ObservationModel):
    """
    距离观测模型 / Range observation model
    
    观测到多个传感器的距离，常用于跟踪和定位问题。
    Observe distances to multiple sensors, commonly used in tracking and localization problems.
    
    y_t = ||x_t - sensor_loc|| + η_t
    """
    
    def __init__(self, sensor_locations: chex.Array, obs_noise_std: float = 0.1):
        """
        Args:
            sensor_locations: 传感器位置 [num_sensors, spatial_dim] / sensor positions
            obs_noise_std: 观测噪声标准差 / observation noise std
        """
        self.sensor_locations = sensor_locations
        self.num_sensors = sensor_locations.shape[0]
        self.R = obs_noise_std**2 * jnp.eye(self.num_sensors)
    
    def get_observation_function(self) -> Callable[[chex.Array], chex.Array]:
        """Returns the range observation function."""
        @jit
        def obs_fn(state: chex.Array) -> chex.Array:
            position = state[:self.sensor_locations.shape[1]]
            distances = jnp.linalg.norm(
                position[None, :] - self.sensor_locations, axis=1
            )
            return distances
        return obs_fn


# ============================================================================
# 模型工厂函数 / Model Factory Functions
# ============================================================================

def create_pendulum_system(dt: float = 0.05,
                          obs_noise_std: float = 0.05,
                          process_noise_std: float = 0.1):
    """
    创建单摆系统 / Create pendulum system
    
    Returns:
        dynamics_model: 单摆动态模型 / pendulum dynamics model
        observation_model: 角度观测模型 / angle observation model
    """
    dynamics = PendulumDynamics(
        dt=dt,
        process_noise_std=process_noise_std
    )
    
    # 只观测角度 / observe only angle
    observation = PartialObservation(
        observed_indices=[0],  # 角度分量
        obs_noise_std=obs_noise_std
    )
    
    return dynamics, observation


def create_lorenz_system(dt: float = 0.01,
                        obs_noise_std: float = 0.1,
                        process_noise_std: float = 0.1):
    """
    创建洛伦兹系统 / Create Lorenz system
    
    Returns:
        dynamics_model: 洛伦兹动态模型 / Lorenz dynamics model
        observation_model: 部分观测模型 / partial observation model
    """
    dynamics = LorenzDynamics(
        dt=dt,
        process_noise_std=process_noise_std
    )
    
    # 观测x和y分量 / observe x and y components
    observation = PartialObservation(
        observed_indices=[0, 1],
        obs_noise_std=obs_noise_std
    )
    
    return dynamics, observation


def create_linear_system(A: chex.Array,
                        H: chex.Array,
                        Q: Optional[chex.Array] = None,
                        R: Optional[chex.Array] = None):
    """
    创建线性系统 / Create linear system
    
    Args:
        A: 转移矩阵 / transition matrix
        H: 观测矩阵 / observation matrix
        Q: 过程噪声协方差 / process noise covariance
        R: 观测噪声协方差 / observation noise covariance
    
    Returns:
        dynamics_model: 线性动态模型 / linear dynamics model
        observation_model: 线性观测模型 / linear observation model
    """
    state_dim = A.shape[0]
    obs_dim = H.shape[0]
    
    if Q is None:
        Q = 0.1 * jnp.eye(state_dim)
    if R is None:
        R = 0.1 * jnp.eye(obs_dim)
    
    dynamics = LinearDynamics(A=A, Q=Q)
    observation = LinearObservation(H=H, R=R)
    
    return dynamics, observation


def create_tracking_system(dt: float = 0.1,
                          sensor_locations: chex.Array = None,
                          obs_noise_std: float = 0.1,
                          process_noise_std: float = 0.1):
    """
    创建目标跟踪系统 / Create target tracking system
    
    4维状态: [x, y, vx, vy] (位置和速度)
    观测: 到传感器的距离
    
    Returns:
        dynamics_model: 常速度运动模型 / constant velocity motion model
        observation_model: 距离观测模型 / range observation model
    """
    # 常速度运动模型 / constant velocity motion model
    A = jnp.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    Q = process_noise_std**2 * jnp.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt]
    ])
    
    dynamics = LinearDynamics(A=A, Q=Q)
    
    # 默认传感器位置 / default sensor positions
    if sensor_locations is None:
        sensor_locations = jnp.array([
            [0, 0],
            [10, 0], 
            [0, 10],
            [10, 10]
        ])
    
    observation = RangeObservation(
        sensor_locations=sensor_locations,
        obs_noise_std=obs_noise_std
    )
    
    return dynamics, observation