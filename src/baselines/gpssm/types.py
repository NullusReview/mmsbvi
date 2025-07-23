"""
GPSSM (Gaussian Process State Space Model) Type Definitions / GPSSM 类型定义
=========================================================================

This module defines all data structures used in the GPSSM implementation,
following best practices for type-safety and clarity using `chex.dataclass`.

此模块使用 `chex.dataclass` 定义了GPSSM实现中用到的所有数据结构，
遵循类型安全和代码清晰的最佳实践。
"""

from typing import NamedTuple, Union
import chex
from flax.struct import dataclass

# ============================================================================
# Configuration Structures / 配置结构
# ============================================================================

@dataclass
class GPSSMConfig:
    """
    Configuration parameters for the GPSSM model.
    GPSSM模型的配置参数。

    Attributes:
        state_dim: Dimensionality of the latent state space (D).
                   潜在状态空间的维度 (D)。
        obs_dim: Dimensionality of the observation space (P).
                 观测空间的维度 (P)。
        num_inducing: Number of inducing points for the sparse GP (M).
                      稀疏GP的诱导点数量 (M)。
        num_particles: Number of particles (samples) for Monte Carlo estimation.
                       用于蒙特卡洛估计的粒子（样本）数量。
        jitter: Small constant added to covariance matrices for numerical stability.
                为保证数值稳定性加到协方差矩阵的抖动项。
    """
    state_dim: int
    obs_dim: int
    num_inducing: int
    num_particles: int = 10
    jitter: float = 1e-4


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    优化器配置。

    Attributes:
        learning_rate: The learning rate for the optimizer.
                       优化器的学习率。
        num_iterations: The total number of optimization iterations.
                        总优化迭代次数。
        clip_norm: Maximum gradient norm for gradient clipping.
                   用于梯度裁剪的最大范数。
    """
    learning_rate: float = 1e-3
    num_iterations: int = 2000
    clip_norm: float = 10.0


# ============================================================================
# Parameter Structures / 参数结构
# ============================================================================

@dataclass
class KernelParams:
    """
    Parameters for the RBF kernel.
    RBF核函数的参数。

    Attributes:
        lengthscale: The lengthscale parameter of the RBF kernel (l).
                     RBF核的长度尺度参数 (l)。
        variance: The signal variance of the RBF kernel (σ_f^2).
                  RBF核的信号方差 (σ_f^2)。
    """
    lengthscale: chex.Array
    variance: chex.Scalar


@dataclass
class InducingPoints:
    """
    Inducing point variables for the sparse GP.
    稀疏GP的诱导点变量。

    Attributes:
        z: Inducing point locations [M, D].
           诱导点位置 Z。
        m: Variational mean of inducing variables [M, D].
           诱导变量的变分均值 m。
        L: Cholesky factor of the variational covariance of inducing variables [M, M].
           诱导变量变分协方差的Cholesky因子 L (S = L @ L.T)。
    """
    z: chex.Array
    m: chex.Array
    L: chex.Array


@dataclass
class GPParams:
    """
    All parameters related to the Gaussian Process.
    与高斯过程相关的所有参数。

    Attributes:
        kernel: Parameters of the kernel function.
                核函数参数。
        inducing: Inducing point variables.
                  诱导点变量。
        obs_noise_variance: Variance of the observation noise (η_t).
                            观测噪声的方差 (η_t)。
    """
    kernel: KernelParams
    inducing: InducingPoints
    obs_noise_variance: chex.Scalar


@dataclass
class VariationalParams:
    """
    Variational parameters for the latent states' approximate posterior.
    潜在状态近似后验的变分参数。

    Attributes:
        q_mu: Mean of the variational distribution q(x_{1:T}) [T, D].
              变分分布 q(x_{1:T}) 的均值。
        q_sqrt: Cholesky factor of the variational covariance q(x_{1:T}) [T, D, D].
                变分分布 q(x_{1:T}) 协方差的Cholesky因子。
    """
    q_mu: chex.Array
    q_sqrt: chex.Array


# ============================================================================
# State and Training Structures / 状态与训练结构
# ============================================================================

@dataclass
class GPSSMState:
    """
    A container for all trainable parameters of the GPSSM.
    包含GPSSM所有可训练参数的容器。
    """
    gp: GPParams
    variational: VariationalParams


@dataclass
class TrainingState:
    """
    A container for the entire state of the training process.
    包含整个训练过程状态的容器。
    """
    params: GPSSMState
    opt_state: chex.ArrayTree
    key: chex.PRNGKey
    iteration: int