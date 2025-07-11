"""
扩展卡尔曼滤波平滑器 / Extended Kalman Filter Smoother
=====================================================

针对大角度单摆系统的EKF平滑器实现，包括：
- 非线性sin(θ)雅可比矩阵计算
- 周期性角度处理
- 前向滤波过程
- 后向平滑过程（RTS平滑器）
- 数值稳定性保证

EKF Smoother implementation for large angle pendulum system, including:
- Nonlinear sin(θ) Jacobian computation
- Periodic angle handling
- Forward filtering process
- Backward smoothing process (RTS smoother)
- Numerical stability guarantees
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, Tuple, Optional, List, NamedTuple
from functools import partial
import chex


class EKFState(NamedTuple):
    """EKF状态表示 / EKF state representation"""
    mean: chex.Array  # 状态均值 [θ, ω] / state mean [angle, angular_velocity]
    covariance: chex.Array  # 状态协方差 (2, 2) / state covariance
    log_likelihood: chex.Scalar  # 对数似然 / log-likelihood


class EKFResult(NamedTuple):
    """EKF结果 / EKF result"""
    filtered_states: List[EKFState]  # 滤波状态序列 / filtered state sequence
    smoothed_states: List[EKFState]  # 平滑状态序列 / smoothed state sequence
    total_log_likelihood: float  # 总对数似然 / total log-likelihood
    runtime: float  # 运行时间 / runtime


class PendulumEKFSmoother:
    """
    大角度单摆系统的EKF平滑器 / EKF smoother for large angle pendulum system
    
    实现完整的EKF平滑算法，包括前向滤波和后向平滑。
    数学原理基于Extended Kalman Filter和Rauch-Tung-Striebel平滑器。
    
    关键特性：
    - 非线性sin(θ)重力项的精确雅可比矩阵
    - 周期性角度处理 θ ∈ [-π, π]
    - 倒立点附近的数值稳定性
    - 角度包装函数确保状态连续性
    
    Implements complete EKF smoothing algorithm with forward filtering and backward smoothing.
    Mathematical foundation based on Extended Kalman Filter and Rauch-Tung-Striebel smoother.
    
    Key features:
    - Exact Jacobian matrix for nonlinear sin(θ) gravity term
    - Periodic angle handling θ ∈ [-π, π]
    - Numerical stability near inverted point
    - Angle wrapping function ensures state continuity
    """
    
    def __init__(
        self,
        dt: float = 0.05,
        g: float = 9.81,
        L: float = 1.0,
        gamma: float = 0.2,
        sigma: float = 0.3,
        process_noise_scale: float = 0.1,
        obs_noise_std: float = 0.05
    ):
        """
        初始化EKF平滑器 / Initialize EKF smoother
        
        Args:
            dt: 时间步长 / time step
            g: 重力加速度 / gravitational acceleration
            L: 单摆长度 / pendulum length
            gamma: 阻尼系数 / damping coefficient
            sigma: 噪声强度 / noise intensity
            process_noise_scale: 过程噪声缩放 / process noise scaling
            obs_noise_std: 观测噪声标准差 / observation noise std
        """
        self.dt = dt
        self.g = g
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.process_noise_scale = process_noise_scale
        self.obs_noise_std = obs_noise_std
        
        # 预编译核心函数 / precompile core functions
        self._state_transition = jax.jit(self._state_transition_impl)
        self._compute_jacobian = jax.jit(self._compute_jacobian_impl)
        self._predict_step = jax.jit(self._predict_step_impl)
        self._update_step = jax.jit(self._update_step_impl)
        self._smooth_step = jax.jit(self._smooth_step_impl)
        
        # 构建过程噪声协方差矩阵 / construct process noise covariance matrix
        # Q矩阵反映单摆系统的随机性：角度无直接噪声，角速度有噪声
        self.Q = jnp.array([
            [0.0, 0.0],  # 角度无直接过程噪声
            [0.0, (self.sigma * self.process_noise_scale)**2]  # 角速度有噪声
        ]) * self.dt
        
        # 观测噪声协方差矩阵 / observation noise covariance matrix
        self.R = jnp.array([[self.obs_noise_std**2]])
        
        # 观测矩阵 / observation matrix (只观测角度)
        self.H = jnp.array([[1.0, 0.0]])
    
    @partial(jax.jit, static_argnums=(0,))
    def _state_transition_impl(self, state: chex.Array) -> chex.Array:
        """
        状态转移函数 / State transition function
        
        实现大角度单摆系统的欧拉积分：
        dθ/dt = ω
        dω/dt = -(g/L)sin(θ) - γω + σ*dW
        
        Args:
            state: 当前状态 [θ, ω] / current state [angle, angular_velocity]
            
        Returns:
            next_state: 下一时刻状态 / next state
        """
        theta, omega = state[0], state[1]
        
        # 大角度单摆动力学 / Large angle pendulum dynamics
        dtheta_dt = omega
        domega_dt = -(self.g/self.L) * jnp.sin(theta) - self.gamma * omega
        
        # 欧拉积分 / Euler integration
        next_theta = theta + self.dt * dtheta_dt
        next_omega = omega + self.dt * domega_dt
        
        # 角度包装到[-π, π] / Wrap angle to [-π, π]
        next_theta = self._wrap_angle(next_theta)
        
        next_state = jnp.array([next_theta, next_omega])
        
        return next_state
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_jacobian_impl(self, state: chex.Array) -> chex.Array:
        """
        计算状态转移雅可比矩阵 / Compute state transition Jacobian matrix
        
        F = ∂f/∂x = [∂f₁/∂θ  ∂f₁/∂ω]
                    [∂f₂/∂θ  ∂f₂/∂ω]
        
        其中 f₁ = θ + dt*ω, f₂ = ω + dt*(-(g/L)sin(θ) - γω)
        
        Args:
            state: 当前状态 [θ, ω] / current state [angle, angular_velocity]
            
        Returns:
            F: 雅可比矩阵 (2, 2) / Jacobian matrix
        """
        theta, omega = state[0], state[1]
        
        # 计算偏导数 / compute partial derivatives
        # ∂f₁/∂θ = 1, ∂f₁/∂ω = dt
        # ∂f₂/∂θ = dt*(-(g/L)cos(θ)), ∂f₂/∂ω = 1 - dt*γ
        # 注意：sin(θ)对θ的导数是cos(θ)
        F = jnp.array([
            [1.0, self.dt],
            [self.dt * (-(self.g/self.L) * jnp.cos(theta)), 1.0 - self.dt * self.gamma]
        ])
        
        return F
    
    @partial(jax.jit, static_argnums=(0,))
    def _predict_step_impl(
        self, 
        prev_state: EKFState
    ) -> EKFState:
        """
        EKF预测步骤 / EKF prediction step
        
        预测均值: x̂[k|k-1] = f(x̂[k-1|k-1])
        预测协方差: P[k|k-1] = F[k-1] * P[k-1|k-1] * F[k-1]ᵀ + Q[k-1]
        
        Args:
            prev_state: 前一时刻状态 / previous state
            
        Returns:
            predicted_state: 预测状态 / predicted state
        """
        # 状态预测 / state prediction
        predicted_mean = self._state_transition(prev_state.mean)
        
        # 雅可比矩阵 / Jacobian matrix
        F = self._compute_jacobian(prev_state.mean)
        
        # 协方差预测 / covariance prediction
        predicted_covariance = F @ prev_state.covariance @ F.T + self.Q
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        predicted_covariance = self._ensure_positive_definite(predicted_covariance)
        
        return EKFState(
            mean=predicted_mean,
            covariance=predicted_covariance,
            log_likelihood=prev_state.log_likelihood
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_step_impl(
        self,
        predicted_state: EKFState,
        observation: float
    ) -> EKFState:
        """
        EKF更新步骤 / EKF update step
        
        卡尔曼增益: K[k] = P[k|k-1] * H[k]ᵀ * (H[k] * P[k|k-1] * H[k]ᵀ + R[k])⁻¹
        状态更新: x̂[k|k] = x̂[k|k-1] + K[k] * (y[k] - H[k] * x̂[k|k-1])
        协方差更新: P[k|k] = (I - K[k] * H[k]) * P[k|k-1]
        
        Args:
            predicted_state: 预测状态 / predicted state
            observation: 观测值 / observation
            
        Returns:
            updated_state: 更新后状态 / updated state
        """
        # 创建观测向量 / create observation vector
        y = jnp.array([observation])
        
        # 预测观测 / predicted observation
        predicted_obs = self.H @ predicted_state.mean
        
        # 观测残差 / observation residual
        innovation = y - predicted_obs
        
        # 创新协方差 / innovation covariance
        S = self.H @ predicted_state.covariance @ self.H.T + self.R
        
        # 确保创新协方差可逆 / ensure innovation covariance is invertible
        S = S + jnp.eye(S.shape[0]) * 1e-8
        
        # 卡尔曼增益 / Kalman gain
        K = predicted_state.covariance @ self.H.T @ jnp.linalg.inv(S)
        
        # 状态更新 / state update
        updated_mean = predicted_state.mean + K @ innovation
        
        # 协方差更新（Joseph形式，数值稳定）/ covariance update (Joseph form, numerically stable)
        I_KH = jnp.eye(2) - K @ self.H
        updated_covariance = I_KH @ predicted_state.covariance @ I_KH.T + K @ self.R @ K.T
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        updated_covariance = self._ensure_positive_definite(updated_covariance)
        
        # 计算对数似然 / compute log-likelihood
        log_likelihood = self._compute_log_likelihood(innovation, S)
        
        return EKFState(
            mean=updated_mean,
            covariance=updated_covariance,
            log_likelihood=predicted_state.log_likelihood + log_likelihood
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _smooth_step_impl(
        self,
        filtered_state: EKFState,
        next_filtered_state: EKFState,
        next_smoothed_state: EKFState
    ) -> EKFState:
        """
        EKF平滑步骤 / EKF smoothing step
        
        RTS平滑器后向递归：
        A[k] = P[k|k] * F[k]ᵀ * P[k+1|k]⁻¹
        x̂[k|T] = x̂[k|k] + A[k] * (x̂[k+1|T] - x̂[k+1|k])
        P[k|T] = P[k|k] + A[k] * (P[k+1|T] - P[k+1|k]) * A[k]ᵀ
        
        Args:
            filtered_state: 滤波状态 / filtered state
            next_filtered_state: 下一时刻滤波状态 / next filtered state
            next_smoothed_state: 下一时刻平滑状态 / next smoothed state
            
        Returns:
            smoothed_state: 平滑状态 / smoothed state
        """
        # 计算雅可比矩阵 / compute Jacobian matrix
        F = self._compute_jacobian(filtered_state.mean)
        
        # 预测协方差 / predicted covariance
        predicted_cov = F @ filtered_state.covariance @ F.T + self.Q
        predicted_cov = self._ensure_positive_definite(predicted_cov)
        
        # 平滑增益 / smoothing gain
        A = filtered_state.covariance @ F.T @ jnp.linalg.inv(predicted_cov)
        
        # 预测状态 / predicted state
        predicted_mean = self._state_transition(filtered_state.mean)
        
        # 平滑状态 / smoothed state
        smoothed_mean = filtered_state.mean + A @ (next_smoothed_state.mean - predicted_mean)
        
        # 平滑协方差 / smoothed covariance
        smoothed_covariance = (
            filtered_state.covariance + 
            A @ (next_smoothed_state.covariance - predicted_cov) @ A.T
        )
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        smoothed_covariance = self._ensure_positive_definite(smoothed_covariance)
        
        return EKFState(
            mean=smoothed_mean,
            covariance=smoothed_covariance,
            log_likelihood=filtered_state.log_likelihood
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _ensure_positive_definite(self, cov: chex.Array) -> chex.Array:
        """
        确保协方差矩阵正定 / Ensure covariance matrix is positive definite
        
        使用eigenvalue decomposition和regularization
        """
        # 对称化 / symmetrize
        cov = 0.5 * (cov + cov.T)
        
        # 特征值分解 / eigenvalue decomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(cov)
        
        # 确保特征值为正 / ensure positive eigenvalues
        eigenvals = jnp.maximum(eigenvals, 1e-8)
        
        # 重构协方差矩阵 / reconstruct covariance matrix
        cov_regularized = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        return cov_regularized
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_log_likelihood(
        self, 
        innovation: chex.Array, 
        innovation_cov: chex.Array
    ) -> chex.Scalar:
        """
        计算对数似然 / Compute log-likelihood
        
        log p(y|x) = -0.5 * [log(2π) + log|S| + innovation^T * S^(-1) * innovation]
        """
        dim = innovation.shape[0]
        
        # 计算行列式的对数 / compute log determinant
        sign, logdet = jnp.linalg.slogdet(innovation_cov)
        
        # 计算二次项 / compute quadratic term
        quadratic_term = innovation.T @ jnp.linalg.solve(innovation_cov, innovation)
        
        # 完整对数似然 / complete log-likelihood
        log_likelihood = -0.5 * (
            dim * jnp.log(2 * jnp.pi) + 
            logdet + 
            quadratic_term
        )
        
        return log_likelihood
    
    @partial(jax.jit, static_argnums=(0,))
    def _wrap_angle(self, theta: chex.Scalar) -> chex.Scalar:
        """
        角度包装函数 / Angle wrapping function
        
        将角度包装到[-π, π]区间，确保周期性边界条件。
        Wrap angle to [-π, π] interval, ensuring periodic boundary conditions.
        
        Args:
            theta: 输入角度 / input angle
            
        Returns:
            wrapped_theta: 包装后的角度 / wrapped angle
        """
        return jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    def smooth(
        self,
        observations: chex.Array,
        initial_mean: Optional[chex.Array] = None,
        initial_cov: Optional[chex.Array] = None
    ) -> EKFResult:
        """
        执行完整的EKF平滑 / Perform complete EKF smoothing
        
        Args:
            observations: 观测序列 (T,) / observation sequence
            initial_mean: 初始状态均值 / initial state mean
            initial_cov: 初始状态协方差 / initial state covariance
            
        Returns:
            result: EKF平滑结果 / EKF smoothing result
        """
        import time
        start_time = time.time()
        
        T = len(observations)
        
        # 设置初始状态 / set initial state
        if initial_mean is None:
            initial_mean = jnp.array([observations[0], 0.0])  # 初始角速度为0
        if initial_cov is None:
            initial_cov = jnp.eye(2) * 1.0  # 初始不确定性
        
        initial_state = EKFState(
            mean=initial_mean,
            covariance=initial_cov,
            log_likelihood=jnp.array(0.0)
        )
        
        # 前向滤波过程 / forward filtering process
        filtered_states = [initial_state]
        
        for t in range(1, T):
            # 预测步骤 / prediction step
            predicted_state = self._predict_step(filtered_states[t-1])
            
            # 更新步骤 / update step
            updated_state = self._update_step(predicted_state, observations[t])
            
            filtered_states.append(updated_state)
        
        # 后向平滑过程 / backward smoothing process
        smoothed_states = [None] * T
        smoothed_states[T-1] = filtered_states[T-1]  # 最后一个状态不变
        
        for t in range(T-2, -1, -1):
            smoothed_states[t] = self._smooth_step(
                filtered_states[t],
                filtered_states[t+1],
                smoothed_states[t+1]
            )
        
        # 计算总对数似然 / compute total log-likelihood
        total_log_likelihood = float(filtered_states[-1].log_likelihood)
        
        runtime = time.time() - start_time
        
        return EKFResult(
            filtered_states=filtered_states,
            smoothed_states=smoothed_states,
            total_log_likelihood=total_log_likelihood,
            runtime=runtime
        )
    
    def extract_estimates(self, result: EKFResult) -> Dict[str, chex.Array]:
        """
        提取状态估计 / Extract state estimates
        
        Returns:
            estimates: 包含位置和速度估计及不确定性 / contains position and velocity estimates with uncertainty
        """
        T = len(result.smoothed_states)
        
        # 提取均值和标准差 / extract means and standard deviations
        theta_means = jnp.array([state.mean[0] for state in result.smoothed_states])
        theta_stds = jnp.array([jnp.sqrt(state.covariance[0, 0]) for state in result.smoothed_states])
        omega_means = jnp.array([state.mean[1] for state in result.smoothed_states])
        omega_stds = jnp.array([jnp.sqrt(state.covariance[1, 1]) for state in result.smoothed_states])
        
        return {
            'theta_mean': theta_means,
            'theta_std': theta_stds,
            'omega_mean': omega_means,
            'omega_std': omega_stds,
            'covariances': [state.covariance for state in result.smoothed_states]
        }