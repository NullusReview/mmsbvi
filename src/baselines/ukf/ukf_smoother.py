"""
无味卡尔曼滤波平滑器 / Unscented Kalman Filter Smoother
===================================================

针对大角度单摆系统的UKF平滑器实现，包括：
- Sigma点生成和权重计算
- 无味变换（Unscented Transform）
- 前向滤波过程
- 后向平滑过程
- 周期性角度处理
- 数值稳定性保证

UKF Smoother implementation for large angle pendulum system, including:
- Sigma point generation and weight computation
- Unscented Transform
- Forward filtering process
- Backward smoothing process
- Periodic angle handling
- Numerical stability guarantees
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, Tuple, Optional, List, NamedTuple
from functools import partial
import chex


class UKFState(NamedTuple):
    """UKF状态表示 / UKF state representation"""
    mean: chex.Array  # 状态均值 [θ, ω] / state mean [angle, angular_velocity]
    covariance: chex.Array  # 状态协方差 (2, 2) / state covariance
    log_likelihood: chex.Scalar  # 对数似然 / log-likelihood


class UKFResult(NamedTuple):
    """UKF结果 / UKF result"""
    filtered_states: List[UKFState]  # 滤波状态序列 / filtered state sequence
    smoothed_states: List[UKFState]  # 平滑状态序列 / smoothed state sequence
    total_log_likelihood: float  # 总对数似然 / total log-likelihood
    runtime: float  # 运行时间 / runtime


class PendulumUKFSmoother:
    """
    大角度单摆系统的UKF平滑器 / UKF smoother for large angle pendulum system
    
    实现完整的UKF平滑算法，使用sigma点近似状态分布。
    数学原理基于Unscented Kalman Filter和RTS平滑器框架。
    
    关键特性：
    - 非线性sin(θ)重力项的精确sigma点变换
    - 周期性角度处理 θ ∈ [-π, π]
    - 倒立点附近的数值稳定性
    - 角度包装确保状态连续性
    
    Implements complete UKF smoothing algorithm using sigma points to approximate state distribution.
    Mathematical foundation based on Unscented Kalman Filter and RTS smoother framework.
    
    Key features:
    - Exact sigma point transform for nonlinear sin(θ) gravity term
    - Periodic angle handling θ ∈ [-π, π]
    - Numerical stability near inverted point
    - Angle wrapping ensures state continuity
    """
    
    def __init__(
        self,
        dt: float = 0.05,
        g: float = 9.81,
        L: float = 1.0,
        gamma: float = 0.2,
        sigma: float = 0.3,
        process_noise_scale: float = 0.1,
        obs_noise_std: float = 0.05,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 1.0
    ):
        """
        初始化UKF平滑器 / Initialize UKF smoother
        
        Args:
            dt: 时间步长 / time step
            g: 重力加速度 / gravitational acceleration
            L: 单摆长度 / pendulum length
            gamma: 阻尼系数 / damping coefficient
            sigma: 噪声强度 / noise intensity
            process_noise_scale: 过程噪声缩放 / process noise scaling
            obs_noise_std: 观测噪声标准差 / observation noise std
            alpha: UKF缩放参数 / UKF scaling parameter
            beta: 高阶矩参数 / higher-order moment parameter
            kappa: 次要缩放参数 / secondary scaling parameter
        """
        self.dt = dt
        self.g = g
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.process_noise_scale = process_noise_scale
        self.obs_noise_std = obs_noise_std
        
        # UKF参数 / UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # 状态维度 / state dimension
        self.n_state = 2  # [θ, ω]
        self.n_obs = 1    # [θ]
        
        # 计算UKF缩放参数 / compute UKF scaling parameters
        self.lambda_param = self.alpha**2 * (self.n_state + self.kappa) - self.n_state
        self.gamma = jnp.sqrt(self.n_state + self.lambda_param)
        
        # 计算权重 / compute weights
        self.weights_mean, self.weights_cov = self._compute_weights()
        
        # 预编译核心函数 / precompile core functions
        self._state_transition = jax.jit(self._state_transition_impl)
        self._observation_function = jax.jit(self._observation_function_impl)
        self._generate_sigma_points = jax.jit(self._generate_sigma_points_impl)
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
    
    def _compute_weights(self) -> Tuple[chex.Array, chex.Array]:
        """
        计算UKF权重 / Compute UKF weights
        
        Returns:
            weights_mean: 均值权重 / mean weights
            weights_cov: 协方差权重 / covariance weights
        """
        n_sigma = 2 * self.n_state + 1
        
        # 均值权重 / mean weights
        weights_mean = jnp.zeros(n_sigma)
        weights_mean = weights_mean.at[0].set(self.lambda_param / (self.n_state + self.lambda_param))
        weights_mean = weights_mean.at[1:].set(1.0 / (2 * (self.n_state + self.lambda_param)))
        
        # 协方差权重 / covariance weights
        weights_cov = jnp.zeros(n_sigma)
        weights_cov = weights_cov.at[0].set(
            self.lambda_param / (self.n_state + self.lambda_param) + 
            (1 - self.alpha**2 + self.beta)
        )
        weights_cov = weights_cov.at[1:].set(1.0 / (2 * (self.n_state + self.lambda_param)))
        
        return weights_mean, weights_cov
    
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
    def _observation_function_impl(self, state: chex.Array) -> chex.Array:
        """
        观测函数 / Observation function
        
        Args:
            state: 状态 [θ, ω] / state [angle, angular_velocity]
            
        Returns:
            observation: 观测值 [θ] / observation [angle]
        """
        return jnp.array([state[0]])  # 只观测角度
    
    @partial(jax.jit, static_argnums=(0,))
    def _generate_sigma_points_impl(
        self, 
        mean: chex.Array, 
        covariance: chex.Array
    ) -> chex.Array:
        """
        生成sigma点 / Generate sigma points
        
        对于n维状态，生成2n+1个sigma点：
        χ₀ = μ
        χᵢ = μ + γ(√P)ᵢ, i = 1, ..., n
        χᵢ = μ - γ(√P)ᵢ₋ₙ, i = n+1, ..., 2n
        
        Args:
            mean: 状态均值 / state mean
            covariance: 状态协方差 / state covariance
            
        Returns:
            sigma_points: sigma点矩阵 (2n+1, n) / sigma points matrix
        """
        n = mean.shape[0]
        n_sigma = 2 * n + 1
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        covariance = self._ensure_positive_definite(covariance)
        
        # 计算协方差矩阵的平方根 / compute square root of covariance matrix
        sqrt_cov = jnp.linalg.cholesky(covariance)
        
        # 生成sigma点 / generate sigma points
        sigma_points = jnp.zeros((n_sigma, n))
        
        # 中心点 / center point
        sigma_points = sigma_points.at[0].set(mean)
        
        # 正向sigma点 / positive sigma points
        for i in range(n):
            sigma_point = mean + self.gamma * sqrt_cov[:, i]
            # 对角度分量进行包装 / wrap angle component
            sigma_point = sigma_point.at[0].set(self._wrap_angle(sigma_point[0]))
            sigma_points = sigma_points.at[i+1].set(sigma_point)
        
        # 负向sigma点 / negative sigma points
        for i in range(n):
            sigma_point = mean - self.gamma * sqrt_cov[:, i]
            # 对角度分量进行包装 / wrap angle component
            sigma_point = sigma_point.at[0].set(self._wrap_angle(sigma_point[0]))
            sigma_points = sigma_points.at[i+n+1].set(sigma_point)
        
        return sigma_points
    
    @partial(jax.jit, static_argnums=(0,))
    def _unscented_transform_state(
        self, 
        sigma_points: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        状态无味变换 / State unscented transform
        
        传播sigma点通过状态转移函数，计算预测均值和协方差。
        
        Args:
            sigma_points: sigma点矩阵 (2n+1, n) / sigma points matrix
            
        Returns:
            pred_mean: 预测均值 / predicted mean
            pred_cov: 预测协方差 / predicted covariance
        """
        n_sigma = sigma_points.shape[0]
        
        # 传播sigma点 / propagate sigma points
        propagated_points = jnp.zeros_like(sigma_points)
        for i in range(n_sigma):
            propagated_points = propagated_points.at[i].set(
                self._state_transition(sigma_points[i])
            )
        
        # 计算预测均值 / compute predicted mean
        pred_mean = jnp.sum(
            self.weights_mean[:, None] * propagated_points, 
            axis=0
        )
        
        # 计算预测协方差 / compute predicted covariance
        pred_cov = jnp.zeros((self.n_state, self.n_state))
        for i in range(n_sigma):
            diff = propagated_points[i] - pred_mean
            pred_cov = pred_cov + self.weights_cov[i] * jnp.outer(diff, diff)
        
        pred_cov = pred_cov + self.Q
        
        return pred_mean, pred_cov
    
    @partial(jax.jit, static_argnums=(0,))
    def _unscented_transform_observation(
        self, 
        sigma_points: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        观测无味变换 / Observation unscented transform
        
        传播sigma点通过观测函数，计算观测预测和协方差。
        
        Args:
            sigma_points: sigma点矩阵 (2n+1, n) / sigma points matrix
            
        Returns:
            obs_mean: 观测预测均值 / predicted observation mean
            obs_cov: 观测预测协方差 / predicted observation covariance
            cross_cov: 交叉协方差 / cross covariance
        """
        n_sigma = sigma_points.shape[0]
        
        # 传播sigma点通过观测函数 / propagate sigma points through observation function
        obs_points = jnp.zeros((n_sigma, self.n_obs))
        for i in range(n_sigma):
            obs_points = obs_points.at[i].set(
                self._observation_function(sigma_points[i])
            )
        
        # 计算观测预测均值 / compute predicted observation mean
        obs_mean = jnp.sum(
            self.weights_mean[:, None] * obs_points, 
            axis=0
        )
        
        # 计算状态预测均值 / compute predicted state mean
        state_mean = jnp.sum(
            self.weights_mean[:, None] * sigma_points, 
            axis=0
        )
        
        # 计算观测预测协方差 / compute predicted observation covariance
        obs_cov = jnp.zeros((self.n_obs, self.n_obs))
        for i in range(n_sigma):
            diff = obs_points[i] - obs_mean
            obs_cov = obs_cov + self.weights_cov[i] * jnp.outer(diff, diff)
        
        obs_cov = obs_cov + self.R
        
        # 计算交叉协方差 / compute cross covariance
        cross_cov = jnp.zeros((self.n_state, self.n_obs))
        for i in range(n_sigma):
            state_diff = sigma_points[i] - state_mean
            obs_diff = obs_points[i] - obs_mean
            cross_cov = cross_cov + self.weights_cov[i] * jnp.outer(state_diff, obs_diff)
        
        return obs_mean, obs_cov, cross_cov
    
    @partial(jax.jit, static_argnums=(0,))
    def _predict_step_impl(
        self, 
        prev_state: UKFState
    ) -> UKFState:
        """
        UKF预测步骤 / UKF prediction step
        
        Args:
            prev_state: 前一时刻状态 / previous state
            
        Returns:
            predicted_state: 预测状态 / predicted state
        """
        # 生成sigma点 / generate sigma points
        sigma_points = self._generate_sigma_points(prev_state.mean, prev_state.covariance)
        
        # 状态无味变换 / state unscented transform
        pred_mean, pred_cov = self._unscented_transform_state(sigma_points)
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        pred_cov = self._ensure_positive_definite(pred_cov)
        
        return UKFState(
            mean=pred_mean,
            covariance=pred_cov,
            log_likelihood=prev_state.log_likelihood
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_step_impl(
        self,
        predicted_state: UKFState,
        observation: float
    ) -> UKFState:
        """
        UKF更新步骤 / UKF update step
        
        Args:
            predicted_state: 预测状态 / predicted state
            observation: 观测值 / observation
            
        Returns:
            updated_state: 更新后状态 / updated state
        """
        # 生成sigma点 / generate sigma points
        sigma_points = self._generate_sigma_points(
            predicted_state.mean, 
            predicted_state.covariance
        )
        
        # 观测无味变换 / observation unscented transform
        obs_mean, obs_cov, cross_cov = self._unscented_transform_observation(sigma_points)
        
        # 创建观测向量 / create observation vector
        y = jnp.array([observation])
        
        # 观测残差 / observation residual
        innovation = y - obs_mean
        
        # 确保观测协方差可逆 / ensure observation covariance is invertible
        obs_cov = obs_cov + jnp.eye(obs_cov.shape[0]) * 1e-8
        
        # 卡尔曼增益 / Kalman gain
        K = cross_cov @ jnp.linalg.inv(obs_cov)
        
        # 状态更新 / state update
        updated_mean = predicted_state.mean + K @ innovation
        
        # 协方差更新 / covariance update
        updated_covariance = predicted_state.covariance - K @ obs_cov @ K.T
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        updated_covariance = self._ensure_positive_definite(updated_covariance)
        
        # 计算对数似然 / compute log-likelihood
        log_likelihood = self._compute_log_likelihood(innovation, obs_cov)
        
        return UKFState(
            mean=updated_mean,
            covariance=updated_covariance,
            log_likelihood=predicted_state.log_likelihood + log_likelihood
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _smooth_step_impl(
        self,
        filtered_state: UKFState,
        next_filtered_state: UKFState,
        next_smoothed_state: UKFState
    ) -> UKFState:
        """
        UKF平滑步骤 / UKF smoothing step
        
        使用类似RTS的后向递归框架，但使用UKF的预测结果。
        
        Args:
            filtered_state: 滤波状态 / filtered state
            next_filtered_state: 下一时刻滤波状态 / next filtered state
            next_smoothed_state: 下一时刻平滑状态 / next smoothed state
            
        Returns:
            smoothed_state: 平滑状态 / smoothed state
        """
        # 生成sigma点 / generate sigma points
        sigma_points = self._generate_sigma_points(
            filtered_state.mean, 
            filtered_state.covariance
        )
        
        # 状态无味变换得到预测结果 / state unscented transform for prediction
        pred_mean, pred_cov = self._unscented_transform_state(sigma_points)
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        pred_cov = self._ensure_positive_definite(pred_cov)
        
        # 计算平滑增益 / compute smoothing gain
        A = filtered_state.covariance @ jnp.linalg.solve(pred_cov, jnp.eye(self.n_state))
        
        # 平滑状态 / smoothed state
        smoothed_mean = filtered_state.mean + A @ (next_smoothed_state.mean - pred_mean)
        
        # 平滑协方差 / smoothed covariance
        smoothed_covariance = (
            filtered_state.covariance + 
            A @ (next_smoothed_state.covariance - pred_cov) @ A.T
        )
        
        # 确保协方差矩阵正定 / ensure positive definite covariance
        smoothed_covariance = self._ensure_positive_definite(smoothed_covariance)
        
        return UKFState(
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
    
    def smooth(
        self,
        observations: chex.Array,
        initial_mean: Optional[chex.Array] = None,
        initial_cov: Optional[chex.Array] = None
    ) -> UKFResult:
        """
        执行完整的UKF平滑 / Perform complete UKF smoothing
        
        Args:
            observations: 观测序列 (T,) / observation sequence
            initial_mean: 初始状态均值 / initial state mean
            initial_cov: 初始状态协方差 / initial state covariance
            
        Returns:
            result: UKF平滑结果 / UKF smoothing result
        """
        import time
        start_time = time.time()
        
        T = len(observations)
        
        # 设置初始状态 / set initial state
        if initial_mean is None:
            initial_mean = jnp.array([observations[0], 0.0])  # 初始角速度为0
        if initial_cov is None:
            initial_cov = jnp.eye(2) * 1.0  # 初始不确定性
        
        initial_state = UKFState(
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
        
        return UKFResult(
            filtered_states=filtered_states,
            smoothed_states=smoothed_states,
            total_log_likelihood=total_log_likelihood,
            runtime=runtime
        )
    
    def extract_estimates(self, result: UKFResult) -> Dict[str, chex.Array]:
        """
        提取状态估计 / Extract state estimates
        
        Returns:
            estimates: 包含角度和角速度估计及不确定性 / contains angle and angular velocity estimates with uncertainty
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