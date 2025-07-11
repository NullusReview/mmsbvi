"""
SVI (Stochastic Variational Inference) 平滑器 / SVI Smoother
=========================================================

实现基于变分推断的大角度单摆系统状态估计器。
Implements variational inference-based state estimator for large angle pendulum system.

数学原理 / Mathematical Principles:
- 使用Gaussian变分分布 q(x_t) = N(μ_t, Σ_t) 近似后验
- 通过最大化ELBO优化变分参数
- 使用reparameterization trick和随机梯度下降

Mathematical principles:
- Use Gaussian variational distribution q(x_t) = N(μ_t, Σ_t) to approximate posterior
- Optimize variational parameters by maximizing ELBO
- Use reparameterization trick and stochastic gradient descent
"""

import jax
import jax.numpy as jnp
import optax
from jax import random, grad, jit, vmap
from typing import NamedTuple, Dict, Any, Optional, Tuple
import chex
from functools import partial
import time


class SVIState(NamedTuple):
    """SVI状态结构 / SVI state structure"""
    means: chex.Array  # (T, 2) 变分均值 / variational means
    log_stds: chex.Array  # (T, 2) 变分对数标准差 / variational log-stds
    total_log_likelihood: chex.Scalar  # 总对数似然 / total log-likelihood
    elbo: chex.Scalar  # ELBO值 / ELBO value
    
    
class SVIParams(NamedTuple):
    """SVI训练参数 / SVI training parameters"""
    means: chex.Array  # (T, 2) 变分均值参数 / variational mean parameters
    log_stds: chex.Array  # (T, 2) 变分对数标准差参数 / variational log-std parameters


class PendulumSVISmoother:
    """
    基于随机变分推断的大角度单摆系统平滑器 / Large Angle Pendulum System Smoother based on SVI
    
    核心思想：
    1. 使用Gaussian变分分布 q(x_t) = N(μ_t, Σ_t) 近似后验
    2. 通过最大化ELBO = E_q[log p(x,y)] - E_q[log q(x)] 优化参数
    3. 使用Adam优化器进行随机梯度下降
    
    Core idea:
    1. Use Gaussian variational distribution q(x_t) = N(μ_t, Σ_t) to approximate posterior
    2. Optimize parameters by maximizing ELBO = E_q[log p(x,y)] - E_q[log q(x)]
    3. Use Adam optimizer for stochastic gradient descent
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
        learning_rate: float = 0.01,
        n_samples: int = 10,
        max_iterations: int = 1000,
        convergence_tol: float = 1e-6
    ):
        """
        初始化SVI平滑器 / Initialize SVI smoother
        
        Args:
            dt: 时间步长 / time step
            g: 重力加速度 / gravitational acceleration
            L: 单摆长度 / pendulum length
            gamma: 阻尼系数 / damping coefficient
            sigma: 噪声强度 / noise intensity
            process_noise_scale: 过程噪声缩放 / process noise scaling
            obs_noise_std: 观测噪声标准差 / observation noise std
            learning_rate: 学习率 / learning rate
            n_samples: 蒙特卡洛样本数 / number of Monte Carlo samples
            max_iterations: 最大迭代次数 / maximum iterations
            convergence_tol: 收敛容差 / convergence tolerance
        """
        self.dt = dt
        self.g = g
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.process_noise_scale = process_noise_scale
        self.obs_noise_std = obs_noise_std
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        
        # 过程噪声协方差矩阵 / process noise covariance matrix
        # 使用更合理的噪声模型：考虑数值稳定性
        angle_noise_var = max((self.process_noise_scale * self.dt)**2, 1e-4)  # 最小方差
        angular_vel_noise_var = max((self.sigma * jnp.sqrt(self.dt))**2, 1e-4)  # 最小方差
        
        self.Q = jnp.diag(jnp.array([angle_noise_var, angular_vel_noise_var]))
        
        # 观测噪声协方差 / observation noise covariance
        self.R = self.obs_noise_std**2
        
        # 编译核心函数 / compile core functions
        self._dynamics_mean = jit(self._dynamics_mean_impl)
        self._log_transition_prob = jit(self._log_transition_prob_impl)
        self._log_observation_prob = jit(self._log_observation_prob_impl)
        self._compute_elbo = jit(self._compute_elbo_impl)
        
        # 初始化优化器 / initialize optimizer
        self.optimizer = optax.adam(learning_rate)
    
    @partial(jit, static_argnums=(0,))
    def _dynamics_mean_impl(self, state: chex.Array) -> chex.Array:
        """
        大角度单摆系统的动态均值 / Large angle pendulum system dynamics mean
        
        dθ = ω * dt
        dω = (-(g/L)sin(θ) - γω) * dt + σ * dW
        
        Args:
            state: 当前状态 [θ, ω] / current state [θ, ω]
            
        Returns:
            next_mean: 下一时刻状态均值 / next state mean
        """
        theta, omega = state[0], state[1]
        
        # 确定性动态 / deterministic dynamics
        dtheta = omega * self.dt
        domega = (-(self.g/self.L) * jnp.sin(theta) - self.gamma * omega) * self.dt
        
        next_theta = theta + dtheta
        next_omega = omega + domega
        
        # 角度包装到[-π, π] / Wrap angle to [-π, π]
        next_theta = jnp.mod(next_theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        next_mean = jnp.array([next_theta, next_omega])
        return next_mean
    
    @partial(jit, static_argnums=(0,))
    def _log_transition_prob_impl(
        self, 
        x_curr: chex.Array, 
        x_prev: chex.Array
    ) -> chex.Scalar:
        """
        计算转移概率的对数 / Compute log transition probability
        
        p(x_t | x_{t-1}) = N(x_t; f(x_{t-1}), Q)
        
        Args:
            x_curr: 当前状态 / current state
            x_prev: 前一状态 / previous state
            
        Returns:
            log_prob: 对数转移概率 / log transition probability
        """
        predicted_mean = self._dynamics_mean_impl(x_prev)
        diff = x_curr - predicted_mean
        
        # 多维高斯对数概率 / multivariate Gaussian log probability
        # log p(x) = -0.5 * [(x-μ)^T Σ^-1 (x-μ) + log|2πΣ|]
        quad_form = jnp.sum(diff * jnp.linalg.solve(self.Q, diff))
        log_det_term = jnp.linalg.slogdet(2 * jnp.pi * self.Q)[1]
        log_prob = -0.5 * (quad_form + log_det_term)
        
        return log_prob
    
    @partial(jit, static_argnums=(0,))
    def _log_observation_prob_impl(
        self, 
        observation: chex.Scalar, 
        state: chex.Array
    ) -> chex.Scalar:
        """
        计算观测概率的对数 / Compute log observation probability
        
        p(y_t | x_t) = N(y_t; x_t[0], R)
        
        Args:
            observation: 观测值 / observation
            state: 状态 [x, v] / state [x, v]
            
        Returns:
            log_prob: 对数观测概率 / log observation probability
        """
        predicted_obs = state[0]  # 观测角度 / observe angle
        diff = observation - predicted_obs
        
        log_prob = -0.5 * diff**2 / self.R
        log_prob -= 0.5 * jnp.log(2 * jnp.pi * self.R)
        
        return log_prob
    
    @partial(jit, static_argnums=(0,))
    def _sample_from_variational(
        self, 
        params: SVIParams, 
        key: chex.PRNGKey
    ) -> chex.Array:
        """
        从变分分布采样 / Sample from variational distribution
        
        Args:
            params: 变分参数 / variational parameters
            key: 随机密钥 / random key
            
        Returns:
            samples: 状态样本 (n_samples, T, 2) / state samples
        """
        T = params.means.shape[0]
        
        # 生成随机噪声 / generate random noise
        noise = random.normal(key, (self.n_samples, T, 2))
        
        # reparameterization trick
        stds = jnp.exp(params.log_stds)
        samples = params.means[None, :, :] + noise * stds[None, :, :]
        
        return samples
    
    @partial(jit, static_argnums=(0,))
    def _compute_elbo_impl(
        self, 
        params: SVIParams, 
        observations: chex.Array,
        key: chex.PRNGKey
    ) -> chex.Scalar:
        """
        计算ELBO（Evidence Lower Bound）/ Compute ELBO
        
        ELBO = E_q[log p(x,y)] - E_q[log q(x)]
             = E_q[log p(x_1) + ∑ log p(x_t|x_{t-1}) + ∑ log p(y_t|x_t)] - E_q[∑ log q(x_t)]
        
        Args:
            params: 变分参数 / variational parameters
            observations: 观测序列 / observation sequence
            key: 随机密钥 / random key
            
        Returns:
            elbo: ELBO值 / ELBO value
        """
        T = len(observations)
        
        # 从变分分布采样 / sample from variational distribution
        samples = self._sample_from_variational(params, key)
        
        # 计算联合对数似然 E_q[log p(x,y)] / compute joint log-likelihood
        def compute_joint_log_likelihood(sample_seq):
            # 先验 log p(x_1) / prior log p(x_1)
            # 假设初始状态为零均值单位协方差 / assume initial state has zero mean unit covariance
            log_prob = -0.5 * jnp.sum(sample_seq[0]**2) - jnp.log(2 * jnp.pi)
            
            # 转移概率 ∑ log p(x_t|x_{t-1}) / transition probabilities
            for t in range(1, T):
                log_prob += self._log_transition_prob_impl(sample_seq[t], sample_seq[t-1])
            
            # 观测概率 ∑ log p(y_t|x_t) / observation probabilities
            for t in range(T):
                log_prob += self._log_observation_prob_impl(observations[t], sample_seq[t])
            
            return log_prob
        
        # 并行计算所有样本的联合对数似然 / compute joint log-likelihood for all samples in parallel
        joint_log_probs = vmap(compute_joint_log_likelihood)(samples)
        expected_joint_log_prob = jnp.mean(joint_log_probs)
        
        # 计算变分分布的熵 E_q[log q(x)] / compute entropy of variational distribution
        def compute_variational_log_prob(sample_seq):
            log_prob = 0.0
            for t in range(T):
                # log q(x_t) = log N(x_t; μ_t, Σ_t)
                diff = sample_seq[t] - params.means[t]
                stds = jnp.exp(params.log_stds[t])
                log_prob += -0.5 * jnp.sum((diff / stds)**2)
                log_prob += -jnp.sum(params.log_stds[t]) - jnp.log(2 * jnp.pi)
            return log_prob
        
        variational_log_probs = vmap(compute_variational_log_prob)(samples)
        expected_variational_log_prob = jnp.mean(variational_log_probs)
        
        # ELBO = E_q[log p(x,y)] - E_q[log q(x)]
        elbo = expected_joint_log_prob - expected_variational_log_prob
        
        return elbo
    
    def smooth(
        self,
        observations: chex.Array,
        initial_mean: chex.Array,
        initial_cov: chex.Array,
        key: Optional[chex.PRNGKey] = None
    ) -> SVIState:
        """
        执行SVI平滑 / Perform SVI smoothing
        
        Args:
            observations: 观测序列 (T,) / observation sequence
            initial_mean: 初始状态均值 (2,) / initial state mean
            initial_cov: 初始状态协方差 (2, 2) / initial state covariance
            key: 随机密钥 / random key
            
        Returns:
            result: SVI平滑结果 / SVI smoothing result
        """
        if key is None:
            key = random.PRNGKey(42)
        
        T = len(observations)
        
        # 初始化变分参数 / initialize variational parameters
        # 使用EKF风格的初始化 / use EKF-style initialization
        init_means = jnp.zeros((T, 2))
        init_means = init_means.at[0].set(initial_mean)
        
        # 简单的前向传播初始化 / simple forward propagation initialization
        for t in range(1, T):
            pred_mean = self._dynamics_mean_impl(init_means[t-1])
            # 使用观测修正位置 / use observation to correct position
            pred_mean = pred_mean.at[0].set(0.5 * pred_mean[0] + 0.5 * observations[t])
            init_means = init_means.at[t].set(pred_mean)
        
        # 初始化对数标准差 / initialize log standard deviations
        init_log_stds = jnp.log(jnp.ones((T, 2)) * 0.1)
        
        # 创建初始参数 / create initial parameters
        params = SVIParams(means=init_means, log_stds=init_log_stds)
        
        # 初始化优化器状态 / initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        # 定义损失函数（负ELBO）/ define loss function (negative ELBO)
        def loss_fn(params_inner, key_inner):
            return -self._compute_elbo_impl(params_inner, observations, key_inner)
        
        # 梯度函数 / gradient function
        grad_fn = jit(grad(loss_fn))
        
        # 优化循环 / optimization loop
        print(f"开始SVI优化，最大迭代次数: {self.max_iterations}")
        
        best_elbo = -jnp.inf
        best_params = params
        patience = 50
        no_improve_count = 0
        
        for iteration in range(self.max_iterations):
            key, subkey = random.split(key)
            
            # 计算梯度 / compute gradients
            grads = grad_fn(params, subkey)
            
            # 更新参数 / update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # 计算当前ELBO / compute current ELBO
            if iteration % 10 == 0:
                key, subkey = random.split(key)
                current_elbo = self._compute_elbo_impl(params, observations, subkey)
                
                if current_elbo > best_elbo:
                    best_elbo = current_elbo
                    best_params = params
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}: ELBO = {current_elbo:.6f}")
                
                # 早停检查 / early stopping check
                if no_improve_count >= patience:
                    print(f"早停于迭代 {iteration}")
                    break
        
        # 使用最佳参数计算最终结果 / compute final result with best parameters
        key, subkey = random.split(key)
        final_elbo = self._compute_elbo_impl(best_params, observations, subkey)
        
        # 计算总对数似然（近似）/ compute total log-likelihood (approximation)
        total_log_likelihood = self._compute_approximate_log_likelihood(
            best_params, observations, subkey
        )
        
        print(f"SVI优化完成，最终ELBO: {final_elbo:.6f}")
        
        return SVIState(
            means=best_params.means,
            log_stds=best_params.log_stds,
            total_log_likelihood=total_log_likelihood,
            elbo=final_elbo
        )
    
    @partial(jit, static_argnums=(0,))
    def _compute_approximate_log_likelihood(
        self, 
        params: SVIParams, 
        observations: chex.Array,
        key: chex.PRNGKey
    ) -> chex.Scalar:
        """
        计算近似对数似然 / Compute approximate log-likelihood
        
        使用重要性采样估计 log p(y) ≈ log ∫ p(y|x)p(x)dx
        Use importance sampling to estimate log p(y) ≈ log ∫ p(y|x)p(x)dx
        """
        T = len(observations)
        
        # 从变分分布采样 / sample from variational distribution
        samples = self._sample_from_variational(params, key)
        
        def compute_log_weight(sample_seq):
            # log p(x,y) - log q(x)
            # 先验 / prior
            log_weight = -0.5 * jnp.sum(sample_seq[0]**2) - jnp.log(2 * jnp.pi)
            
            # 转移概率 / transition probabilities
            for t in range(1, T):
                log_weight += self._log_transition_prob_impl(sample_seq[t], sample_seq[t-1])
            
            # 观测概率 / observation probabilities
            for t in range(T):
                log_weight += self._log_observation_prob_impl(observations[t], sample_seq[t])
            
            # 减去变分分布概率 / subtract variational distribution probability
            for t in range(T):
                diff = sample_seq[t] - params.means[t]
                stds = jnp.exp(params.log_stds[t])
                log_weight -= -0.5 * jnp.sum((diff / stds)**2)
                log_weight -= -jnp.sum(params.log_stds[t]) - jnp.log(2 * jnp.pi)
            
            return log_weight
        
        log_weights = vmap(compute_log_weight)(samples)
        
        # 使用log-sum-exp技巧计算对数似然 / use log-sum-exp trick to compute log-likelihood
        max_log_weight = jnp.max(log_weights)
        log_likelihood = max_log_weight + jnp.log(jnp.mean(jnp.exp(log_weights - max_log_weight)))
        
        return log_likelihood
    
    def extract_estimates(self, result: SVIState) -> Dict[str, chex.Array]:
        """
        提取状态估计 / Extract state estimates
        
        Args:
            result: SVI结果 / SVI result
            
        Returns:
            estimates: 状态估计字典 / state estimates dictionary
        """
        # 提取均值和标准差 / extract means and standard deviations
        theta_mean = result.means[:, 0]
        theta_std = jnp.exp(result.log_stds[:, 0])
        omega_mean = result.means[:, 1]
        omega_std = jnp.exp(result.log_stds[:, 1])
        
        return {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'omega_mean': omega_mean,
            'omega_std': omega_std
        }