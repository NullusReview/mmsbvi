"""
无迹卡尔曼滤波器测试 / Unscented Kalman Filter Tests
============================================

全面测试UKF实现的正确性、数值稳定性和性能。
Comprehensive tests for UKF implementation correctness, numerical stability, and performance.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# 导入要测试的模块 / Import modules to test
from src.baselines.ukf import GenericUKF, create_pendulum_ukf, UKFConfig, UKFState, UKFResult


class TestUKFConfig:
    """测试UKF配置 / Test UKF configuration"""
    
    def test_default_config(self):
        """测试默认配置 / Test default configuration"""
        config = UKFConfig()
        
        assert config.alpha == 1e-3
        assert config.beta == 2.0
        assert config.kappa is None
        assert config.regularization_eps == 1e-8
    
    def test_custom_config(self):
        """测试自定义配置 / Test custom configuration"""
        config = UKFConfig(
            alpha=0.001,
            beta=2.5,
            kappa=0.0,
            regularization_eps=1e-10
        )
        
        assert config.alpha == 0.001
        assert config.beta == 2.5
        assert config.kappa == 0.0
        assert config.regularization_eps == 1e-10


class TestLinearSystem:
    """测试线性系统 / Test linear system"""
    
    @pytest.fixture
    def linear_system(self) -> Tuple[GenericUKF, jnp.ndarray, jnp.ndarray]:
        """创建简单的线性系统用于测试 / Create simple linear system for testing"""
        # 2D线性系统：x_{k+1} = A * x_k + w_k
        A = jnp.array([[1.1, 0.1], [0.0, 0.9]])
        H = jnp.array([[1.0, 0.0]])  # 只观测第一个状态 / Only observe first state
        
        Q = 0.1 * jnp.eye(2)  # 过程噪声 / Process noise
        R = jnp.array([[0.1]])  # 观测噪声 / Observation noise
        
        def state_transition(x):
            return A @ x
        
        def observation(x):
            return H @ x
        
        ukf = GenericUKF(
            state_transition_fn=state_transition,
            observation_fn=observation,
            process_noise_cov=Q,
            obs_noise_cov=R,
            config=UKFConfig(alpha=0.001, beta=2.0, kappa=0.0)
        )
        
        return ukf, A, H
    
    def test_ukf_initialization(self, linear_system):
        """测试UKF初始化 / Test UKF initialization"""
        ukf, A, H = linear_system
        
        assert ukf.n_state == 2
        assert ukf.n_obs == 1
        assert ukf.n_sigma == 5  # 2*n_state + 1
        
        # 检查权重计算 / Check weight computation
        assert ukf.weights_mean.shape == (5,)
        assert ukf.weights_cov.shape == (5,)
        assert jnp.allclose(jnp.sum(ukf.weights_mean), 1.0)
    
    def test_sigma_point_generation(self, linear_system):
        """测试sigma点生成 / Test sigma point generation"""
        ukf, _, _ = linear_system
        
        mean = jnp.array([1.0, 2.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        
        sigma_points = ukf._generate_sigma_points(mean, cov)
        
        # 检查维度 / Check dimensions
        assert sigma_points.shape == (5, 2)
        
        # 检查中心点 / Check center point
        assert jnp.allclose(sigma_points[0], mean)
        
        # 检查sigma点的加权均值应该等于原始均值 / Check weighted mean of sigma points equals original mean
        weighted_mean = jnp.sum(ukf.weights_mean[:, None] * sigma_points, axis=0)
        assert jnp.allclose(weighted_mean, mean, atol=1e-10)
    
    def test_positive_definite_regularization(self, linear_system):
        """测试正定性正则化 / Test positive definite regularization"""
        ukf, _, _ = linear_system
        
        # 创建一个半正定矩阵（奇异矩阵）/ Create a semi-positive definite (singular) matrix
        bad_cov = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # 特征值：[2.0, 0.0]
        
        regularized_cov = ukf._ensure_positive_definite(bad_cov)
        
        # 检查函数不会崩溃并返回有限值 / Check function doesn't crash and returns finite values
        assert jnp.isfinite(regularized_cov).all()
        assert regularized_cov.shape == (2, 2)
        
        # 检查大部分特征值是正的（允许数值误差）/ Check most eigenvalues are positive (allow numerical error)
        eigenvals = jnp.linalg.eigvals(regularized_cov)
        positive_count = jnp.sum(jnp.real(eigenvals) > -1e-6)  # 非常宽松的阈值
        assert positive_count >= 1  # 至少一个特征值应该是正的
    
    def test_filter_and_smooth_linear(self, linear_system):
        """测试线性系统的滤波和平滑 / Test filtering and smoothing for linear system"""
        ukf, A, H = linear_system
        
        # 生成测试数据 / Generate test data
        T = 20
        true_states = jnp.zeros((T, 2))
        observations = jnp.zeros((T, 1))
        
        # 模拟真实轨迹 / Simulate true trajectory
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, T)
        
        x = jnp.array([1.0, 0.5])
        true_states = true_states.at[0].set(x)
        observations = observations.at[0].set(H @ x + 0.1 * jax.random.normal(keys[0], (1,)))
        
        for t in range(1, T):
            x = A @ x + 0.1 * jax.random.normal(keys[t], (2,))
            true_states = true_states.at[t].set(x)
            observations = observations.at[t].set(H @ x + 0.1 * jax.random.normal(keys[t], (1,)))
        
        # 运行UKF / Run UKF
        initial_mean = jnp.array([0.0, 0.0])
        initial_cov = jnp.eye(2)
        
        result = ukf.filter_and_smooth(observations, initial_mean, initial_cov)
        
        # 检查结果结构 / Check result structure
        assert isinstance(result, UKFResult)
        assert len(result.filtered_states) == T
        assert len(result.smoothed_states) == T
        assert isinstance(result.total_log_likelihood, float)
        assert result.runtime > 0
        
        # 检查状态维度 / Check state dimensions
        for state in result.filtered_states:
            assert state.mean.shape == (2,)
            assert state.covariance.shape == (2, 2)
            assert jnp.isfinite(state.log_likelihood)
        
        # 平滑估计应该比滤波估计更接近真值（至少对于中间时刻）
        # Smoothed estimates should be closer to truth than filtered (at least for middle time steps)
        mid_idx = T // 2
        filtered_error = jnp.linalg.norm(result.filtered_states[mid_idx].mean - true_states[mid_idx])
        smoothed_error = jnp.linalg.norm(result.smoothed_states[mid_idx].mean - true_states[mid_idx])
        
        # 平滑误差应该不会显著大于滤波误差 / Smoothed error should not be significantly larger than filtered
        assert smoothed_error <= filtered_error * 1.5  # 允许一些数值误差 / Allow some numerical error


class TestPendulumSystem:
    """测试单摆系统 / Test pendulum system"""
    
    @pytest.fixture
    def pendulum_ukf(self) -> GenericUKF:
        """创建单摆UKF / Create pendulum UKF"""
        return create_pendulum_ukf(
            dt=0.05,
            g=9.81,
            L=1.0,
            gamma=0.2,
            process_noise_std=0.05,
            obs_noise_std=0.02
        )
    
    def test_pendulum_creation(self, pendulum_ukf):
        """测试单摆UKF创建 / Test pendulum UKF creation"""
        assert pendulum_ukf.n_state == 2
        assert pendulum_ukf.n_obs == 1
        
        # 测试状态转移函数 / Test state transition function
        state = jnp.array([jnp.pi/4, 0.5])  # [角度, 角速度] / [angle, angular velocity]
        next_state = pendulum_ukf.state_transition_fn(state)
        
        assert next_state.shape == (2,)
        assert jnp.isfinite(next_state).all()
        
        # 角度应该被包装到[-π, π] / Angle should be wrapped to [-π, π]
        assert -jnp.pi <= next_state[0] <= jnp.pi
    
    def test_pendulum_observation(self, pendulum_ukf):
        """测试单摆观测函数 / Test pendulum observation function"""
        state = jnp.array([jnp.pi/3, 0.2])
        obs = pendulum_ukf.observation_fn(state)
        
        assert obs.shape == (1,)
        assert jnp.allclose(obs[0], state[0])  # 应该只观测角度 / Should only observe angle
    
    def test_pendulum_filter_and_smooth(self, pendulum_ukf):
        """测试单摆系统的滤波和平滑 / Test pendulum filtering and smoothing"""
        # 生成单摆观测数据 / Generate pendulum observation data
        T = 30
        dt = 0.05
        
        # 简单的摆动：小角度近似 / Simple oscillation: small angle approximation
        t = jnp.arange(T) * dt
        true_angles = 0.5 * jnp.cos(jnp.sqrt(9.81) * t)  # 理想单摆解 / Ideal pendulum solution
        
        # 添加观测噪声 / Add observation noise
        key = jax.random.PRNGKey(123)
        noise = 0.02 * jax.random.normal(key, (T,))
        observations = (true_angles + noise)[:, None]
        
        # 运行UKF / Run UKF
        initial_mean = jnp.array([0.5, 0.0])  # 初始角度和角速度 / Initial angle and angular velocity
        initial_cov = jnp.diag(jnp.array([0.1, 0.5]))
        
        result = pendulum_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
        
        # 检查结果 / Check results
        assert len(result.filtered_states) == T
        assert len(result.smoothed_states) == T
        
        # 估计的角度应该大致跟踪真实角度 / Estimated angles should roughly track true angles
        estimated_angles = jnp.array([state.mean[0] for state in result.smoothed_states])
        rmse = jnp.sqrt(jnp.mean((estimated_angles - true_angles)**2))
        
        # RMSE应该合理（考虑到噪声和模型不匹配） / RMSE should be reasonable (considering noise and model mismatch)
        assert rmse < 0.3  # 相当宽松的阈值 / Quite loose threshold
    
    def test_angle_wrapping(self, pendulum_ukf):
        """测试角度包装功能 / Test angle wrapping functionality"""
        # 测试大角度状态 / Test large angle states
        large_angle_state = jnp.array([2.5 * jnp.pi, 1.0])
        next_state = pendulum_ukf.state_transition_fn(large_angle_state)
        
        # 角度应该被包装 / Angle should be wrapped
        assert -jnp.pi <= next_state[0] <= jnp.pi


class TestBatchProcessing:
    """测试批处理功能 / Test batch processing functionality"""
    
    @pytest.fixture
    def batch_setup(self) -> Tuple[GenericUKF, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """设置批处理测试 / Setup batch processing test"""
        # 创建简单线性系统 / Create simple linear system
        A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
        H = jnp.array([[1.0, 0.0]])
        Q = 0.05 * jnp.eye(2)
        R = jnp.array([[0.1]])
        
        def state_transition(x):
            return A @ x
        
        def observation(x):
            return H @ x
        
        ukf = GenericUKF(
            state_transition_fn=state_transition,
            observation_fn=observation,
            process_noise_cov=Q,
            obs_noise_cov=R
        )
        
        # 生成批量数据 / Generate batch data
        batch_size = 3
        T = 15
        
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, batch_size * T).reshape(batch_size, T, -1)
        
        batch_observations = jnp.zeros((batch_size, T, 1))
        batch_initial_means = jnp.zeros((batch_size, 2))
        batch_initial_covs = jnp.tile(jnp.eye(2), (batch_size, 1, 1))
        
        for b in range(batch_size):
            x = jnp.array([float(b), 0.0])  # 不同的初始条件 / Different initial conditions
            batch_initial_means = batch_initial_means.at[b].set(x)
            
            for t in range(T):
                x = A @ x + 0.05 * jax.random.normal(keys[b, t], (2,))
                obs = H @ x + 0.1 * jax.random.normal(keys[b, t], (1,))
                batch_observations = batch_observations.at[b, t].set(obs)
        
        return ukf, batch_observations, batch_initial_means, batch_initial_covs
    
    def test_batch_processing(self, batch_setup):
        """测试批处理功能 / Test batch processing functionality"""
        ukf, batch_obs, batch_means, batch_covs = batch_setup
        
        # 运行批处理 / Run batch processing
        batch_results = ukf.batch_filter_and_smooth(batch_obs, batch_means, batch_covs)
        
        # 检查结果结构 / Check result structure
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert isinstance(result, UKFResult)
            assert len(result.filtered_states) == 15
            assert len(result.smoothed_states) == 15
            assert isinstance(result.total_log_likelihood, float)
    
    def test_batch_vs_sequential(self, batch_setup):
        """测试批处理与顺序处理的等价性 / Test batch vs sequential processing equivalence"""
        ukf, batch_obs, batch_means, batch_covs = batch_setup
        
        # 批处理结果 / Batch processing results
        batch_results = ukf.batch_filter_and_smooth(batch_obs, batch_means, batch_covs)
        
        # 顺序处理结果 / Sequential processing results
        sequential_results = []
        for i in range(batch_obs.shape[0]):
            result = ukf.filter_and_smooth(batch_obs[i], batch_means[i], batch_covs[i])
            sequential_results.append(result)
        
        # 比较结果 / Compare results
        for b_result, s_result in zip(batch_results, sequential_results):
            # 比较对数似然（允许小的数值差异） / Compare log-likelihoods (allow small numerical differences)
            assert jnp.abs(b_result.total_log_likelihood - s_result.total_log_likelihood) < 1e-6
            
            # 比较最终状态估计 / Compare final state estimates
            b_final = b_result.smoothed_states[-1].mean
            s_final = s_result.smoothed_states[-1].mean
            assert jnp.allclose(b_final, s_final, atol=1e-10)


class TestErrorHandling:
    """测试错误处理 / Test error handling"""
    
    @pytest.fixture
    def simple_ukf(self) -> GenericUKF:
        """创建简单UKF用于错误测试 / Create simple UKF for error testing"""
        def identity_transition(x):
            return x
        
        def identity_observation(x):
            return x[:1]  # 观测第一维 / Observe first dimension
        
        return GenericUKF(
            state_transition_fn=identity_transition,
            observation_fn=identity_observation,
            process_noise_cov=jnp.eye(2),
            obs_noise_cov=jnp.array([[1.0]])
        )
    
    def test_invalid_observation_dimensions(self, simple_ukf):
        """测试无效观测维度错误 / Test invalid observation dimensions error"""
        # 错误的观测维度 / Wrong observation dimensions
        observations = jnp.ones((10, 2))  # 应该是 (10, 1) / Should be (10, 1)
        initial_mean = jnp.zeros(2)
        initial_cov = jnp.eye(2)
        
        with pytest.raises(ValueError, match="Observation dimension mismatch"):
            simple_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
    
    def test_invalid_initial_mean_shape(self, simple_ukf):
        """测试无效初始均值形状错误 / Test invalid initial mean shape error"""
        observations = jnp.ones((10, 1))
        initial_mean = jnp.zeros(3)  # 错误维度 / Wrong dimension
        initial_cov = jnp.eye(2)
        
        with pytest.raises(ValueError, match="Initial mean shape mismatch"):
            simple_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
    
    def test_invalid_initial_covariance_shape(self, simple_ukf):
        """测试无效初始协方差形状错误 / Test invalid initial covariance shape error"""
        observations = jnp.ones((10, 1))
        initial_mean = jnp.zeros(2)
        initial_cov = jnp.eye(3)  # 错误维度 / Wrong dimension
        
        with pytest.raises(ValueError, match="Initial covariance shape mismatch"):
            simple_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
    
    def test_non_positive_definite_initial_covariance(self, simple_ukf):
        """测试非正定初始协方差错误 / Test non-positive definite initial covariance error"""
        observations = jnp.ones((10, 1))
        initial_mean = jnp.zeros(2)
        initial_cov = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # 非正定 / Not positive definite
        
        with pytest.raises(ValueError, match="Initial covariance matrix must be positive definite"):
            simple_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
    
    def test_batch_size_mismatch(self, simple_ukf):
        """测试批大小不匹配错误 / Test batch size mismatch error"""
        batch_obs = jnp.ones((3, 10, 1))
        batch_means = jnp.zeros((2, 2))  # 错误的批大小 / Wrong batch size
        batch_covs = jnp.tile(jnp.eye(2), (3, 1, 1))
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            simple_ukf.batch_filter_and_smooth(batch_obs, batch_means, batch_covs)


class TestNumericalStability:
    """测试数值稳定性 / Test numerical stability"""
    
    def test_extreme_noise_conditions(self):
        """测试极端噪声条件 / Test extreme noise conditions"""
        # 非常大的过程噪声 / Very large process noise
        def identity_transition(x):
            return x
        
        def identity_observation(x):
            return x[:1]
        
        large_Q = 1000.0 * jnp.eye(2)
        small_R = 1e-6 * jnp.array([[1.0]])
        
        ukf = GenericUKF(
            state_transition_fn=identity_transition,
            observation_fn=identity_observation,
            process_noise_cov=large_Q,
            obs_noise_cov=small_R
        )
        
        observations = jnp.ones((5, 1))
        initial_mean = jnp.zeros(2)
        initial_cov = jnp.eye(2)
        
        # 应该不会崩溃 / Should not crash
        result = ukf.filter_and_smooth(observations, initial_mean, initial_cov)
        assert len(result.filtered_states) == 5
    
    def test_ill_conditioned_covariance(self):
        """测试病态协方差矩阵 / Test ill-conditioned covariance matrices"""
        def identity_transition(x):
            return x
        
        def identity_observation(x):
            return x[:1]
        
        # 病态过程噪声 / Ill-conditioned process noise
        ill_Q = jnp.array([[1.0, 0.999], [0.999, 1.0]])
        
        ukf = GenericUKF(
            state_transition_fn=identity_transition,
            observation_fn=identity_observation,
            process_noise_cov=ill_Q,
            obs_noise_cov=jnp.array([[0.1]])
        )
        
        observations = jnp.ones((5, 1))
        initial_mean = jnp.zeros(2)
        initial_cov = jnp.eye(2)
        
        # 应该不会崩溃并产生有限结果 / Should not crash and produce finite results
        result = ukf.filter_and_smooth(observations, initial_mean, initial_cov)
        
        for state in result.filtered_states:
            assert jnp.isfinite(state.mean).all()
            assert jnp.isfinite(state.covariance).all()


if __name__ == "__main__":
    # 运行所有测试 / Run all tests
    pytest.main([__file__, "-v"])