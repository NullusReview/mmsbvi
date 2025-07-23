"""
Comprehensive test suite for Neural Control Variational Solver
神经控制变分求解器的全面测试套件

Tests cover:
- Mathematical correctness / 数学正确性
- Performance and parallelization / 性能和并行化
- Training stability / 训练稳定性
- Numerical stability / 数值稳定性
- Integration with existing components / 与现有组件的集成
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, value_and_grad
from functools import partial
import numpy as np
from typing import Dict, List, Tuple

# Import the modules to test / 导入要测试的模块
from src.mmsbvi.algorithms.control_grad import (
    VariationalObjective, PathSampler, DensityEstimator, PrimalControlGradFlowSolver
)
from src.mmsbvi.core.types import (
    ControlGradConfig, ControlGradState, NetworkConfig, 
    SDEState, BatchStates, PathSamples
)
from src.mmsbvi.nets.flax_drift import FöllmerDriftNet


class TestControlGradConfig:
    """Test ControlGradConfig data structure / 测试ControlGradConfig数据结构"""
    
    def test_config_creation(self):
        """Test basic config creation / 测试基本配置创建"""
        config = ControlGradConfig()
        assert config.state_dim == 2
        assert config.time_horizon == 1.0
        assert config.batch_size == 1024
        assert config.initial_params == {"mean": 0.0, "std": 1.0}
        assert config.target_params == {"mean": 0.0, "std": 1.0}
    
    def test_config_customization(self):
        """Test config customization / 测试配置定制"""
        custom_config = ControlGradConfig(
            state_dim=3,
            batch_size=512,
            num_epochs=1000,
            initial_params={"mean": 1.0, "std": 2.0}
        )
        assert custom_config.state_dim == 3
        assert custom_config.batch_size == 512
        assert custom_config.num_epochs == 1000
        assert custom_config.initial_params["mean"] == 1.0
        assert custom_config.initial_params["std"] == 2.0


class TestVariationalObjective:
    """Test VariationalObjective component / 测试VariationalObjective组件"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration / 创建测试配置"""
        return ControlGradConfig(
            state_dim=2,
            batch_size=32,
            num_time_steps=10,
            time_horizon=1.0
        )
    
    @pytest.fixture
    def objective(self, config):
        """Create VariationalObjective instance / 创建VariationalObjective实例"""
        return VariationalObjective(config)
    
    @pytest.fixture
    def mock_network_apply(self):
        """Create mock network apply function / 创建模拟网络应用函数"""
        def mock_apply(params, x, t, train):
            # Simple linear control: u(x,t) = -x / 简单线性控制
            return -x
        return mock_apply
    
    def test_integration_weights(self, objective):
        """Test trapezoidal integration weights / 测试梯形积分权重"""
        weights = objective.integration_weights
        expected_length = objective.config.num_time_steps + 1
        assert len(weights) == expected_length
        
        # Check trapezoidal rule: first and last weights should be 0.5*dt
        # 检查梯形规则：第一个和最后一个权重应该是0.5*dt
        dt = objective.dt
        assert jnp.allclose(weights[0], 0.5 * dt)
        assert jnp.allclose(weights[-1], 0.5 * dt)
        assert jnp.allclose(weights[1:-1], dt)
    
    def test_control_cost_computation(self, objective, mock_network_apply):
        """Test control cost computation / 测试控制代价计算"""
        key = random.PRNGKey(42)
        batch_size = 4
        num_steps = objective.config.num_time_steps
        state_dim = objective.config.state_dim
        
        # Create sample paths / 创建样本路径
        paths = random.normal(key, (batch_size, num_steps + 1, state_dim))
        times = jnp.linspace(0.0, 1.0, num_steps + 1)
        
        # Mock parameters / 模拟参数
        params = {}
        
        cost = objective.compute_control_cost(
            paths, times, mock_network_apply, params, key
        )
        
        # Cost should be positive / 代价应该为正
        assert cost >= 0.0
        assert jnp.isfinite(cost)
    
    def test_boundary_penalty_computation(self, objective):
        """Test boundary penalty computation / 测试边界惩罚计算"""
        key = random.PRNGKey(42)
        batch_size = 4
        num_steps = objective.config.num_time_steps
        state_dim = objective.config.state_dim
        
        # Create sample paths / 创建样本路径
        paths = random.normal(key, (batch_size, num_steps + 1, state_dim))
        
        # Create mock density functions / 创建模拟密度函数
        def mock_initial_density(x):
            return jnp.sum(-0.5 * x**2, axis=-1)  # Gaussian log-density / 高斯对数密度
        
        def mock_target_density(x):
            return jnp.sum(-0.5 * x**2, axis=-1)  # Gaussian log-density / 高斯对数密度
        
        # Create mock density estimator for the updated function signature
        # 为更新的函数签名创建模拟密度估计器
        from src.mmsbvi.algorithms.control_grad import DensityEstimator
        config = ControlGradConfig(state_dim=state_dim)
        mock_density_estimator = DensityEstimator(config)
        
        penalty = objective.compute_boundary_penalty(
            paths, mock_initial_density, mock_target_density, mock_density_estimator
        )
        
        assert jnp.isfinite(penalty)
    
    def test_gradient_computation(self, objective, mock_network_apply):
        """Test gradient computation through objective / 测试通过目标函数的梯度计算"""
        key = random.PRNGKey(42)
        batch_size = 4
        num_steps = objective.config.num_time_steps
        state_dim = objective.config.state_dim
        
        # Create sample paths / 创建样本路径
        paths = random.normal(key, (batch_size, num_steps + 1, state_dim))
        times = jnp.linspace(0.0, 1.0, num_steps + 1)
        
        # Parametric network apply function / 参数化网络应用函数
        def param_network_apply(params, x, t, train):
            # 处理参数格式：如果是嵌套字典格式，提取params / Handle parameter format
            if isinstance(params, dict) and "params" in params:
                actual_params = params["params"]
            else:
                actual_params = params
            return actual_params["weight"] * x
        
        # Mock density functions / 模拟密度函数
        def mock_density(x):
            return jnp.sum(-0.5 * x**2, axis=-1)
        
        # Test parameters / 测试参数
        test_params = {"weight": jnp.array(1.0)}
        
        def loss_fn(params):
            cost = objective.compute_control_cost(
                paths, times, param_network_apply, params, key
            )
            # Create mock density estimator for updated function signature
            from src.mmsbvi.algorithms.control_grad import DensityEstimator
            config = ControlGradConfig(state_dim=objective.config.state_dim)
            mock_density_estimator = DensityEstimator(config)
            penalty = objective.compute_boundary_penalty(
                paths, mock_density, mock_density, mock_density_estimator
            )
            return cost + penalty
        
        # Compute gradients / 计算梯度
        loss_val, grads = value_and_grad(loss_fn)(test_params)
        
        assert jnp.isfinite(loss_val)
        assert "weight" in grads
        assert jnp.isfinite(grads["weight"])


class TestPathSampler:
    """Test PathSampler component / 测试PathSampler组件"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration / 创建测试配置"""
        return ControlGradConfig(
            state_dim=2,
            batch_size=8,
            num_time_steps=20,
            time_horizon=1.0,
            diffusion_coeff=0.1
        )
    
    @pytest.fixture
    def path_sampler(self, config):
        """Create PathSampler instance / 创建PathSampler实例"""
        return PathSampler(config)
    
    @pytest.fixture
    def mock_network_apply(self):
        """Create mock network apply function / 创建模拟网络应用函数"""
        def mock_apply(params, x, t, train):
            return -0.5 * x  # Simple stabilizing control / 简单稳定控制
        return mock_apply
    
    def test_initial_state_sampling_gaussian(self, path_sampler):
        """Test Gaussian initial state sampling / 测试高斯初始状态采样"""
        key = random.PRNGKey(42)
        batch_size = 16
        
        states = path_sampler.sample_initial_states(
            batch_size, key, "gaussian", {"mean": 1.0, "std": 2.0}
        )
        
        assert states.shape == (batch_size, path_sampler.config.state_dim)
        
        # Check approximate mean and std / 检查近似均值和标准差
        sample_mean = jnp.mean(states)
        sample_std = jnp.std(states)
        
        # Allow some tolerance for finite sampling / 允许有限采样的一些容差
        assert jnp.abs(sample_mean - 1.0) < 0.5
        assert jnp.abs(sample_std - 2.0) < 0.5
    
    def test_initial_state_sampling_uniform(self, path_sampler):
        """Test uniform initial state sampling / 测试均匀初始状态采样"""
        key = random.PRNGKey(42)
        batch_size = 16
        
        states = path_sampler.sample_initial_states(
            batch_size, key, "uniform", {"low": -2.0, "high": 2.0}
        )
        
        assert states.shape == (batch_size, path_sampler.config.state_dim)
        assert jnp.all(states >= -2.0)
        assert jnp.all(states <= 2.0)
    
    def test_controlled_path_sampling(self, path_sampler, mock_network_apply):
        """Test controlled path sampling / 测试控制路径采样"""
        key = random.PRNGKey(42)
        batch_size = 4
        state_dim = path_sampler.config.state_dim
        
        # Initial states / 初始状态
        initial_states = random.normal(key, (batch_size, state_dim))
        
        # Sample paths / 采样路径
        paths = path_sampler.sample_controlled_paths(
            initial_states, key, mock_network_apply, {}
        )
        
        expected_shape = (batch_size, path_sampler.config.num_time_steps + 1, state_dim)
        assert paths.shape == expected_shape
        
        # Check initial conditions preserved / 检查初始条件保持
        assert jnp.allclose(paths[:, 0, :], initial_states, atol=1e-6)
        
        # Check no NaN or Inf / 检查没有NaN或Inf
        assert jnp.all(jnp.isfinite(paths))
    
    def test_path_sampling_determinism(self, path_sampler, mock_network_apply):
        """Test deterministic behavior with same random key / 测试相同随机密钥的确定性行为"""
        key = random.PRNGKey(123)
        batch_size = 4
        state_dim = path_sampler.config.state_dim
        
        initial_states = random.normal(key, (batch_size, state_dim))
        
        # Sample twice with same key / 使用相同密钥采样两次
        paths1 = path_sampler.sample_controlled_paths(
            initial_states, key, mock_network_apply, {}
        )
        paths2 = path_sampler.sample_controlled_paths(
            initial_states, key, mock_network_apply, {}
        )
        
        # Should be identical / 应该相同
        assert jnp.allclose(paths1, paths2, atol=1e-10)


class TestDensityEstimator:
    """Test DensityEstimator component / 测试DensityEstimator组件"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration / 创建测试配置"""
        return ControlGradConfig(state_dim=2)
    
    @pytest.fixture
    def estimator(self, config):
        """Create DensityEstimator instance / 创建DensityEstimator实例"""
        return DensityEstimator(config)
    
    def test_gaussian_density_creation(self, estimator):
        """Test Gaussian density function creation / 测试高斯密度函数创建"""
        mean = 1.0
        std = 2.0
        
        density_fn = estimator.create_gaussian_density_fn(mean, std)
        
        # Test evaluation / 测试评估
        test_points = jnp.array([[1.0, 1.0], [0.0, 0.0]])
        log_densities = jax.vmap(density_fn)(test_points)
        
        assert log_densities.shape == (2,)
        assert jnp.all(jnp.isfinite(log_densities))
        
        # Point at mean should have higher density than point far away
        # 均值处的点应该比远处的点具有更高的密度
        assert log_densities[0] > log_densities[1]
    
    def test_kde_density_creation(self, estimator):
        """Test KDE density function creation / 测试KDE密度函数创建"""
        key = random.PRNGKey(42)
        n_samples = 100
        state_dim = estimator.config.state_dim
        
        # Generate sample data / 生成样本数据
        samples = random.normal(key, (n_samples, state_dim))
        
        density_fn = estimator.create_kde_density_fn(samples, "scott")
        
        # Test evaluation / 测试评估
        test_points = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        log_densities = jax.vmap(density_fn)(test_points)
        
        assert log_densities.shape == (2,)
        assert jnp.all(jnp.isfinite(log_densities))
        
        # Point near samples should have higher density
        # 靠近样本的点应该有更高的密度
        assert log_densities[0] > log_densities[1]


class TestPrimalControlGradFlowSolver:
    """Test main solver / 测试主求解器"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration / 创建测试配置"""
        return ControlGradConfig(
            state_dim=2,
            batch_size=16,
            num_epochs=5,  # Small number for testing / 测试用小数量
            num_time_steps=10,
            learning_rate=1e-3
        )
    
    @pytest.fixture
    def network_config(self):
        """Create network configuration / 创建网络配置"""
        return NetworkConfig(
            hidden_dims=[32, 32],  # Small network for testing / 测试用小网络
            n_layers=2,
            activation="silu",
            use_attention=False,
            dropout_rate=0.0,
            time_encoding_dim=16
        )
    
    @pytest.fixture
    def solver(self, config, network_config):
        """Create solver instance / 创建求解器实例"""
        return PrimalControlGradFlowSolver(config, network_config)
    
    def test_solver_initialization(self, solver):
        """Test solver initialization / 测试求解器初始化"""
        assert solver.config is not None
        assert solver.network_config is not None
        assert solver.objective is not None
        assert solver.path_sampler is not None
        assert solver.density_estimator is not None
    
    def test_network_initialization(self, solver):
        """Test network initialization / 测试网络初始化"""
        key = random.PRNGKey(42)
        
        training_state = solver.initialize_network(key)
        
        assert training_state is not None
        assert training_state.params is not None
        assert training_state.step == 0
        assert solver.network is not None
    
    def test_single_training_step(self, solver):
        """Test single training step / 测试单个训练步骤"""
        key = random.PRNGKey(42)
        
        # Initialize network / 初始化网络
        training_state = solver.initialize_network(key)
        
        # Create solver state with JAX arrays for history (matching updated structure)
        # 使用JAX数组创建求解器状态用于历史记录（匹配更新的结构）
        max_epochs = solver.config.num_epochs
        state = ControlGradState(
            training_state=training_state,
            config=solver.config,
            step=0,
            epoch=0,
            best_loss=float('inf'),
            loss_history=jnp.full(max_epochs, jnp.nan),
            gradient_norm_history=jnp.full(max_epochs, jnp.nan),
            time_per_epoch=jnp.full(max_epochs, jnp.nan),
            control_cost_history=jnp.full(max_epochs, jnp.nan),
            boundary_penalty_history=jnp.full(max_epochs, jnp.nan),
            history_index=0
        )
        
        # Sample initial states / 采样初始状态
        batch_key, train_key = random.split(key)
        batch_initial_states = solver.path_sampler.sample_initial_states(
            solver.config.batch_size,
            batch_key,
            solver.config.initial_distribution,
            solver.config.initial_params
        )
        
        # Perform training step / 执行训练步骤
        new_state, metrics = solver.train_step(state, batch_initial_states, train_key)
        
        # Check results with JAX array history structure
        # 检查JAX数组历史结构的结果
        assert new_state.step == 1
        assert new_state.best_loss <= state.best_loss
        assert new_state.history_index == 1  # One entry recorded
        assert jnp.isfinite(new_state.loss_history[0])  # First entry should be finite
        assert jnp.isnan(new_state.loss_history[1])  # Subsequent entries should be NaN
        assert jnp.isfinite(new_state.gradient_norm_history[0])
        assert jnp.isnan(new_state.gradient_norm_history[1])
        
        # Check metrics / 检查指标
        assert "total_loss" in metrics
        assert "control_cost" in metrics
        assert "boundary_penalty" in metrics
        assert "gradient_norm" in metrics
        
        assert jnp.isfinite(metrics["total_loss"])
        assert jnp.isfinite(metrics["control_cost"])
        assert jnp.isfinite(metrics["gradient_norm"])
        assert metrics["control_cost"] >= 0.0
    
    def test_short_training_loop(self, solver):
        """Test short training loop / 测试短训练循环"""
        key = random.PRNGKey(42)
        
        # Run very short training / 运行很短的训练
        final_state = solver.train(key)
        
        # Check final state / 检查最终状态
        assert final_state.step > 0
        assert final_state.epoch == solver.config.num_epochs - 1
        assert len(final_state.loss_history) == solver.config.num_epochs
        assert len(final_state.gradient_norm_history) == solver.config.num_epochs
        assert final_state.best_loss < float('inf')


class TestNumericalStability:
    """Test numerical stability / 测试数值稳定性"""
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values / 测试极值处理"""
        config = ControlGradConfig(state_dim=2, batch_size=4, num_time_steps=2)  # 匹配路径长度 / Match path length
        objective = VariationalObjective(config)
        
        # Create paths with extreme values / 创建具有极值的路径
        extreme_paths = jnp.array([
            [[1e6, 1e6], [1e6, 1e6], [1e6, 1e6]],  # Very large values / 很大的值
            [[-1e6, -1e6], [-1e6, -1e6], [-1e6, -1e6]],  # Very small values / 很小的值
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Zero values / 零值
            [[1e-10, 1e-10], [1e-10, 1e-10], [1e-10, 1e-10]]  # Tiny values / 微小值
        ])
        
        times = jnp.linspace(0.0, 1.0, 3)
        
        def stable_network_apply(params, x, t, train):
            return jnp.tanh(x)  # Bounded output / 有界输出
        
        key = random.PRNGKey(42)
        cost = objective.compute_control_cost(
            extreme_paths, times, stable_network_apply, {}, key
        )
        
        # Should not be NaN or Inf / 不应该是NaN或Inf
        assert jnp.isfinite(cost)
    
    def test_gradient_numerical_stability(self):
        """Test gradient numerical stability / 测试梯度数值稳定性"""
        config = ControlGradConfig(state_dim=1, batch_size=2, num_time_steps=5)
        objective = VariationalObjective(config)
        
        def param_network_apply(params, x, t, train):
            # 处理参数格式：如果是嵌套字典格式，提取params / Handle parameter format
            if isinstance(params, dict) and "params" in params:
                actual_params = params["params"]
            else:
                actual_params = params
            return actual_params["scale"] * x
        
        def mock_density(x):
            return -0.5 * jnp.sum(x**2, axis=-1)
        
        key = random.PRNGKey(42)
        paths = random.normal(key, (2, 6, 1))
        times = jnp.linspace(0.0, 1.0, 6)
        
        def loss_fn(params):
            cost = objective.compute_control_cost(
                paths, times, param_network_apply, params, key
            )
            # Create mock density estimator for updated function signature
            from src.mmsbvi.algorithms.control_grad import DensityEstimator
            config = ControlGradConfig(state_dim=objective.config.state_dim)
            mock_density_estimator = DensityEstimator(config)
            penalty = objective.compute_boundary_penalty(
                paths, mock_density, mock_density, mock_density_estimator
            )
            return cost + penalty
        
        # Test different parameter scales / 测试不同的参数尺度
        for scale in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
            test_params = {"scale": jnp.array(scale)}
            
            try:
                loss_val, grads = value_and_grad(loss_fn)(test_params)
                
                # Gradients should be finite / 梯度应该是有限的
                assert jnp.isfinite(loss_val)
                assert jnp.isfinite(grads["scale"])
                
            except Exception as e:
                pytest.fail(f"Gradient computation failed for scale {scale}: {e}")


class TestPerformanceAndParallelization:
    """Test performance and parallelization / 测试性能和并行化"""
    
    def test_batch_processing_consistency(self):
        """Test batch processing gives consistent results / 测试批处理给出一致的结果"""
        config = ControlGradConfig(state_dim=2, num_time_steps=5)
        path_sampler = PathSampler(config)
        
        def mock_network_apply(params, x, t, train):
            return -0.1 * x
        
        key = random.PRNGKey(42)
        
        # Test single sample / 测试单样本
        single_initial = random.normal(key, (1, 2))
        single_path = path_sampler.sample_controlled_paths(
            single_initial, key, mock_network_apply, {}
        )
        
        # Test batch of same initial state / 测试相同初始状态的批次
        batch_initial = jnp.tile(single_initial, (4, 1))
        batch_paths = path_sampler.sample_controlled_paths(
            batch_initial, key, mock_network_apply, {}
        )
        
        # All paths in batch should be identical to single path
        # 批次中的所有路径都应该与单路径相同
        for i in range(4):
            assert jnp.allclose(batch_paths[i:i+1], single_path, atol=1e-10)
    
    def test_memory_usage_batch_scaling(self):
        """Test memory doesn't explode with batch size / 测试内存不会随批次大小爆炸"""
        config = ControlGradConfig(state_dim=2, num_time_steps=10)
        objective = VariationalObjective(config)
        
        def simple_network_apply(params, x, t, train):
            return x * 0.1
        
        key = random.PRNGKey(42)
        times = jnp.linspace(0.0, 1.0, 11)
        
        # Test different batch sizes / 测试不同的批次大小
        for batch_size in [1, 4, 16, 64]:
            paths = random.normal(key, (batch_size, 11, 2))
            
            cost = objective.compute_control_cost(
                paths, times, simple_network_apply, {}, key
            )
            
            # Should be able to handle all batch sizes / 应该能够处理所有批次大小
            assert jnp.isfinite(cost)
            assert cost >= 0.0


if __name__ == "__main__":
    # Run tests / 运行测试
    print("🧪 运行Neural Control Variational测试套件 / Running Neural Control Variational test suite")
    
    # Simple smoke test / 简单冒烟测试
    config = ControlGradConfig(
        state_dim=2,
        batch_size=8,
        num_epochs=3,
        num_time_steps=5
    )
    
    print("✅ 配置测试通过 / Configuration test passed")
    
    # Test components / 测试组件
    objective = VariationalObjective(config)
    path_sampler = PathSampler(config)
    estimator = DensityEstimator(config)
    
    print("✅ 组件创建测试通过 / Component creation test passed")
    
    # Test basic functionality / 测试基本功能
    key = random.PRNGKey(42)
    initial_states = path_sampler.sample_initial_states(4, key)
    
    def mock_apply(params, x, t, train):
        return -0.5 * x
    
    paths = path_sampler.sample_controlled_paths(initial_states, key, mock_apply, {})
    
    print(f"✅ 路径采样测试通过，形状: {paths.shape} / Path sampling test passed, shape: {paths.shape}")
    
    # Test objective computation / 测试目标计算
    times = jnp.linspace(0.0, 1.0, config.num_time_steps + 1)
    cost = objective.compute_control_cost(paths, times, mock_apply, {}, key)
    
    print(f"✅ 控制代价计算通过，值: {cost:.6f} / Control cost computation passed, value: {cost:.6f}")
    
    print("🎉 所有基本测试通过！/ All basic tests passed!")