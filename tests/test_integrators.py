"""
SDE Integrators Test Suite
SDE积分器测试套件

Comprehensive tests for SDE numerical integration methods.
SDE数值积分方法的全面测试。
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from typing import Callable, Tuple
import sys
import pathlib

# Add src to path for imports
root_dir = pathlib.Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from mmsbvi.integrators.integrators import (
    EulerMaruyamaIntegrator,
    HeunIntegrator,
    MilsteinIntegrator,
    create_integrator
)
from mmsbvi.core.types import SDEIntegratorConfig
from mmsbvi.core.registry import (
    get_integrator,
    list_integrators,
    clear_registry
)

jax.config.update('jax_enable_x64', True)


# ============================================================================
# Test SDE Problems / 测试SDE问题
# ============================================================================

def ornstein_uhlenbeck_drift(x: jnp.ndarray, t: float, theta: float = 1.0, mu: float = 0.0) -> jnp.ndarray:
    """
    Ornstein-Uhlenbeck drift function
    Ornstein-Uhlenbeck漂移函数
    
    dX_t = θ(μ - X_t)dt + σdW_t
    """
    return theta * (mu - x)


def ornstein_uhlenbeck_diffusion(x: jnp.ndarray, t: float, sigma: float = 1.0) -> jnp.ndarray:
    """
    Ornstein-Uhlenbeck diffusion function
    Ornstein-Uhlenbeck扩散函数
    """
    return jnp.ones_like(x) * sigma


def geometric_brownian_drift(x: jnp.ndarray, t: float, mu: float = 0.05) -> jnp.ndarray:
    """
    Geometric Brownian motion drift
    几何布朗运动漂移
    
    dX_t = μX_t dt + σX_t dW_t
    """
    return mu * x


def geometric_brownian_diffusion(x: jnp.ndarray, t: float, sigma: float = 0.2) -> jnp.ndarray:
    """
    Geometric Brownian motion diffusion
    几何布朗运动扩散
    """
    return sigma * x


def ou_analytical_solution(x0: float, t: float, theta: float = 1.0, mu: float = 0.0, sigma: float = 1.0) -> Tuple[float, float]:
    """
    Analytical solution for Ornstein-Uhlenbeck process
    Ornstein-Uhlenbeck过程的解析解
    
    Returns:
        (mean, variance) at time t
    """
    mean = mu + (x0 - mu) * jnp.exp(-theta * t)
    variance = (sigma**2 / (2 * theta)) * (1 - jnp.exp(-2 * theta * t))
    return mean, variance


# ============================================================================
# Fixtures / 测试夹具
# ============================================================================

@pytest.fixture
def integrator_config():
    """Basic integrator configuration / 基础积分器配置"""
    return SDEIntegratorConfig(
        method="euler_maruyama",
        adaptive=False,
        rtol=1e-3,
        atol=1e-6,
        max_steps=10000
    )


@pytest.fixture
def random_key():
    """Random key for reproducible tests / 可重现测试的随机密钥"""
    return jax.random.PRNGKey(42)


@pytest.fixture(params=["euler_maruyama", "heun"])  # 跳过milstein和amed_euler
def integrator_name(request):
    """Parameterized integrator names / 参数化的积分器名称"""
    return request.param

@pytest.fixture(params=["euler_maruyama", "heun", "milstein"])
def integrator_name_with_milstein(request):
    """Parameterized integrator names including milstein / 包含milstein的参数化积分器名称"""
    return request.param


# ============================================================================
# Basic Functionality Tests / 基础功能测试
# ============================================================================

class TestIntegratorCreation:
    """Test integrator creation and registry / 测试积分器创建和注册"""
    
    def test_create_integrator_factory(self, integrator_config):
        """Test factory function / 测试工厂函数"""
        integrator = create_integrator("euler_maruyama", config=integrator_config)
        assert integrator is not None
        assert isinstance(integrator, EulerMaruyamaIntegrator)
    
    def test_registry_get_integrator(self, integrator_config):
        """Test registry get_integrator / 测试注册表get_integrator"""
        integrator = get_integrator("heun", config=integrator_config)
        assert integrator is not None
        assert isinstance(integrator, HeunIntegrator)
    
    def test_list_integrators(self):
        """Test listing registered integrators / 测试列出注册的积分器"""
        integrators = list_integrators()
        assert "euler_maruyama" in integrators
        assert "heun" in integrators
        assert "milstein" in integrators
    
    def test_unknown_integrator_error(self):
        """Test error for unknown integrator / 测试未知积分器的错误"""
        with pytest.raises(ValueError, match="Unknown SDE integrator"):
            get_integrator("nonexistent_method")


class TestIntegratorInterface:
    """Test integrator interface compliance / 测试积分器接口合规性"""
    
    def test_step_method_exists(self, integrator_name, integrator_config):
        """Test that step method exists / 测试step方法存在"""
        integrator = get_integrator(integrator_name, config=integrator_config)
        assert hasattr(integrator, 'step')
        assert callable(integrator.step)
    
    def test_integrate_method_exists(self, integrator_name, integrator_config):
        """Test that integrate method exists / 测试integrate方法存在"""
        integrator = get_integrator(integrator_name, config=integrator_config)
        assert hasattr(integrator, 'integrate')
        assert callable(integrator.integrate)
    
    def test_integrate_batch_method_exists(self, integrator_name, integrator_config):
        """Test that integrate_batch method exists / 测试integrate_batch方法存在"""
        integrator = get_integrator(integrator_name, config=integrator_config)
        assert hasattr(integrator, 'integrate_batch')
        assert callable(integrator.integrate_batch)


# ============================================================================
# Mathematical Correctness Tests / 数学正确性测试
# ============================================================================

class TestMathematicalCorrectness:
    """Test mathematical correctness of integrators / 测试积分器的数学正确性"""
    
    def test_single_step_shapes(self, integrator_name_with_milstein, integrator_config, random_key):
        """Test output shapes for single step / 测试单步输出形状"""
        # 对于milstein，使用create_integrator提供扩散函数 / For milstein, use create_integrator with diffusion function
        if integrator_name_with_milstein == "milstein":
            integrator = create_integrator(integrator_name_with_milstein, config=integrator_config, diffusion_fn=ornstein_uhlenbeck_diffusion)
        else:
            integrator = get_integrator(integrator_name_with_milstein, config=integrator_config)
        
        # 1D state / 1D状态
        state_1d = jnp.array([1.0])
        result_1d = integrator.step(
            0.0,  # 时间参数 / time parameter
            state_1d,
            ornstein_uhlenbeck_drift,
            ornstein_uhlenbeck_diffusion,
            0.01,
            random_key
        )
        assert result_1d.shape == state_1d.shape
        
        # 2D state / 2D状态
        state_2d = jnp.array([1.0, 2.0])
        result_2d = integrator.step(
            0.0,  # 时间参数 / time parameter
            state_2d,
            ornstein_uhlenbeck_drift,
            ornstein_uhlenbeck_diffusion,
            0.01,
            random_key
        )
        assert result_2d.shape == state_2d.shape
    
    def test_integrate_shapes(self, integrator_name_with_milstein, integrator_config, random_key):
        """Test output shapes for integration / 测试积分输出形状"""
        # 对于milstein，使用create_integrator提供扩散函数 / For milstein, use create_integrator with diffusion function
        if integrator_name_with_milstein == "milstein":
            integrator = create_integrator(integrator_name_with_milstein, config=integrator_config, diffusion_fn=ornstein_uhlenbeck_diffusion)
        else:
            integrator = get_integrator(integrator_name_with_milstein, config=integrator_config)
        
        initial_state = jnp.array([1.0, 2.0])
        time_grid = jnp.linspace(0.0, 1.0, 11)  # 11 time points
        
        trajectory = integrator.integrate(
            initial_state,
            ornstein_uhlenbeck_drift,
            ornstein_uhlenbeck_diffusion,
            time_grid,
            random_key
        )
        
        expected_shape = (len(time_grid), len(initial_state))
        assert trajectory.shape == expected_shape
    
    def test_batch_integration_shapes(self, integrator_name_with_milstein, integrator_config, random_key):
        """Test batch integration shapes / 测试批量积分形状"""
        # 对于milstein，使用create_integrator提供扩散函数 / For milstein, use create_integrator with diffusion function
        if integrator_name_with_milstein == "milstein":
            integrator = create_integrator(integrator_name_with_milstein, config=integrator_config, diffusion_fn=ornstein_uhlenbeck_diffusion)
        else:
            integrator = get_integrator(integrator_name_with_milstein, config=integrator_config)
        
        batch_size = 5
        state_dim = 3
        n_steps = 10
        
        initial_states = jnp.ones((batch_size, state_dim))
        time_grid = jnp.linspace(0.0, 1.0, n_steps + 1)
        
        trajectories = integrator.integrate_batch(
            initial_states,
            ornstein_uhlenbeck_drift,
            ornstein_uhlenbeck_diffusion,
            time_grid,
            random_key
        )
        
        expected_shape = (batch_size, len(time_grid), state_dim)
        assert trajectories.shape == expected_shape
    
    def test_deterministic_drift_only(self, integrator_name_with_milstein, integrator_config):
        """Test deterministic evolution (drift only) / 测试确定性演化（仅漂移）"""
        def zero_diffusion(x, t):
            return jnp.zeros_like(x)
            
        # 对于milstein，使用create_integrator提供扩散函数 / For milstein, use create_integrator with diffusion function  
        if integrator_name_with_milstein == "milstein":
            integrator = create_integrator(integrator_name_with_milstein, config=integrator_config, diffusion_fn=zero_diffusion)
        else:
            integrator = get_integrator(integrator_name_with_milstein, config=integrator_config)
        
        def linear_drift(x, t):
            return -x  # dx/dt = -x, solution: x(t) = x0 * exp(-t)
        
        initial_state = jnp.array([1.0])
        dt = 0.01
        key = jax.random.PRNGKey(42)
        
        # Single step
        result = integrator.step(0.0, initial_state, linear_drift, zero_diffusion, dt, key)
        expected = initial_state * jnp.exp(-dt)
        
        # Allow some numerical error / 允许一些数值误差
        assert jnp.allclose(result, expected, atol=1e-3)


class TestOrnsteinUhlenbeckConvergence:
    """Test convergence for Ornstein-Uhlenbeck process / 测试Ornstein-Uhlenbeck过程的收敛性"""
    
    @pytest.mark.parametrize("integrator_name", ["euler_maruyama", "heun"])
    def test_ou_mean_convergence(self, integrator_name, integrator_config):
        """Test convergence of mean for OU process / 测试OU过程均值的收敛性"""
        integrator = get_integrator(integrator_name, config=integrator_config)
        
        # OU parameters / OU参数
        x0 = 2.0
        T = 1.0
        theta = 1.0
        mu = 0.0
        sigma = 1.0
        
        # Create OU drift and diffusion / 创建OU漂移和扩散
        def ou_drift(x, t):
            return ornstein_uhlenbeck_drift(x, t, theta, mu)
        
        def ou_diffusion(x, t):
            return ornstein_uhlenbeck_diffusion(x, t, sigma)
        
        # Monte Carlo simulation / 蒙特卡洛模拟
        n_paths = 1000
        n_steps = 100
        
        time_grid = jnp.linspace(0.0, T, n_steps + 1)
        initial_states = jnp.full((n_paths, 1), x0)  # 确保是2D数组 (n_paths, 1)
        
        key = jax.random.PRNGKey(123)
        trajectories = integrator.integrate_batch(
            initial_states,
            ou_drift,
            ou_diffusion,
            time_grid,
            key
        )
        
        # Final values / 最终值
        final_values = trajectories[:, -1]
        empirical_mean = jnp.mean(final_values)
        
        # Analytical solution / 解析解
        analytical_mean, _ = ou_analytical_solution(x0, T, theta, mu, sigma)
        
        # Check convergence / 检查收敛性
        assert jnp.abs(empirical_mean - analytical_mean) < 0.1


# ============================================================================
# Performance Tests / 性能测试
# ============================================================================

class TestPerformance:
    """Test integrator performance / 测试积分器性能"""
    
    @pytest.mark.parametrize("integrator_name", ["euler_maruyama", "heun"])
    def test_jit_compilation(self, integrator_name, integrator_config, random_key):
        """Test JIT compilation works / 测试JIT编译工作"""
        integrator = get_integrator(integrator_name, config=integrator_config)
        
        state = jnp.array([1.0])
        dt = 0.01
        
        # First call (compilation) / 第一次调用（编译）
        result1 = integrator.step(
            0.0, state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, dt, random_key
        )
        
        # Second call (should be fast) / 第二次调用（应该很快）
        result2 = integrator.step(
            0.0, state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, dt, random_key
        )
        
        # Results should be deterministic with same key / 使用相同密钥结果应该是确定性的
        assert jnp.allclose(result1, result2)
    
    def test_batch_processing(self, integrator_config, random_key):
        """Test batch processing efficiency / 测试批量处理效率"""
        integrator = get_integrator("euler_maruyama", config=integrator_config)
        
        batch_size = 100
        state_dim = 10
        n_steps = 50
        
        initial_states = jax.random.normal(random_key, (batch_size, state_dim))
        time_grid = jnp.linspace(0.0, 1.0, n_steps + 1)
        
        # This should not raise any errors / 这不应该引发任何错误
        trajectories = integrator.integrate_batch(
            initial_states,
            ornstein_uhlenbeck_drift,
            ornstein_uhlenbeck_diffusion,
            time_grid,
            random_key
        )
        
        assert trajectories.shape == (batch_size, n_steps + 1, state_dim)


# ============================================================================
# Milstein-Specific Tests / Milstein特定测试
# ============================================================================

class TestMilsteinSpecific:
    """Tests specific to Milstein integrator / Milstein积分器特定测试"""
    
    def test_milstein_with_diffusion_derivative(self, integrator_config):
        """Test Milstein with provided diffusion derivative / 测试提供扩散导数的Milstein"""
        
        def diffusion_derivative(x, t):
            """Derivative of σ(x) = σ * x (for geometric Brownian motion)"""
            return jnp.ones_like(x) * 0.2  # σ = 0.2
        
        integrator = MilsteinIntegrator(
            config=integrator_config,
            diffusion_derivative=diffusion_derivative
        )
        
        state = jnp.array([1.0])
        dt = 0.01
        key = jax.random.PRNGKey(42)
        
        result = integrator.step(
            0.0,  # 时间参数 / time parameter
            state,
            geometric_brownian_drift,
            geometric_brownian_diffusion,
            dt,
            key
        )
        
        assert result.shape == state.shape
        assert jnp.isfinite(result).all()
    
    def test_milstein_numerical_derivative(self, integrator_config):
        """Test Milstein with automatic derivative / 测试自动导数的Milstein"""
        # 使用create_integrator自动计算导数 / Use create_integrator for automatic derivative
        integrator = create_integrator('milstein', config=integrator_config, diffusion_fn=geometric_brownian_diffusion)
        
        state = jnp.array([1.0])
        dt = 0.01
        key = jax.random.PRNGKey(42)
        
        # Should work with automatic diffusion derivative / 自动扩散导数应该工作
        result = integrator.step(
            0.0,  # 时间参数 / time parameter
            state,
            geometric_brownian_drift,
            geometric_brownian_diffusion,
            dt,
            key
        )
        
        assert result.shape == state.shape
        assert jnp.isfinite(result).all()


# ============================================================================
# Edge Cases and Error Handling / 边缘情况和错误处理
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling / 测试边缘情况和错误处理"""
    
    def test_zero_time_step(self, integrator_config, random_key):
        """Test behavior with zero time step / 测试零时间步长的行为"""
        integrator = get_integrator("euler_maruyama", config=integrator_config)
        
        state = jnp.array([1.0])
        dt = 0.0
        
        result = integrator.step(
            0.0, state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, dt, random_key
        )
        
        # Should remain unchanged / 应该保持不变
        assert jnp.allclose(result, state)
    
    def test_large_time_step_stability(self, integrator_config, random_key):
        """Test stability with large time steps / 测试大时间步长的稳定性"""
        integrator = get_integrator("euler_maruyama", config=integrator_config)
        
        state = jnp.array([1.0])
        dt = 1.0  # Large time step / 大时间步长
        
        result = integrator.step(
            0.0, state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, dt, random_key
        )
        
        # Should not produce NaN or infinite values / 不应产生NaN或无限值
        assert jnp.isfinite(result).all()


# ============================================================================
# Integration Tests / 集成测试
# ============================================================================

class TestIntegration:
    """Integration tests with existing codebase / 与现有代码库的集成测试"""
    
    def test_registry_integration(self):
        """Test integration with registry system / 测试与注册系统的集成"""
        # Test that integrators are properly registered / 测试积分器是否正确注册
        integrators = list_integrators()
        assert "euler_maruyama" in integrators
        assert "heun" in integrators
        assert "milstein" in integrators
        
        # Test that we can get integrator classes / 测试我们可以获取积分器类
        euler_cls = integrators["euler_maruyama"]
        assert euler_cls.__name__ == "EulerMaruyamaIntegrator"
    
    def test_config_integration(self):
        """Test integration with configuration system / 测试与配置系统的集成"""
        config = SDEIntegratorConfig(
            method="heun",
            adaptive=True,
            rtol=1e-4,
            atol=1e-7
        )
        
        integrator = get_integrator("heun", config=config)
        assert integrator.config.rtol == 1e-4
        assert integrator.config.atol == 1e-7


if __name__ == "__main__":
    # Run basic tests if script is executed directly / 如果直接执行脚本则运行基础测试
    print("🧪 Running basic SDE integrator tests / 运行基础SDE积分器测试")
    
    # Test integrator creation / 测试积分器创建
    integrator = create_integrator("euler_maruyama")
    print(f"✅ Created {integrator.__class__.__name__}")
    
    # Test registry / 测试注册表
    integrators = list_integrators()
    print(f"✅ Registered integrators: {list(integrators.keys())}")
    
    # Test basic functionality / 测试基础功能
    state = jnp.array([1.0])
    dt = 0.01
    key = jax.random.PRNGKey(42)
    
    result = integrator.step(
        0.0, state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, dt, key
    )
    print(f"✅ Single step: {state} → {result}")
    
    # Test integration / 测试积分
    time_grid = jnp.linspace(0.0, 1.0, 11)
    trajectory = integrator.integrate(
        state, ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion, time_grid, key
    )
    print(f"✅ Integration trajectory shape: {trajectory.shape}")
    
    print("🎉 All basic tests passed! / 所有基础测试通过！")