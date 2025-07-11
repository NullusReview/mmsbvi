"""
SVI实现测试 / SVI Implementation Test
=================================

测试SVI变分推断的数学实现正确性。
Test the mathematical correctness of SVI variational inference implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import sys
import pathlib

# 添加项目根目录到路径 / add project root to path
root_dir = pathlib.Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# 确保JAX配置 / ensure JAX configuration
jax.config.update('jax_enable_x64', True)

from src.baselines.svi import DuffingSVISmoother


def generate_test_data(key: jax.random.PRNGKey, T: int = 20) -> dict:
    """
    生成测试用的Duffing系统轨迹 / Generate test Duffing system trajectory
    
    Args:
        key: 随机密钥 / random key
        T: 时间步数 / number of time steps
        
    Returns:
        data: 包含真实状态和观测的字典 / dictionary containing true states and observations
    """
    dt = 0.05
    duffing_mu = 0.35
    duffing_sigma = 0.3
    obs_noise_std = 0.05
    
    # 初始状态 / initial state
    x0 = jnp.array([0.5, 0.0])  # [位置, 速度]
    
    # 生成真实轨迹 / generate true trajectory
    states = [x0]
    key_seq = random.split(key, T)
    
    for t in range(1, T):
        prev_state = states[-1]
        x, v = prev_state[0], prev_state[1]
        
        # Duffing动态 + 噪声 / Duffing dynamics + noise
        dx = v * dt
        dv = (-x**3 + x - duffing_mu * v) * dt + duffing_sigma * jnp.sqrt(dt) * random.normal(key_seq[t])
        
        next_state = prev_state + jnp.array([dx, dv])
        states.append(next_state)
    
    true_states = jnp.array(states)
    
    # 生成观测（位置 + 噪声）/ generate observations (position + noise)
    obs_key = random.split(key, T)[0]
    obs_noise = obs_noise_std * random.normal(obs_key, (T,))
    observations = true_states[:, 0] + obs_noise
    
    return {
        'x': true_states,
        'y': observations,
        'dt': dt,
        'T': T
    }


def test_svi_initialization():
    """测试SVI平滑器初始化 / Test SVI smoother initialization"""
    smoother = DuffingSVISmoother(
        dt=0.05,
        duffing_mu=0.35,
        duffing_sigma=0.3,
        learning_rate=0.01,
        max_iterations=100
    )
    
    # 检查参数设置 / check parameter settings
    assert smoother.dt == 0.05
    assert smoother.duffing_mu == 0.35
    assert smoother.duffing_sigma == 0.3
    assert smoother.learning_rate == 0.01
    assert smoother.max_iterations == 100
    
    # 检查协方差矩阵 / check covariance matrices
    assert smoother.Q.shape == (2, 2)
    assert smoother.R > 0
    
    print("✅ SVI初始化测试通过")


def test_svi_dynamics_mean():
    """测试Duffing动态均值计算 / Test Duffing dynamics mean computation"""
    smoother = DuffingSVISmoother(dt=0.05, duffing_mu=0.35)
    
    # 测试状态 / test state
    state = jnp.array([1.0, 0.5])  # [x=1.0, v=0.5]
    
    # 计算下一时刻均值 / compute next time step mean
    next_mean = smoother._dynamics_mean_impl(state)
    
    # 预期结果 / expected result
    x, v = state[0], state[1]
    expected_dx = v * smoother.dt
    expected_dv = (-x**3 + x - smoother.duffing_mu * v) * smoother.dt
    expected_next = state + jnp.array([expected_dx, expected_dv])
    
    # 验证结果 / verify result
    assert jnp.allclose(next_mean, expected_next, atol=1e-10)
    
    print("✅ Duffing动态均值计算测试通过")


def test_svi_log_probabilities():
    """测试对数概率计算 / Test log probability computations"""
    smoother = DuffingSVISmoother(
        dt=0.05,
        duffing_mu=0.35,
        duffing_sigma=0.3,
        process_noise_scale=0.1,
        obs_noise_std=0.05
    )
    
    # 测试转移概率 / test transition probability
    # 使用更合理的状态转移：从动态模型预测的状态附近
    x_prev = jnp.array([0.5, 0.2])
    predicted_mean = smoother._dynamics_mean_impl(x_prev)
    # 在预测均值附近添加小扰动
    x_curr = predicted_mean + jnp.array([0.001, 0.001])  # 小扰动
    
    log_trans_prob = smoother._log_transition_prob_impl(x_curr, x_prev)
    print(f"转移概率测试: prev={x_prev}, curr={x_curr}, predicted={predicted_mean}, log_prob={log_trans_prob}")
    assert jnp.isfinite(log_trans_prob)
    # 注意：概率密度可以大于1，所以对数概率可以为正 / Note: probability density can be > 1, so log prob can be positive
    
    # 测试观测概率 / test observation probability
    state = jnp.array([0.5, 0.2])
    observation = state[0] + 0.001  # 位置观测加小噪声
    
    log_obs_prob = smoother._log_observation_prob_impl(observation, state)
    print(f"观测概率测试: state={state}, obs={observation}, log_prob={log_obs_prob}")
    assert jnp.isfinite(log_obs_prob)
    # 同样，观测概率密度也可以大于1 / Similarly, observation probability density can be > 1
    
    print("✅ 对数概率计算测试通过")


def test_svi_variational_sampling():
    """测试变分分布采样 / Test variational distribution sampling"""
    smoother = DuffingSVISmoother(n_samples=10)
    
    T = 5
    means = jnp.ones((T, 2)) * 0.5
    log_stds = jnp.log(jnp.ones((T, 2)) * 0.1)
    
    from src.baselines.svi.svi_smoother import SVIParams
    params = SVIParams(means=means, log_stds=log_stds)
    
    key = random.PRNGKey(42)
    samples = smoother._sample_from_variational(params, key)
    
    # 检查采样形状 / check sample shape
    assert samples.shape == (smoother.n_samples, T, 2)
    
    # 检查采样分布（大致）/ check sample distribution (roughly)
    sample_means = jnp.mean(samples, axis=0)
    sample_stds = jnp.std(samples, axis=0)
    
    # 采样均值应该接近变分均值 / sample means should be close to variational means
    assert jnp.allclose(sample_means, means, atol=0.2)
    
    print("✅ 变分分布采样测试通过")


def test_svi_elbo_computation():
    """测试ELBO计算 / Test ELBO computation"""
    smoother = DuffingSVISmoother(
        dt=0.05,
        n_samples=20,
        max_iterations=50
    )
    
    # 生成测试数据 / generate test data
    key = random.PRNGKey(123)
    test_data = generate_test_data(key, T=10)
    observations = test_data['y']
    
    T = len(observations)
    
    # 创建测试变分参数 / create test variational parameters
    means = jnp.zeros((T, 2))
    log_stds = jnp.log(jnp.ones((T, 2)) * 0.2)
    
    from src.baselines.svi.svi_smoother import SVIParams
    params = SVIParams(means=means, log_stds=log_stds)
    
    # 计算ELBO / compute ELBO
    key_elbo = random.PRNGKey(456)
    elbo = smoother._compute_elbo_impl(params, observations, key_elbo)
    
    # ELBO应该是有限的 / ELBO should be finite
    assert jnp.isfinite(elbo)
    
    print(f"✅ ELBO计算测试通过，ELBO值: {elbo:.4f}")


def test_svi_smoothing_basic():
    """测试SVI平滑基本功能 / Test SVI smoothing basic functionality"""
    print("\n=== 开始SVI平滑基本功能测试 ===")
    
    # 创建SVI平滑器（较小的参数用于快速测试）/ create SVI smoother with small parameters for fast testing
    smoother = DuffingSVISmoother(
        dt=0.05,
        duffing_mu=0.35,
        duffing_sigma=0.3,
        learning_rate=0.02,
        n_samples=10,
        max_iterations=50,
        convergence_tol=1e-4
    )
    
    # 生成测试数据 / generate test data
    key = random.PRNGKey(789)
    test_data = generate_test_data(key, T=15)
    
    observations = jnp.array(test_data['y'])
    true_states = test_data['x']
    
    print(f"测试数据: T={len(observations)}, 观测范围: [{jnp.min(observations):.3f}, {jnp.max(observations):.3f}]")
    
    # 执行SVI平滑 / perform SVI smoothing
    initial_mean = jnp.array([observations[0], 0.0])
    initial_cov = jnp.eye(2) * 1.0
    
    key_smooth = random.PRNGKey(101112)
    result = smoother.smooth(
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        key=key_smooth
    )
    
    # 检查结果结构 / check result structure
    assert hasattr(result, 'means')
    assert hasattr(result, 'log_stds')
    assert hasattr(result, 'total_log_likelihood')
    assert hasattr(result, 'elbo')
    
    # 检查结果形状 / check result shapes
    assert result.means.shape == (len(observations), 2)
    assert result.log_stds.shape == (len(observations), 2)
    assert jnp.isfinite(result.total_log_likelihood)
    assert jnp.isfinite(result.elbo)
    
    # 提取估计 / extract estimates
    estimates = smoother.extract_estimates(result)
    
    # 检查估计结构 / check estimates structure
    assert 'x_mean' in estimates
    assert 'x_std' in estimates
    assert 'v_mean' in estimates
    assert 'v_std' in estimates
    
    # 计算RMSE / compute RMSE
    x_true = true_states[:, 0]
    x_pred = estimates['x_mean']
    rmse = float(jnp.sqrt(jnp.mean((x_true - x_pred)**2)))
    
    print(f"SVI结果:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  ELBO: {result.elbo:.4f}")
    print(f"  总对数似然: {result.total_log_likelihood:.4f}")
    print(f"  位置估计范围: [{jnp.min(x_pred):.3f}, {jnp.max(x_pred):.3f}]")
    print(f"  不确定性范围: [{jnp.min(estimates['x_std']):.3f}, {jnp.max(estimates['x_std']):.3f}]")
    
    # 基本合理性检查 / basic sanity checks
    assert rmse < 1.0  # RMSE应该合理
    assert jnp.all(estimates['x_std'] > 0)  # 标准差应该为正
    assert jnp.all(jnp.isfinite(estimates['x_mean']))  # 估计应该有限
    
    print("✅ SVI平滑基本功能测试通过")
    
    return {
        'rmse': rmse,
        'elbo': result.elbo,
        'total_log_likelihood': result.total_log_likelihood,
        'estimates': estimates,
        'true_states': true_states,
        'observations': observations
    }


def test_svi_convergence():
    """测试SVI收敛性 / Test SVI convergence"""
    print("\n=== 开始SVI收敛性测试 ===")
    
    # 使用更多迭代测试收敛 / use more iterations to test convergence
    smoother = DuffingSVISmoother(
        dt=0.05,
        learning_rate=0.01,
        n_samples=15,
        max_iterations=200,
        convergence_tol=1e-5
    )
    
    # 生成简单测试数据 / generate simple test data
    key = random.PRNGKey(131415)
    test_data = generate_test_data(key, T=10)
    
    observations = jnp.array(test_data['y'])
    
    # 运行SVI / run SVI
    result = smoother.smooth(
        observations=observations,
        initial_mean=jnp.array([observations[0], 0.0]),
        initial_cov=jnp.eye(2) * 0.5,
        key=random.PRNGKey(161718)
    )
    
    # 检查ELBO是否有限且合理 / check if ELBO is finite and reasonable
    assert jnp.isfinite(result.elbo)
    assert result.elbo > -1000  # 不应该太负
    
    print(f"收敛测试结果: ELBO={result.elbo:.4f}")
    print("✅ SVI收敛性测试通过")


def main():
    """运行所有SVI测试 / Run all SVI tests"""
    print("🧪 开始SVI实现测试套件")
    print("=" * 50)
    
    try:
        # 基础测试 / basic tests
        test_svi_initialization()
        test_svi_dynamics_mean()
        test_svi_log_probabilities()
        test_svi_variational_sampling()
        test_svi_elbo_computation()
        
        # 功能测试 / functional tests
        result = test_svi_smoothing_basic()
        test_svi_convergence()
        
        print("\n" + "=" * 50)
        print("🎉 所有SVI测试通过！")
        print("=" * 50)
        
        # 返回测试结果用于分析 / return test results for analysis
        return result
        
    except Exception as e:
        print(f"\n❌ SVI测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()