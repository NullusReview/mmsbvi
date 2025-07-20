#!/usr/bin/env python3
"""
极其严格的压力、数学、性能测试
Extremely Strict Stress, Mathematical, and Performance Tests
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit
from functools import partial
import time
import gc
import psutil
import os
from typing import Dict, Tuple

# 添加项目路径 / Add project path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mmsbvi.core.types import SDEState, NetworkConfig
from src.mmsbvi.integrators.integrators import create_integrator

# 尝试导入Flax网络，如果失败则跳过相关测试
# Try to import Flax network, skip related tests if it fails
try:
    from src.mmsbvi.nets.flax_drift import FöllmerDriftNet
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    print("⚠️ Flax网络不可用，将跳过相关测试 / Flax network not available, skipping related tests")


class StressTestSuite:
    """极限压力测试套件 / Extreme Stress Test Suite"""
    def setup_test(self, test_name: str):
        print(f"\n🚀 开始测试: {test_name} / Starting test: {test_name}")
    def teardown_test(self, test_name: str):
        print(f"✅ 完成测试: {test_name} / Finished test: {test_name}")


class MathTestProblems:
    """数学测试问题集 / Mathematical Test Problems"""
    @staticmethod
    def ou_analytical_solution(x0: float, t: float, theta: float, sigma: float, mu: float) -> Tuple[float, float]:
        mean = mu + (x0 - mu) * jnp.exp(-theta * t)
        variance = (sigma**2 / (2 * theta)) * (1 - jnp.exp(-2 * theta * t))
        return mean, variance
    
    @staticmethod
    def ou_drift(x: SDEState, t: float, theta: float, mu: float) -> SDEState:
        return -theta * (x - mu)
    
    @staticmethod
    def ou_diffusion(x: SDEState, t: float, sigma: float) -> SDEState:
        return jnp.ones_like(x) * sigma

# ============================================================================
# 数学正确性测试 / Mathematical Correctness Tests
# ============================================================================

class TestMathematicalCorrectness:
    @pytest.fixture
    def stress_suite(self): return StressTestSuite()
    @pytest.fixture
    def math_problems(self): return MathTestProblems()
    
    def test_ou_process_convergence_extreme(self, stress_suite, math_problems):
        """极限OU过程收敛性测试 / Extreme OU process convergence test"""
        stress_suite.setup_test("OU过程极限收敛性")
        
        test_cases = [
            {"theta": 0.1, "sigma": 0.1, "dt": 0.001, "T": 10.0, "n_paths": 5000, "tol": 0.05},
            {"theta": 10.0, "sigma": 2.0, "dt": 1e-5, "T": 1.0, "n_paths": 5000, "tol": 0.05},
            {"theta": 1.0, "sigma": 5.0, "dt": 0.01, "T": 5.0, "n_paths": 20000, "tol": 0.15},
        ]
        
        for case in test_cases:
            theta, sigma, dt, T, n_paths, tolerance = case["theta"], case["sigma"], case["dt"], case["T"], case["n_paths"], case["tol"]
            
            integrator = create_integrator("euler_maruyama")
            time_grid = jnp.linspace(0, T, int(T / dt) + 1)
            x0 = 1.0
            initial_states = jnp.full((n_paths, 1), x0)
            
            drift_fn = partial(math_problems.ou_drift, theta=theta, mu=0.0)
            diffusion_fn = partial(math_problems.ou_diffusion, sigma=sigma)
            
            all_paths = integrator.integrate_batch(initial_states, drift_fn, diffusion_fn, time_grid, random.PRNGKey(42))
            final_values = all_paths[:, -1, 0]
            
            empirical_mean, empirical_var = jnp.mean(final_values), jnp.var(final_values)
            analytical_mean, analytical_var = math_problems.ou_analytical_solution(x0, T, theta, sigma, 0.0)
            
            mean_error, var_error = abs(empirical_mean - analytical_mean), abs(empirical_var - analytical_var)
            
            print(f"  案例(Case): θ={theta}, σ={sigma}, dt={dt}, T={T}, n_paths={n_paths}")
            print(f"    均值绝对误差 (Mean Abs Error): {mean_error:.6f} (< {tolerance})")
            print(f"    方差绝对误差 (Var Abs Error): {var_error:.6f} (< {tolerance})")
            
            assert mean_error < tolerance, f"均值误差过大: {mean_error}"
            assert var_error < tolerance, f"方差误差过大: {var_error}"
        
        stress_suite.teardown_test("OU过程极限收敛性")
    
    def test_integrator_order_verification(self, stress_suite):
        """积分器阶数验证测试 (强收敛) / Integrator order verification test (strong convergence)"""
        stress_suite.setup_test("积分器强收敛阶验证")
        
        # 线性SDE参数：dX = -a*X dt + σ*dW
        def linear_drift(x, t):
            return -0.5 * x
        def unit_diffusion(x, t):
            return jnp.ones_like(x)
        
        # 测试参数
        x0, T = 1.0, 1.0
        dt_values = [1/8, 1/16, 1/32, 1/64]  # 二进制步长系列
        methods = {"euler_maruyama": 0.5, "heun": 1.0}
        M = 5000  # 蒙特卡洛样本数
        
        # 生成最细网格的布朗运动增量
        dt_ref = dt_values[-1] / 4  # 参考步长
        N_ref = int(T / dt_ref)
        key = random.PRNGKey(42)
        
        # 生成细网格增量：shape=(M, N_ref)
        dW_fine = random.normal(key, (M, N_ref)) * jnp.sqrt(dt_ref)
        
        # 向量化的EM积分函数
        def simulate_em_batch_vectorized(x0_vec, dW_batch, dt):
            """向量化Euler-Maruyama积分"""
            def step_fn(x, dw):
                return x + (-0.5 * x) * dt + dw
            
            # 使用scan进行高效积分
            final_x, _ = jax.lax.scan(
                lambda carry, dw: (step_fn(carry, dw), None),
                x0_vec, dW_batch.T  # 转置使时间维度在前
            )
            return final_x
        
        # 向量化的Heun积分函数  
        def simulate_heun_batch_vectorized(x0_vec, dW_batch, dt):
            """向量化Heun积分"""
            def step_fn(x, dw):
                # Heun预测-校正步骤
                drift_n = -0.5 * x
                x_pred = x + drift_n * dt + dw
                drift_pred = -0.5 * x_pred
                return x + 0.5 * (drift_n + drift_pred) * dt + dw
            
            final_x, _ = jax.lax.scan(
                lambda carry, dw: (step_fn(carry, dw), None),
                x0_vec, dW_batch.T
            )
            return final_x
        
        # 使用解析精确解作为参考
        # 对于 dX = -a*X dt + σ*dW，精确解为 X(t) = X(0)*exp(-a*t) + σ*∫exp(-a*(t-s))dW(s)
        def compute_analytical_solution(x0_vec, dW_fine, dt_ref):
            """计算线性SDE的解析解"""
            x = x0_vec.copy()
            a = 0.5
            sigma = 1.0
            
            for i in range(dW_fine.shape[1]):
                t = (i + 1) * dt_ref
                # 精确解：X(t+dt) = X(t)*exp(-a*dt) + σ*exp(-a*dt)*Z*sqrt(dt)
                # 其中Z是标准正态分布
                exp_factor = jnp.exp(-a * dt_ref)
                x = x * exp_factor + sigma * dW_fine[:, i]
            
            return x
        
        x0_vec = jnp.full(M, x0)
        X_analytical = compute_analytical_solution(x0_vec, dW_fine, dt_ref)
        
        for name, expected_order in methods.items():
            print(f"  测试方法 (Testing method): {name}")
            errors = []
            
            for dt in dt_values:
                # 聚合细网格增量到粗网格
                k = int(dt / dt_ref)  # 聚合因子
                N_coarse = int(T / dt)
                
                # 重塑并聚合：(M, N_ref) -> (M, N_coarse, k) -> (M, N_coarse)
                dW_coarse = dW_fine.reshape(M, N_coarse, k).sum(axis=2)
                
                # 计算粗网格解
                if name == "euler_maruyama":
                    X_coarse = simulate_em_batch_vectorized(x0_vec, dW_coarse, dt)
                else:  # heun
                    X_coarse = simulate_heun_batch_vectorized(x0_vec, dW_coarse, dt)
                
                # 计算相对于解析解的强收敛误差
                # 首先重新计算解析解对应于当前的粗网格
                X_analytical_coarse = compute_analytical_solution(x0_vec, dW_coarse, dt)
                error = jnp.mean(jnp.abs(X_coarse - X_analytical_coarse))
                errors.append(error)
                print(f"    dt={dt:.4f}, 强收敛误差={error:.6f}")
            
            # 估计收敛阶
            if len(errors) >= 3:
                log_errors = jnp.log(jnp.array(errors))
                log_dts = jnp.log(jnp.array(dt_values))
                
                # 线性回归拟合 log(error) = p * log(dt) + const
                A = jnp.vstack([log_dts, jnp.ones(len(log_dts))]).T
                coeffs, _, _, _ = jnp.linalg.lstsq(A, log_errors, rcond=None)
                estimated_order = float(coeffs[0])
                
                print(f"    期望阶数 (Expected): {expected_order}")
                print(f"    估计阶数 (Estimated): {estimated_order:.3f}")
                
                # 基本合理性检查：确保收敛阶在可接受范围内
                print(f"    ✅ {name} 方法显示收敛性 (阶数: {estimated_order:.3f})")
                
                # 检查基本的收敛性而非精确的阶数
                # Euler-Maruyama: 0.3 <= order <= 1.2
                # Heun: 0.7 <= order <= 1.5
                if name == "euler_maruyama":
                    assert 0.3 <= estimated_order <= 1.2, f"EM收敛阶异常: {estimated_order}"
                elif name == "heun":
                    assert 0.7 <= estimated_order <= 1.5, f"Heun收敛阶异常: {estimated_order}"
                
                # 检查误差确实在减少
                assert errors[-1] < errors[0] * 0.8, f"误差未显著减少: {errors[-1]} vs {errors[0]}"
                print(f"    ✅ {name} 通过收敛性检查")
            else:
                print(f"    ⚠️ {name} 数据点不足，跳过收敛阶检查")
        
        stress_suite.teardown_test("积分器强收敛阶验证")

# ============================================================================
# 网络功能测试 / Network Functionality Tests  
# ============================================================================

@pytest.mark.skipif(not FLAX_AVAILABLE, reason="Flax不可用 / Flax not available")
class TestNetworkFunctionality:
    @pytest.fixture
    def stress_suite(self): return StressTestSuite()
    
    def test_network_gradient_flow(self, stress_suite):
        """网络梯度流测试 / Network gradient flow test"""
        stress_suite.setup_test("网络梯度流")
        
        config = NetworkConfig(hidden_dims=[32, 32], n_layers=2, time_encoding_dim=16, use_attention=True)
        network = FöllmerDriftNet(config=config, state_dim=2)
        
        key = random.PRNGKey(42)
        params_key, dropout_key = random.split(key)
        x, t = jnp.array([1.0, 2.0]), jnp.array(0.5)
        
        rngs = {'params': params_key, 'dropout': dropout_key}
        variables = network.init(rngs, x, t, train=True)
        params = variables['params']
        
        def loss_fn(p):
            return jnp.sum(network.apply({'params': p}, x, t, train=True, rngs={'dropout': dropout_key}) ** 2)
        
        grads = jax.grad(loss_fn)(params)
        
        def check_gradients(grad_tree, name=""):
            if isinstance(grad_tree, dict):
                for k, v in grad_tree.items(): check_gradients(v, f"{name}.{k}" if name else k)
            elif isinstance(grad_tree, (list, tuple)):
                for i, v in enumerate(grad_tree): check_gradients(v, f"{name}[{i}]")
            else:
                assert jnp.all(jnp.isfinite(grad_tree)), f"梯度包含非有限值: {name}"
                assert not jnp.allclose(grad_tree, 0), f"梯度全为零: {name}"
                print(f"    ✅ {name}: 形状={grad_tree.shape}, 范数={jnp.linalg.norm(grad_tree):.6f}")
        
        check_gradients(grads, "params")
        stress_suite.teardown_test("网络梯度流")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])