#!/usr/bin/env python3
"""
极致性能基准测试
Ultra Performance Benchmark Tests

比较原版积分器与极致优化版本的性能差异
Compares performance between original and ultra-optimized integrators
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import time
import gc
from functools import partial

# 添加项目路径
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mmsbvi.integrators.integrators import create_integrator

class PerformanceBenchmark:
    """性能基准测试套件 / Performance benchmark suite"""
    
    def setup_test(self, test_name: str):
        print(f"\n🚀 性能基准: {test_name} / Performance Benchmark: {test_name}")
        # 强制JIT编译和内存清理
        jax.clear_caches()
        gc.collect()
    
    def teardown_test(self, test_name: str):
        print(f"✅ 基准完成: {test_name} / Benchmark completed: {test_name}")

class TestPerformanceBenchmark:
    @pytest.fixture
    def benchmark(self): 
        return PerformanceBenchmark()
    
    def test_euler_maruyama_performance_comparison(self, benchmark):
        """Euler-Maruyama性能对比测试 / Euler-Maruyama performance comparison"""
        benchmark.setup_test("Euler-Maruyama性能对比")
        
        # 测试参数 - 大规模问题
        def linear_drift(x, t): return -0.5 * x
        def unit_diffusion(x, t): return jnp.ones_like(x)
        
        x0, T = 1.0, 1.0
        batch_sizes = [1000, 5000, 10000]  # 不同批量大小
        n_steps = 1000  # 时间步数
        n_runs = 3  # 重复次数取平均
        
        for batch_size in batch_sizes:
            print(f"\n  批量大小 (Batch size): {batch_size}")
            
            # 准备数据
            initial_states = jnp.full((batch_size, 1), x0)
            time_grid = jnp.linspace(0, T, n_steps + 1)
            key = random.PRNGKey(42)
            
            # 测试原版积分器
            integrator_original = create_integrator("euler_maruyama")
            
            # 预热JIT编译
            _ = integrator_original.integrate_batch(initial_states[:10], linear_drift, unit_diffusion, time_grid[:11], key)
            
            # 计时原版
            times_original = []
            for run in range(n_runs):
                start_time = time.perf_counter()
                result_original = integrator_original.integrate_batch(
                    initial_states, linear_drift, unit_diffusion, time_grid, key
                )
                jax.block_until_ready(result_original)  # 确保计算完成
                end_time = time.perf_counter()
                times_original.append(end_time - start_time)
            
            avg_time_original = sum(times_original) / len(times_original)
            
            # 测试Ultra版本
            integrator_ultra = create_integrator("euler_maruyama_ultra")
            
            # 预热JIT编译
            _ = integrator_ultra.integrate_batch_ultra(initial_states[:10], -0.5, 1.0, time_grid[:11], key)
            
            # 计时Ultra版本
            times_ultra = []
            for run in range(n_runs):
                start_time = time.perf_counter()
                result_ultra = integrator_ultra.integrate_batch_ultra(
                    initial_states, -0.5, 1.0, time_grid, key
                )
                jax.block_until_ready(result_ultra)
                end_time = time.perf_counter()
                times_ultra.append(end_time - start_time)
            
            avg_time_ultra = sum(times_ultra) / len(times_ultra)
            
            # 计算加速比
            speedup = avg_time_original / avg_time_ultra
            
            print(f"    原版时间 (Original): {avg_time_original:.4f}s")
            print(f"    Ultra时间 (Ultra): {avg_time_ultra:.4f}s")
            print(f"    加速比 (Speedup): {speedup:.2f}x")
            
            # 验证结果一致性
            error = jnp.mean(jnp.abs(result_original - result_ultra))
            print(f"    结果误差 (Result error): {error:.2e}")
            
            # 基本性能要求：Ultra版本应该更快
            assert speedup > 1.0, f"Ultra版本未获得加速: {speedup:.2f}x"
            assert error < 1e-10, f"结果误差过大: {error}"
        
        benchmark.teardown_test("Euler-Maruyama性能对比")
    
    def test_heun_performance_comparison(self, benchmark):
        """Heun方法性能对比测试 / Heun method performance comparison"""
        benchmark.setup_test("Heun方法性能对比")
        
        # 测试参数
        x0, T = 1.0, 1.0
        batch_size = 5000
        n_steps = 500
        
        initial_states = jnp.full((batch_size, 1), x0)
        time_grid = jnp.linspace(0, T, n_steps + 1)
        key = random.PRNGKey(42)
        
        def linear_drift(x, t): return -0.5 * x
        def unit_diffusion(x, t): return jnp.ones_like(x)
        
        # 原版Heun
        integrator_original = create_integrator("heun")
        
        # 预热
        _ = integrator_original.integrate_batch(initial_states[:10], linear_drift, unit_diffusion, time_grid[:11], key)
        
        start_time = time.perf_counter()
        result_original = integrator_original.integrate_batch(
            initial_states, linear_drift, unit_diffusion, time_grid, key
        )
        jax.block_until_ready(result_original)
        time_original = time.perf_counter() - start_time
        
        # Ultra Heun
        integrator_ultra = create_integrator("heun_ultra")
        
        # 预热
        _ = integrator_ultra.integrate_batch_ultra(initial_states[:10], -0.5, 1.0, time_grid[:11], key)
        
        start_time = time.perf_counter()
        result_ultra = integrator_ultra.integrate_batch_ultra(
            initial_states, -0.5, 1.0, time_grid, key
        )
        jax.block_until_ready(result_ultra)
        time_ultra = time.perf_counter() - start_time
        
        speedup = time_original / time_ultra
        error = jnp.mean(jnp.abs(result_original - result_ultra))
        
        print(f"  原版Heun时间: {time_original:.4f}s")
        print(f"  Ultra Heun时间: {time_ultra:.4f}s")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  结果误差: {error:.2e}")
        
        assert speedup > 1.0, f"Ultra Heun未获得加速: {speedup:.2f}x"
        assert error < 1e-10, f"Heun结果误差过大: {error}"
        
        benchmark.teardown_test("Heun方法性能对比")
    
    def test_memory_efficiency(self, benchmark):
        """内存效率测试 / Memory efficiency test"""
        benchmark.setup_test("内存效率分析")
        
        # 大批量测试内存使用
        batch_size = 20000
        n_steps = 2000
        x0, T = 1.0, 1.0
        
        initial_states = jnp.full((batch_size, 1), x0)
        time_grid = jnp.linspace(0, T, n_steps + 1)
        key = random.PRNGKey(42)
        
        def linear_drift(x, t): return -0.5 * x
        def unit_diffusion(x, t): return jnp.ones_like(x)
        
        print(f"  大规模测试: 批量={batch_size}, 步数={n_steps}")
        
        # 测试Ultra版本是否能处理大规模问题
        integrator_ultra = create_integrator("euler_maruyama_ultra")
        
        try:
            start_time = time.perf_counter()
            result = integrator_ultra.integrate_batch_ultra(
                initial_states, -0.5, 1.0, time_grid, key
            )
            jax.block_until_ready(result)
            elapsed_time = time.perf_counter() - start_time
            
            print(f"  Ultra版本成功处理大规模问题")
            print(f"  处理时间: {elapsed_time:.2f}s")
            print(f"  吞吐量: {batch_size * n_steps / elapsed_time / 1e6:.2f} M步/秒")
            
            assert result.shape == (batch_size, n_steps + 1, 1), "输出形状错误"
            
        except Exception as e:
            pytest.fail(f"大规模测试失败: {e}")
        
        benchmark.teardown_test("内存效率分析")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])