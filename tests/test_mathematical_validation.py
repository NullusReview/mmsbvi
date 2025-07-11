"""
Comprehensive Mathematical Validation Test Suite
全面数学验证测试套件

Tests all fixed components with rigorous mathematical checks.
使用严格的数学检查测试所有修复的组件。

This test suite MUST pass before any numerical experiments can be trusted.
在任何数值实验可以被信任之前，这个测试套件必须通过。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import pytest
import numpy as np

# Enable 64-bit precision for accurate tests / 启用64位精度以进行精确测试
jax.config.update('jax_enable_x64', True)

# JAX-compatible integration function / JAX兼容的积分函数  
def jax_trapz(y: jnp.ndarray, dx: float) -> float:
    """JAX-compatible trapezoidal integration."""
    return dx * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))

from src.mmsbvi.core import (
    GridConfig1D, OUProcessParams, MMSBProblem, IPFPConfig
)
from src.mmsbvi.solvers.pde_solver_1d import (
    solve_onsager_fokker_pde_fixed,
    operator_adjointness_error,
    validate_pde_solution_fixed,
    create_test_problem_1d_fixed,
    _gradient_neumann_1d_fixed,
    _divergence_neumann_1d_fixed,
)
from src.mmsbvi.solvers.gaussian_kernel_1d import (
    apply_ou_kernel_1d_fixed,
    validate_ou_kernel_properties,
    compare_with_analytical_ou,
    _build_ou_transition_matrix,
)
from src.mmsbvi.algorithms.ipfp_1d import (
    solve_mmsb_ipfp_1d_fixed,
    validate_ipfp_solution_fixed,
)


class TestPDESolverFixed:
    """Test suite for fixed PDE solver / 修复PDE求解器的测试套件"""
    
    def test_operator_adjointness_property(self):
        """Test that ∇* = -div exactly / 测试 ∇* = -div 精确成立"""
        # Test with various functions / 用各种函数测试
        test_cases = [
            jnp.array([1.0, 2.0, 3.0, 2.0, 1.0]),  # Smooth
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # Constant
            jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]), # Oscillatory
            jnp.array([0.0, 1.0, 2.0, 1.0, 0.0]),  # Zero boundaries
        ]
        
        h = 0.1
        tolerance = 1e-12
        
        for i, u in enumerate(test_cases):
            v = test_cases[(i + 1) % len(test_cases)]
            error = operator_adjointness_error(u, v, h)
            assert error < tolerance, f"Adjointness test {i} failed: error = {error:.2e}"
    
    def test_neumann_boundary_conditions(self):
        """Test Neumann boundary conditions implementation / 测试Neumann边界条件实现"""
        # Test that gradient satisfies proper boundary conditions
        # 测试梯度满足正确的边界条件
        
        # Linear function should have constant gradient / 线性函数应该有常数梯度
        x = jnp.linspace(0, 1, 11)
        h = x[1] - x[0]
        u_linear = 2 * x + 3  # Linear: du/dx = 2
        
        grad_u = _gradient_neumann_1d_fixed(u_linear, h)
        
        # Check interior gradient is constant and boundary is zero / 检查内部梯度为常数且边界为零
        
        # Interior points should have gradient = 2 / 内部点梯度应该为2
        interior_grad = grad_u[1:-1]
        interior_mean = jnp.mean(interior_grad)
        interior_std = jnp.std(interior_grad)
        
        assert jnp.abs(interior_mean - 2.0) < 1e-10, f"Interior gradient mean wrong: {interior_mean}"
        assert interior_std < 1e-10, f"Interior gradient not constant: std = {interior_std}"
        
        # Boundary gradients should be zero for Neumann BC / Neumann边界条件下边界梯度应为零
        assert jnp.abs(grad_u[0]) < 1e-10, f"Left boundary gradient not zero: {grad_u[0]}"
        assert jnp.abs(grad_u[-1]) < 1e-10, f"Right boundary gradient not zero: {grad_u[-1]}"
    
    def test_pde_with_known_solution(self):
        """Test PDE solver with known analytical solution / 用已知解析解测试PDE求解器"""
        # Create test problem with known solution / 创建具有已知解的测试问题
        # 使用奇数网格点可避免中心差分 + Neumann 条件下出现的棋盘格零空间 /
        # Using an odd number of points avoids the checkerboard null-mode of the centered-difference
        # Neumann Laplacian, ensuring machine-precision accuracy.
        rho, sigma, grid, phi_exact = create_test_problem_1d_fixed(n_points=51)
        h = grid[1] - grid[0]
        
        # Solve PDE / 求解PDE
        phi_computed = solve_onsager_fokker_pde_fixed(rho, sigma, h)
        
        # Remove gauge freedom (both should have zero mean)
        # 去除规范自由度（两者都应该有零均值）
        phi_computed = phi_computed - jnp.mean(phi_computed)
        phi_exact = phi_exact - jnp.mean(phi_exact)
        
        # Check L2 error / 检查L2误差
        l2_error = jnp.sqrt(jax_trapz((phi_computed - phi_exact)**2, dx=h))
        assert l2_error < 1e-6, f"PDE solution error too large: {l2_error:.2e}"
        
        # Check residual / 检查残差
        residual = validate_pde_solution_fixed(rho, sigma, phi_computed, h)
        assert residual < 1e-8, f"PDE residual too large: {residual:.2e}"
    
    def test_solution_uniqueness_up_to_constant(self):
        """Test that solution is unique up to additive constant / 测试解在加性常数意义下唯一"""
        rho, sigma, grid, _ = create_test_problem_1d_fixed(n_points=30)
        h = grid[1] - grid[0]
        
        # Solve twice with different initial guesses / 用不同初值求解两次
        phi1 = solve_onsager_fokker_pde_fixed(rho, sigma, h)
        phi2 = solve_onsager_fokker_pde_fixed(rho, sigma + 1e-10, h)  # Slight perturbation
        
        # Remove means / 去除均值
        phi1 = phi1 - jnp.mean(phi1)
        phi2 = phi2 - jnp.mean(phi2)
        
        # Should be very close / 应该非常接近
        diff = jnp.max(jnp.abs(phi1 - phi2))
        assert diff < 1e-8, f"Solution not unique up to constant: diff = {diff:.2e}"


class TestOUKernelFixed:
    """Test suite for fixed OU kernel / 修复OU核的测试套件"""
    
    def test_probability_conservation(self):
        """Test that OU kernel conserves probability / 测试OU核保存概率"""
        grid = jnp.linspace(-3, 3, 50)
        dt = 0.5
        ou_params = OUProcessParams(
            mean_reversion=1.0,
            diffusion=1.0,
            equilibrium_mean=0.0
        )
        
        # Test with various initial densities / 用各种初始密度测试
        test_densities = [
            jnp.exp(-0.5 * grid**2) / jnp.sqrt(2 * jnp.pi),  # Gaussian
            jnp.exp(-jnp.abs(grid)),  # Laplacian
            jnp.ones_like(grid),  # Uniform
        ]
        
        h = grid[1] - grid[0]
        
        for i, density in enumerate(test_densities):
            # Normalize / 归一化
            density = density / jax_trapz(density, dx=h)
            
            # Apply kernel / 应用核
            evolved = apply_ou_kernel_1d_fixed(density, dt, ou_params, grid)
            
            # Check mass conservation / 检查质量守恒
            initial_mass = jax_trapz(density, dx=h)
            final_mass = jax_trapz(evolved, dx=h)
            mass_error = jnp.abs(final_mass - initial_mass)
            
            assert mass_error < 1e-10, f"Mass not conserved for density {i}: error = {mass_error:.2e}"
    
    def test_transition_matrix_properties(self):
        """Test mathematical properties of transition matrix / 测试转移矩阵的数学性质"""
        grid = jnp.linspace(-2, 2, 30)
        dt = 0.3
        ou_params = OUProcessParams(
            mean_reversion=0.5,
            diffusion=1.0,
            equilibrium_mean=0.0
        )
        
        K = _build_ou_transition_matrix(grid, dt, ou_params)
        h = grid[1] - grid[0]
        
        # Test 1: Each column should integrate to 1 (using trapezoidal rule)
        # 测试1：每列应该积分为1（使用梯形法则）
        column_integrals = h * (jnp.sum(K, axis=0) - 0.5 * (K[0, :] + K[-1, :]))
        max_error = jnp.max(jnp.abs(column_integrals - 1.0))
        assert max_error < 1e-8, f"Column integrals not 1: max_error = {max_error:.2e}"
        
        # Test 2: All entries should be non-negative / 所有元素应该非负
        min_value = jnp.min(K)
        assert min_value >= 0, f"Negative kernel values: min = {min_value}"
        
        # Test 3: Kernel should be smooth / 核应该是光滑的
        # (No sharp discontinuities)
        grad_K = jnp.gradient(K, axis=0)
        max_grad = jnp.max(jnp.abs(grad_K))
        assert max_grad < 100, f"Kernel not smooth: max_gradient = {max_grad}"
    
    def test_analytical_comparison(self):
        """Compare with analytical OU process statistics / 与解析OU过程统计量比较"""
        x0 = 1.0
        t = 0.8
        ou_params = OUProcessParams(
            mean_reversion=2.0,
            diffusion=1.5,
            equilibrium_mean=0.5
        )
        
        comparison = compare_with_analytical_ou(x0, t, ou_params)
        
        # Mean should match within numerical precision / 均值应该在数值精度内匹配
        assert comparison["mean_error"] < 1e-6, \
               f"Mean error too large: {comparison['mean_error']:.2e}"
        
        # Variance should match within numerical precision / 方差应该在数值精度内匹配
        assert comparison["variance_error"] < 1e-5, \
               f"Variance error too large: {comparison['variance_error']:.2e}"
    
    def test_equilibrium_limit(self):
        """Test approach to equilibrium distribution / 测试向平衡分布的逼近"""
        grid = jnp.linspace(-5, 5, 100)
        h = grid[1] - grid[0]
        
        ou_params = OUProcessParams(
            mean_reversion=1.0,
            diffusion=1.0,
            equilibrium_mean=0.0
        )
        
        # Start with point mass / 从点质量开始
        initial_density = jnp.zeros_like(grid)
        center_idx = len(grid) // 2
        initial_density = initial_density.at[center_idx].set(1.0 / h)
        
        # Evolve for long time / 长时间演化
        density = initial_density
        long_dt = 5.0  # Long time step
        
        for _ in range(3):  # Multiple steps
            density = apply_ou_kernel_1d_fixed(density, long_dt, ou_params, grid)
        
        # Should approach Gaussian equilibrium / 应该接近高斯平衡态
        # For θ=σ=1, equilibrium is N(0, σ²/(2θ)) = N(0, 0.5)
        equilibrium_var = 0.5
        expected_density = jnp.exp(-0.5 * grid**2 / equilibrium_var) / jnp.sqrt(2 * jnp.pi * equilibrium_var)
        expected_density = expected_density / jax_trapz(expected_density, dx=h)
        
        # Check L1 distance to equilibrium / 检查到平衡态的L1距离
        l1_distance = jax_trapz(jnp.abs(density - expected_density), dx=h)
        assert l1_distance < 0.1, f"Not approaching equilibrium: L1 distance = {l1_distance:.2e}"


class TestIPFPFixed:
    """Test suite for fixed IPFP algorithm / 修复IPFP算法的测试套件"""
    
    def test_two_marginal_convergence(self):
        """Test IPFP convergence for two marginals / 测试两边际IPFP收敛"""
        # Create simple 2-marginal problem / 创建简单的2边际问题
        grid_config = GridConfig1D.create(n_points=50, bounds=(-3.0, 3.0))
        h = grid_config.spacing
        
        # Create well-separated Gaussians / 创建分离良好的高斯分布
        grid = grid_config.points
        rho_0 = jnp.exp(-0.5 * (grid + 1.5)**2 / 0.3) 
        rho_1 = jnp.exp(-0.5 * (grid - 1.5)**2 / 0.3)
        
        # Normalize / 归一化
        rho_0 = rho_0 / jax_trapz(rho_0, dx=h)
        rho_1 = rho_1 / jax_trapz(rho_1, dx=h)
        
        # OU parameters / OU参数
        ou_params = OUProcessParams(
            mean_reversion=0.5,
            diffusion=1.0,
            equilibrium_mean=0.0
        )
        
        # Problem setup / 问题设置
        problem = MMSBProblem(
            observation_times=jnp.array([0.0, 1.0]),
            observed_marginals=[rho_0, rho_1],
            ou_params=ou_params,
            grid=grid_config
        )
        
        # Algorithm config / 算法配置
        config = IPFPConfig(
            max_iterations=200,
            tolerance=1e-7,
            check_interval=10,
            verbose=False
        )
        
        # Solve / 求解
        solution = solve_mmsb_ipfp_1d_fixed(problem, config)
        
        # Check convergence / 检查收敛
        assert solution.final_error < config.tolerance, \
               f"IPFP did not converge: final_error = {solution.final_error:.2e}"
        
        # Validate solution / 验证解
        metrics = validate_ipfp_solution_fixed(solution, problem)
        
        for k in range(2):
            l1_error = metrics[f"l1_marginal_{k}"]
            assert l1_error < 1e-6, f"Marginal {k} constraint not satisfied: L1 error = {l1_error:.2e}"
    
    def test_three_marginal_basic(self):
        """Test basic IPFP functionality for three marginals / 测试三边际的基本IPFP功能"""
        # Smaller problem for 3-marginal case / 三边际情况的较小问题
        grid_config = GridConfig1D.create(n_points=30, bounds=(-2.0, 2.0))
        h = grid_config.spacing
        grid = grid_config.points
        
        # Create three Gaussians / 创建三个高斯分布
        rho_0 = jnp.exp(-0.5 * (grid + 1.0)**2 / 0.2)
        rho_1 = jnp.exp(-0.5 * grid**2 / 0.3)
        rho_2 = jnp.exp(-0.5 * (grid - 1.0)**2 / 0.2)
        
        # Normalize / 归一化
        marginals = []
        for rho in [rho_0, rho_1, rho_2]:
            rho_norm = rho / jax_trapz(rho, dx=h)
            marginals.append(rho_norm)
        
        # OU parameters / OU参数
        ou_params = OUProcessParams(
            mean_reversion=1.0,
            diffusion=0.8,
            equilibrium_mean=0.0
        )
        
        # Problem setup / 问题设置
        problem = MMSBProblem(
            observation_times=jnp.array([0.0, 0.5, 1.0]),
            observed_marginals=marginals,
            ou_params=ou_params,
            grid=grid_config
        )
        
        # Algorithm config (looser tolerance for 3-marginal)
        # 算法配置（三边际的较松容差）
        config = IPFPConfig(
            max_iterations=100,
            tolerance=1e-5,
            check_interval=10,
            verbose=False
        )
        
        # Solve / 求解
        solution = solve_mmsb_ipfp_1d_fixed(problem, config)
        
        # Basic checks / 基本检查
        assert solution.n_iterations < config.max_iterations, "IPFP did not converge in time"
        assert len(solution.potentials) == 3, "Wrong number of potentials"
        assert len(solution.path_densities) == 3, "Wrong number of path densities"
        
        # Check that path densities are normalized / 检查路径密度是否归一化
        for k, density in enumerate(solution.path_densities):
            mass = jax_trapz(density, dx=h)
            assert jnp.abs(mass - 1.0) < 1e-6, f"Path density {k} not normalized: mass = {mass}"

        # Enhanced validation of marginal constraints / 增强对边际约束的验证
        metrics = validate_ipfp_solution_fixed(solution, problem)
        for k in range(3):
            l1_error = metrics[f"l1_marginal_{k}"]
            assert l1_error < 1e-4, f"Marginal {k} constraint not satisfied: L1 error = {l1_error:.2e}"

    def test_four_marginal_convergence(self):
        """Test IPFP convergence for four marginals / 测试四边际IPFP收敛"""
        grid_config = GridConfig1D.create(n_points=40, bounds=(-3.0, 3.0))
        h = grid_config.spacing
        grid = grid_config.points

        # Create four Gaussians / 创建四个高斯分布
        rho_0 = jnp.exp(-0.5 * (grid + 2.0)**2 / 0.2)
        rho_1 = jnp.exp(-0.5 * (grid + 0.7)**2 / 0.2)
        rho_2 = jnp.exp(-0.5 * (grid - 0.7)**2 / 0.2)
        rho_3 = jnp.exp(-0.5 * (grid - 2.0)**2 / 0.2)
        
        marginals = []
        for rho in [rho_0, rho_1, rho_2, rho_3]:
            rho_norm = rho / jax_trapz(rho, dx=h)
            marginals.append(rho_norm)

        ou_params = OUProcessParams(
            mean_reversion=0.8,
            diffusion=0.9,
            equilibrium_mean=0.0
        )

        problem = MMSBProblem(
            observation_times=jnp.array([0.0, 0.3, 0.7, 1.0]),
            observed_marginals=marginals,
            ou_params=ou_params,
            grid=grid_config
        )

        config = IPFPConfig(
            max_iterations=250,
            tolerance=1e-5,
            check_interval=10,
            verbose=False
        )

        solution = solve_mmsb_ipfp_1d_fixed(problem, config)

        assert solution.final_error < config.tolerance, \
               f"IPFP did not converge for K=4: final_error = {solution.final_error:.2e}"

        metrics = validate_ipfp_solution_fixed(solution, problem)
        for k in range(4):
            l1_error = metrics[f"l1_marginal_{k}"]
            assert l1_error < 1e-4, f"Marginal {k} (K=4) not satisfied: L1 error = {l1_error:.2e}"


def run_full_validation_suite():
    """
    Run the complete mathematical validation suite.
    运行完整的数学验证套件。
    
    This must pass before trusting any numerical results!
    在信任任何数值结果之前这必须通过！
    """
    print("=" * 80)
    print("COMPREHENSIVE MATHEMATICAL VALIDATION SUITE")
    print("全面数学验证套件")
    print("=" * 80)
    
    # Test classes / 测试类
    test_classes = [
        ("PDE Solver Fixed", TestPDESolverFixed),
        ("OU Kernel Fixed", TestOUKernelFixed), 
        ("IPFP Algorithm Fixed", TestIPFPFixed),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for class_name, test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Testing: {class_name}")
        print(f"测试: {class_name}")
        print(f"{'='*50}")
        
        # Get all test methods / 获取所有测试方法
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        class_instance = test_class()
        
        for method_name in test_methods:
            total_tests += 1
            test_method = getattr(class_instance, method_name)
            
            print(f"\nRunning: {method_name}")
            try:
                test_method()
                print(f"✓ PASSED")
                passed_tests += 1
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                failed_tests.append(f"{class_name}.{method_name}: {str(e)}")
    
    # Summary / 总结
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY / 验证总结")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print(f"\nFAILED TESTS:")
        for test in failed_tests:
            print(f"  - {test}")
        print(f"\n⚠️  VALIDATION FAILED - DO NOT PROCEED WITH EXPERIMENTS!")
        print(f"⚠️  验证失败 - 不要进行实验!")
        return False
    else:
        print(f"\n✅ ALL TESTS PASSED - SAFE TO PROCEED WITH EXPERIMENTS!")
        print(f"✅ 所有测试通过 - 可以安全进行实验!")
        return True


if __name__ == "__main__":
    success = run_full_validation_suite()
    sys.exit(0 if success else 1)