#!/usr/bin/env python3
"""
Test OU-to-Linear mapping consistency
测试OU到线性系统的映射一致性
"""

import jax
import jax.numpy as jnp
import math

jax.config.update('jax_enable_x64', True)

def test_ou_linear_consistency():
    """Test the mapping between linear Gaussian system and OU process"""
    
    print("=== Testing OU-Linear System Mapping ===")
    
    # Linear system parameters
    A = 0.8
    Q = 0.1
    dt = 1.0  # discrete time step
    
    # Method 1: From demo (potentially incorrect)
    theta_demo = -math.log(A) / dt
    sigma_demo = math.sqrt(2 * theta_demo * Q / (1 - A ** 2))
    
    # Method 2: Theoretical correct mapping
    # For OU: dX = -theta * X * dt + sigma * dW
    # Discrete: X_{k+1} = exp(-theta*dt) * X_k + noise
    # So: A = exp(-theta*dt) => theta = -log(A)/dt
    theta_correct = -math.log(A) / dt
    
    # For variance: Var[X_{k+1}|X_k] = Q
    # In OU: stationary_var = sigma^2/(2*theta)
    # Transition var = stationary_var * (1 - exp(-2*theta*dt))
    # So: Q = sigma^2/(2*theta) * (1 - A^2)
    # => sigma^2 = 2*theta*Q / (1 - A^2)
    sigma_correct = math.sqrt(2 * theta_correct * Q / (1 - A**2))
    
    print(f"Linear system: A={A}, Q={Q}, dt={dt}")
    print(f"Demo mapping: theta={theta_demo:.6f}, sigma={sigma_demo:.6f}")
    print(f"Correct mapping: theta={theta_correct:.6f}, sigma={sigma_correct:.6f}")
    print(f"Mapping difference: theta_diff={abs(theta_demo-theta_correct):.2e}, sigma_diff={abs(sigma_demo-sigma_correct):.2e}")
    
    # Verify by checking transition properties
    # OU transition mean: x_next = x * exp(-theta*dt) = x * A
    # OU transition var: sigma^2/(2*theta) * (1 - exp(-2*theta*dt)) = sigma^2/(2*theta) * (1 - A^2)
    
    ou_mean_coeff = math.exp(-theta_correct * dt)
    ou_var = (sigma_correct**2 / (2 * theta_correct)) * (1 - math.exp(-2 * theta_correct * dt))
    
    print(f"\nVerification:")
    print(f"OU mean coefficient: {ou_mean_coeff:.6f} (should equal A={A})")
    print(f"OU transition variance: {ou_var:.6f} (should equal Q={Q})")
    print(f"Mean coefficient error: {abs(ou_mean_coeff - A):.2e}")
    print(f"Variance error: {abs(ou_var - Q):.2e}")
    
    return abs(ou_mean_coeff - A) < 1e-10 and abs(ou_var - Q) < 1e-10

def test_multiple_systems():
    """Test multiple parameter combinations"""
    
    test_cases = [
        {"A": 0.8, "Q": 0.1, "name": "Case 1"},
        {"A": 0.6065, "Q": 0.095, "name": "Case 2"},
        {"A": 0.9, "Q": 0.05, "name": "Case 3"},
    ]
    
    print("\n=== Testing Multiple Systems ===")
    
    all_passed = True
    for case in test_cases:
        A, Q = case["A"], case["Q"]
        dt = 1.0
        
        theta = -math.log(A) / dt
        sigma = math.sqrt(2 * theta * Q / (1 - A**2))
        
        # Verify
        ou_mean_coeff = math.exp(-theta * dt)
        ou_var = (sigma**2 / (2 * theta)) * (1 - math.exp(-2 * theta * dt))
        
        mean_error = abs(ou_mean_coeff - A)
        var_error = abs(ou_var - Q)
        
        passed = mean_error < 1e-10 and var_error < 1e-10
        all_passed = all_passed and passed
        
        print(f"{case['name']}: A={A}, Q={Q} -> theta={theta:.4f}, sigma={sigma:.4f}")
        print(f"  Errors: mean={mean_error:.2e}, var={var_error:.2e} {'✓' if passed else '✗'}")
    
    return all_passed

if __name__ == "__main__":
    consistency_ok = test_ou_linear_consistency()
    multiple_ok = test_multiple_systems()
    
    if consistency_ok and multiple_ok:
        print("\n🎉 ALL OU-LINEAR MAPPINGS ARE CONSISTENT!")
    else:
        print("\n⚠️ OU-Linear mapping has issues!")