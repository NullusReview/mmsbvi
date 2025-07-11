"""
Solvers module for MMSBVI
MMSBVI求解器模块

This module provides PDE solvers and related numerical methods.
本模块提供PDE求解器和相关数值方法。
"""

from .pde_solver_1d import (
    solve_onsager_fokker_pde_fixed,
    compute_onsager_fokker_metric_fixed,
    validate_pde_solution_fixed,
    create_test_problem_1d_fixed,
    operator_adjointness_error,
)

from .gaussian_kernel_1d import (
    apply_ou_kernel_1d_fixed,
    apply_backward_ou_kernel_1d_fixed,
    compute_log_transition_kernel_1d_fixed,
    estimate_kernel_bandwidth_fixed,
    validate_ou_kernel_properties,
    compare_with_analytical_ou,
)

__all__ = [
    # PDE solver (FIXED) / PDE求解器（修复版）
    "solve_onsager_fokker_pde_fixed",
    "compute_onsager_fokker_metric_fixed", 
    "validate_pde_solution_fixed",
    "create_test_problem_1d_fixed",
    "operator_adjointness_error",
    
    # OU kernel (FIXED) / OU核（修复版）
    "apply_ou_kernel_1d_fixed",
    "apply_backward_ou_kernel_1d_fixed",
    "compute_log_transition_kernel_1d_fixed",
    "estimate_kernel_bandwidth_fixed",
    "validate_ou_kernel_properties",
    "compare_with_analytical_ou",
]