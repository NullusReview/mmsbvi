"""
Algorithms module for MMSBVI
MMSBVI��!W

This module provides the main algorithms for solving MMSB problems.
,!WЛB�MMSB;���
"""

from .ipfp_1d import (
    solve_mmsb_ipfp_1d_fixed,
    validate_ipfp_solution_fixed,
)

__all__ = [
    "solve_mmsb_ipfp_1d_fixed",
    "validate_ipfp_solution_fixed",
]