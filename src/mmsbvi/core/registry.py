"""
Global Component Registry
全局组件注册表

Lightweight registry system for pluggable solvers, networks, and samplers.
可插拔求解器、网络和采样器的轻量级注册表系统。
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from functools import wraps
import inspect
import jax
import jax.numpy as jnp
import flax.linen as nn

from .types import (
    SDEIntegrator, SDEIntegratorConfig, DriftNetwork, NetworkConfig, 
    TrainingConfig, PerformanceConfig, NetworkParams, NetworkTrainingState
)


# ============================================================================
# Global Registries / 全局注册表
# ============================================================================

INTEGRATOR_REGISTRY: Dict[str, Type[SDEIntegrator]] = {}
SOLVER_REGISTRY: Dict[str, Callable] = {}
NETWORK_REGISTRY: Dict[str, Callable] = {}
SAMPLER_REGISTRY: Dict[str, Callable] = {}


# ============================================================================
# Registration Decorators / 注册装饰器
# ============================================================================

def register_integrator(name: str) -> Callable:
    """Decorator to register SDE integrators"""
    def decorator(cls: Type[SDEIntegrator]) -> Type[SDEIntegrator]:
        if name in INTEGRATOR_REGISTRY:
            raise ValueError(f"SDE integrator '{name}' already registered")
        INTEGRATOR_REGISTRY[name] = cls
        return cls
    return decorator

def register_solver(name: str) -> Callable:
    """Decorator to register PDE solvers"""
    def decorator(cls_or_fn: Union[Type, Callable]) -> Union[Type, Callable]:
        if name in SOLVER_REGISTRY:
            raise ValueError(f"Solver '{name}' already registered")
        SOLVER_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator

def register_network(name: str) -> Callable:
    """Decorator to register neural networks"""
    def decorator(cls_or_fn: Union[Type, Callable]) -> Union[Type, Callable]:
        if name in NETWORK_REGISTRY:
            raise ValueError(f"Network '{name}' already registered")
        NETWORK_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator

def register_sampler(name: str) -> Callable:
    """Decorator to register samplers"""
    def decorator(cls_or_fn: Union[Type, Callable]) -> Union[Type, Callable]:
        if name in SAMPLER_REGISTRY:
            raise ValueError(f"Sampler '{name}' already registered")
        SAMPLER_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


# ============================================================================
# Factory Functions / 工厂函数
# ============================================================================

def get_integrator_class(name: str) -> Type[SDEIntegrator]:
    """Get SDE integrator class by name."""
    if name not in INTEGRATOR_REGISTRY:
        available = list(INTEGRATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown SDE integrator: '{name}'. Available: {available}"
        )
    return INTEGRATOR_REGISTRY[name]

def get_integrator(
    name: str, 
    config: Optional[SDEIntegratorConfig] = None,
    **kwargs
) -> SDEIntegrator:
    """
    Create SDE integrator by name
    """
    integrator_cls = get_integrator_class(name)
    
    init_kwargs = {"config": config}
    
    # Pass kwargs only if they are in the constructor signature
    sig = inspect.signature(integrator_cls.__init__)
    for param in sig.parameters:
        if param in kwargs:
            init_kwargs[param] = kwargs[param]
            
    return integrator_cls(**init_kwargs)


def get_solver(name: str, **kwargs) -> Any:
    """Create solver by name"""
    if name not in SOLVER_REGISTRY:
        available = list(SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver: '{name}'. Available: {available}")
    
    solver_cls = SOLVER_REGISTRY[name]
    return solver_cls(**kwargs) if inspect.isclass(solver_cls) else solver_cls

def get_network(
    name: str,
    config: Optional[NetworkConfig] = None,
    state_dim: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Create network by name"""
    if name not in NETWORK_REGISTRY:
        available = list(NETWORK_REGISTRY.keys())
        raise ValueError(f"Unknown network: '{name}'. Available: {available}")
    
    network_cls = NETWORK_REGISTRY[name]
    
    if inspect.isclass(network_cls):
        init_kwargs = {"config": config, "state_dim": state_dim, **kwargs}
        # Filter kwargs to only those accepted by the constructor
        sig = inspect.signature(network_cls.__init__)
        accepted_kwargs = {k: v for k, v in init_kwargs.items() if k in sig.parameters}
        return network_cls(**accepted_kwargs)
    return network_cls

# ============================================================================
# Registry Information / 注册表信息
# ============================================================================

def list_integrators() -> Dict[str, Type[SDEIntegrator]]:
    """List all registered SDE integrators"""
    return INTEGRATOR_REGISTRY.copy()

def list_solvers() -> Dict[str, Callable]:
    """List all registered solvers"""
    return SOLVER_REGISTRY.copy()

def list_networks() -> Dict[str, Callable]:
    """List all registered networks"""
    return NETWORK_REGISTRY.copy()

def list_samplers() -> Dict[str, Callable]:
    """List all registered samplers"""
    return SAMPLER_REGISTRY.copy()

def clear_registry(registry_type: str = "all") -> None:
    """Clear registry contents (mainly for testing)"""
    if registry_type in ["integrator", "all"]:
        INTEGRATOR_REGISTRY.clear()
    if registry_type in ["solver", "all"]:
        SOLVER_REGISTRY.clear()
    if registry_type in ["network", "all"]:
        NETWORK_REGISTRY.clear()
    if registry_type in ["sampler", "all"]:
        SAMPLER_REGISTRY.clear()