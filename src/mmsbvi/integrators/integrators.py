"""
SDE Numerical Integration Methods
SDE数值积分方法

High-performance JAX-optimized SDE integrators with GPU acceleration.
基于JAX优化的高性能SDE积分器，支持GPU加速。

Implements Euler-Maruyama, Heun, and a correct diagonal Milstein scheme.
实现Euler-Maruyama、Heun以及一个正确的对角Milstein格式。
"""

import jax
import jax.numpy as jnp
import jax.random
from jax import jit, vmap, lax
from functools import partial
import inspect
from typing import Optional, Tuple, Callable
import chex

from ..core.types import (
    SDEState, DriftFunction, DiffusionFunction, DiffusionDerivative,
    SDEIntegratorConfig, Float, Array
)
from ..core.registry import register_integrator, get_integrator_class

# ============================================================================
# Base Integrator Class / 基础积分器类
# ============================================================================

class BaseSDEIntegrator:
    """SDE积分器基类 / Base class for SDE integrators."""
    def __init__(self, config: Optional[SDEIntegratorConfig] = None):
        self.config = config or SDEIntegratorConfig()

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        raise NotImplementedError("Subclasses must implement _step_impl.")

    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        # This method is decorated with JIT in subclasses.
        # 此方法在子类中被JIT装饰。
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def integrate(self, initial_state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, time_grid: Float[Array, "n"], key: jax.random.PRNGKey) -> Float[Array, "n d"]:
        """使用lax.scan进行高效的多步积分 / Multi-step integration using lax.scan for efficiency."""
        def scan_fn(carry, scan_input):
            state, t_current = carry
            t_next, subkey = scan_input
            dt = t_next - t_current
            new_state = self.step(t_current, state, drift_fn, diffusion_fn, dt, subkey)
            return (new_state, t_next), new_state
        
        n_steps = len(time_grid) - 1
        keys = jax.random.split(key, n_steps)
        scan_inputs = (time_grid[1:], keys)
        init_carry = (initial_state, time_grid[0])
        _, trajectory_steps = lax.scan(scan_fn, init_carry, scan_inputs)
        return jnp.concatenate([jnp.expand_dims(initial_state, 0), trajectory_steps], axis=0)

    def integrate_batch(self, initial_states: Float[Array, "batch d"], drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, time_grid: Float[Array, "n"], key: jax.random.PRNGKey) -> Float[Array, "batch n d"]:
        """内存高效的批量积分 / Memory-efficient batch integration."""
        def scan_fn(carry_states, scan_input):
            t_current, t_next, subkey = scan_input
            dt = t_next - t_current
            new_states = self.step(t_current, carry_states, drift_fn, diffusion_fn, dt, subkey)
            chex.assert_equal_shape([new_states, carry_states])
            return new_states, new_states

        n_steps = len(time_grid) - 1
        keys = jax.random.split(key, n_steps)
        scan_inputs = (time_grid[:-1], time_grid[1:], keys)
        _, trajectory_steps = lax.scan(scan_fn, initial_states, scan_inputs)
        trajectory_steps = jnp.transpose(trajectory_steps, (1, 0, 2))
        initial_states_expanded = jnp.expand_dims(initial_states, axis=1)
        return jnp.concatenate([initial_states_expanded, trajectory_steps], axis=1)

# ============================================================================
# Concrete Integrators / 具体积分器实现
# ============================================================================

@register_integrator("euler_maruyama")
class EulerMaruyamaIntegrator(BaseSDEIntegrator):
    @partial(jit, static_argnums=(0, 3, 4))
    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        drift = drift_fn(state, t)
        noise = jax.random.normal(key, state.shape)
        diffusion = diffusion_fn(state, t)
        return state + drift * dt + diffusion * jnp.sqrt(dt) * noise

@register_integrator("heun")
class HeunIntegrator(BaseSDEIntegrator):
    @partial(jit, static_argnums=(0, 3, 4))
    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        dW = jax.random.normal(key, state.shape) * jnp.sqrt(dt)
        drift_n = drift_fn(state, t)
        diffusion_n = diffusion_fn(state, t)
        x_pred = state + drift_n * dt + diffusion_n * dW
        drift_pred = drift_fn(x_pred, t + dt)
        diffusion_pred = diffusion_fn(x_pred, t + dt)
        return state + 0.5 * (drift_n + drift_pred) * dt + 0.5 * (diffusion_n + diffusion_pred) * dW

@register_integrator("milstein")
class MilsteinIntegrator(BaseSDEIntegrator):
    def __init__(self, config: Optional[SDEIntegratorConfig] = None, diffusion_derivative: Optional[DiffusionDerivative] = None):
        super().__init__(config)
        assert diffusion_derivative is not None, "MilsteinIntegrator requires a diffusion_derivative function."
        self.diffusion_derivative = diffusion_derivative

    @partial(jit, static_argnums=(0, 3, 4))
    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        dW = jax.random.normal(key, state.shape) * jnp.sqrt(dt)
        drift_term = drift_fn(state, t) * dt
        diffusion_val = diffusion_fn(state, t)
        diffusion_term = diffusion_val * dW
        
        diffusion_grad = self.diffusion_derivative(state, t)
        
        if state.ndim > 0 and hasattr(diffusion_grad, 'ndim') and diffusion_grad.ndim == 2:
            diffusion_grad_diag = jnp.diag(diffusion_grad)
            levy_area = 0.5 * diffusion_val * diffusion_grad_diag * (dW**2 - dt)
        else:
            levy_area = 0.5 * diffusion_val * diffusion_grad * (dW**2 - dt)
            
        new_state = state + drift_term + diffusion_term + levy_area
        return jnp.reshape(new_state, state.shape)

# ============================================================================
# Ultra High-Performance Integrators / 极致高性能积分器
# ============================================================================

@register_integrator("euler_maruyama_ultra")
class UltraEulerMaruyamaIntegrator(BaseSDEIntegrator):
    """极致优化的Euler-Maruyama积分器 / Ultra-optimized Euler-Maruyama integrator"""
    
    @partial(jit, static_argnums=(0, 3, 4))
    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        """标准step接口，保持兼容性 / Standard step interface for compatibility"""
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        """标准实现 / Standard implementation"""
        drift = drift_fn(state, t)
        noise = jax.random.normal(key, state.shape)
        diffusion = diffusion_fn(state, t)
        return state + drift * dt + diffusion * jnp.sqrt(dt) * noise
    
    @partial(jit, static_argnums=(0,))
    def integrate_batch_ultra(self, initial_states: Float[Array, "batch d"], 
                             drift_coeff: float, diffusion_coeff: float,
                             time_grid: Float[Array, "n"], 
                             key: jax.random.PRNGKey) -> Float[Array, "batch n d"]:
        """
        超高性能批量积分，专门针对线性SDE：dX = a*X*dt + σ*dW
        Pre-generates all random numbers and uses fused operations
        """
        batch_size, state_dim = initial_states.shape
        n_steps = len(time_grid) - 1
        
        # 生成与原版积分器一致的随机数序列
        dt_values = jnp.diff(time_grid)
        sqrt_dt_values = jnp.sqrt(dt_values)
        
        # 分裂key以匹配原版的随机数生成模式
        step_keys = jax.random.split(key, n_steps)
        all_noise = jax.vmap(
            lambda k: jax.random.normal(k, (batch_size, state_dim))
        )(step_keys) * sqrt_dt_values[:, None, None]
        
        # 融合的向量化步进函数
        @jit
        def fused_step_fn(states, inputs):
            dt, noise = inputs
            # 融合的Euler-Maruyama步骤：已经包含sqrt(dt)
            drift_term = drift_coeff * states * dt
            diffusion_term = diffusion_coeff * noise  
            return states + drift_term + diffusion_term
        
        # 使用scan进行时间步进
        def scan_fn(carry_states, scan_inputs):
            dt, batch_noise = scan_inputs
            new_states = fused_step_fn(carry_states, (dt, batch_noise))
            return new_states, new_states
        
        scan_inputs = (dt_values, all_noise)
        _, trajectory = lax.scan(scan_fn, initial_states, scan_inputs)
        
        # 直接构造结果，避免转置和连接
        result = jnp.concatenate([
            initial_states[None, :, :], 
            trajectory
        ], axis=0).transpose(1, 0, 2)
        
        return result

@register_integrator("heun_ultra")  
class UltraHeunIntegrator(BaseSDEIntegrator):
    """极致优化的Heun积分器 / Ultra-optimized Heun integrator"""
    
    @partial(jit, static_argnums=(0, 3, 4))
    def step(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        """标准step接口，保持兼容性 / Standard step interface for compatibility"""
        return self._step_impl(t, state, drift_fn, diffusion_fn, dt, key)

    def _step_impl(self, t: float, state: SDEState, drift_fn: DriftFunction, diffusion_fn: DiffusionFunction, dt: float, key: jax.random.PRNGKey) -> SDEState:
        """标准Heun实现 / Standard Heun implementation"""
        dW = jax.random.normal(key, state.shape) * jnp.sqrt(dt)
        drift_n = drift_fn(state, t)
        diffusion_n = diffusion_fn(state, t)
        x_pred = state + drift_n * dt + diffusion_n * dW
        drift_pred = drift_fn(x_pred, t + dt)
        diffusion_pred = diffusion_fn(x_pred, t + dt)
        return state + 0.5 * (drift_n + drift_pred) * dt + 0.5 * (diffusion_n + diffusion_pred) * dW
    
    @partial(jit, static_argnums=(0,))
    def integrate_batch_ultra(self, initial_states: Float[Array, "batch d"],
                             drift_coeff: float, diffusion_coeff: float, 
                             time_grid: Float[Array, "n"],
                             key: jax.random.PRNGKey) -> Float[Array, "batch n d"]:
        """超高性能Heun积分 / Ultra-high performance Heun integration"""
        batch_size, state_dim = initial_states.shape
        n_steps = len(time_grid) - 1
        
        # 生成与原版积分器一致的随机数序列
        dt_values = jnp.diff(time_grid)
        sqrt_dt_values = jnp.sqrt(dt_values)
        
        # 分裂key以匹配原版的随机数生成模式
        step_keys = jax.random.split(key, n_steps)
        all_noise = jax.vmap(
            lambda k: jax.random.normal(k, (batch_size, state_dim))
        )(step_keys) * sqrt_dt_values[:, None, None]
        
        @jit
        def fused_heun_step(states, inputs):
            dt, noise = inputs
            
            # 预测步骤
            drift_n = drift_coeff * states
            diffusion_term = diffusion_coeff * noise  # 已包含sqrt(dt)
            x_pred = states + drift_n * dt + diffusion_term
            
            # 校正步骤  
            drift_pred = drift_coeff * x_pred
            final_drift = 0.5 * (drift_n + drift_pred) * dt
            
            return states + final_drift + diffusion_term
        
        def scan_fn(carry_states, scan_inputs):
            dt, batch_noise = scan_inputs
            new_states = fused_heun_step(carry_states, (dt, batch_noise))
            return new_states, new_states
            
        scan_inputs = (dt_values, all_noise)
        _, trajectory = lax.scan(scan_fn, initial_states, scan_inputs)
        
        result = jnp.concatenate([
            initial_states[None, :, :],
            trajectory  
        ], axis=0).transpose(1, 0, 2)
        
        return result

# ============================================================================
# Factory Function / 工厂函数
# ============================================================================

def create_integrator(
    method: str = "euler_maruyama",
    config: Optional[SDEIntegratorConfig] = None,
    diffusion_fn: Optional[DiffusionFunction] = None,
    **kwargs
) -> BaseSDEIntegrator:
    """创建SDE积分器的工厂函数 / Factory function to create SDE integrators."""
    integrator_cls = get_integrator_class(method)
    
    init_kwargs = {"config": config, **kwargs}
    
    if method == "milstein" and "diffusion_derivative" not in init_kwargs:
        assert diffusion_fn is not None, "diffusion_fn must be provided for Milstein integrator."
        try:
            output_shape = jax.eval_shape(lambda x: diffusion_fn(x, 0.0), jnp.ones((1,))).shape
            is_scalar_output = (len(output_shape) == 0) or (len(output_shape) == 1 and output_shape[0] == 1)
        except Exception:
            is_scalar_output = False
            
        if is_scalar_output:
            init_kwargs["diffusion_derivative"] = jax.grad(lambda x, t: diffusion_fn(x, t).sum())
        else:
            init_kwargs["diffusion_derivative"] = jax.jacfwd(diffusion_fn, argnums=0)
            
    sig = inspect.signature(integrator_cls.__init__)
    accepted_kwargs = {k: v for k, v in init_kwargs.items() if k in sig.parameters}
    
    return integrator_cls(**accepted_kwargs)