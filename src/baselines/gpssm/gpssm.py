"""
Gaussian Process State-Space Model (GPSSM) Solver / GPSSM求解器
================================================================

This module provides the main GPSSM solver class, which orchestrates the
components from `types.py`, `gpr.py`, and `inference.py` to perform
end-to-end training and prediction.

此模块提供了主要的GPSSM求解器类，它负责协调 `types.py`, `gpr.py`,
和 `inference.py` 中的组件，以执行端到端的训练和预测。
"""

import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from functools import partial
from typing import Callable, Tuple, Dict, Any
import time
import chex
from sklearn.cluster import KMeans

from .types import (
    GPSSMConfig, OptimizerConfig, GPSSMState, KernelParams,
    InducingPoints, GPParams, VariationalParams, TrainingState
)
from . import inference
from . import gpr

class GPSSMSolver:
    """
    A generic solver for Gaussian Process State-Space Models.
    一个通用的高斯过程状态空间模型求解器。
    """
    def __init__(
        self,
        config: GPSSMConfig,
        opt_config: OptimizerConfig,
        dynamics_fn: Callable[[chex.Array], chex.Array],
        observation_fn: Callable[[chex.Array], chex.Array]
    ):
        """
        Initializes the GPSSM solver.

        Args:
            config: GPSSM model configuration. / GPSSM模型配置。
            opt_config: Optimizer configuration. / 优化器配置。
            dynamics_fn: The deterministic part of the dynamics, f_det(x).
                         动态模型的确定性部分，f_det(x)。
            observation_fn: The observation function, h(x).
                            观测函数，h(x)。
        """
        self.config = config
        self.opt_config = opt_config
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(opt_config.clip_norm),
            optax.adam(opt_config.learning_rate)
        )

    def initialize(
        self,
        key: chex.PRNGKey,
        observations: chex.Array
    ) -> TrainingState:
        """
        Initializes the parameters and optimizer state.
        初始化参数和优化器状态。

        Args:
            key: JAX random key.
            observations: The sequence of observations [T, P] for smart init.
                          用于智能初始化的观测序列 [T, P]。

        Returns:
            The initial training state.
        """
        key_gp, key_var, key_init = jax.random.split(key, 3)
        T, P = observations.shape
        D = self.config.state_dim
        M = self.config.num_inducing

        # Initialize inducing points using k-means on observations
        # This requires a bit of care if obs_dim != state_dim
        # For now, we use a simple heuristic if P < D
        if P >= D:
            kmeans = KMeans(n_clusters=M, random_state=0, n_init='auto').fit(observations)
            inducing_locs = jnp.array(kmeans.cluster_centers_)
            if P > D:
                inducing_locs = inducing_locs[:, :D] # Truncate
        else:
            # Fallback to random initialization around data mean
            obs_mean = jnp.mean(observations, axis=0)
            pad = jnp.zeros(D - P)
            center = jnp.concatenate([obs_mean, pad])
            inducing_locs = center + jax.random.normal(key_gp, (M, D))

        # GP Parameters
        gp_params = GPParams(
            kernel=KernelParams(
                lengthscale=jnp.ones(D),
                variance=1.0
            ),
            inducing=InducingPoints(
                z=inducing_locs,
                m=jnp.zeros((M, D)),
                L=jnp.eye(M)
            ),
            obs_noise_variance=0.1
        )

        # Variational Parameters
        var_params = VariationalParams(
            q_mu=jnp.zeros((T, D)),
            q_sqrt=jnp.tile(jnp.eye(D) * 0.1, (T, 1, 1))
        )

        params = GPSSMState(gp=gp_params, variational=var_params)
        opt_state = self.optimizer.init(params)

        return TrainingState(
            params=params,
            opt_state=opt_state,
            key=key_init,
            iteration=0
        )

    def _update_step(
        self,
        train_state: TrainingState,
        observations: chex.Array
    ) -> Tuple[TrainingState, Dict[str, chex.Scalar]]:
        """
        Performs a single optimization step.
        执行单步优化。
        """
        key, loss_key = jax.random.split(train_state.key)

        # Define the loss function to be JIT-compiled
        @partial(jit, static_argnames=('dynamics_fn', 'observation_fn', 'config'))
        def loss_fn(params, obs, key, dynamics_fn, observation_fn, config):
            return -inference.compute_elbo(
                params, obs, key, dynamics_fn, observation_fn, config
            )
        
        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fn)(
            train_state.params,
            observations,
            loss_key,
            self.dynamics_fn,
            self.observation_fn,
            self.config
        )
        
        updates, new_opt_state = self.optimizer.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)

        metrics = {'elbo': -loss, 'loss': loss}
        
        new_train_state = TrainingState(
            params=new_params,
            opt_state=new_opt_state,
            key=key,
            iteration=train_state.iteration + 1
        )
        return new_train_state, metrics

    def fit(
        self,
        key: chex.PRNGKey,
        observations: chex.Array
    ) -> Tuple[GPSSMState, Dict[str, Any]]:
        """
        Fits the GPSSM model to the data.
        将GPSSM模型拟合到数据。

        Args:
            key: JAX random key.
            observations: The sequence of observations [T, P].

        Returns:
            A tuple containing:
            - The final optimized parameters.
            - A history of training metrics.
        """
        train_state = self.initialize(key, observations)
        history = {'elbo': [], 'loss': [], 'time': []}

        print(f"--- Starting GPSSM Training ---")
        print(f"Total iterations: {self.opt_config.num_iterations}")
        print(f"State dim: {self.config.state_dim}, Obs dim: {self.config.obs_dim}")
        print(f"Num. inducing: {self.config.num_inducing}")
        
        start_time = time.time()
        for i in range(self.opt_config.num_iterations):
            iter_start_time = time.time()
            train_state, metrics = self._update_step(train_state, observations)
            
            iter_time = time.time() - iter_start_time
            history['elbo'].append(metrics['elbo'])
            history['loss'].append(metrics['loss'])
            history['time'].append(iter_time)

            if i % 100 == 0:
                print(f"Iter {i:4d}/{self.opt_config.num_iterations} | "
                      f"ELBO: {metrics['elbo']:.4f} | "
                      f"Time/iter: {iter_time:.3f}s")

        total_time = time.time() - start_time
        print(f"--- Training Finished ---")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final ELBO: {history['elbo'][-1]:.4f}")

        return train_state.params, history

    @partial(jit, static_argnums=(0, 2))
    def predict(
        self,
        params: GPSSMState,
        num_steps: int,
        initial_state: chex.Array
    ) -> chex.Array:
        """
        Predicts future states by rolling out the learned dynamics.
        通过展开学习到的动态来预测未来状态。

        Args:
            params: The trained model parameters.
            num_steps: The number of future steps to predict.
            initial_state: The starting state for prediction [D].

        Returns:
            The sequence of predicted state means [num_steps, D].
        """
        def rollout_step(state, _):
            f_mean, _ = gpr.predict_f(
                state[None, :], params.gp.inducing, params.gp.kernel, self.config, self.config.state_dim
            )
            next_state = self.dynamics_fn(state) + f_mean.squeeze(0)
            return next_state, next_state

        _, predictions = jax.lax.scan(
            rollout_step, initial_state, None, length=num_steps
        )
        return predictions