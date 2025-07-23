"""
大角度单摆数据生成器 / Large Angle Pendulum Data Generator
=====================================================

为高维非线性状态空间模型实验生成鲁棒的、具挑战性的测试数据集。
Generates robust and challenging test datasets for high-dimensional nonlinear state-space model experiments.

核心特性 / Key Features:
- 完整非线性动力学: θ̈ = -(g/L)sin(θ) - γθ̇ + τ(t)
- 可选的外部驱动力: 可模拟受驱单摆等非自治系统。
- 结构化观测策略: 可模拟传感器在特定阶段数据缺失的场景。
- 多样化的初始条件: 覆盖振荡和旋转等多种运动模式。
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from typing import NamedTuple, Dict, List, Optional, Tuple, Callable
import chex
from functools import partial
import time
import pickle
import pathlib

jax.config.update('jax_enable_x64', True)


class PendulumParams(NamedTuple):
    """
    大角度单摆物理参数 / Large angle pendulum physical parameters
    
    Attributes:
        g: 重力加速度 / gravitational acceleration [m/s²]
        L: 摆长 / pendulum length [m]
        gamma: 阻尼系数 / damping coefficient [1/s]
        sigma: 过程噪声强度 (扭矩噪声标准差) / process noise intensity (torque noise std) [N·m]
        forcing_amplitude: 外部驱动力振幅 / amplitude of external forcing torque [N·m]
        forcing_frequency: 外部驱动力频率 / frequency of external forcing torque [rad/s]
    """
    g: float = 9.81
    L: float = 1.0
    gamma: float = 0.1
    sigma: float = 0.05
    forcing_amplitude: float = 0.0  # 默认为0，即无外部驱动力
    forcing_frequency: float = 0.0 # 默认为0


class ObservationConfig(NamedTuple):
    """观测配置 / Observation configuration"""
    obs_times: chex.Array
    obs_noise_std: float = 0.1
    strategy: str = "dense"


class PendulumTrajectory(NamedTuple):
    """单摆轨迹数据结构 / Pendulum trajectory data structure"""
    times: chex.Array
    states: chex.Array
    observations: chex.Array
    obs_times: chex.Array
    true_obs_values: chex.Array
    params: PendulumParams
    obs_config: ObservationConfig


class LargeAnglePendulumGenerator:
    """
    大角度单摆轨迹生成器 / Large angle pendulum trajectory generator

    为评估非线性滤波器和学习算法而设计，能够生成具有丰富动力学行为的数据集。
    可配置外部驱动力以模拟非自治系统，并支持不同的观测策略以测试模型鲁棒性。
    """

    def __init__(
        self,
        params: Optional[PendulumParams] = None,
        dt: float = 0.01,
        total_time: float = 10.0
    ):
        self.params = params or PendulumParams()
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        self._dynamics_step = jax.jit(self._dynamics_step_impl)
        self._generate_trajectory = jax.jit(self._generate_trajectory_impl)

    @partial(jax.jit, static_argnums=(0,))
    def _dynamics_step_impl(
        self,
        state: chex.Array,
        t: float,
        key: chex.PRNGKey
    ) -> chex.Array:
        """
        单步动力学积分，包含可选的时变外部扭矩。
        Single step dynamics integration, including optional time-varying external torque.
        """
        theta, omega = state[0], state[1]
        p = self.params

        # 外部驱动力 (如果振幅>0)
        external_torque = p.forcing_amplitude * jnp.sin(p.forcing_frequency * t)

        gravity_torque = -(p.g / p.L) * jnp.sin(theta)
        damping_torque = -p.gamma * omega
        noise_torque = p.sigma * random.normal(key) / jnp.sqrt(self.dt)

        alpha = gravity_torque + damping_torque + external_torque + noise_torque
        new_omega = omega + alpha * self.dt
        new_theta = theta + new_omega * self.dt
        new_theta = jnp.mod(new_theta + jnp.pi, 2 * jnp.pi) - jnp.pi

        return jnp.array([new_theta, new_omega])

    @partial(jax.jit, static_argnums=(0,))
    def _generate_trajectory_impl(
        self,
        initial_state: chex.Array,
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array]:
        """生成完整轨迹 / Generate complete trajectory"""
        times = jnp.arange(0, self.total_time, self.dt)
        keys = random.split(key, self.n_steps)

        def step_fn(state, scan_input):
            t, key_i = scan_input
            next_state = self._dynamics_step_impl(state, t, key_i)
            return next_state, next_state

        final_state, trajectory = jax.lax.scan(step_fn, initial_state, (times, keys))
        full_trajectory = jnp.concatenate([initial_state[None, :], trajectory], axis=0)
        full_times = jnp.arange(0, self.total_time + self.dt, self.dt)
        return full_times[:full_trajectory.shape[0]], full_trajectory

    def generate_scenario(
        self,
        key: chex.PRNGKey,
        initial_dist_type: str = "uniform",
        obs_strategy: str = "dense",
    ) -> PendulumTrajectory:
        """
        生成一个随机场景 / Generate a random scenario.

        Args:
            key: 随机数密钥 / random key
            initial_dist_type: 初始分布类型 ("uniform", "bimodal")
            obs_strategy: 观测策略 ("dense", "structured_sparse")

        Returns:
            trajectory: 完整轨迹数据 / complete trajectory data
        """
        key_init, key_traj, key_obs = random.split(key, 3)

        # 1. 初始条件采样
        if initial_dist_type == "bimodal":
            # 生成双峰分布，模拟系统从两个明确的状态簇开始
            key_choice, key_noise = random.split(key_init)
            sign = random.choice(key_choice, jnp.array([-1.0, 1.0]))
            theta_0 = sign * (jnp.pi / 2) + random.normal(key_noise) * 0.2
            omega_0 = -sign * 0.5 + random.normal(key_noise) * 0.2
        else: # uniform
            max_angle_rad = jnp.deg2rad(170.0)
            theta_0 = random.uniform(key_init, shape=(), minval=-max_angle_rad, maxval=max_angle_rad)
            omega_0 = random.uniform(key_init, shape=(), minval=-1.0, maxval=1.0)
        
        initial_state = jnp.array([theta_0, omega_0])

        # 2. 生成轨迹
        times, states = self._generate_trajectory_impl(initial_state, key_traj)

        # 3. 创建观测配置
        if obs_strategy == "structured_sparse":
            obs_config = self._create_structured_sparse_obs_config()
        else: # dense
            obs_config = self._create_dense_obs_config()

        # 4. 生成观测
        observations, obs_times, true_obs = self._generate_observations(
            times, states, obs_config, key_obs
        )

        return PendulumTrajectory(
            times=times, states=states, observations=observations,
            obs_times=obs_times, true_obs_values=true_obs,
            params=self.params, obs_config=obs_config
        )

    def _create_dense_obs_config(self, obs_freq: float = 20.0) -> ObservationConfig:
        """创建密集观测配置 / Create dense observation configuration"""
        obs_dt = 1.0 / obs_freq
        obs_times = jnp.arange(0, self.total_time, obs_dt)
        return ObservationConfig(obs_times=obs_times, obs_noise_std=0.1, strategy="dense")

    def _create_structured_sparse_obs_config(self) -> ObservationConfig:
        """
        创建结构化稀疏观测配置，模拟真实世界中传感器可能在某些阶段失效或采样率降低。
        Create a structured sparse observation configuration, simulating real-world scenarios
        where sensors might fail or have a lower sampling rate during certain phases.
        """
        t_max = self.total_time
        # 在轨迹的初始和末尾阶段进行较密集的观测
        obs_times_start = jnp.arange(0, t_max * 0.2, 0.25)
        obs_times_end = jnp.arange(t_max * 0.8, t_max, 0.25)
        
        # 在轨迹中间的关键动态演化区域进行非常稀疏的观测
        obs_times_middle = jnp.array([t_max * 0.4, t_max * 0.6])
        
        obs_times = jnp.concatenate([obs_times_start, obs_times_middle, obs_times_end])
        obs_times = jnp.unique(obs_times)

        return ObservationConfig(obs_times=obs_times, obs_noise_std=0.1, strategy="structured_sparse")

    def _generate_observations(
        self,
        times: chex.Array, states: chex.Array,
        obs_config: ObservationConfig, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """生成观测数据 / Generate observation data"""
        true_obs_values = jnp.interp(obs_config.obs_times, times, states[:, 0])
        obs_noise = obs_config.obs_noise_std * random.normal(key, shape=true_obs_values.shape)
        observations = true_obs_values + obs_noise
        observations = vmap(lambda theta: jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi)(observations)
        return observations, obs_config.obs_times, true_obs_values

    def generate_and_save_dataset(
        self,
        n_trajectories: int,
        base_key: chex.PRNGKey,
        data_dir: str,
        scenario_type: str = "challenging"
    ):
        """
        生成并保存完整数据集 / Generate and save the full dataset
        
        Args:
            n_trajectories: 轨迹数量
            base_key: 随机密钥
            data_dir: 数据保存目录
            scenario_type: 场景类型 ("standard", "challenging")
        """
        keys = random.split(base_key, n_trajectories)
        trajectories = []

        print(f"🚀 生成 {n_trajectories} 个 '{scenario_type}' 场景轨迹...")
        print(f"   Generating {n_trajectories} trajectories for the '{scenario_type}' scenario...")

        for i, key in enumerate(keys):
            if scenario_type == "challenging":
                trajectory = self.generate_scenario(key, initial_dist_type="bimodal", obs_strategy="structured_sparse")
            else: # standard
                trajectory = self.generate_scenario(key, initial_dist_type="uniform", obs_strategy="dense")
            
            trajectories.append(trajectory)
            if (i + 1) % 20 == 0 or (i + 1) == n_trajectories:
                print(f"  ...完成 {i+1}/{n_trajectories}")

        self._save_trajectories(trajectories, data_dir)
        print(f"✅ 成功生成 {len(trajectories)} 个轨迹")
        return trajectories

    def _save_trajectories(self, trajectories: List[PendulumTrajectory], data_dir: str):
        """保存轨迹数据 / Save trajectory data"""
        data_path = pathlib.Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        for i, traj in enumerate(trajectories):
            file_path = data_path / f"pendulum_traj_{i:04d}.pkl"
            traj_dict = {
                'times': np.array(traj.times),
                'states': np.array(traj.states),
                'observations': np.array(traj.observations),
                'obs_times': np.array(traj.obs_times),
                'true_obs_values': np.array(traj.true_obs_values),
                'params': traj.params._asdict(),
                'obs_config': {
                    'obs_times': np.array(traj.obs_config.obs_times),
                    'obs_noise_std': traj.obs_config.obs_noise_std,
                    'strategy': traj.obs_config.strategy,
                }
            }
            with open(file_path, 'wb') as f:
                pickle.dump(traj_dict, f)
        print(f"💾 轨迹数据已保存到 {data_path}")


if __name__ == "__main__":
    print("🚀 开始生成大角度单摆数据集...")
    print("   将生成一个更具挑战性的数据集，用于测试模型在非自治和稀疏观测下的性能。")
    start_time = time.time()

    # 1. 配置生成器，包含外部驱动力以模拟非自治系统
    # 1. Configure generator with external forcing to simulate a non-autonomous system.
    generator = LargeAnglePendulumGenerator(
        params=PendulumParams(
            gamma=0.1,
            sigma=0.05,
            forcing_amplitude=1.5,
            forcing_frequency=2.0
        ),
        dt=0.01,
        total_time=10.0
    )

    # 2. 生成256条挑战性轨迹
    #    - 初始分布为双峰
    #    - 观测策略为结构化稀疏
    # 2. Generate 256 challenging trajectories
    #    - Initial distribution: bimodal
    #    - Observation strategy: structured_sparse
    key = random.PRNGKey(42)
    trajectories = generator.generate_and_save_dataset(
        n_trajectories=256,
        base_key=key,
        data_dir="data/driven_pendulum_v1",
        scenario_type="challenging"
    )

    # 3. 分析一条样本轨迹
    print(f"\n📊 分析样本轨迹 #0:")
    sample_trajectory = trajectories[0]
    print(f"   初始状态 (θ, ω): ({sample_trajectory.states[0, 0]:.2f}, {sample_trajectory.states[0, 1]:.2f})")
    print(f"   最终状态 (θ, ω): ({sample_trajectory.states[-1, 0]:.2f}, {sample_trajectory.states[-1, 1]:.2f})")
    print(f"   观测点数量: {len(sample_trajectory.obs_times)}")
    print(f"   观测策略: {sample_trajectory.obs_config.strategy}")

    end_time = time.time()
    print(f"\n✅ 数据集生成完成，耗时 {end_time - start_time:.2f} 秒。")