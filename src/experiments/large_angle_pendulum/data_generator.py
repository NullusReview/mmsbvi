"""
大角度单摆数据生成器 / Large Angle Pendulum Data Generator
=====================================================

生成接近倒立点的大角度单摆轨迹，创造天然多模态后验分布场景。
Generate large angle pendulum trajectories near unstable equilibrium for natural multi-modal posteriors.

核心特性 / Key Features:
- 完整非线性动力学: θ̈ = -(g/L)sin(θ) - γθ̇ + σξ(t) 
- 周期性状态空间: θ ∈ [-π, π]
- 关键倒立点场景: 初始条件接近 θ = π
- 稀疏观测策略: 跳过关键转折时刻
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import NamedTuple, Dict, List, Optional, Tuple
import chex
from functools import partial
import time

jax.config.update('jax_enable_x64', True)


class PendulumParams(NamedTuple):
    """大角度单摆物理参数 / Large angle pendulum physical parameters"""
    g: float = 9.81      # 重力加速度 / gravitational acceleration [m/s²]
    L: float = 1.0       # 摆长 / pendulum length [m]  
    gamma: float = 0.2   # 阻尼系数 / damping coefficient [1/s]
    sigma: float = 0.3   # 过程噪声强度 / process noise intensity [rad/s²]


class ObservationConfig(NamedTuple):
    """观测配置 / Observation configuration"""
    obs_times: chex.Array      # 观测时刻 / observation times
    obs_noise_std: float = 0.1 # 观测噪声标准差 / observation noise std [rad]
    sparse_strategy: str = "skip_unstable"  # 稀疏策略 / sparsity strategy
    

class PendulumTrajectory(NamedTuple):
    """单摆轨迹数据结构 / Pendulum trajectory data structure"""
    times: chex.Array           # 时间序列 / time series
    states: chex.Array          # 状态轨迹 (T, 2) [θ, ω] / state trajectory
    observations: chex.Array    # 观测序列 / observation sequence  
    obs_times: chex.Array       # 观测时刻 / observation times
    true_obs_values: chex.Array # 真实观测值（无噪声）/ true observation values
    params: PendulumParams      # 物理参数 / physical parameters
    obs_config: ObservationConfig # 观测配置 / observation configuration


class LargeAnglePendulumGenerator:
    """
    大角度单摆轨迹生成器 / Large angle pendulum trajectory generator
    
    设计用于创造多模态后验分布的测试场景：
    Designed to create test scenarios with multi-modal posterior distributions:
    
    1. 倒立点不稳定性：θ ≈ π 附近微小扰动导致不同运动模式
    2. 周期性边界：θ ∈ [-π, π] 的拓扑复杂性
    3. 强非线性：sin(θ) 项在大角度时显著
    4. 稀疏观测：关键时刻缺失增加后验不确定性
    """
    
    def __init__(
        self,
        params: Optional[PendulumParams] = None,
        dt: float = 0.02,  # 积分时间步长 / integration time step
        total_time: float = 3.0  # 总仿真时间 / total simulation time
    ):
        """
        初始化生成器 / Initialize generator
        
        Args:
            params: 物理参数 / physical parameters
            dt: 数值积分步长 / numerical integration step
            total_time: 仿真总时长 / total simulation time
        """
        self.params = params or PendulumParams()
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        
        # 编译核心函数以提高性能 / compile core functions for performance
        self._dynamics_step = jax.jit(self._dynamics_step_impl)
        self._generate_trajectory = jax.jit(self._generate_trajectory_impl)
        
    @partial(jax.jit, static_argnums=(0,))
    def _dynamics_step_impl(
        self, 
        state: chex.Array, 
        key: chex.PRNGKey
    ) -> chex.Array:
        """
        单步动力学积分 / Single step dynamics integration
        
        大角度单摆方程 / Large angle pendulum equation:
        θ̈ = -(g/L)sin(θ) - γθ̇ + σξ(t)
        
        Args:
            state: 当前状态 [θ, ω] / current state [θ, ω]
            key: 随机数密钥 / random key
            
        Returns:
            next_state: 下一时刻状态 / next state
        """
        theta, omega = state[0], state[1]
        
        # 非线性重力项（关键！）/ Nonlinear gravity term (crucial!)
        gravity_torque = -(self.params.g / self.params.L) * jnp.sin(theta)
        
        # 线性阻尼 / Linear damping
        damping_torque = -self.params.gamma * omega
        
        # 随机强迫 / Stochastic forcing
        noise = self.params.sigma * random.normal(key) * jnp.sqrt(self.dt)
        
        # 角加速度 / Angular acceleration
        alpha = gravity_torque + damping_torque + noise
        
        # 欧拉积分 / Euler integration
        new_omega = omega + alpha * self.dt
        new_theta = theta + new_omega * self.dt
        
        # 周期性边界条件：θ ∈ [-π, π] / Periodic boundary: θ ∈ [-π, π]
        new_theta = self._wrap_angle(new_theta)
        
        return jnp.array([new_theta, new_omega])
    
    @partial(jax.jit, static_argnums=(0,))
    def _wrap_angle(self, theta: float) -> float:
        """
        角度包装到 [-π, π] / Wrap angle to [-π, π]
        
        处理周期性边界条件，确保状态空间的拓扑正确性。
        Handle periodic boundary conditions for correct topological behavior.
        """
        return jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    @partial(jax.jit, static_argnums=(0,))
    def _generate_trajectory_impl(
        self, 
        initial_state: chex.Array, 
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array]:
        """
        生成完整轨迹 / Generate complete trajectory
        
        Args:
            initial_state: 初始状态 [θ₀, ω₀] / initial state
            key: 随机数密钥 / random key
            
        Returns:
            times: 时间序列 / time series
            states: 状态轨迹 (T, 2) / state trajectory
        """
        times = jnp.arange(0, self.total_time, self.dt)
        keys = random.split(key, self.n_steps - 1)
        
        def step_fn(state, key_i):
            next_state = self._dynamics_step_impl(state, key_i)
            return next_state, next_state
        
        # 扫描积分 / Scan integration
        final_state, trajectory = jax.lax.scan(
            step_fn, initial_state, keys
        )
        
        # 添加初始状态 / Add initial state
        full_trajectory = jnp.concatenate([
            initial_state[None, :], trajectory
        ], axis=0)
        
        return times, full_trajectory
    
    def generate_unstable_scenario(
        self, 
        key: chex.PRNGKey,
        theta_perturbation: float = 0.05,
        omega_perturbation: float = 0.02
    ) -> PendulumTrajectory:
        """
        生成倒立点附近的不稳定场景 / Generate unstable scenario near inverted equilibrium
        
        这是创造多模态后验的关键：从倒立点开始的微小扰动
        会导致截然不同的运动模式（向左倒、向右倒、翻越）。
        
        Key for multi-modal posteriors: small perturbations from inverted state
        lead to drastically different motion patterns (fall left, fall right, flip over).
        
        Args:
            key: 随机数密钥 / random key
            theta_perturbation: 角度扰动幅度 / angle perturbation magnitude
            omega_perturbation: 角速度扰动幅度 / angular velocity perturbation magnitude
            
        Returns:
            trajectory: 完整轨迹数据 / complete trajectory data
        """
        key_init, key_traj, key_obs = random.split(key, 3)
        
        # 关键：接近倒立点的随机初始条件 / Critical: random initial conditions near inverted point
        theta_0 = jnp.pi + theta_perturbation * random.normal(key_init)
        omega_0 = omega_perturbation * random.normal(key_init)
        initial_state = jnp.array([theta_0, omega_0])
        
        # 生成轨迹 / Generate trajectory
        times, states = self._generate_trajectory_impl(initial_state, key_traj)
        
        # 设计稀疏观测策略 / Design sparse observation strategy
        obs_config = self._create_sparse_observation_config()
        
        # 生成观测 / Generate observations
        observations, obs_times, true_obs = self._generate_observations(
            times, states, obs_config, key_obs
        )
        
        return PendulumTrajectory(
            times=times,
            states=states,
            observations=observations,
            obs_times=obs_times,
            true_obs_values=true_obs,
            params=self.params,
            obs_config=obs_config
        )
    
    def _create_sparse_observation_config(self) -> ObservationConfig:
        """
        创建稀疏观测配置 / Create sparse observation configuration
        
        关键策略：跳过倒立点附近的关键时刻，增加后验不确定性。
        Key strategy: skip critical moments near inverted point to increase posterior uncertainty.
        """
        # 基础观测时刻 / Base observation times
        dense_times = jnp.arange(0, self.total_time, 0.3)
        
        # 稀疏策略：跳过 [0.8, 1.4] 区间（预期倒立翻转时刻）
        # Sparsity strategy: skip [0.8, 1.4] interval (expected inversion time)
        sparse_mask = jnp.logical_or(
            dense_times < 0.8,
            dense_times > 1.4
        )
        sparse_times = dense_times[sparse_mask]
        
        return ObservationConfig(
            obs_times=sparse_times,
            obs_noise_std=0.1,
            sparse_strategy="skip_unstable"
        )
    
    def _generate_observations(
        self,
        times: chex.Array,
        states: chex.Array, 
        obs_config: ObservationConfig,
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        生成观测数据 / Generate observation data
        
        Args:
            times: 仿真时间序列 / simulation time series
            states: 状态轨迹 / state trajectory
            obs_config: 观测配置 / observation configuration
            key: 随机数密钥 / random key
            
        Returns:
            observations: 含噪声观测 / noisy observations
            obs_times: 观测时刻 / observation times  
            true_obs: 真实观测值 / true observation values
        """
        # 插值获取观测时刻的真实状态 / Interpolate true states at observation times
        true_obs_values = jnp.interp(obs_config.obs_times, times, states[:, 0])
        
        # 添加观测噪声 / Add observation noise
        obs_noise = obs_config.obs_noise_std * random.normal(
            key, shape=true_obs_values.shape
        )
        observations = true_obs_values + obs_noise
        
        # 处理观测的周期性 / Handle observation periodicity
        observations = jax.vmap(self._wrap_angle)(observations)
        
        return observations, obs_config.obs_times, true_obs_values
    
    def generate_multiple_scenarios(
        self, 
        n_trajectories: int,
        base_key: chex.PRNGKey,
        save_data: bool = True,
        data_dir: Optional[str] = None
    ) -> List[PendulumTrajectory]:
        """
        生成多个测试场景 / Generate multiple test scenarios
        
        Args:
            n_trajectories: 轨迹数量 / number of trajectories
            base_key: 基础随机密钥 / base random key
            save_data: 是否保存数据 / whether to save data
            data_dir: 数据保存目录 / data save directory
            
        Returns:
            trajectories: 轨迹列表 / list of trajectories
        """
        keys = random.split(base_key, n_trajectories)
        trajectories = []
        
        print(f"生成 {n_trajectories} 个大角度单摆测试轨迹...")
        print(f"Generating {n_trajectories} large angle pendulum test trajectories...")
        
        for i, key in enumerate(keys):
            # 轻微变化扰动参数以增加多样性 / Slightly vary perturbation for diversity
            theta_pert = 0.05 + 0.02 * (i / n_trajectories - 0.5)
            omega_pert = 0.02 + 0.01 * (i / n_trajectories - 0.5)
            
            trajectory = self.generate_unstable_scenario(
                key, theta_pert, omega_pert
            )
            trajectories.append(trajectory)
            
            if i % 5 == 0:
                print(f"  完成 {i+1}/{n_trajectories}")
        
        if save_data and data_dir:
            self._save_trajectories(trajectories, data_dir)
        
        print(f"✅ 成功生成 {len(trajectories)} 个多模态测试轨迹")
        return trajectories
    
    def _save_trajectories(self, trajectories: List[PendulumTrajectory], data_dir: str):
        """保存轨迹数据 / Save trajectory data"""
        import pickle
        import pathlib
        
        data_path = pathlib.Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        for i, traj in enumerate(trajectories):
            file_path = data_path / f"pendulum_traj_{i:03d}.pkl"
            
            # 转换为numpy以便保存 / Convert to numpy for saving
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
                    'sparse_strategy': traj.obs_config.sparse_strategy
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(traj_dict, f)
        
        print(f"✅ 轨迹数据已保存到 {data_path}")


def analyze_trajectory_multimodality(trajectory: PendulumTrajectory) -> Dict[str, float]:
    """
    分析轨迹的多模态特征 / Analyze trajectory multi-modality characteristics
    
    Args:
        trajectory: 单摆轨迹 / pendulum trajectory
        
    Returns:
        analysis: 多模态分析结果 / multi-modality analysis results
    """
    states = trajectory.states
    theta_traj = states[:, 0]
    omega_traj = states[:, 1]
    
    # 检测倒立点穿越 / Detect inverted point crossings
    inverted_crossings = jnp.sum(jnp.abs(theta_traj) > 2.5)
    
    # 分析角速度变化模式 / Analyze angular velocity patterns  
    omega_reversals = jnp.sum(jnp.diff(jnp.sign(omega_traj)) != 0)
    
    # 计算相空间覆盖 / Compute phase space coverage
    theta_range = jnp.max(theta_traj) - jnp.min(theta_traj)
    omega_range = jnp.max(omega_traj) - jnp.min(omega_traj)
    
    # 估计轨迹复杂度 / Estimate trajectory complexity
    complexity = float(inverted_crossings + 0.1 * omega_reversals)
    
    return {
        'inverted_crossings': float(inverted_crossings),
        'omega_reversals': float(omega_reversals), 
        'theta_range': float(theta_range),
        'omega_range': float(omega_range),
        'complexity_score': complexity,
        'is_multimodal_candidate': complexity > 1.0
    }


if __name__ == "__main__":
    # 测试大角度单摆生成器 / Test large angle pendulum generator
    print("🧪 测试大角度单摆数据生成器")
    print("🧪 Testing Large Angle Pendulum Data Generator")
    
    generator = LargeAnglePendulumGenerator(
        params=PendulumParams(gamma=0.15, sigma=0.25),
        dt=0.01,
        total_time=4.0
    )
    
    # 生成单个测试轨迹 / Generate single test trajectory
    key = random.PRNGKey(42)
    trajectory = generator.generate_unstable_scenario(key)
    
    # 分析多模态特征 / Analyze multi-modal characteristics
    analysis = analyze_trajectory_multimodality(trajectory)
    
    print(f"\n📊 轨迹分析结果:")
    print(f"   倒立点穿越次数: {analysis['inverted_crossings']}")
    print(f"   角速度反转次数: {analysis['omega_reversals']}")
    print(f"   角度范围: {analysis['theta_range']:.2f} rad")
    print(f"   角速度范围: {analysis['omega_range']:.2f} rad/s")
    print(f"   复杂度评分: {analysis['complexity_score']:.2f}")
    print(f"   多模态候选: {analysis['is_multimodal_candidate']}")
    
    print(f"\n✅ 大角度单摆数据生成器测试完成")