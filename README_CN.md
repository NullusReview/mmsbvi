<div align="center">
<h1>MMSBVI: Multi-Marginal Schrödinger Bridge Variational Inference</h1>
<h3>多边际薛定谔桥变分推断</h3>
</div>

<p align="center">
  <a href="#核心概念">核心概念</a> •
  <a href="#架构特色">架构特色</a> •
  <a href="#安装">安装</a> •
  <a href="#复现验证">复现验证</a> •
  <a href="#代码结构">代码结构</a>
</p>

---

本仓库是 ICLR 2026 论文 ***Geometric Variational Inference via Multi-Marginal Schrödinger Bridge*** 的 JAX 实现。项目旨在提供一个框架，用于探索和验证“路径空间变分推断” (Variational Inference in Path Space) 与“多边际最优传输” (Multi-Marginal Optimal Transport) 之间的理论对偶性。

## 核心概念

“多边际薛定谔桥” (Multi-Marginal Schrödinger Bridge, MMSB) 问题本质为寻找一个随机过程，其在多个指定时间点的边缘分布 (marginal distribution) 与给定的目标分布相匹配，同时使该过程的路径测度与一个先验参考过程（通常是布朗运动或Ornstein-Uhlenbeck过程）的KL散度最小。

形式上，给定在时间点 $t_0, t_1, \dots, t_K$ 的一系列目标边际分布 $\rho_0, \rho_1, \dots, \rho_K$，我们寻找一个路径测度 $\mathbb{P}$，以解决以下优化问题：

$$
\mathbb{P}^* = \arg\min_{\mathbb{P}} \text{KL}(\mathbb{P} || \mathbb{Q}) \quad \text{s.t.} \quad X_{t_k} \sim \rho_k, \forall k \in \{0, \dots, K\}
$$

其中 $\mathbb{Q}$ 是一个先验参考过程（如OU过程）的路径测度。此问题等价于一个随机控制问题，其解由一组耦合的非线性偏微分方程（薛定谔系统）描述。

本项目探索了两种求解该问题的范式：
1.  **经典数值方法**：通过迭代比例拟合过程 (IPFP) 在离散网格上求解对偶问题。
2.  **现代机器学习方法**：将问题转化为随机控制问题，并使用神经网络参数化控制策略进行端到端优化。

## 架构特色

本项目的架构设计融合了学术研究的原则与现代机器学习工程实践。

1.  **双核求解器架构 (Dual-Core Solver Architecture)**
    *   **经典网格求解器 (`ipfp_1d.py`)**: 基于 Sinkhorn 算法的迭代比例拟合过程 (IPFP)，为低维问题提供高精度解，用于理论验证。
    *   **神经控制求解器 (`control_grad.py`)**: 将 MMSB 问题重构为随机控制问题，使用神经网络 (`FöllmerDriftNet`) 参数化随机微分方程 (SDE) 的漂移项，并通过变分推断进行端到端优化。此求解器旨在扩展到高维问题。

2.  **高度模块化与可扩展性 (Highly Modular & Extensible)**
    *   **类型系统 (`types.py`)**: 使用 `chex.dataclass` 和 `jaxtyping` 定义类型系统，将核心概念如问题定义 (`MMSBProblem`)、算法配置 (`IPFPConfig`, `ControlGradConfig`) 和解 (`MMSBSolution`) 等进行解耦。
    *   **组件注册表 (`registry.py`)**: 采用工厂模式，允许通过字符串名称动态注册和加载不同的求解器、网络和积分器，并通过配置文件（如 Hydra）进行管理。

3.  **高性能计算 (High-Performance Computing)**
    *   整个代码库基于 JAX 构建，使用其 `jit`, `vmap`, `pmap` 等变换进行并行计算和GPU加速。
    *   在神经求解器中，应用了梯度检查点 (gradient checkpointing)、混合精度训练 (mixed precision) 等，以提升计算和内存效率，同时保持数值精度。

## 安装

### 环境配置
我们推荐使用 `pip` 管理依赖。要设置环境，请运行：
```bash
# 安装依赖
pip install -r requirements-cpu.txt requirements-gpu.txt
```

### 核心依赖
*   **JAX Ecosystem**: `jax`, `jaxlib`, `flax`, `optax`, `chex`
*   **Optimal Transport**: `ott-jax`
*   **Scientific Computing**: `numpy`, `scipy`
*   **Configuration**: `hydra-core`

### 运行核心测试
为确保环境配置正确，请运行测试套件：
```bash
pytest tests/
```
所有测试用例均应通过。

## 复现验证

论文中的关键理论验证和图表，可通过 `automation/` 目录下的脚本复现。

### 完整的验证套件
要按顺序运行所有验证工作流，请执行主脚本。这将复现图表和数值结果。
```bash
chmod +x automation/run_complete_validation_suite.sh
./automation/run_complete_validation_suite.sh
```

### 单个验证工作流
您也可以独立运行每个验证工作流：
*   **RTS等价性验证**: 验证MMSB解在特定条件下与Rauch-Tung-Striebel (RTS)平滑器的一致性。
    ```bash
    ```bash
    ./automation/run_rts_equivalence_workflow.sh
    ```
*   **几何极限验证**: 探索当噪声趋于零时，薛定谔桥如何收敛到确定性的最优传输路径。
    ```bash
    ./automation/run_geometric_limits_workflow.sh
    ```
*   **参数敏感性分析**: 分析模型性能对关键参数（如正则化强度、时间步长）的敏感度。
    ```bash
    ./automation/run_parameter_sensitivity_workflow.sh
    ```
生成的结果将按实验类型保存在 `results/` 目录中。

## 代码结构

项目结构旨在将核心算法与实验验证脚本清晰分离。

```
src/mmsbvi/
├── core/                    # 核心类型定义、配置和组件注册表
├── algorithms/              # 核心算法实现 (IPFP, Neural Control)
├── solvers/                 # 数值求解器 (PDE, Gaussian Kernel)
├── integrators/             # SDE数值积分格式
├── nets/                    # 神经网络架构 (Flax)
├── utils/                   # 工具函数 (日志、配置)
└── configs/                 # Hydra配置文件

theoretical_verification/    # 1D理论验证实验脚本
tests/                       # 单元和集成测试
automation/                  # 一键复现工作流的Shell脚本
```

---

<div align="center">
本仓库基于MIT许可证。
</div>