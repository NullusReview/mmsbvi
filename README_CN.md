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

本仓库是论文 ***Geometric Variational Inference via Multi-Marginal Schrödinger Bridge*** 的官方 JAX 实现。本项目旨在建立并数值验证一个核心理论：路径空间中的变分推断 (Variational Inference) 与一个多边际薛定谔桥 (Multi-Marginal Schrödinger Bridge, MMSB) 问题存在基础等价性。这一发现将经典的贝叶斯平滑问题置于最优传输与信息几何的统一视角下进行审视。

## 核心思想

本工作的核心论点是：**先验即几何 (The Prior is the Geometry)**。我们证明了连续时间系统中的贝叶斯平滑问题，等价于在某个黎曼流形上寻找一条测地线，而该流形的度量完全由先验参考过程决定。

这一思想由我们的核心成果 **定理1 (VI-MMSB 等价性)** 精确阐述。该定理证明了，最小化变分自由能的目标，与求解一个多边际薛定谔桥问题是完全等价的。该问题的目标是寻找一个路径测度 $Q$，使其与一个参考过程 $P_{\text{ref}}$ (例如 Ornstein-Uhlenbeck 过程) 的 KL 散度 (Kullback-Leibler divergence) 最小，同时满足其在一系列观测时间点的边际分布恰好是给定的目标边际 $\{\rho_{t_k}^{\text{obs}}\}$。

形式化描述如下：
$$
Q^* = \operatorname*{arg\,min}_{\substack{Q: Q_{t_0}=\rho_0 \\ Q_{t_k}=\rho_{t_k}^{\text{obs}}, k=1,\dots,K}} \mathrm{KL}(Q \,\|\, P_{\text{ref}})
$$
此问题的解，即后验路径测度 $Q^*$，其演化轨迹是在由 **Onsager-Fokker 度量** 所赋予几何结构的概率分布空间中的一条测地线。该理论框架统一了经典与现代观点，不仅能在线性高斯设定下精确恢复经典的 Rauch-Tung-Striebel (RTS) 平滑器，还能在参数极限下自然地内插 Wasserstein 几何与 Fisher-Rao 几何。

本仓库提供了一个高精度的**迭代比例拟合算法 (IPFP)** 实现，作为严格数值验证上述理论发现的核心工具。同时，我们也勾勒了一个基于神经网络的随机控制方法，作为扩展至高维问题的方向。

## 架构亮点

本项目的架构设计融合了学术研究的严谨性与现代机器学习的工程实践。

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
我们推荐使用 `pip` 管理依赖。设置环境，请运行：
```bash
# 安装依赖
pip install -r requirements-cpu.txt requirements-gpu.txt
```

### 核心依赖
*   **JAX 生态**: `jax`, `jaxlib`, `flax`, `optax`, `chex`
*   **最优运输**: `ott-jax`
*   **科学计算**: `numpy`, `scipy`
*   **配置**: `hydra-core`

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