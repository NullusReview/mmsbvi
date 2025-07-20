<div align="center">
<h1>MMSBVI</h1>
<h2>Multi-Marginal Schrödinger Bridge Variational Inference</h2>
</div>

本仓库是 ICLR 2026 论文 ***Geometric Variational Inference via Multi-Marginal Schrödinger Bridge*** 的 JAX 实现。本项目提供了一个严格的数学验证框架，用以展示路径空间变分推断与多边际最优传输之间的理论等价性。

本仓库的核心是一个为精确数学验证而设计的1D原型。

---

## 目录

1. [安装](#安装)
2. [复现实验](#复现实验)
3. [代码结构](#代码结构)
4. [引用](#引用)

---

## 安装

### 环境配置

我们使用 `pip` 管理依赖项。要配置环境，请运行：

```bash
# 安装依赖项 (CPU版本足以运行所有实验)
pip install -r requirements-cpu.txt
```

### 核心依赖

该框架同时支持1D理论验证和可扩展的高维神经求解器：

**核心JAX生态系统：**
- **JAX 0.6.2**: 高性能计算框架
- **JAXLib 0.6.2**: JAX后端实现
- **NumPy 1.26.3**: 数值计算基础

**神经架构组件：**
- **Flax 0.8.5**: 用于Föllmer漂移参数化的神经网络架构
- **Optax 0.1.9**: 基于梯度的优化框架
- **Chex 0.1.85**: 可靠的JAX测试和调试工具
- **Einops 0.7.0**: 具有可读记号的张量操作

**科学计算：**
- **SciPy 1.12.0**: 科学计算算法
- **OTT-JAX 0.4.5**: 最优传输计算
- **BlackJAX 1.2.5**: MCMC采样（用于基线比较）

### 核心测试
为确认环境已正确配置，请运行核心测试套件：

```bash
pytest tests/
```
所有17个测试都应通过。

---

## 复现实验

论文中的主要理论论点和图表，均可通过 `automation/` 目录下的shell脚本进行复现。

### 完整的验证套件

要按顺序运行所有验证工作流，请执行主脚本。这将复现所有的关键图表和数值结果。

```bash
chmod +x automation/run_complete_validation_suite.sh
./automation/run_complete_validation_suite.sh
```

### 单个验证工作流

您也可以独立运行每个验证工作流：

- **RTS等价性验证**:
  ```bash
  ./automation/run_rts_equivalence_workflow.sh
  ```
- **几何极限验证**:
  ```bash
  ./automation/run_geometric_limits_workflow.sh
  ```
- **参数敏感性分析**:
  ```bash
  ./automation/run_parameter_sensitivity_workflow.sh
  ```

生成的图表和数据将按实验类型保存在 `results/` 目录中。

---

## 代码结构

项目结构旨在将核心算法与实验验证脚本分离。

```
src/mmsbvi/
├── core/                    # 核心类型定义和配置
│   ├── types.py            # 主要数据结构 (Grid1D, MMSBProblem等)
│   └── registry.py         # 可插拔架构的组件注册表
├── algorithms/              # 核心算法实现
│   ├── ipfp_1d.py          # 1D多边际IPFP主算法
│   ├── control_grad.py     # 原初控制梯度流求解器
│   └── score_sb.py         # 基于分数的薛定谔桥（占位符）
├── solvers/                 # 数值求解器
│   ├── pde_solver_1d.py    # Onsager-Fokker PDE求解器
│   └── gaussian_kernel_1d.py # OU转移核计算
├── integrators/             # SDE数值积分方法
│   └── integrators.py      # Euler、Heun、Milstein、AMED-Euler格式
├── nets/                    # 神经网络架构
│   └── flax_drift.py       # 基于Flax的Föllmer漂移网络
├── utils/                   # 工具函数
│   ├── logger.py           # 日志系统
│   └── config.py           # YAML配置管理
├── configs/                 # 配置文件
│   ├── baseline.yaml       # 默认配置
│   ├── lorenz_cfg.yaml     # 洛伦兹系统配置
│   └── physionet_cfg.yaml  # PhysioNet数据集配置
├── cli/                     # 命令行界面
│   └── train.py            # 统一训练入口点
└── visualization/           # 可视化模块

theoretical_verification/    # 1D理论验证实验
tests/                       # 用于验证的单元和集成测试
automation/                  # 用于运行验证工作流的shell脚本
```

---

## 许可证

本项目基于MIT许可证 - 详见 [LICENSE](LICENSE) 文件。