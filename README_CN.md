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
├── algorithms/              # 核心算法实现
│   ├── ipfp_1d.py          # 1D多边际IPFP主算法
├── solvers/                 # 数值求解器
│   ├── pde_solver_1d.py    # Onsager-Fokker PDE求解器
│   ├── gaussian_kernel_1d.py # OU转移核计算
├── utils/                   # 工具函数
└── visualization/           # 可视化模块

experiments/                 # 用于生成论文图表的脚本
tests/                       # 用于验证的单元和集成测试
automation/                  # 用于运行验证工作流的shell脚本
```

---

## 许可证

本项目基于MIT许可证 - 详见 [LICENSE](LICENSE) 文件。