<div align="center">
<h1>Multi-Marginal Schrödinger Bridge Variational Inference</h1>
<h3>MMSBVI</h3>
</div>

<p align="center">
  <a href="README_CN.md">中文</a> 


<p align="center">
  <a href="#core-concepts">Core Concepts</a> •
  <a href="#architectural-highlights">Architectural Highlights</a> •
  <a href="#installation">Installation</a> •
  <a href="#reproducing-validation">Reproducing Validation</a> •
  <a href="#code-structure">Code Structure</a>
</p>

---

This repository contains the JAX implementation for the paper, ***Geometric Variational Inference via Multi-Marginal Schrödinger Bridge***. This project establishes and numerically validates a fundamental equivalence between Variational Inference in path space and a Multi-Marginal Schrödinger Bridge (MMSB) problem, reframing Bayesian smoothing through the lens of optimal transport and information geometry.

## Core Concepts

The central thesis of this work is that **the prior is the geometry**. We demonstrate that Bayesian smoothing for continuous-time systems can be viewed as finding a geodesic on a Riemannian manifold whose metric is determined by the reference process.

This is formalized by **Theorem 1 (VI-MMSB Equivalence)**, which proves that minimizing the variational free energy is equivalent to solving a multi-marginal Schrödinger Bridge problem. The objective is to find a path measure $Q$ that minimizes the Kullback-Leibler (KL) divergence to a reference process $P_{\text{ref}}$ (e.g., an Ornstein-Uhlenbeck process), subject to matching a set of target marginals $\{\rho_{t_k}^{\text{obs}}\}$ derived from observations.

Formally, this constrained optimization is expressed as:

$$
Q^{*} = 
\operatorname*{arg\,min}_{Q} \left\{
  \mathrm{KL}\!\left( Q \,\|\, P_{\text{ref}} \right)
  \;\middle|\;
  Q_{t_k} = \rho_{t_k}^{\text{obs}},\; k = 0,\dots,K
\right\}
$$

The solution to this problem, the posterior path measure $Q^*$, traces a geodesic on the space of probability distributions endowed with the **Onsager-Fokker metric**. This framework unifies classical and modern perspectives, recovering the Rauch-Tung-Striebel (RTS) smoother in the linear-Gaussian case and interpolating between Wasserstein and Fisher-Rao geometries.

This repository provides a high-precision implementation of the **Iterative Proportional Fitting Procedure (IPFP)** to solve this problem, serving as a tool for the rigorous numerical validation of these theoretical findings. A neural-network-based control approach is outlined as a direction to tackle higher-dimensional problems.

## Architectural Highlights

The architecture of this project integrates principles of academic research with modern machine learning engineering.

1.  **Dual-Core Solver Architecture**
    *   **Classical Grid Solver (`ipfp_1d.py`)**: An Iterative Proportional Fitting Procedure (IPFP) based on the Sinkhorn algorithm. It provides a high-precision solution for low-dimensional problems, which is used for theoretical validation.
    *   **Neural Control Solver (`control_grad.py`)**: Reformulates the MMSB problem as a stochastic control task. It uses a neural network (`FöllmerDriftNet`) to parameterize the drift term of a Stochastic Differential Equation (SDE) and performs end-to-end optimization via variational inference. This solver is designed for extensibility to high-dimensional problems.

2.  **Highly Modular & Extensible**
    *   **Type System (`types.py`)**: Utilizes `chex.dataclass` and `jaxtyping` to define the type system, decoupling core concepts like the problem definition (`MMSBProblem`), algorithm configurations (`IPFPConfig`, `ControlGradConfig`), and the solution (`MMSBSolution`).
    *   **Component Registry (`registry.py`)**: Implements a factory pattern that allows for dynamic registration and loading of different solvers, networks, and integrators via string names, managed through configuration files (e.g., Hydra).

3.  **High-Performance Computing**
    *   The entire codebase is built on JAX, using its `jit`, `vmap`, and `pmap` transformations for parallel computing and GPU acceleration.
    *   In the neural solver, techniques such as gradient checkpointing and mixed-precision training are applied to improve computational and memory efficiency while maintaining numerical accuracy.

## Installation

### Environment Setup
We recommend using `pip` to manage dependencies. To set up the environment, please run:
```bash
# Install dependencies
pip install -r requirements-cpu.txt requirements-gpu.txt
```

### Core Dependencies
*   **JAX Ecosystem**: `jax`, `jaxlib`, `flax`, `optax`, `chex`
*   **Optimal Transport**: `ott-jax`
*   **Scientific Computing**: `numpy`, `scipy`
*   **Configuration**: `hydra-core`

### Running Core Tests
To verify that the environment is set up correctly, please run the test suite:
```bash
pytest tests/
```
All test cases should pass.

## Reproducing Validation

The key theoretical validations and figures from the paper can be reproduced with scripts located in the `automation/` directory.

### Complete Validation Suite
To run all validation workflows in sequence, execute the main script. This will reproduce the figures and numerical results.
```bash
chmod +x automation/run_complete_validation_suite.sh
./automation/run_complete_validation_suite.sh
```

### Individual Validation Workflows
You can also run each validation workflow independently:
*   **RTS Equivalence Validation**: Verifies the consistency of the MMSB solution with the Rauch-Tung-Striebel (RTS) smoother under specific conditions.
    ```bash
    ./automation/run_rts_equivalence_workflow.sh
    ```
*   **Geometric Limits Validation**: Explores how the Schrödinger bridge converges to a deterministic optimal transport path as the noise term approaches zero.
    ```bash
    ./automation/run_geometric_limits_workflow.sh
    ```
*   **Parameter Sensitivity Analysis**: Analyzes the sensitivity of the model's performance to key parameters, such as regularization strength and time step size.
    ```bash
    ./automation/run_parameter_sensitivity_workflow.sh
    ```
The generated results will be saved in the `results/` directory, organized by experiment type.

## Code Structure

The project is structured to separate the core algorithms from the experimental validation scripts.

```
src/mmsbvi/
├── core/                    # Core type definitions, configs, and component registry
├── algorithms/              # Core algorithm implementations (IPFP, Neural Control)
├── solvers/                 # Numerical solvers (PDE, Gaussian Kernel)
├── integrators/             # SDE numerical integration schemes
├── nets/                    # Neural network architectures (Flax)
├── utils/                   # Utility functions (logging, config)
└── configs/                 # Hydra configuration files

theoretical_verification/    # Scripts for 1D theoretical validation experiments
tests/                       # Unit and integration tests
automation/                  # Shell scripts for validation workflows
```

---

<div align="center">
This repository is licensed under the MIT License.
</div>