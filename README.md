<div align="center">
<h1>MMSBVI</h1>
<h2>Multi-Marginal Schrödinger Bridge Variational Inference</h2>
</div>

<p align="center"><a href="README_CN.md">中文</a></p>

This repository contains the official JAX implementation for the ICLR 2026 paper: *Geometric Variational Inference via Multi-Marginal Schrödinger Bridge*. This work provides a rigorous mathematical validation framework demonstrating the theoretical equivalence between path-space variational inference and multi-marginal optimal transport.

The core of this repository is a 1D prototype designed for the precise mathematical validation of our theoretical claims.

---

## Index

1. [Installation](#installation)
2. [Reproducing Experiments](#reproducing-experiments)
3. [Code Structure](#code-structure)
4. [Reference](#reference)

---

## Installation

### Environment Setup

We use `pip` for managing dependencies. To set up the environment, run:

```bash
# Install dependencies (CPU version is sufficient for all experiments)
pip install -r requirements-cpu.txt
```

### Core Dependencies

The framework includes support for both 1D theoretical validation and extensible high-dimensional neural solvers:

**Core JAX Ecosystem:**
- **JAX 0.6.2**: High-performance computing framework
- **JAXLib 0.6.2**: JAX backend implementation
- **NumPy 1.26.3**: Numerical computing foundation

**Neural Architecture Components:**
- **Flax 0.8.5**: Neural network architecture for Föllmer drift parametrization  
- **Optax 0.1.9**: Gradient-based optimization framework
- **Chex 0.1.85**: Reliable JAX testing and debugging utilities
- **Einops 0.7.0**: Tensor operations with readable notation

**Scientific Computing:**
- **SciPy 1.12.0**: Scientific computing algorithms
- **OTT-JAX 0.4.5**: Optimal transport computations
- **BlackJAX 1.2.5**: MCMC sampling (for baseline comparisons)

### Core Tests
To confirm that the environment is set up correctly, run the core test suite:

```bash
pytest tests/
```
All 17 tests should pass.

---

## Reproducing Experiments

The main theoretical claims and figures in the paper can be reproduced using the shell scripts located in the `automation/` directory.

### Complete Validation Suite

To run all validation workflows sequentially, execute the master script. This will reproduce all key figures and numerical results.

```bash
chmod +x automation/run_complete_validation_suite.sh
./automation/run_complete_validation_suite.sh
```

### Individual Workflows

Alternatively, you can run each validation workflow independently:

- **RTS Equivalence Validation**:
  ```bash
  ./automation/run_rts_equivalence_workflow.sh
  ```
- **Geometric Limits Validation**:
  ```bash
  ./automation/run_geometric_limits_workflow.sh
  ```
- **Parameter Sensitivity Analysis**:
  ```bash
  ./automation/run_parameter_sensitivity_workflow.sh
  ```

The resulting figures and data will be saved in the `results/` directory, organized by experiment type.

---

## Code Structure

The project is structured to separate the core algorithms from the experimental validation scripts.

```
src/mmsbvi/
├── core/                    # Core type definitions and configurations
│   ├── types.py            # Main data structures (Grid1D, MMSBProblem, etc.)
│   └── registry.py         # Component registry for pluggable architecture
├── algorithms/              # Core algorithm implementations
│   ├── ipfp_1d.py          # 1D Multi-Marginal IPFP main algorithm
│   ├── control_grad.py     # Primal-Control Gradient Flow solver 
│   └── score_sb.py         # Score-based Schrödinger Bridge (placeholder)
├── solvers/                 # Numerical solvers
│   ├── pde_solver_1d.py    # Onsager-Fokker PDE solver
│   └── gaussian_kernel_1d.py # OU transition kernel computation
├── integrators/             # SDE numerical integration methods 
│   └── integrators.py      # Euler, Heun, Milstein, AMED-Euler schemes
├── nets/                    # Neural network architectures 
│   └── flax_drift.py       # Flax-based Föllmer drift networks
├── utils/                   # Utility functions
│   ├── logger.py           # Logging system
│   └── config.py           # YAML configuration management 
├── configs/                 # Configuration files 
│   ├── baseline.yaml       # Default configuration
│   ├── lorenz_cfg.yaml     # Lorenz system configuration
│   └── physionet_cfg.yaml  # PhysioNet dataset configuration
├── cli/                     # Command line interface 
│   └── train.py            # Unified training entry point
└── visualization/           # Visualization modules

theoretical_verification/    # 1D theoretical validation experiments
tests/                       # Unit and integration tests for validation
automation/                  # Shell scripts for running validation workflows
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.