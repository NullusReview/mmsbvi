# MMSBVI 环境安装指南

## 快速安装

### 方法1: 自动安装（推荐）
```bash
python setup_environment.py
```
脚本会自动检测你的硬件（Mac、CUDA GPU或普通CPU）并安装对应依赖。

### 方法2: 手动安装

#### Mac / CPU 环境
```bash
pip install -r requirements-cpu.txt
```

#### CUDA GPU 环境
```bash
pip install -r requirements-gpu.txt
```

## 环境验证

安装完成后验证：
```python
import jax
import jax.numpy as jnp

# 检查设备
print("JAX devices:", jax.devices())

# 测试64位精度（理论验证必需）
jax.config.update('jax_enable_x64', True)
x = jnp.array([1.0], dtype=jnp.float64)
print("64-bit support:", x.dtype)

# 测试基本计算
result = jnp.sum(jnp.array([1.0, 2.0, 3.0])**2)
print("Basic computation:", result)
```

期望输出：
- **Mac**: `[CpuDevice(id=0)]`
- **CUDA GPU**: `[GpuDevice(id=0)]` 或 `[CudaDevice(id=0)]`

## 依赖说明

### 核心依赖
- `jax==0.4.25`: 核心计算框架
- `ott-jax==0.4.5`: Google的最优传输库（替代自己实现OT）
- `blackjax>=1.0.0`: MCMC基线对比
- `optax==0.1.9`: 标准优化器 + Anderson加速

### 自定义实现部分
本项目需要自己实现以下核心组件：
- **Onsager-Fokker度量**: 需要求解PDE `-∇·(ρ∇φ) = σ`
- **FFT加速卷积**: 用于高斯核的快速计算
- **Multi-marginal IPFP扩展**: 基于OTT-JAX进行扩展

### 差异说明
| 依赖 | CPU版本 | GPU版本 | 说明 |
|------|---------|---------|------|
| JAX | `jax==0.4.25` | `jax[cuda12]==0.4.25` | GPU版本包含CUDA支持 |
| 监控工具 | 基础监控 | GPU监控 | GPU版本额外包含GPU内存监控 |
| 数值计算 | 标准精度 | 混合精度支持 | GPU版本可选择性能优化 |

## 故障排除

### 常见问题

1. **CUDA版本不匹配**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   # 如果是CUDA 11.x，改用：
   pip install jax[cuda11_pip]==0.4.25
   ```

2. **Mac内存不足**
   ```python
   # 降低内存使用
   jax.config.update('jax_memory_fraction', 0.5)
   ```

3. **导入错误**
   ```bash
   # 清理安装
   pip uninstall jax jaxlib -y
   pip install -r requirements-[cpu/gpu].txt --force-reinstall
   ```

## 性能配置

### Mac优化
```python
import os
import jax

# CPU优化配置
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
```

### CUDA GPU优化
```python
import os
import jax

# GPU优化配置
jax.config.update('jax_enable_x64', True)  # 精度优先
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true'
)
```

## 开发环境

推荐使用conda创建独立环境：
```bash
# 创建环境
conda create -n mmsbvi python=3.11 -y
conda activate mmsbvi

# 安装依赖
python setup_environment.py
```