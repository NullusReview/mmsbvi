#!/usr/bin/env python3
"""
MMSBVI Environment Setup Script
自动检测硬件并安装对应依赖
"""

import platform
import subprocess
import sys
import os

def detect_hardware():
    """检测硬件环境"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    
    # 检测是否有NVIDIA GPU
    has_nvidia_gpu = False
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            has_nvidia_gpu = True
            print("NVIDIA GPU detected")
            print(result.stdout.split('\n')[0])  # GPU信息首行
        else:
            print("No NVIDIA GPU detected")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available")
    
    # 判断环境类型
    if has_nvidia_gpu:
        return "gpu"
    elif system == "Darwin" and "arm" in machine.lower():
        return "m2_mac"
    else:
        return "cpu"

def install_requirements(env_type):
    """安装对应环境的依赖"""
    if env_type == "gpu":
        requirements_file = "requirements-gpu.txt"
        print("Installing GPU environment (CUDA)...")
    else:
        requirements_file = "requirements-cpu.txt"
        if env_type == "m2_mac":
            print("Installing M2 Mac CPU environment...")
        else:
            print("Installing CPU environment...")
    
    # 检查文件是否存在
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found!")
        return False
    
    # 安装依赖
    try:
        print(f"Installing from {requirements_file}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], check=True)
        print("Installation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\nVerifying installation...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # 测试基本功能
        x = jnp.array([1.0, 2.0, 3.0])
        result = jnp.sum(x**2)
        
        print(f"JAX working: {result}")
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX version: {jax.__version__}")
        
        # 测试64位精度
        jax.config.update('jax_enable_x64', True)
        x64 = jnp.array([1.0], dtype=jnp.float64)
        print(f"64-bit precision: {x64.dtype}")
        
        # 测试OTT-JAX
        try:
            import ott
            print(f"OTT-JAX version: {ott.__version__}")
        except ImportError:
            print("Warning: OTT-JAX not installed")
        
        # 测试Optax
        try:
            import optax
            print(f"Optax version: {optax.__version__}")
        except ImportError:
            print("Warning: Optax not installed")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Verification error: {e}")
        return False

def main():
    """主函数"""
    print("MMSBVI Environment Setup")
    print("=" * 40)
    
    # 检测硬件
    env_type = detect_hardware()
    print(f"\nDetected environment: {env_type.upper()}")
    
    # 确认安装
    response = input(f"\n继续安装 {env_type} 环境? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # 安装依赖
    if install_requirements(env_type):
        # 验证安装
        if verify_installation():
            print("\nEnvironment setup completed successfully!")
            if env_type == "gpu":
                print("Remember to configure CUDA environment variables if needed.")
            elif env_type == "m2_mac":
                print("M2 Mac optimized for CPU-only computation.")
        else:
            print("\nInstallation completed but verification failed.")
    else:
        print("\nEnvironment setup failed.")

if __name__ == "__main__":
    main()