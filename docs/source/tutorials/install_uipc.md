# Installing TacEx [UIPC]
# RTX 5090 适配 TacEx UIPC 说明

## 核心修改（3 处）

### 1. 修改编译并行度
**文件**: `source/tacex_uipc/setup.py` (第 93 行)
```python
build_args += ["-j24"]  # 原来是 -j4
```

### 2. 更新依赖版本
**文件**: `source/tacex_uipc/libuipc/conda/env.yaml`
```yaml
cmake=3.26        # 原来是 3.22
cuda-toolkit=12.8 # 原来是 12.4
```

### 3. 设置环境变量（安装时）
```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CMAKE_CUDA_ARCHITECTURES=89
```

## 为什么需要修改？

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 编译失败：找不到 `<span>` | GCC 9.5 不完全支持 C++20 | 使用系统 GCC 11 |
| CUDA 架构不匹配 | RTX 5090 是新架构 (Blackwell) | 使用 sm_89 向下兼容 |
| vcpkg 依赖安装失败 | vcpkg 版本过旧 | 更新 vcpkg 到最新版 |
| 编译太慢 | 只用 4 个线程 | 增加到 24 个线程 |

## 完整安装命令

```bash
# 1. 更新 vcpkg
cd ~/Toolchain/vcpkg
git pull
./bootstrap-vcpkg.sh -disableMetrics

# 2. 进入项目目录
cd /path/to/TacEx_base

# 3. 激活环境并设置变量
conda activate tacex
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CMAKE_CUDA_ARCHITECTURES=89

# 4. 安装
pip install -e source/tacex_uipc -v

# 5. 验证
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py \
    --num_envs 1 --debug_vis --env uipc
```

## GPU 对比

| 显卡 | 架构 | CUDA 架构代码 | 驱动版本 | CUDA 版本 |
|-----|------|-------------|---------|----------|
| RTX 4090 | Ada Lovelace | 89 | 525+ | 12.4+ |
| RTX 5090 | Blackwell | 89* | 570+ | 12.8+ |

*注：理论上是 100，但使用 89 兼容性更好

## 详细文档

- `UIPC_RTX5090_SETUP.md` - 完整安装指南（英文）
- `UIPC_MODIFICATIONS_SUMMARY.md` - 详细修改说明（英文）
- `uipc.md` - 原始文档（已更新）

## 预计时间

- 编译时间：10-15 分钟（24 线程）
- 性能提升：相比 RTX 4090 约 20-30%

## 常见错误

1. **找不到 `<span>`** → 使用 GCC 11
2. **vcpkg 配置失败** → 更新 vcpkg
3. **NVCC 不支持 C++20** → 设置 CUDAHOSTCXX
4. **编译太慢** → 增加 -j 参数

---

**测试环境**: RTX 5090, Driver 570.169, CUDA 12.8, Ubuntu 22.04  
**状态**: ✅ 编译成功，测试通过
The tacex_uipc package is responsible for the UIPC simulation in TacEx.

## Prerequisites

- **Python**: 3.11.x (tested with 3.11.14)
- **Isaac Sim**: 4.5.0, 5.0.0, or 5.1.0
- **System GCC**: 11.4+ (required for C++20 support)
- **CMake**: 3.26+
- **CUDA**: 12.4 (optional, for GPU support)

## Installation Steps

### 1. Install vcpkg (if not already installed)

```bash
mkdir -p ~/Toolchain
cd ~/Toolchain
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
```

Set the system environment variable `CMAKE_TOOLCHAIN_FILE`:

```bash
# Add to ~/.bashrc
export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Reload bashrc
source ~/.bashrc
```

### 2. Install libuipc dependencies

Install CMake and CUDA into the tacex conda environment:

```bash
# Inside the root dir of TacEx repo
conda activate tacex
conda env update -n tacex --file ./source/tacex_uipc/libuipc/conda/env.yaml -y
```

**Note**: If CUDA 12.4 doesn't work, try updating your Nvidia drivers or use an older CUDA version by adjusting the `env.yaml` file (e.g., CUDA 12.2).

### 3. Install tacex_uipc

**IMPORTANT**: The conda environment includes GCC 9.5.0, which does **NOT** support C++20's `<span>` header required by libuipc. You **MUST** use system GCC 11.4+.

```bash
conda activate tacex

# Method 1: Using conda run with explicit compiler settings (RECOMMENDED)
conda run -n tacex bash -c "export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CMAKE_PREFIX_PATH=\$CONDA_PREFIX:\$CMAKE_PREFIX_PATH && pip install -e source/tacex_uipc -v"

# Method 2: Direct activation (if conda run has issues)
# conda activate tacex
# export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
# pip install -e source/tacex_uipc -v
```

**Alternative**: Install all TacEx packages at once:
```bash
./tacex.sh -i all
```

### 4. Verify installation

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

---

## Troubleshooting

### Error: `fatal error: span: No such file or directory`

**Cause**: This error occurs in two scenarios:
1. CMake is using conda's GCC 9.5.0 which doesn't support C++20 `<span>` header (requires GCC 10+)
2. **CUDA files** (`.cu`) are being compiled, and `nvcc` is using conda's GCC 9.5 as the host compiler

**Solution**: 
1. Ensure you have system GCC 11+ installed:
   ```bash
   sudo apt install g++-11
   g++-11 --version  # Should show GCC 11.4 or higher
   ```

2. Clean the build directory and reinstall with explicit compiler settings:
   ```bash
   cd source/tacex_uipc
   rm -rf build tacex_uipc.egg-info
   cd ../..
   
   # Set CC, CXX for both C/C++ and CUDA host compiler
   conda run -n tacex bash -c "export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CMAKE_PREFIX_PATH=\$CONDA_PREFIX:\$CMAKE_PREFIX_PATH && pip install -e source/tacex_uipc -v"
   ```

   **Note**: The `setup.py` has been updated to automatically set `CMAKE_CUDA_HOST_COMPILER` to match `CXX`, ensuring CUDA files also use GCC 11+.
   ```bash
   cd source/tacex_uipc
   rm -rf build
   cd ../..
   
   # Method 1: Use conda run with explicit compiler override
   conda run -n tacex bash -c "unset CC CXX && export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CMAKE_PREFIX_PATH=\$CONDA_PREFIX:\$CMAKE_PREFIX_PATH && pip install -e source/tacex_uipc -v"
   ```

3. Verify the correct compiler is being used by checking the CMake cache:
   ```bash
   grep CMAKE_CXX_COMPILER source/tacex_uipc/build/CMakeCache.txt
   # Should show: CMAKE_CXX_COMPILER:STRING=/usr/bin/g++-11
   # NOT: CMAKE_CXX_COMPILER:STRING=/home/ly/miniconda3/envs/tacex/bin/x86_64-conda-linux-gnu-c++
   ```

### Checking Dependencies

Verify that all required dependencies are installed:
```bash
# Check GCC version
g++-11 --version

# Check CMake version  
conda run -n tacex cmake --version

# Check CUDA version (optional, for GPU support)
conda run -n tacex nvcc --version

# Check vcpkg installation
ls $CMAKE_TOOLCHAIN_FILE
```

