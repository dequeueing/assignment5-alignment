# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

> **Note**: If you encounter a build error for `flash-attn` stating "FlashAttention is only supported on CUDA 11.7 and above", it means your CUDA Toolkit is older than 11.7. You have two options:
> 
> - **Option A (Recommended for assignment)**: Comment out `flash-attn` from `pyproject.toml` since it's not required for this assignment. The core code doesn't directly use it.
> - **Option B (For production use)**: Upgrade your CUDA Toolkit to 11.7+ (see [CUDA Upgrade Guide](#cuda-upgrade-guide) below)

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Module System (HPC Cluster Environment)

This environment uses **Environment Modules** (Lmod) to manage software versions. This is typical of HPC clusters and virtual machines.

### Understanding Your Setup

- **Default NVCC**: System-installed CUDA 11.5 (via `/usr/bin/nvcc`) ✓
- **Module System**: Available at `/software/u22/modulefiles/`
- **Session-based**: Module changes only affect the current terminal session

### Loading a Different CUDA Version

To use a newer CUDA version (e.g., CUDA 12.9 for flash-attn compatibility):

```sh
# Load CUDA 12.9 module (recommended for this assignment)
module load nvhpc/25.7/nvhpc-hpcx-cuda12/25.7

# Verify the change
nvcc -V  # Should show CUDA 12.9

# Check all loaded modules
module list
```

### Switching Between Versions

```sh
# Switch from one module to another
module switch nvhpc/25.7/nvhpc-hpcx-cuda12/25.7 nvhpc/25.7/nvhpc-hpcx-cuda11/25.7

# Unload a specific module
module unload nvhpc/25.7/nvhpc-hpcx-cuda12/25.7

# Reset to system defaults
module reset
```

### Available CUDA Modules

```sh
# List all available CUDA-related modules
module avail cuda

# Currently available:
# - nvhpc/25.7/nvhpc-hpcx-cuda12/25.7  (CUDA 12.9, recommended)
# - nvhpc/25.7/nvhpc-hpcx-cuda11/25.7  (CUDA 11.7+)
# - nvhpc/23.11/nvhpc-hpcx-cuda12      (CUDA 12, older)
# - nvhpc/23.11/nvhpc-hpcx-cuda11      (CUDA 11, older)
```

---

## CUDA Upgrade Guide

If you need to use flash-attn or encounter CUDA version compatibility issues, you have two options:

### Option 1: Use Module System (Recommended)

Use the module system described above to load CUDA 12.9 without modifying system files:

```sh
module load nvhpc/25.7/nvhpc-hpcx-cuda12/25.7
rm -rf .venv
uv sync
```

### Option 2: Manual CUDA Toolkit Installation

If you need to permanently upgrade the system CUDA Toolkit to 11.7 or higher:

#### 1. Check Current CUDA Version
```sh
nvcc -V
```

#### 2. Download CUDA Toolkit 11.8+ (Recommended)
Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) and select:
- OS: Linux
- Architecture: x86_64
- Distribution: Ubuntu (or your distro)
- Version: 22.04 (or your version)
- Installer Type: runfile (local)

#### 3. Install CUDA Toolkit
```sh
# Download the installer (e.g., cuda_11.8.0_linux.run)
chmod +x cuda_11.8.0_linux.run
sudo ./cuda_11.8.0_linux.run --silent --driver --toolkit
```

#### 4. Set Environment Variables
Add to `~/.bashrc` or `~/.zshrc`:
```sh
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then reload:
```sh
source ~/.bashrc  # or source ~/.zshrc
```

#### 5. Verify Installation
```sh
nvcc -V
```

#### 6. Reinstall Dependencies
```sh
rm -rf .venv
uv sync
```

