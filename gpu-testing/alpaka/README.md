# RAMBO Monte Carlo Integrator - Alpaka

High-performance **multi-vendor GPU** implementation using the Alpaka 2.0.0 portability library.

## Overview

Alpaka provides a single-source programming model that can target NVIDIA, AMD, and Intel GPUs, as well as CPUs. The backend is selected at compile time via CMake flags.

## Supported Backends

| Backend | Vendor | CMake Flag | Auto-Detection |
|---------|--------|------------|----------------|
| **CUDA** | NVIDIA | `-DALPAKA_BACKEND=CUDA` | ✅ via `native` |
| **HIP** | AMD | `-DALPAKA_BACKEND=HIP` | ✅ via rocminfo |
| **SYCL** | Intel | `-DALPAKA_BACKEND=SYCL` | ✅ automatic |
| **CPU** | Any | `-DALPAKA_BACKEND=CPU` | N/A |
| **OMP** | Any | `-DALPAKA_BACKEND=OMP` | N/A |

## Requirements

- CMake ≥ 3.18
- C++20 compatible compiler
- Alpaka 2.0.0+ library (built with your desired backend)
- Backend-specific:
  - **CUDA**: CUDA Toolkit 11.0+
  - **HIP**: ROCm 5.0+
  - **SYCL**: Intel oneAPI or AdaptiveCpp

## Build Instructions

### NVIDIA GPUs (CUDA)
```bash
mkdir build && cd build

# Auto-detect GPU architecture
cmake -DALPAKA_BACKEND=CUDA -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
```

### AMD GPUs (HIP)
```bash
# Auto-detect GPU architecture
cmake -DALPAKA_BACKEND=HIP -Dalpaka_ROOT=/path/to/alpaka ..
make -j4

# Or specify architecture manually
cmake -DALPAKA_BACKEND=HIP -DHIP_GPU_ARCH=gfx1100 -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
```

### Intel GPUs (SYCL)
```bash
cmake -DALPAKA_BACKEND=SYCL -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
```

### CPU Serial (default)
```bash
cmake -DALPAKA_BACKEND=CPU -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
```

### OpenMP (multi-threaded CPU)
```bash
cmake -DALPAKA_BACKEND=OMP -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
```

## Run

```bash
./rambo_alpaka [num_events] [seed]

# Examples:
./rambo_alpaka                  # Default: 100k events, seed 5489
./rambo_alpaka 10000000         # 10M events
./rambo_alpaka 10000000 42      # 10M events with custom seed
```

## GPU Architecture Reference

### NVIDIA (CUDA)
Architecture is auto-detected via CMake's `native` setting.

### AMD (HIP)
| Architecture | GPUs |
|--------------|------|
| gfx906 | MI50, Radeon VII |
| gfx908 | MI100 |
| gfx90a | MI210, MI250 |
| gfx1030 | RX 6800, 6900 |
| gfx1100 | RX 7900 |

## Backend Selection Summary

| Backend | CMake Flag | Description |
|---------|------------|-------------|
| CUDA | `-DALPAKA_BACKEND=CUDA` | NVIDIA GPU |
| HIP | `-DALPAKA_BACKEND=HIP` | AMD GPU |
| SYCL | `-DALPAKA_BACKEND=SYCL` | Intel GPU (or portable) |
| CPU Serial | `-DALPAKA_BACKEND=CPU` | Single-threaded CPU |
| OpenMP | `-DALPAKA_BACKEND=OMP` | Multi-threaded CPU |

## Important Notes

1. **Alpaka must be built with the corresponding backend enabled**. If you want to use HIP, your Alpaka installation must have been configured with `-Dalpaka_ACC_GPU_HIP_ENABLE=ON`.

2. The backend is selected at **compile time**, not runtime. To switch backends, you need to reconfigure and rebuild.

3. For best performance, ensure your Alpaka installation matches your target hardware.
