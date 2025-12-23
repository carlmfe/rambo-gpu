# RAMBO SYCL Implementation

High-performance **multi-vendor GPU** implementation of the RAMBO Monte Carlo integrator using SYCL.

## Overview

This implementation uses SYCL to perform parallel Monte Carlo integration on **NVIDIA, AMD, and Intel GPUs**. SYCL provides portability across GPU vendors while maintaining high performance. The structure mirrors the base/, kokkos/, alpaka/, and cuda/ implementations for easy comparison and maintenance.

## Supported GPU Vendors

| Vendor | Backend | Compiler | Auto-Detection |
|--------|---------|----------|----------------|
| **NVIDIA** | CUDA | Intel DPC++/LLVM (clang++) | ✅ via nvidia-smi |
| **AMD** | HIP | Intel DPC++/LLVM (clang++) | ✅ via rocminfo |
| **Intel** | Native | Intel DPC++ (icpx) | ✅ automatic |

## File Structure

```
sycl/
├── CMakeLists.txt      # Multi-vendor GPU build configuration
├── integrands.hpp      # Physics integrands (DrellYan, Eggholder, etc.)
├── integrator.hpp      # RamboIntegrator class with SYCL parallel_for
├── main.cpp            # Entry point with benchmark and verification
├── rambo_sycl.hpp      # Phase space generator using RAMBO algorithm
└── README.md           # This file
```

## Prerequisites

- SYCL compiler:
  - **NVIDIA**: Intel DPC++/LLVM with CUDA support, or AdaptiveCpp
  - **AMD**: Intel DPC++/LLVM with HIP support, or AdaptiveCpp
  - **Intel**: Intel oneAPI DPC++ (icpx)
- CMake 3.18+

## Build Instructions

### NVIDIA GPUs (CUDA backend)
```bash
cd gpu-testing/sycl
mkdir build && cd build

# Auto-detect GPU architecture
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ -DSYCL_BACKEND=CUDA ..
make

# Or specify architecture manually
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ \
      -DSYCL_BACKEND=CUDA \
      -DCUDA_GPU_ARCH=sm_89 ..
make
```

### AMD GPUs (HIP backend)
```bash
# Auto-detect GPU architecture
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ -DSYCL_BACKEND=HIP ..
make

# Or specify architecture manually
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ \
      -DSYCL_BACKEND=HIP \
      -DHIP_GPU_ARCH=gfx1100 ..
make
```

### Intel GPUs
```bash
cmake -DCMAKE_CXX_COMPILER=icpx -DSYCL_BACKEND=INTEL ..
make
```

### AdaptiveCpp (formerly hipSYCL) - All vendors
```bash
cmake -DCMAKE_CXX_COMPILER=acpp \
      -DSYCL_BACKEND=ADAPTIVECPP \
      -DADAPTIVECPP_TARGETS="cuda:sm_89" ..  # or "hip:gfx1100"
make
```

## Backend Selection

| Backend | Vendor | CMake Flags | Architecture |
|---------|--------|-------------|--------------|
| CUDA | NVIDIA | `-DSYCL_BACKEND=CUDA` | `-DCUDA_GPU_ARCH=sm_XX` |
| HIP | AMD | `-DSYCL_BACKEND=HIP` | `-DHIP_GPU_ARCH=gfxXXXX` |
| INTEL | Intel | `-DSYCL_BACKEND=INTEL` | Auto-detected |
| ADAPTIVECPP | Any | `-DSYCL_BACKEND=ADAPTIVECPP` | `-DADAPTIVECPP_TARGETS=...` |

## Usage

```
./rambo_sycl [nEvents] [seed]
```

- `nEvents`: Number of Monte Carlo events (default: 100000)
- `seed`: Random seed for reproducibility (default: 5489)

## GPU Architecture Reference

### NVIDIA (CUDA)
| Architecture | GPUs |
|--------------|------|
| sm_80 | A100, RTX 30xx (GA102) |
| sm_86 | RTX 30xx (GA10x) |
| sm_89 | RTX 40xx, RTX Ada |
| sm_90 | H100 |

Note: CUDA 13.0+ removed support for sm_70 (Volta) and sm_75 (Turing).

### AMD (HIP)
| Architecture | GPUs |
|--------------|------|
| gfx906 | MI50, Radeon VII |
| gfx908 | MI100 |
| gfx90a | MI210, MI250 |
| gfx1030 | RX 6800, 6900 |
| gfx1100 | RX 7900 |

## Physics

The default integrand computes the parton-level Drell-Yan cross-section:

```
q + qbar → γ* → e+ e-
```

At the Z-pole (91.2 GeV), the analytic cross-section for up-quarks is:
```
σ = 4πα²e_q² / (3s) × ℏ²c²
```

The Monte Carlo result should converge to this value as the number of events increases.

## Implementation Details

### Key Components

1. **PhaseSpaceGenerator** (`rambo_sycl.hpp`): Generates random 4-momenta using the RAMBO algorithm
2. **Integrands** (`integrands.hpp`): Various physics functions to integrate
3. **RamboIntegrator** (`integrator.hpp`): SYCL-based parallel Monte Carlo integration

### RNG Strategy

Uses `std::minstd_rand` for portability across GPU vendors. Each work-item gets a unique seed based on its global index.

### Performance

SYCL provides near-native performance while maintaining portability:
- NVIDIA: Comparable to native CUDA
- AMD: Comparable to native HIP
- Intel: Native performance on Intel GPUs
