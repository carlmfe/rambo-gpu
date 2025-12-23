# RAMBO SYCL Implementation

High-performance GPU implementation of the RAMBO Monte Carlo integrator using SYCL with CUDA backend.

## Overview

This implementation uses SYCL (via clang++ with CUDA backend) to perform parallel Monte Carlo integration on NVIDIA GPUs. The structure mirrors the base/, kokkos/, alpaka/, and cuda/ implementations for easy comparison and maintenance.

## File Structure

```
sycl/
├── CMakeLists.txt      # Build configuration for SYCL/CUDA
├── integrands.hpp      # Physics integrands (DrellYan, Eggholder, etc.)
├── integrator.hpp      # RamboIntegrator class with SYCL parallel_for
├── main.cpp            # Entry point with benchmark and verification
├── rambo_sycl.hpp      # Phase space generator using RAMBO algorithm
└── README.md           # This file
```

## Prerequisites

- SYCL compiler (clang++ with CUDA support, Intel DPC++, or AdaptiveCpp)
- CUDA-capable NVIDIA GPU (for CUDA backend)
- CMake 3.18+

## Build Instructions

```bash
cd gpu-testing/sycl
mkdir build && cd build

# Specify SYCL compiler (Intel DPC++/LLVM with CUDA backend)
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ ..
make

# Alternative: Specify backend explicitly
cmake -DCMAKE_CXX_COMPILER=/path/to/sycl/clang++ -DSYCL_BACKEND=CUDA ..
make

# Intel GPU backend
cmake -DCMAKE_CXX_COMPILER=icpx -DSYCL_BACKEND=INTEL ..
make

# Run with default parameters (100000 events, seed 5489)
./rambo_sycl

# Run with custom parameters
./rambo_sycl 1000000 12345
```

## Backend Selection

| Backend | Compiler | CMake Flags |
|---------|----------|-------------|
| CUDA | clang++ (SYCL) | `-DSYCL_BACKEND=CUDA` |
| Intel GPU | icpx | `-DSYCL_BACKEND=INTEL` |
| AdaptiveCpp | syclcc | `-DSYCL_BACKEND=ADAPTIVECPP` |

## Usage

```
./rambo_sycl [nEvents] [seed]
```

- `nEvents`: Number of Monte Carlo events (default: 100000)
- `seed`: Random seed for reproducibility (default: 5489)

## GPU Architecture

By default, CUDA GPU architecture is **auto-detected** using `nvidia-smi`. To specify manually:

```bash
cmake -DCUDA_GPU_ARCH=sm_89 ..  # For RTX 40xx / Ada
cmake -DCUDA_GPU_ARCH=sm_86 ..  # For RTX 30xx
cmake -DCUDA_GPU_ARCH=sm_80 ..  # For A100 / RTX 30xx
cmake -DCUDA_GPU_ARCH=native .. # Auto-detect (default)
```

Note: CUDA 13.0+ removed support for `sm_70` (Volta) and `sm_75` (Turing).

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

### SYCL Features Used

- `sycl::queue` with GPU selector
- USM (Unified Shared Memory) for device memory allocation
- `parallel_for` with `nd_range` for work distribution
- `atomic_ref` for thread-safe accumulation
- Grid-stride loops for handling large event counts

### Performance Considerations

- Uses XorShift64 RNG for fast, reproducible random numbers
- Hash-based seed mixing to avoid RNG correlation between threads
- Atomic operations for sum reduction (simple but effective)
- Warmup run before timed benchmark

## Comparison with Other Implementations

| Feature | SYCL | CUDA | Kokkos | Alpaka |
|---------|------|------|--------|--------|
| Language | C++ | CUDA C++ | C++ | C++ |
| Memory Model | USM | Explicit | Views | Buffers |
| Kernel Launch | parallel_for | <<<>>> | parallel_reduce | exec<> |
| Reduction | atomic_ref | atomicAdd | Sum reducer | Tree reduction |
| Portability | High | NVIDIA only | High | High |

## Troubleshooting

### Wrong GPU architecture
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture
cmake -DCUDA_GPU_ARCH=sm_XX ..
```

### Compiler not found
```bash
# Verify clang++ is in PATH after loading module
which clang++

# Check SYCL support
clang++ --version
```
