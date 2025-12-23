# RAMBO Monte Carlo Integrator - Alpaka

GPU-accelerated implementation using the Alpaka 2.0.0 portability library.

## Requirements

- CMake ≥ 3.18
- C++20 compatible compiler
- Alpaka 2.0.0 library
- For CUDA backend: CUDA Toolkit 11.0+

## Build

```bash
mkdir build && cd build

# CUDA backend (requires alpaka installation path)
cmake -DALPAKA_BACKEND=CUDA -Dalpaka_ROOT=/path/to/alpaka ..

# CPU Serial backend (default)
cmake -DALPAKA_BACKEND=CPU -Dalpaka_ROOT=/path/to/alpaka ..

# OpenMP backend
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

## Output

```
======================================
RAMBO Monte Carlo Integrator (Alpaka)
======================================
Compiled backend: CUDA GPU
Number of events: 10000000
Random seed: 5489
Center-of-mass energy: 1000 GeV
Number of particles: 3

Available accelerator tags:
Tags: TagGpuCudaRt

----------------------------------------
Backend: CUDA GPU
----------------------------------------
  Mean: 313.253
  Error: 306.7
  Time: 85.0 ms
  Throughput: 1.18e+08 events/sec

======================================
Benchmark complete.
======================================
```

## Backend Selection

| Backend | CMake Flag | Description |
|---------|------------|-------------|
| CUDA | `-DALPAKA_BACKEND=CUDA` | NVIDIA GPU (default) |
| CPU Serial | `-DALPAKA_BACKEND=CPU` | Single-threaded CPU |
| OpenMP | `-DALPAKA_BACKEND=OMP` | Multi-threaded CPU |
