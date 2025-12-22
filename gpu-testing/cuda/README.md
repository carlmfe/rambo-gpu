# RAMBO Monte Carlo Integrator - CUDA

GPU-accelerated implementation using pure CUDA.

## Requirements

- CMake ≥ 3.18
- CUDA Toolkit 11.0+ (tested with CUDA 13.0)
- NVIDIA GPU with compute capability 5.0+
- C++17 compatible host compiler

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

To specify a different GPU architecture:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..  # Ada Lovelace (RTX 40xx)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # Ampere (RTX 30xx)
cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..  # Turing (RTX 20xx)
```

## Run

```bash
./rambo_cuda [num_events] [seed]

# Examples:
./rambo_cuda                  # Default: 100k events, seed 5489
./rambo_cuda 10000000         # 10M events
./rambo_cuda 10000000 42      # 10M events with custom seed
```

## Output

```
======================================
RAMBO Monte Carlo Integrator (CUDA)
======================================
Compiled backend: CUDA GPU
Device: NVIDIA RTX 2000 Ada Generation Laptop GPU
Number of events: 10000000
Random seed: 5489
Center-of-mass energy: 1000 GeV
Number of particles: 3

----------------------------------------
Backend: CUDA GPU
----------------------------------------
  Mean: -753.552
  Error: 306.4
  Time: 70.0 ms
  Throughput: 1.43e+08 events/sec

======================================
Benchmark complete.
======================================
```

## Notes

- Uses XorShift64 RNG for fast, reproducible random number generation
- Grid-stride loop pattern for handling arbitrary event counts
- Atomic operations for global sum reduction
