# RAMBO Monte Carlo Integrator - Kokkos

GPU-accelerated implementation using the Kokkos performance portability library.

## Requirements

- CMake ≥ 3.16
- C++17 compatible compiler
- Kokkos library (with CUDA and/or OpenMP backend)
- Environment module: `kokkos/dev`

## Build

```bash
# Load the Kokkos module (handled automatically by CMake)
mkdir build && cd build
cmake ..
make -j4
```

The CMake configuration automatically loads the `kokkos/dev` module via `find_package(EnvModules)`.

## Run

```bash
./rambo_kokkos [num_events] [seed]

# Examples:
./rambo_kokkos                  # Default: 100k events, seed 5489
./rambo_kokkos 10000000         # 10M events
./rambo_kokkos 10000000 42      # 10M events with custom seed
```

## Output

```
======================================
RAMBO Monte Carlo Integrator (Kokkos)
======================================
Compiled backend: Cuda
Number of events: 10000000
Random seed: 5489
Center-of-mass energy: 1000 GeV
Number of particles: 3

----------------------------------------
Backend: Cuda
----------------------------------------
  Mean: 1812.93
  Error: 306.5
  Time: 120.5 ms
  Throughput: 8.3e+07 events/sec

======================================
Benchmark complete.
======================================
```

## Notes

- The backend (CUDA, OpenMP, Serial) is determined by how Kokkos was built
- Uses `Kokkos::Random_XorShift64_Pool` for thread-safe RNG
