# RAMBO Monte Carlo Integrator - Base (Serial CPU)

Reference implementation using only standard C++ with no parallelization libraries.

## Requirements

- CMake ≥ 3.16
- C++17 compatible compiler (GCC 9+, Clang 10+)

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Run

```bash
./rambo_base [num_events] [seed]

# Examples:
./rambo_base                  # Default: 100k events, seed 5489
./rambo_base 1000000          # 1M events
./rambo_base 1000000 42       # 1M events with custom seed
```

## Output

```
======================================
RAMBO Monte Carlo Integrator (Base)
======================================
Compiled backend: CPU Serial
Number of events: 1000000
Random seed: 5489
Center-of-mass energy: 1000 GeV
Number of particles: 3

----------------------------------------
Backend: CPU Serial
----------------------------------------
  Mean: 884.412
  Error: 969.332
  Time: 350.0 ms
  Throughput: 2.86e+06 events/sec

======================================
Benchmark complete.
======================================
```
