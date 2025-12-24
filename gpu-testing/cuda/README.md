# RAMBO CUDA Implementation

Header-only CUDA library for RAMBO phase space generation and Monte Carlo integration.

## Quick Start

```bash
mkdir build && cd build
cmake ..
make
./rambo_cuda 1000000 5489
```

## Library Usage

```cmake
find_package(rambo-cuda REQUIRED)
target_link_libraries(my_app PRIVATE rambo::cuda)
```

```cpp
#include <rambo/rambo.cuh>

rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
rambo::RamboIntegrator<rambo::DrellYanIntegrand, 2> integrator(1000000, integrand);

double masses[2] = {0.000511, 0.000511};
double mean, error;
integrator.run(91.2, masses, mean, error, 5489);
```

## Custom Integrands

```cpp
struct MyIntegrand {
    double scale;
    __host__ __device__ MyIntegrand(double s = 1.0) : scale(s) {}
    
    __device__ auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i=particle, mu=0:E,1:px,2:py,3:pz
        return myCalculation(momenta) * scale;
    }
};
```

## Requirements

- CMake ≥ 3.18
- CUDA Toolkit
- C++17 compiler

## Files

```
include/rambo/
├── rambo.cuh         # Main include
├── phase_space.cuh   # PhaseSpaceGenerator, RamboAlgorithm
├── integrator.cuh    # RamboIntegrator
└── integrands.cuh    # Example integrands
```
