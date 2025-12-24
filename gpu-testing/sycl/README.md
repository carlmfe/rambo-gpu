# RAMBO SYCL Implementation

Header-only SYCL library for RAMBO phase space generation and Monte Carlo integration.
Supports multiple backends: CUDA (NVIDIA), HIP (AMD), Intel, AdaptiveCpp.

## Quick Start

```bash
module load sycl/cuda  # Load SYCL compiler
mkdir build && cd build
cmake -DCUDA_GPU_ARCH=sm_89 ..
make
./rambo_sycl 1000000 5489
```

## Library Usage

```cmake
find_package(rambo-sycl REQUIRED)
target_link_libraries(my_app PRIVATE rambo::sycl)
```

```cpp
#include <rambo/rambo.hpp>

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
    MyIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // Use sycl:: math functions for device code
        return sycl::sin(momenta[0][0]) * scale;
    }
};
```

## Backend Selection

| Backend | CMake Flags |
|---------|-------------|
| CUDA | `-DSYCL_BACKEND=CUDA -DCUDA_GPU_ARCH=sm_XX` |
| HIP | `-DSYCL_BACKEND=HIP -DHIP_GPU_ARCH=gfxXXXX` |
| Intel | `-DSYCL_BACKEND=INTEL` |
| AdaptiveCpp | `-DSYCL_BACKEND=ADAPTIVECPP` |

## Requirements

- SYCL compiler (Intel DPC++, AdaptiveCpp)
- CMake ≥ 3.18
- C++20 compiler

## Files

```
include/rambo/
├── rambo.hpp         # Main include
├── phase_space.hpp   # PhaseSpaceGenerator, RamboAlgorithm
├── integrator.hpp    # RamboIntegrator
└── integrands.hpp    # Example integrands
```
