# RAMBO Alpaka Implementation

Header-only Alpaka library for RAMBO phase space generation and Monte Carlo integration.
Supports multiple backends: CUDA, HIP, SYCL, CPU Serial, OpenMP.

## Quick Start

```bash
module load alpaka/2.0.0_cuda  # Or your alpaka installation
mkdir build && cd build
cmake -DALPAKA_BACKEND=CUDA ..
make
./rambo_alpaka 1000000 5489
```

## Library Usage

```cmake
find_package(rambo-alpaka REQUIRED)
target_link_libraries(my_app PRIVATE rambo::alpaka)
```

```cpp
#include <rambo/rambo.hpp>

using AccTag = alpaka::TagGpuCudaRt;  // Or other backend

rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
rambo::RamboIntegrator<AccTag, rambo::DrellYanIntegrand, 2> integrator(1000000, integrand);

double masses[2] = {0.000511, 0.000511};
double mean, error;
integrator.run(91.2, masses, mean, error, 5489);
```

## Custom Integrands

```cpp
struct MyIntegrand {
    double scale;
    ALPAKA_FN_HOST_ACC MyIntegrand(double s = 1.0) : scale(s) {}
    
    ALPAKA_FN_HOST_ACC auto evaluate(const double momenta[][4]) const -> double {
        return myCalculation(momenta) * scale;
    }
};
```

## Backend Selection

| Backend | CMake Flag | AccTag |
|---------|-----------|--------|
| CUDA | `-DALPAKA_BACKEND=CUDA` | `alpaka::TagGpuCudaRt` |
| HIP | `-DALPAKA_BACKEND=HIP` | `alpaka::TagGpuHipRt` |
| CPU Serial | `-DALPAKA_BACKEND=CPU` | `alpaka::TagCpuSerial` |
| OpenMP | `-DALPAKA_BACKEND=OMP` | `alpaka::TagCpuOmp2Blocks` |

## Requirements

- Alpaka 2.0.0+
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
