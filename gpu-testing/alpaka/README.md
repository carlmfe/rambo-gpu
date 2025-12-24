# RAMBO Monte Carlo Integrator - Alpaka

Header-only Alpaka library for portable GPU/CPU phase space generation.
Supports CUDA, HIP, SYCL, CPU Serial, and OpenMP backends.

## Requirements

- **CMake** ≥ 3.18
- **C++20** compiler (GCC ≥10, Clang ≥12, NVCC ≥11)
- **Alpaka** ≥ 2.0.0, pre-built with desired backend

## Installation

```bash
mkdir build && cd build
cmake -Dalpaka_ROOT=/path/to/alpaka -DALPAKA_BACKEND=CUDA \
      -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

## Usage

### CMake
```cmake
find_package(rambo-alpaka REQUIRED)
target_link_libraries(my_app PRIVATE rambo::alpaka)
```

### Code
```cpp
#include <rambo/rambo.hpp>

int main() {
    using AccTag = alpaka::TagGpuCudaRt;  // Or TagCpuSerial, TagCpuOmp2Blocks
    
    constexpr int nParticles = 2;
    double masses[nParticles] = {0.000511, 0.000511};
    
    rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
    rambo::RamboIntegrator<AccTag, rambo::DrellYanIntegrand, nParticles> 
        integrator(1000000, integrand);
    
    double mean, error;
    integrator.run(91.2, masses, mean, error, 5489);
}
```

## Custom Integrands

Wrap any function by creating a struct with an `evaluate()` method:

```cpp
struct MyIntegrand {
    double scale;
    
    ALPAKA_FN_HOST_ACC MyIntegrand(double s = 1.0) : scale(s) {}
    
    ALPAKA_FN_HOST_ACC auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle index, mu = 0:E, 1:px, 2:py, 3:pz
        return myMatrixElement(momenta[0], momenta[1]) * scale;
    }
};

// Use it:
MyIntegrand integrand(1.0);
rambo::RamboIntegrator<AccTag, MyIntegrand, 2> integrator(nEvents, integrand);
```

## Backend Selection

| Backend | CMake Flag | AccTag |
|---------|------------|--------|
| CUDA | `-DALPAKA_BACKEND=CUDA` | `alpaka::TagGpuCudaRt` |
| HIP | `-DALPAKA_BACKEND=HIP` | `alpaka::TagGpuHipRt` |
| CPU Serial | `-DALPAKA_BACKEND=CPU` | `alpaka::TagCpuSerial` |
| OpenMP | `-DALPAKA_BACKEND=OMP` | `alpaka::TagCpuOmp2Blocks` |
