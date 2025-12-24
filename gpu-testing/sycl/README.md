# RAMBO Monte Carlo Integrator - SYCL

Header-only SYCL library for portable GPU phase space generation.
Supports CUDA (NVIDIA), HIP (AMD), Intel, and AdaptiveCpp backends.

## Requirements

- **CMake** ≥ 3.18
- **C++20** compiler with SYCL support (Intel DPC++, AdaptiveCpp)

## Installation

```bash
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCUDA_GPU_ARCH=sm_89 \
      -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

## Usage

### CMake
```cmake
find_package(rambo-sycl REQUIRED)
target_link_libraries(my_app PRIVATE rambo::sycl)
```

### Code
```cpp
#include <rambo/rambo.hpp>

int main() {
    constexpr int nParticles = 2;
    double masses[nParticles] = {0.000511, 0.000511};
    
    rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
    rambo::RamboIntegrator<rambo::DrellYanIntegrand, nParticles> 
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
    
    MyIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle index, mu = 0:E, 1:px, 2:py, 3:pz
        // Use sycl:: math functions for device compatibility
        return sycl::sin(momenta[0][0]) * scale;
    }
};

// Use it:
MyIntegrand integrand(1.0);
rambo::RamboIntegrator<MyIntegrand, 2> integrator(nEvents, integrand);
```

## Backend Selection

| Backend | CMake Flags |
|---------|-------------|
| CUDA | `-DSYCL_BACKEND=CUDA -DCUDA_GPU_ARCH=sm_XX` |
| HIP | `-DSYCL_BACKEND=HIP -DHIP_GPU_ARCH=gfxXXXX` |
| Intel | `-DSYCL_BACKEND=INTEL` |
| AdaptiveCpp | `-DSYCL_BACKEND=ADAPTIVECPP` |
