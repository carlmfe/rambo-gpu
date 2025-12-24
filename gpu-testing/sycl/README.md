# RAMBO Monte Carlo Integrator - SYCL

Header-only SYCL library for cross-platform GPU phase space generation.
Supports Intel GPUs (DPC++), NVIDIA GPUs (via CUDA backend), and AMD GPUs (via HIP backend).

## Requirements

- **CMake** ≥ 3.18
- **C++20** SYCL compiler (Intel DPC++, AdaptiveCpp)
- **SYCL runtime** for target device

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
    sycl::queue q{sycl::gpu_selector_v};
    
    constexpr int nParticles = 2;
    double masses[nParticles] = {0.000511, 0.000511};
    
    rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
    rambo::RamboIntegrator<rambo::DrellYanIntegrand, nParticles> 
        integrator(1000000, integrand, q);
    
    double mean, error;
    integrator.run(91.2, masses, mean, error, 5489);
}
```

## Custom Integrands

Use `sycl::` math functions for device-compatible code:

```cpp
struct MyIntegrand {
    double scale;
    MyIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // Use sycl::sqrt, sycl::exp, etc. instead of std:: versions
        return myMatrixElement(momenta[0], momenta[1]) * scale;
    }
};
```
