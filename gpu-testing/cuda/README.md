# RAMBO Monte Carlo Integrator - CUDA

Header-only CUDA library for GPU-accelerated phase space generation.

## Requirements

- **CMake** ≥ 3.18
- **CUDA Toolkit** ≥ 11.0
- **C++17** host compiler (GCC ≥9, Clang ≥10)

## Usage

### CMake
```cmake
find_package(rambo-cuda REQUIRED)
target_link_libraries(my_app PRIVATE rambo::cuda)
```

### Code
```cpp
#include <rambo/rambo.cuh>

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

Use `__host__ __device__` / `__device__` decorators for GPU compatibility:

```cpp
struct MyIntegrand {
    double scale;
    __host__ __device__ MyIntegrand(double s = 1.0) : scale(s) {}
    
    __device__ auto evaluate(const double momenta[][4]) const -> double {
        return myMatrixElement(momenta[0], momenta[1]) * scale;
    }
};
```
