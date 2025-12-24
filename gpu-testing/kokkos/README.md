# RAMBO Monte Carlo Integrator - Kokkos

Header-only, GPU-accelerated phase space generation library using Kokkos.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, NVCC ≥11)
- **Kokkos** ≥ 4.0, pre-built with desired backend (Serial, OpenMP, CUDA, HIP)

## Usage

### CMake
```cmake
find_package(rambo-kokkos REQUIRED)
target_link_libraries(my_app PRIVATE rambo::kokkos)
```

### Code
```cpp
#include <Kokkos_Core.hpp>
#include <rambo/rambo.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int nParticles = 2;
        double masses[nParticles] = {0.000511, 0.000511};
        
        rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
        rambo::RamboIntegrator<rambo::DrellYanIntegrand, nParticles> 
            integrator(1000000, integrand);
        
        double mean, error;
        integrator.run(91.2, masses, mean, error, 5489);
    }
    Kokkos::finalize();
}
```

## Custom Integrands

Use `KOKKOS_FUNCTION` / `KOKKOS_INLINE_FUNCTION` decorators for GPU compatibility:

```cpp
struct MyIntegrand {
    double scale;
    KOKKOS_FUNCTION MyIntegrand(double s = 1.0) : scale(s) {}
    
    KOKKOS_INLINE_FUNCTION 
    auto evaluate(const double momenta[][4]) const -> double {
        return myMatrixElement(momenta[0], momenta[1]) * scale;
    }
};
```

**Note**: Any function called from `evaluate()` must also have `KOKKOS_INLINE_FUNCTION`.
