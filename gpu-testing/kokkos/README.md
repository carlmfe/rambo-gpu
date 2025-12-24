# RAMBO Monte Carlo Integrator - Kokkos

Header-only, GPU-accelerated phase space generation library using Kokkos.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, NVCC ≥11)
- **Kokkos** ≥ 4.0, pre-built with desired backend (Serial, OpenMP, CUDA, HIP)

## Installation

```bash
mkdir build && cd build
cmake -DKokkos_ROOT=/path/to/kokkos -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

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

Wrap any function by creating a struct with an `evaluate()` method:

```cpp
// Suppose you have an existing function:
double myMatrixElement(double p1[4], double p2[4], double p3[4]) {
    // ... physics calculation ...
    return result;
}

// Wrap it as an integrand:
struct MyIntegrand {
    double scale;
    
    KOKKOS_FUNCTION MyIntegrand(double s = 1.0) : scale(s) {}
    
    KOKKOS_INLINE_FUNCTION 
    auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle index, mu = 0:E, 1:px, 2:py, 3:pz
        double p1[4] = {momenta[0][0], momenta[0][1], momenta[0][2], momenta[0][3]};
        double p2[4] = {momenta[1][0], momenta[1][1], momenta[1][2], momenta[1][3]};
        double p3[4] = {momenta[2][0], momenta[2][1], momenta[2][2], momenta[2][3]};
        
        // Call your physics function (must be KOKKOS_INLINE_FUNCTION if on GPU)
        return myMatrixElement(p1, p2, p3) * scale;
    }
};

// Use it:
MyIntegrand integrand(1.0);
rambo::RamboIntegrator<MyIntegrand, 3> integrator(nEvents, integrand);
```

**Note**: For GPU execution, any function called from `evaluate()` must also be decorated with `KOKKOS_INLINE_FUNCTION`.

## Running the Example

```bash
./rambo_kokkos [num_events] [seed]
./rambo_kokkos 10000000 5489
```

## Notes

- User must call `Kokkos::initialize()` / `Kokkos::finalize()`
- Backend (CPU/GPU) is determined by Kokkos build configuration
- 4-momenta use metric signature (+,−,−,−)
