# RAMBO Monte Carlo Integrator - Base (Serial CPU)

Header-only, serial CPU reference implementation. No external dependencies.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, MSVC 2019+)

## Installation

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

## Usage

### CMake
```cmake
find_package(rambo-base REQUIRED)
target_link_libraries(my_app PRIVATE rambo::base)
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

```cpp
struct MyIntegrand {
    double scale;
    MyIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle, mu = 0:E, 1:px, 2:py, 3:pz
        return myFunction(momenta[0], momenta[1]) * scale;
    }
};
```

## Running the Example

```bash
./rambo_base [num_events] [seed]
./rambo_base 1000000 5489
```

## Notes

- Serial execution (single-threaded)
- Uses XorShift64 RNG
- 4-momenta use metric signature (+,−,−,−)
