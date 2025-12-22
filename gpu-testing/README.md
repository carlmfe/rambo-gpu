# RAMBO GPU Testing

Monte Carlo phase space integration using the RAMBO algorithm, implemented across multiple GPU frameworks for performance comparison.

## Project Structure

```
gpu-testing/
├── base/                 # Serial CPU reference implementation (C++17)
│   ├── rambo_base.hpp    # Phase space generator
│   ├── integrands.hpp    # Integrand definitions
│   ├── integrator.hpp    # Monte Carlo integrator
│   └── main.cpp          # Entry point
│
├── kokkos/               # Kokkos implementation (CUDA/OpenMP)
│   ├── rambo_kokkos.hpp
│   ├── integrands.hpp
│   ├── integrator.hpp
│   └── main.cpp
│
├── alpaka/               # Alpaka 2.0.0 implementation (CUDA/CPU/OpenMP)
│   ├── rambo_alpaka.hpp
│   ├── integrands.hpp
│   ├── integrator.hpp
│   └── main.cpp
│
├── cuda/                 # Pure CUDA implementation
│   ├── rambo_cuda.cuh
│   ├── integrands.cuh
│   ├── integrator.cuh
│   └── main.cu
│
├── benchmark.sh          # Performance comparison script
└── check_gpu.sh          # GPU utilization verification
```

## The Eggholder Integrand

The Eggholder integrand is a physics-inspired test function computed from Lorentz-invariant quantities. For a 3-particle final state with 4-momenta $p_1$, $p_2$, $p_3$:

### Mandelstam-like Invariants

$$s_{ij} = (p_i - p_j)^2 = (E_i - E_j)^2 - |\vec{p}_i - \vec{p}_j|^2$$

Using the Minkowski metric signature $(+,-,-,-)$:

$$s_{12} = (p_1 - p_2)^\mu (p_1 - p_2)_\mu$$
$$s_{13} = (p_1 - p_3)^\mu (p_1 - p_3)_\mu$$
$$s_{23} = (p_2 - p_3)^\mu (p_2 - p_3)_\mu$$

### Integrand Function

$$f(p_1, p_2, p_3) = \sin\left(\sqrt{\frac{|s_{12} - s_{23}|}{\lambda^2}}\right) \cdot \cos\left(\sqrt{\frac{|s_{13}|}{\lambda^2}}\right)$$

where $\lambda^2$ is a scale parameter (default: $1000^2 = 10^6$).

This integrand:
- Is Lorentz-invariant (depends only on invariant masses)
- Has oscillatory behavior testing numerical precision
- Produces non-trivial values across phase space

## Shell Scripts

### benchmark.sh

Builds and benchmarks all implementations, comparing throughput.

```bash
# Basic usage (10M events, 3 runs each)
./benchmark.sh

# Custom event count
./benchmark.sh 100000000

# Custom event count and seed
./benchmark.sh 100000000 42

# Custom runs
./benchmark.sh 10000000 5489 5
```

**Output includes:**
- Build status for each implementation
- GPU utilization verification
- Throughput comparison (events/sec)
- Relative performance ranking

### check_gpu.sh

Verifies that a program actually uses the GPU during execution.

```bash
# Check GPU utilization for any executable
./check_gpu.sh ./alpaka/build/rambo_alpaka 10000000 5489
```

**Output includes:**
- Initial GPU state
- Maximum GPU utilization during execution
- Confirmation message: `✓ GPU WAS UTILIZED` or `✗ GPU was NOT utilized`

## Quick Start

```bash
# Build all implementations
cd gpu-testing
for dir in base kokkos alpaka cuda; do
    cd $dir && mkdir -p build && cd build && cmake .. && make -j4 && cd ../..
done

# Run benchmark comparison
./benchmark.sh 10000000

# Verify GPU usage
./check_gpu.sh ./cuda/build/rambo_cuda 10000000 5489
```

## Performance Reference

| Implementation | Throughput | Hardware |
|---------------|------------|----------|
| Base (Serial) | ~3M ev/s | CPU single-thread |
| Kokkos (CUDA) | ~85M ev/s | RTX 2000 Ada |
| Alpaka (CUDA) | ~115M ev/s | RTX 2000 Ada |
| CUDA | ~140M ev/s | RTX 2000 Ada |

*Results vary by hardware and event count.*
