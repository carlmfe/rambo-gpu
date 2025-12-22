// RAMBO Monte Carlo Integrator using Alpaka 2.0.0
// This implementation is agnostic to the integrand kernel and supports multiple backends
//
// Build with different backends:
//   cmake -DALPAKA_BACKEND=CUDA ..   # NVIDIA GPU (default)
//   cmake -DALPAKA_BACKEND=CPU ..    # CPU serial
//   cmake -DALPAKA_BACKEND=OMP ..    # OpenMP parallel

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>

#include <alpaka/alpaka.hpp>

#include "integrator.hpp"
#include "integrands.hpp"

// =============================================================================
// Backend Selection (compile-time)
// =============================================================================
#if defined(ALPAKA_USE_CUDA)
    using DefaultAccTag = alpaka::TagGpuCudaRt;
    constexpr const char* BACKEND_NAME = "CUDA GPU";
#elif defined(ALPAKA_USE_CPU_OMP)
    using DefaultAccTag = alpaka::TagCpuOmp2Blocks;
    constexpr const char* BACKEND_NAME = "CPU OpenMP";
#elif defined(ALPAKA_USE_CPU_SERIAL)
    using DefaultAccTag = alpaka::TagCpuSerial;
    constexpr const char* BACKEND_NAME = "CPU Serial";
#else
    // Fallback: use first available tag
    using DefaultAccTag = std::tuple_element_t<0, alpaka::EnabledAccTags>;
    constexpr const char* BACKEND_NAME = "Auto-detected";
#endif

// =============================================================================
// Benchmark helper
// =============================================================================
template <typename AccTag, IntegrandConcept Integrand, int nParticles>
void runBenchmark(const std::string& backendName, 
                  int64_t nEvents, 
                  double cmEnergy, 
                  const double* masses,
                  const Integrand& integrand,
                  uint64_t seed) {
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Backend: " << backendName << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    double mean = 0.0;
    double error = 0.0;
    
    // Warmup run (smaller)
    {
        RamboIntegrator<AccTag, Integrand, nParticles> warmup(
            std::min(nEvents / 10, int64_t(10000)), integrand);
        warmup.run(cmEnergy, masses, mean, error, seed);
    }
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    RamboIntegrator<AccTag, Integrand, nParticles> integrator(nEvents, integrand);
    integrator.run(cmEnergy, masses, mean, error, seed);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = duration.count() / 1000.0;
    
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Error: " << error << std::endl;
    std::cout << "  Time: " << timeMs << " ms" << std::endl;
    std::cout << "  Throughput: " << (nEvents / timeMs * 1000.0) << " events/sec" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse command line arguments
    const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 100000;
    const uint64_t seed = (argc > 2) ? std::stoull(argv[2]) : 5489ULL;
    const double cmEnergy = 1000.0;  // Center-of-mass energy in GeV
    constexpr int nParticles = 3;
    
    std::cout << "======================================" << std::endl;
    std::cout << "RAMBO Monte Carlo Integrator (Alpaka)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: " << BACKEND_NAME << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
    std::cout << std::endl;
    
    // Print available accelerators
    std::cout << "Available accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    std::cout << std::endl;
    
    // Set up particle masses (massless particles)
    double masses[nParticles] = {0.0, 0.0, 0.0};
    
    // Create integrand
    EggholderIntegrand integrand(1.0);
    
    // Run benchmark with compiled backend
    runBenchmark<DefaultAccTag, EggholderIntegrand, nParticles>(
        BACKEND_NAME, nEvents, cmEnergy, masses, integrand, seed);
    
    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}
