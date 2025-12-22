// =============================================================================
// RAMBO Monte Carlo Integration - Base Serial Implementation
// =============================================================================
// Reference implementation using only standard C++ (no parallelization).
// Useful for validation and as a performance baseline.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <algorithm>

#include "rambo_base.hpp"
#include "integrands.hpp"
#include "integrator.hpp"

// =============================================================================
// Benchmark helper
// =============================================================================
template <typename Integrand, int nParticles>
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
        RamboIntegrator<Integrand, nParticles> warmup(
            std::min(nEvents / 10, int64_t(10000)), integrand);
        warmup.run(cmEnergy, masses, mean, error, seed);
    }
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    RamboIntegrator<Integrand, nParticles> integrator(nEvents, integrand);
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
    std::cout << "RAMBO Monte Carlo Integrator (Base)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: CPU Serial" << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
    std::cout << std::endl;
    
    // Set up particle masses (non-zero for testing massive RAMBO)
    // Using realistic masses: ~electron, ~muon, ~pion scale in GeV
    double masses[nParticles] = {0.5, 100.0, 140.0};
    
    // Create integrand (uses default lambdaSquared = 1000^2)
    EggholderIntegrand integrand;
    
    // Run benchmark
    runBenchmark<EggholderIntegrand, nParticles>(
        "CPU Serial", nEvents, cmEnergy, masses, integrand, seed);
    
    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}