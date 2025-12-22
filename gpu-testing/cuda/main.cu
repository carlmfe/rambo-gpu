// RAMBO Monte Carlo Integrator using pure CUDA
// This implementation is agnostic to the integrand kernel
//
// Build:
//   mkdir build && cd build
//   cmake ..
//   make
//
// Or directly with nvcc:
//   nvcc -O3 -o rambo_cuda main.cu

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>

#include "integrator.cuh"
#include "integrands.cuh"

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
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "======================================" << std::endl;
    std::cout << "RAMBO Monte Carlo Integrator (CUDA)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: CUDA GPU" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
    std::cout << std::endl;
    
    // Set up particle masses (massless particles)
    double masses[nParticles] = {0.0, 0.0, 0.0};
    
    // Create integrand
    EggholderIntegrand integrand(1.0);
    
    // Run benchmark
    runBenchmark<EggholderIntegrand, nParticles>(
        "CUDA GPU", nEvents, cmEnergy, masses, integrand, seed);
    
    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}
