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
    const double cmEnergy = 91.2;  // Z-pole energy in GeV (classic Drell-Yan)
    constexpr int nParticles = 2;   // e+ e- final state
    
    std::cout << "======================================" << std::endl;
    std::cout << "RAMBO Monte Carlo Integrator (Base)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: CPU Serial" << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
    std::cout << std::endl;
    
    // Set up particle masses for e+ e- (electron mass in GeV)
    // Using massless approximation for high-energy limit
    constexpr double electronMass = 0.000511;  // GeV
    double masses[nParticles] = {electronMass, electronMass};
    
    // Create Drell-Yan integrand
    // Using up-quark charge (2/3) as default
    const double quarkCharge = 2.0 / 3.0;  // up-quark
    const double alphaEM = 1.0 / 137.035999;
    DrellYanIntegrand integrand(quarkCharge, alphaEM);
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Drell-Yan Process: q qbar -> gamma* -> e+ e-" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Quark charge (e_q): " << quarkCharge << std::endl;
    std::cout << "Fine structure constant (alpha): " << alphaEM << std::endl;
    std::cout << std::endl;
    
    // Run benchmark
    runBenchmark<DrellYanIntegrand, nParticles>(
        "CPU Serial", nEvents, cmEnergy, masses, integrand, seed);
    
    // ==========================================================================
    // Analytic verification
    // ==========================================================================
    std::cout << "========================================" << std::endl;
    std::cout << "Analytic Verification" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Analytic cross-section: sigma = 4*pi*alpha^2*e_q^2 / (3*s) * hbarc^2
    double s = cmEnergy * cmEnergy;  // GeV^2
    double analyticSigma = DrellYanIntegrand::analyticCrossSection(s, quarkCharge, alphaEM);
    
    // Convert to convenient units
    // 1 mb = 10^-27 cm^2, 1 nb = 10^-33 cm^2, 1 pb = 10^-36 cm^2
    double analyticSigma_nb = analyticSigma * 1e6;  // mb -> nb
    double analyticSigma_pb = analyticSigma * 1e9;  // mb -> pb
    
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Analytic cross-section:" << std::endl;
    std::cout << "  sigma = 4*pi*alpha^2*e_q^2 / (3*s) * hbarc^2" << std::endl;
    std::cout << "  s = " << s << " GeV^2" << std::endl;
    std::cout << "  sigma = " << analyticSigma << " mb" << std::endl;
    std::cout << "  sigma = " << analyticSigma_nb << " nb" << std::endl;
    std::cout << "  sigma = " << analyticSigma_pb << " pb" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Note: The Monte Carlo result should converge to the" << std::endl;
    std::cout << "analytic value as the number of events increases." << std::endl;
    std::cout << "The MC integral includes the phase space weight from RAMBO." << std::endl;
    std::cout << std::endl;
    
    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}