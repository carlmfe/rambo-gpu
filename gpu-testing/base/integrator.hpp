#pragma once

#include <cstdint>
#include <cmath>

#include "rambo_base.hpp"

// =============================================================================
// Result structure
// =============================================================================

struct IntegrationResult {
    double mean = 0.0;
    double error = 0.0;
    double sum = 0.0;
    double sumSquared = 0.0;
    int64_t nEvents = 0;
    
    void computeStatistics() {
        mean = sum / static_cast<double>(nEvents);
        double variance = (sumSquared / static_cast<double>(nEvents)) - (mean * mean);
        error = std::sqrt(std::fabs(variance) / static_cast<double>(nEvents));
    }
};

// =============================================================================
// Integrator Class
// =============================================================================

// Integrand-agnostic Monte Carlo integrator using serial CPU execution
// Integrand must have: auto evaluate(const double momenta[][4]) const -> double
template <typename Integrand, int NumParticles>
class RamboIntegrator {
public:
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    // Main entry point
    void run(double cmEnergy, const double* masses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        // Launch Monte Carlo kernel
        launchMonteCarloKernel(cmEnergy, masses, result.sum, result.sumSquared, seed);
        
        // Compute statistics
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }

private:
    int64_t nEvents_;
    Integrand integrand_;
    
    // Launch Monte Carlo kernel (serial execution)
    void launchMonteCarloKernel(double cmEnergy, const double* masses,
                                 double& sum, double& sumSquared, uint64_t seed) {
        
        // Create phase space generator
        PhaseSpaceGenerator<NumParticles> generator(cmEnergy, masses);
        
        // Initialize RNG state (using XorShift64 from rambo_base.hpp)
        uint64_t rngState = seed;
        if (rngState == 0) rngState = 1;  // Avoid zero state
        
        // Accumulators
        double localSum = 0.0;
        double localSum2 = 0.0;
        
        // Local momenta storage
        double momenta[NumParticles][4];
        
        // Main Monte Carlo loop (serial)
        for (int64_t i = 0; i < nEvents_; ++i) {
            // Generate phase space point
            double logWeight = generator(rngState, momenta);
            
            // Evaluate integrand and compute weighted contribution
            double fx = integrand_.evaluate(momenta);
            double weightedValue = fx * std::exp(logWeight);
            
            // Accumulate
            localSum += weightedValue;
            localSum2 += weightedValue * weightedValue;
        }
        
        sum = localSum;
        sumSquared = localSum2;
    }
};
