#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "rambo_kokkos.hpp"

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

// Integrand-agnostic Monte Carlo integrator using Kokkos
// Integrand must have: KOKKOS_INLINE_FUNCTION auto evaluate(const double momenta[][4]) const -> double
template <typename Integrand, int NumParticles>
class RamboIntegrator {
public:
    using RngPool = Kokkos::Random_XorShift64_Pool<>;
    
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    // Main entry point
    void run(double cmEnergy, const double* hostMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        // Allocate device memory for masses
        Kokkos::View<double*> deviceMasses("masses", NumParticles);
        
        // Copy masses to device
        copyMassesToDevice(hostMasses, deviceMasses);
        
        // Launch Monte Carlo kernel with built-in reduction
        launchMonteCarloKernel(cmEnergy, deviceMasses, result.sum, result.sumSquared, seed);
        
        // Compute statistics
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
    // Overload accepting Kokkos::View for backwards compatibility
    void run(double cmEnergy, Kokkos::View<double*> deviceMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        // Launch Monte Carlo kernel with built-in reduction
        launchMonteCarloKernel(cmEnergy, deviceMasses, result.sum, result.sumSquared, seed);
        
        // Compute statistics
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
    // Copy masses array to device (public for CUDA lambda access)
    void copyMassesToDevice(const double* hostMasses, Kokkos::View<double*>& deviceMasses) {
        auto hostMassesMirror = Kokkos::create_mirror_view(deviceMasses);
        for (int i = 0; i < NumParticles; ++i) {
            hostMassesMirror(i) = hostMasses[i];
        }
        Kokkos::deep_copy(deviceMasses, hostMassesMirror);
    }
    
    // Launch Monte Carlo kernel (public for CUDA lambda access)
    void launchMonteCarloKernel(double cmEnergy, Kokkos::View<double*> deviceMasses,
                                 double& sum, double& sumSquared, uint64_t seed) {
        
        RngPool rngPool(seed);
        const Integrand integrandCopy = integrand_;  // Capture for device lambda
        const int64_t nEvents = nEvents_;
        
        double sumVal = 0.0;
        double sum2Val = 0.0;
        
        Kokkos::parallel_reduce("MonteCarloKernel", nEvents,
            KOKKOS_LAMBDA(const int64_t idx, double& localSum, double& localSum2) {
                // Get thread-local RNG state
                auto rng = rngPool.get_state();
                
                // Copy masses to local array
                double massesLocal[NumParticles];
                for (int j = 0; j < NumParticles; ++j) {
                    massesLocal[j] = deviceMasses(j);
                }
                
                // Generate phase space point
                double momenta[NumParticles][4];
                PhaseSpaceGenerator<NumParticles> rambo(cmEnergy, massesLocal);
                double logWeight = rambo(rng, momenta);
                
                // Evaluate integrand and compute weighted contribution
                double fx = integrandCopy.evaluate(momenta);
                double weightedValue = fx * Kokkos::exp(logWeight);
                
                // Accumulate for reduction
                localSum += weightedValue;
                localSum2 += weightedValue * weightedValue;
                
                // Return RNG state to pool
                rngPool.free_state(rng);
            },
            Kokkos::Sum<double>(sumVal),
            Kokkos::Sum<double>(sum2Val)
        );
        
        sum = sumVal;
        sumSquared = sum2Val;
    }

private:
    int64_t nEvents_;
    Integrand integrand_;
};
