#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "rambo_sycl.hpp"

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

// Integrand-agnostic Monte Carlo integrator using SYCL
// Integrand must have: auto evaluate(const double momenta[][4]) const -> double
template <typename Integrand, int NumParticles>
class RamboIntegrator {
public:
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    // Main entry point
    void run(double cmEnergy, const double* hostMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        // Create SYCL queue with GPU selector
        sycl::queue queue{sycl::gpu_selector_v};
        
        // Allocate device memory using USM
        double* deviceMasses = sycl::malloc_device<double>(NumParticles, queue);
        double* deviceSum = sycl::malloc_device<double>(1, queue);
        double* deviceSum2 = sycl::malloc_device<double>(1, queue);
        
        // Copy masses to device and initialize sums
        queue.memcpy(deviceMasses, hostMasses, sizeof(double) * NumParticles);
        queue.memset(deviceSum, 0, sizeof(double));
        queue.memset(deviceSum2, 0, sizeof(double));
        queue.wait();
        
        // Launch Monte Carlo kernel
        launchMonteCarloKernel(queue, cmEnergy, deviceMasses, deviceSum, deviceSum2, seed);
        
        // Copy results back
        queue.memcpy(&result.sum, deviceSum, sizeof(double));
        queue.memcpy(&result.sumSquared, deviceSum2, sizeof(double));
        queue.wait();
        
        // Free device memory
        sycl::free(deviceMasses, queue);
        sycl::free(deviceSum, queue);
        sycl::free(deviceSum2, queue);
        
        // Compute statistics
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
private:
    int64_t nEvents_;
    Integrand integrand_;
    
    // Launch Monte Carlo kernel using SYCL parallel_for
    void launchMonteCarloKernel(sycl::queue& queue, double cmEnergy, 
                                 double* deviceMasses, double* deviceSum, 
                                 double* deviceSum2, uint64_t baseSeed) {
        
        // Get device properties for workgroup sizing
        auto device = queue.get_device();
        size_t maxWorkGroupSize = device.get_info<sycl::info::device::max_work_group_size>();
        size_t workGroupSize = std::min(maxWorkGroupSize, size_t(256));
        
        // Round up nEvents to multiple of workGroupSize for efficient execution
        size_t globalSize = ((nEvents_ + workGroupSize - 1) / workGroupSize) * workGroupSize;
        
        // Capture integrand and parameters by value for the kernel
        Integrand integrand = integrand_;
        int64_t nEvents = nEvents_;
        
        // Submit kernel
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>{globalSize, workGroupSize},
                [=](sycl::nd_item<1> item) {
                    int64_t idx = item.get_global_id(0);
                    
                    // Initialize RNG state per thread with hash-like mixing
                    // to avoid correlation between adjacent threads
                    uint64_t rngState = baseSeed ^ (static_cast<uint64_t>(idx) * 2685821657736338717ULL);
                    if (rngState == 0) rngState = baseSeed + 1;  // Avoid zero state
                    
                    double localSum = 0.0;
                    double localSum2 = 0.0;
                    
                    // Grid-stride loop for handling more events than threads
                    int64_t gridSize = item.get_global_range(0);
                    while (idx < nEvents) {
                        // Generate phase space point
                        double momenta[NumParticles][4];
                        PhaseSpaceGenerator<NumParticles> rambo(cmEnergy, deviceMasses);
                        double logWeight = rambo(rngState, momenta);
                        
                        // Evaluate integrand and compute weighted contribution
                        double fx = integrand.evaluate(momenta);
                        double weightedValue = fx * sycl::exp(logWeight);
                        
                        // Accumulate locally
                        localSum += weightedValue;
                        localSum2 += weightedValue * weightedValue;
                        
                        idx += gridSize;
                    }
                    
                    // Atomic add to global sum using SYCL atomics
                    auto sumRef = sycl::atomic_ref<double, 
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*deviceSum);
                    sumRef.fetch_add(localSum);
                    
                    auto sum2Ref = sycl::atomic_ref<double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*deviceSum2);
                    sum2Ref.fetch_add(localSum2);
                });
        });
        
        queue.wait();
    }
};
