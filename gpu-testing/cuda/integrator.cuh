#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "rambo_cuda.cuh"

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
// Monte Carlo Kernel
// =============================================================================

template <typename Integrand, int NumParticles>
__global__ void MonteCarloKernel(
    double cmEnergy,
    const double* masses,
    Integrand integrand,
    double* deviceSum,
    double* deviceSum2,
    uint64_t baseSeed,
    int64_t nEvents
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize RNG state per thread with better seed mixing
    // Use hash-like mixing to avoid correlation between adjacent threads
    uint64_t rngState = baseSeed ^ (uint64_t(idx) * 2685821657736338717ULL);
    if (rngState == 0) rngState = baseSeed + 1;  // Avoid zero state
    
    double localSum = 0.0;
    double localSum2 = 0.0;
    
    // Grid-stride loop for handling more events than threads
    while (idx < nEvents) {
        // Generate phase space point
        double momenta[NumParticles][4];
        PhaseSpaceGenerator<NumParticles> rambo(cmEnergy, masses);
        double logWeight = rambo(rngState, momenta);
        
        // Evaluate integrand and compute weighted contribution
        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * exp(logWeight);
        
        // Accumulate locally
        localSum += weightedValue;
        localSum2 += weightedValue * weightedValue;
        
        idx += gridDim.x * blockDim.x;
    }
    
    // Atomic add to global sum
    atomicAdd(deviceSum, localSum);
    atomicAdd(deviceSum2, localSum2);
}

// =============================================================================
// Integrator Class
// =============================================================================

// Integrand-agnostic Monte Carlo integrator using CUDA
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
        
        // Get device properties for kernel configuration
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Configure kernel launch
        int blockSize = 256;
        int numBlocks = std::min(
            static_cast<int>((nEvents_ + blockSize - 1) / blockSize),
            prop.maxGridSize[0]
        );
        
        // Allocate device memory
        double* deviceMasses = nullptr;
        double* deviceSum = nullptr;
        double* deviceSum2 = nullptr;
        
        cudaMalloc(&deviceMasses, sizeof(double) * NumParticles);
        cudaMalloc(&deviceSum, sizeof(double));
        cudaMalloc(&deviceSum2, sizeof(double));
        
        // Copy masses to device and initialize sums
        copyMassesToDevice(hostMasses, deviceMasses);
        cudaMemset(deviceSum, 0, sizeof(double));
        cudaMemset(deviceSum2, 0, sizeof(double));
        
        // Launch kernel
        launchMonteCarloKernel(cmEnergy, deviceMasses, deviceSum, deviceSum2, 
                               seed, numBlocks, blockSize);
        
        // Synchronize and check for errors
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Copy results back
        result.sum = copyScalarFromDevice(deviceSum);
        result.sumSquared = copyScalarFromDevice(deviceSum2);
        
        // Free device memory
        cudaFree(deviceMasses);
        cudaFree(deviceSum);
        cudaFree(deviceSum2);
        
        // Compute statistics
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
private:
    int64_t nEvents_;
    Integrand integrand_;
    
    // Copy masses array to device
    void copyMassesToDevice(const double* hostMasses, double* deviceMasses) {
        cudaMemcpy(deviceMasses, hostMasses, sizeof(double) * NumParticles, 
                   cudaMemcpyHostToDevice);
    }
    
    // Launch Monte Carlo kernel
    void launchMonteCarloKernel(double cmEnergy, double* deviceMasses,
                                 double* deviceSum, double* deviceSum2,
                                 uint64_t seed, int numBlocks, int blockSize) {
        MonteCarloKernel<Integrand, NumParticles><<<numBlocks, blockSize>>>(
            cmEnergy,
            deviceMasses,
            integrand_,
            deviceSum,
            deviceSum2,
            seed,
            nEvents_
        );
    }
    
    // Copy single scalar result from device
    double copyScalarFromDevice(double* devicePtr) {
        double hostValue;
        cudaMemcpy(&hostValue, devicePtr, sizeof(double), cudaMemcpyDeviceToHost);
        return hostValue;
    }
};
