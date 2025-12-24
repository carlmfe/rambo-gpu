#pragma once
#ifndef RAMBO_ALPAKA_INTEGRATOR_HPP
#define RAMBO_ALPAKA_INTEGRATOR_HPP

#include <alpaka/alpaka.hpp>
#include <cmath>
#include <cstdint>
#include <concepts>

#include "phase_space.hpp"

namespace rambo {

// =============================================================================
// Concepts
// =============================================================================

template <typename T>
concept IntegrandConcept = requires(T integrand, const double momenta[][4]) {
    { integrand.evaluate(momenta) } -> std::convertible_to<double>;
};

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
// Kernels
// =============================================================================

template <IntegrandConcept Integrand, int NumParticles>
struct MonteCarloKernel {
    double energy;
    const double* masses;
    Integrand integrand;
    double* sums;
    double* sumsSquared;
    uint64_t baseSeed;
    int64_t nEvents;
    
    ALPAKA_FN_HOST_ACC MonteCarloKernel(
        double e, const double* m, const Integrand& integ,
        double* s, double* s2, uint64_t seed, int64_t n)
        : energy(e), masses(m), integrand(integ),
          sums(s), sumsSquared(s2), baseSeed(seed), nEvents(n) {}
    
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const linearIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];
        
        if (static_cast<int64_t>(linearIdx) >= nEvents) return;
        
        auto engine = alpaka::rand::engine::createDefault(
            acc, 
            static_cast<uint32_t>(baseSeed + linearIdx),
            static_cast<uint32_t>((baseSeed + linearIdx) >> 32)
        );
        
        alpaka::rand::RandDefault rand;
        auto dist = alpaka::rand::distribution::createUniformReal<double>(rand);
        
        double momenta[NumParticles][4];
        PhaseSpaceGenerator<NumParticles> rambo(energy, masses);
        double logWeight = rambo(engine, dist, momenta);
        
        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * std::exp(logWeight);
        
        sums[linearIdx] = weightedValue;
        sumsSquared[linearIdx] = weightedValue * weightedValue;
    }
};

struct ReductionKernel {
    double* data;
    int64_t size;
    int64_t stride;
    
    ALPAKA_FN_HOST_ACC ReductionKernel(double* d, int64_t n, int64_t s)
        : data(d), size(n), stride(s) {}
    
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const idx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];
        
        int64_t i = static_cast<int64_t>(idx) * stride * 2;
        int64_t j = i + stride;
        
        if (j < size) {
            data[i] += data[j];
        }
    }
};

// =============================================================================
// Integrator Class
// =============================================================================

template <typename TAccTag, IntegrandConcept Integrand, int NumParticles>
class RamboIntegrator {
public:
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    using DevAcc = alpaka::Dev<Acc>;
    using DevHost = alpaka::DevCpu;
    
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    void run(double energy, const double* hostMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        auto const platformAcc = alpaka::Platform<Acc>{};
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);
        Queue queue(devAcc);
        
        auto deviceMasses = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(NumParticles));
        auto deviceSums = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(nEvents_));
        auto deviceSums2 = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(nEvents_));
        
        copyMassesToDevice(devHost, queue, hostMasses, deviceMasses);
        launchMonteCarloKernel(devAcc, queue, energy, deviceMasses, deviceSums, deviceSums2, seed);
        performReduction(devAcc, queue, deviceSums, deviceSums2);
        
        result.sum = copyScalarFromDevice(devHost, queue, deviceSums);
        result.sumSquared = copyScalarFromDevice(devHost, queue, deviceSums2);
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
private:
    int64_t nEvents_;
    Integrand integrand_;
    
    template <typename MassBuf>
    void copyMassesToDevice(const DevHost& devHost, Queue& queue, 
                            const double* hostMasses, MassBuf& deviceMasses) {
        auto hostMassesBuf = alpaka::allocBuf<double, Idx>(devHost, static_cast<Idx>(NumParticles));
        for (int i = 0; i < NumParticles; ++i) {
            hostMassesBuf[static_cast<Idx>(i)] = hostMasses[i];
        }
        alpaka::memcpy(queue, deviceMasses, hostMassesBuf);
        alpaka::wait(queue);
    }
    
    template <typename MassBuf, typename SumBuf>
    void launchMonteCarloKernel(const DevAcc& devAcc, Queue& queue, double energy,
                                 MassBuf& deviceMasses, SumBuf& deviceSums, 
                                 SumBuf& deviceSums2, uint64_t seed) {
        alpaka::Vec<Dim, Idx> const extent{static_cast<Idx>(nEvents_)};
        
        MonteCarloKernel<Integrand, NumParticles> kernel(
            energy,
            alpaka::getPtrNative(deviceMasses),
            integrand_,
            alpaka::getPtrNative(deviceSums),
            alpaka::getPtrNative(deviceSums2),
            seed,
            nEvents_
        );
        
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, alpaka::Vec<Dim, Idx>{1}};
        auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, kernel);
        alpaka::exec<Acc>(queue, workDiv, kernel);
        alpaka::wait(queue);
    }
    
    template <typename SumBuf>
    void performReduction(const DevAcc& devAcc, Queue& queue, 
                          SumBuf& deviceSums, SumBuf& deviceSums2) {
        double* sumsPtr = alpaka::getPtrNative(deviceSums);
        double* sums2Ptr = alpaka::getPtrNative(deviceSums2);
        
        int64_t stride = 1;
        while (stride < nEvents_) {
            int64_t numThreads = (nEvents_ + stride * 2 - 1) / (stride * 2);
            
            ReductionKernel reduceKernel1(sumsPtr, nEvents_, stride);
            ReductionKernel reduceKernel2(sums2Ptr, nEvents_, stride);
            
            alpaka::Vec<Dim, Idx> const reduceExtent{static_cast<Idx>(numThreads)};
            alpaka::KernelCfg<Acc> const reduceCfg = {reduceExtent, alpaka::Vec<Dim, Idx>{1}};
            auto const reduceWorkDiv = alpaka::getValidWorkDiv(reduceCfg, devAcc, reduceKernel1);
            
            alpaka::exec<Acc>(queue, reduceWorkDiv, reduceKernel1);
            alpaka::exec<Acc>(queue, reduceWorkDiv, reduceKernel2);
            alpaka::wait(queue);
            
            stride *= 2;
        }
    }
    
    template <typename SumBuf>
    double copyScalarFromDevice(const DevHost& devHost, Queue& queue, SumBuf& deviceBuf) {
        auto hostResult = alpaka::allocBuf<double, Idx>(devHost, static_cast<Idx>(1));
        alpaka::memcpy(queue, hostResult, deviceBuf, static_cast<Idx>(1));
        alpaka::wait(queue);
        return hostResult[0];
    }
};

} // namespace rambo

#endif // RAMBO_ALPAKA_INTEGRATOR_HPP
