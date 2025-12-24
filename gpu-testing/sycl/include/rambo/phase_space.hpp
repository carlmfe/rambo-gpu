#pragma once
#ifndef RAMBO_SYCL_PHASE_SPACE_HPP
#define RAMBO_SYCL_PHASE_SPACE_HPP

#include <sycl/sycl.hpp>
#include <cstdint>

namespace rambo {

// =============================================================================
// Random Number Generation (XorShift64)
// =============================================================================

inline uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

inline auto uniformRandom(uint64_t& state) -> double {
    return static_cast<double>(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

// =============================================================================
// RAMBO Algorithm Implementation
// =============================================================================

template <int nParticles>
struct RamboAlgorithm {
    static constexpr double tolerance = 1e-14;
    static constexpr int maxIterations = 1000;
    
    auto generate(double cmEnergy, const double* masses, 
                  uint64_t& rngState, double momenta[][4]) const -> double {
        double q[nParticles][4];
        double p[nParticles][4];
        double zCoeff[nParticles];
        double totalMom[4];
        double boostVec[3];
        double momSq[nParticles];
        double massSq[nParticles];
        double energies[nParticles];
        double virtMom[nParticles];
        
        const double twoPi = 8.0 * sycl::atan(1.0);
        const double logPiOver2 = sycl::log(twoPi / 4.0);
        
        if (nParticles < 1 || nParticles > 10) return 0.0;
        
        zCoeff[0] = 0.0;
        if (nParticles > 1) {
            zCoeff[1] = 0.0;
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k-1] + logPiOver2 - 2.0 * sycl::log(static_cast<double>(k-1));
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k] - sycl::log(static_cast<double>(k));
        }
        
        double totalMass = 0.0;
        int nMassive = 0;
        for (int i = 0; i < nParticles; i++) {
            if (masses[i] != 0.0) nMassive++;
            totalMass += sycl::fabs(masses[i]);
        }
        
        if (totalMass > cmEnergy) return 0.0;
        
        for (int i = 0; i < nParticles; i++) {
            double rand1 = uniformRandom(rngState);
            double cosTheta = 2.0 * rand1 - 1.0;
            double sinTheta = sycl::sqrt(1.0 - cosTheta * cosTheta);
            double phi = twoPi * uniformRandom(rngState);
            rand1 = uniformRandom(rngState);
            double rand2 = uniformRandom(rngState);
            
            q[i][0] = -sycl::log(rand1 * rand2);
            q[i][3] = q[i][0] * cosTheta;
            q[i][2] = q[i][0] * sinTheta * sycl::cos(phi);
            q[i][1] = q[i][0] * sinTheta * sycl::sin(phi);
        }
        
        for (int mu = 0; mu < 4; mu++) totalMom[mu] = 0.0;
        for (int i = 0; i < nParticles; i++) {
            for (int mu = 0; mu < 4; mu++) {
                totalMom[mu] += q[i][mu];
            }
        }
        
        double invariantMass = sycl::sqrt(totalMom[0]*totalMom[0] - totalMom[1]*totalMom[1] 
                                         - totalMom[2]*totalMom[2] - totalMom[3]*totalMom[3]);
        for (int k = 0; k < 3; k++) {
            boostVec[k] = -totalMom[k+1] / invariantMass;
        }
        double gamma = totalMom[0] / invariantMass;
        double boostFactor = 1.0 / (1.0 + gamma);
        double scaleFactor = cmEnergy / invariantMass;
        
        for (int i = 0; i < nParticles; ++i) {
            double bDotQ = boostVec[0]*q[i][1] + boostVec[1]*q[i][2] + boostVec[2]*q[i][3];
            for (int k = 1; k < 4; ++k) {
                p[i][k] = scaleFactor * (q[i][k] + boostVec[k-1] * (q[i][0] + boostFactor * bDotQ));
            }
            p[i][0] = scaleFactor * (gamma * q[i][0] + bDotQ);
        }
        
        double logWeight = logPiOver2;
        if (nParticles != 2) {
            logWeight += (2.0 * nParticles - 4.0) * sycl::log(cmEnergy) + zCoeff[nParticles-1];
        }
        
        if (nMassive == 0) {
            for (int i = 0; i < nParticles; ++i) {
                for (int mu = 0; mu < 4; ++mu) {
                    momenta[i][mu] = p[i][mu];
                }
            }
            return logWeight;
        }
        
        double xMax = sycl::sqrt(1.0 - (totalMass * totalMass) / (cmEnergy * cmEnergy));
        for (int i = 0; i < nParticles; ++i) {
            massSq[i] = masses[i] * masses[i];
            momSq[i] = p[i][0] * p[i][0];
        }
        
        int iteration = 0;
        double x = xMax;
        double accuracyGoal = cmEnergy * tolerance;
        
        while (true) {
            double f = -cmEnergy;
            double df = 0.0;
            double x2 = x * x;
            for (int i = 0; i < nParticles; ++i) {
                energies[i] = sycl::sqrt(massSq[i] + x2 * momSq[i]);
                f += energies[i];
                df += momSq[i] / energies[i];
            }
            if (sycl::fabs(f) <= accuracyGoal) break;
            if (++iteration > maxIterations) break;
            x = x - f / (df * x);
        }
        
        for (int i = 0; i < nParticles; i++) {
            virtMom[i] = x * p[i][0];
            for (int k = 1; k < 4; k++) {
                p[i][k] *= x;
            }
            p[i][0] = energies[i];
        }
        
        double weightProduct = 1.0;
        double weightSum = 0.0;
        for (int i = 0; i < nParticles; i++) {
            weightProduct *= virtMom[i] / energies[i];
            weightSum += (virtMom[i] * virtMom[i]) / energies[i];
        }
        double logWeightMassive = (2.0 * nParticles - 3.0) * sycl::log(x) 
                                 + sycl::log(weightProduct / weightSum * cmEnergy);
        
        logWeight += logWeightMassive;
        
        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                momenta[i][mu] = p[i][mu];
            }
        }
        
        return logWeight;
    }
};

// =============================================================================
// Phase Space Generator (Wrapper)
// =============================================================================

template <int nParticles, typename Algorithm = RamboAlgorithm<nParticles>>
struct PhaseSpaceGenerator {
    double cmEnergy;
    const double* masses;
    Algorithm algorithm;
    
    PhaseSpaceGenerator(double energy, const double* m) 
        : cmEnergy(energy), masses(m), algorithm() {}
    
    auto operator()(uint64_t& rngState, double momenta[][4]) const -> double {
        return algorithm.generate(cmEnergy, masses, rngState, momenta);
    }
};

template <int nParticles>
using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

} // namespace rambo

#endif // RAMBO_SYCL_PHASE_SPACE_HPP
