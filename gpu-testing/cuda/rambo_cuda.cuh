#pragma once

#include <cmath>
#include <cstdint>

// =============================================================================
// RAMBO Phase Space Generator for CUDA
// =============================================================================
// Generates nParticles 4-momenta with the given total center-of-mass energy
// and particle masses using the RAMBO algorithm.
//
// Reference: R. Kleiss, W.J. Stirling, S.D. Ellis, Comp. Phys. Comm. 40 (1986) 359
// =============================================================================

// XorShift64 RNG for CUDA
__device__ __forceinline__ uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

__device__ __forceinline__ double uniformRandom(uint64_t& state) {
    return (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

template <int nParticles>
struct PhaseSpaceGenerator {
    double cmEnergy;          // Center-of-mass energy
    const double* masses;     // Particle masses array
    
    __device__ PhaseSpaceGenerator(double energy, const double* m) 
        : cmEnergy(energy), masses(m) {}
    
    // Generate momenta and return the log of the phase space weight
    __device__ double operator()(uint64_t& rngState, double momenta[][4]) const {
        // Local arrays for intermediate calculations
        double q[nParticles][4];      // Isotropic momenta
        double p[nParticles][4];      // Boosted momenta
        double zCoeff[nParticles];    // Weight coefficients
        double totalMom[4];           // Total 4-momentum
        double boostVec[3];           // Boost vector
        
        // Arrays for massive particle rescaling
        double momSq[nParticles];     // |p|^2 for each particle
        double massSq[nParticles];    // m^2 for each particle
        double energies[nParticles];  // Energies after rescaling
        double virtMom[nParticles];   // Virtual momenta
        
        // Algorithm constants
        constexpr double tolerance = 1e-14;
        constexpr int maxIterations = 1000;
        const double twoPi = 8.0 * atan(1.0);
        const double logPiOver2 = log(twoPi / 4.0);
        
        // Validate particle count
        if (nParticles < 1 || nParticles > 10) return 0.0;
        
        // Pre-compute weight coefficients
        zCoeff[0] = 0.0;
        if (nParticles > 1) {
            zCoeff[1] = logPiOver2;
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k-1] + logPiOver2 - 2.0 * log(double(k-1));
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k] - log(double(k));
        }
        
        // Calculate total mass and count massive particles
        double totalMass = 0.0;
        int nMassive = 0;
        for (int i = 0; i < nParticles; i++) {
            if (masses[i] != 0.0) nMassive++;
            totalMass += fabs(masses[i]);
        }
        
        // Check kinematic threshold
        if (totalMass > cmEnergy) return 0.0;
        
        // Generate nParticles isotropic massless momenta
        for (int i = 0; i < nParticles; i++) {
            double rand1 = uniformRandom(rngState);
            double cosTheta = 2.0 * rand1 - 1.0;
            double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            double phi = twoPi * uniformRandom(rngState);
            rand1 = uniformRandom(rngState);
            double rand2 = uniformRandom(rngState);
            
            // Energy from exponential distribution
            q[i][0] = -log(rand1 * rand2);
            q[i][3] = q[i][0] * cosTheta;
            q[i][2] = q[i][0] * sinTheta * cos(phi);
            q[i][1] = q[i][0] * sinTheta * sin(phi);
        }
        
        // Calculate total 4-momentum
        for (int mu = 0; mu < 4; mu++) totalMom[mu] = 0.0;
        for (int i = 0; i < nParticles; i++) {
            for (int mu = 0; mu < 4; mu++) {
                totalMom[mu] += q[i][mu];
            }
        }
        
        // Compute boost parameters
        double invariantMass = sqrt(totalMom[0]*totalMom[0] - totalMom[1]*totalMom[1] 
                                   - totalMom[2]*totalMom[2] - totalMom[3]*totalMom[3]);
        for (int k = 0; k < 3; k++) {
            boostVec[k] = -totalMom[k+1] / invariantMass;
        }
        double gamma = totalMom[0] / invariantMass;
        double boostFactor = 1.0 / (1.0 + gamma);
        double scaleFactor = cmEnergy / invariantMass;
        
        // Apply Lorentz boost to scale to desired CM energy
        for (int i = 0; i < nParticles; ++i) {
            double bDotQ = boostVec[0]*q[i][1] + boostVec[1]*q[i][2] + boostVec[2]*q[i][3];
            for (int k = 1; k < 4; ++k) {
                p[i][k] = scaleFactor * (q[i][k] + boostVec[k-1] * (q[i][0] + boostFactor * bDotQ));
            }
            p[i][0] = scaleFactor * (gamma * q[i][0] + bDotQ);
        }
        
        // Calculate log-weight for massless case
        double logWeight = logPiOver2;
        if (nParticles != 2) {
            logWeight += (2.0 * nParticles - 4.0) * log(cmEnergy) + zCoeff[nParticles-1];
        }
        
        // Return early for massless particles
        if (nMassive == 0) {
            for (int i = 0; i < nParticles; ++i) {
                for (int mu = 0; mu < 4; ++mu) {
                    momenta[i][mu] = p[i][mu];
                }
            }
            return logWeight;
        }
        
        // Rescale momenta for massive particles using Newton-Raphson
        double xMax = sqrt(1.0 - (totalMass * totalMass) / (cmEnergy * cmEnergy));
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
                energies[i] = sqrt(massSq[i] + x2 * momSq[i]);
                f += energies[i];
                df += momSq[i] / energies[i];
            }
            if (fabs(f) <= accuracyGoal) break;
            if (++iteration > maxIterations) break;
            x = x - f / (df * x);
        }
        
        // Apply rescaling
        for (int i = 0; i < nParticles; i++) {
            virtMom[i] = x * p[i][0];
            for (int k = 1; k < 4; k++) {
                p[i][k] *= x;
            }
            p[i][0] = energies[i];
        }
        
        // Calculate massive weight correction
        double weightProduct = 1.0;
        double weightSum = 0.0;
        for (int i = 0; i < nParticles; i++) {
            weightProduct *= virtMom[i] / energies[i];
            weightSum += (virtMom[i] * virtMom[i]) / energies[i];
        }
        double logWeightMassive = (2.0 * nParticles - 3.0) * log(x) 
                                 + log(weightProduct / weightSum * cmEnergy);
        
        logWeight += logWeightMassive;
        
        // Copy final momenta to output
        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                momenta[i][mu] = p[i][mu];
            }
        }
        
        return logWeight;
    }
};

// Backwards compatibility alias
template <int NP>
using RamboDevice = PhaseSpaceGenerator<NP>;
