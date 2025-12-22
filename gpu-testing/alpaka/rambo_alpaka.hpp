#pragma once

#include <alpaka/alpaka.hpp>
#include <cmath>

// =============================================================================
// Phase Space Generation Framework for Alpaka 2.0.0
// =============================================================================
// Modular design allowing different phase space generation algorithms.
// The PhaseSpaceGenerator wraps an algorithm (e.g., RamboAlgorithm) and
// provides a uniform interface for generating particle momenta.
//
// To use a different algorithm, change the Algorithm template parameter.
// =============================================================================

// =============================================================================
// RAMBO Algorithm Implementation
// =============================================================================
// Reference: R. Kleiss, W.J. Stirling, S.D. Ellis, Comp. Phys. Comm. 40 (1986) 359
//
// Generates nParticles 4-momenta with uniform phase space distribution.
// Supports both massless and massive particles.
// =============================================================================

template <int nParticles>
struct RamboAlgorithm {
    // Algorithm constants
    static constexpr double tolerance = 1e-14;
    static constexpr int maxIterations = 1000;
    
    // Generate momenta and return the log of the phase space weight
    // Parameters:
    //   cmEnergy: Center-of-mass energy
    //   masses: Array of particle masses
    //   engine: Alpaka RNG engine (modified)
    //   dist: Alpaka uniform distribution [0,1)
    //   momenta: Output array [nParticles][4] for 4-momenta
    // Returns: log(weight) or 0.0 if generation failed
    template <typename TEngine, typename TDist>
    ALPAKA_FN_HOST_ACC auto generate(double cmEnergy, const double* masses,
                                     TEngine& engine, TDist& dist, 
                                     double momenta[][4]) const -> double {
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
        
        // Constants
        const double twoPi = 8.0 * std::atan(1.0);
        const double logPiOver2 = std::log(twoPi / 4.0);
        
        // Validate particle count
        if (nParticles < 1 || nParticles > 10) return 0.0;
        
        // Pre-compute weight coefficients
        zCoeff[0] = 0.0;
        if (nParticles > 1) {
            zCoeff[1] = logPiOver2;
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k-1] + logPiOver2 - 2.0 * std::log(double(k-1));
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k] - std::log(double(k));
        }
        
        // Calculate total mass and count massive particles
        double totalMass = 0.0;
        int nMassive = 0;
        for (int i = 0; i < nParticles; i++) {
            if (masses[i] != 0.0) nMassive++;
            totalMass += std::fabs(masses[i]);
        }
        
        // Check kinematic threshold
        if (totalMass > cmEnergy) return 0.0;
        
        // Generate nParticles isotropic massless momenta
        for (int i = 0; i < nParticles; i++) {
            double rand1 = dist(engine);
            double cosTheta = 2.0 * rand1 - 1.0;
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            double phi = twoPi * dist(engine);
            rand1 = dist(engine);
            double rand2 = dist(engine);
            
            // Energy from exponential distribution
            q[i][0] = -std::log(rand1 * rand2);
            q[i][3] = q[i][0] * cosTheta;
            q[i][2] = q[i][0] * sinTheta * std::cos(phi);
            q[i][1] = q[i][0] * sinTheta * std::sin(phi);
        }
        
        // Calculate total 4-momentum
        for (int mu = 0; mu < 4; mu++) totalMom[mu] = 0.0;
        for (int i = 0; i < nParticles; i++) {
            for (int mu = 0; mu < 4; mu++) {
                totalMom[mu] += q[i][mu];
            }
        }
        
        // Compute boost parameters
        double invariantMass = std::sqrt(totalMom[0]*totalMom[0] - totalMom[1]*totalMom[1] 
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
            logWeight += (2.0 * nParticles - 4.0) * std::log(cmEnergy) + zCoeff[nParticles-1];
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
        double xMax = std::sqrt(1.0 - (totalMass * totalMass) / (cmEnergy * cmEnergy));
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
                energies[i] = std::sqrt(massSq[i] + x2 * momSq[i]);
                f += energies[i];
                df += momSq[i] / energies[i];
            }
            if (std::fabs(f) <= accuracyGoal) break;
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
        double logWeightMassive = (2.0 * nParticles - 3.0) * std::log(x) 
                                 + std::log(weightProduct / weightSum * cmEnergy);
        
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

// =============================================================================
// Phase Space Generator (Wrapper)
// =============================================================================
// Template wrapper that delegates to a specific algorithm implementation.
// Provides a uniform callable interface for the integrator.
//
// Template Parameters:
//   nParticles: Number of final-state particles
//   Algorithm: Phase space algorithm (default: RamboAlgorithm)
// =============================================================================

template <int nParticles, typename Algorithm = RamboAlgorithm<nParticles>>
struct PhaseSpaceGenerator {
    double cmEnergy;          // Center-of-mass energy
    const double* masses;     // Particle masses array
    Algorithm algorithm;      // The underlying algorithm
    
    ALPAKA_FN_HOST_ACC PhaseSpaceGenerator(double energy, const double* m) 
        : cmEnergy(energy), masses(m), algorithm() {}
    
    // Generate momenta and return the log of the phase space weight
    // TEngine: RNG engine type (e.g., Philox)
    // TDist: Uniform distribution type [0,1)
    template <typename TEngine, typename TDist>
    ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine, TDist& dist, double momenta[][4]) const -> double {
        return algorithm.generate(cmEnergy, masses, engine, dist, momenta);
    }
};

// Convenience type alias using RAMBO as the default algorithm
template <int nParticles>
using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

// Backwards compatibility alias
template <int NP>
using RamboDevice = DefaultPhaseSpaceGenerator<NP>;
