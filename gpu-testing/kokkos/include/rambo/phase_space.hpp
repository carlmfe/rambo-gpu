#pragma once
#ifndef RAMBO_PHASE_SPACE_HPP
#define RAMBO_PHASE_SPACE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>

namespace rambo {

// =============================================================================
// Phase Space Generation Framework for Kokkos
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
    //   rng: Kokkos RNG generator (modified)
    //   momenta: Output array [nParticles][4] for 4-momenta
    // Returns: log(weight) or 0.0 if generation failed
    template <typename TRng>
    KOKKOS_FUNCTION auto generate(double cmEnergy, const double* masses, 
                                  TRng& rng, double momenta[][4]) const -> double {
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
        const double twoPi = 8.0 * Kokkos::atan(1.0);
        const double logPiOver2 = Kokkos::log(twoPi / 4.0);
        
        // Validate particle count
        if (nParticles < 1 || nParticles > 10) return 0.0;
        
        // Pre-compute weight coefficients
        zCoeff[0] = 0.0;
        if (nParticles > 1) {
            zCoeff[1] = 0.0;  // Not logPiOver2 - that's added separately in logWeight
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k-1] + logPiOver2 - 2.0 * Kokkos::log(double(k-1));
            for (int k = 2; k < nParticles; k++)
                zCoeff[k] = zCoeff[k] - Kokkos::log(double(k));
        }
        
        // Calculate total mass and count massive particles
        double totalMass = 0.0;
        int nMassive = 0;
        for (int i = 0; i < nParticles; i++) {
            if (masses[i] != 0.0) nMassive++;
            totalMass += Kokkos::fabs(masses[i]);
        }
        
        // Check kinematic threshold
        if (totalMass > cmEnergy) return 0.0;
        
        // Generate nParticles isotropic massless momenta
        for (int i = 0; i < nParticles; i++) {
            double rand1 = rng.drand();
            double cosTheta = 2.0 * rand1 - 1.0;
            double sinTheta = Kokkos::sqrt(1.0 - cosTheta * cosTheta);
            double phi = twoPi * rng.drand();
            rand1 = rng.drand();
            double rand2 = rng.drand();
            
            // Energy from exponential distribution
            q[i][0] = -Kokkos::log(rand1 * rand2);
            q[i][3] = q[i][0] * cosTheta;
            q[i][2] = q[i][0] * sinTheta * Kokkos::cos(phi);
            q[i][1] = q[i][0] * sinTheta * Kokkos::sin(phi);
        }
        
        // Calculate total 4-momentum
        for (int mu = 0; mu < 4; mu++) totalMom[mu] = 0.0;
        for (int i = 0; i < nParticles; i++) {
            for (int mu = 0; mu < 4; mu++) {
                totalMom[mu] += q[i][mu];
            }
        }
        
        // Compute boost parameters
        double invariantMass = Kokkos::sqrt(totalMom[0]*totalMom[0] - totalMom[1]*totalMom[1] 
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
            logWeight += (2.0 * nParticles - 4.0) * Kokkos::log(cmEnergy) + zCoeff[nParticles-1];
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
        double xMax = Kokkos::sqrt(1.0 - (totalMass * totalMass) / (cmEnergy * cmEnergy));
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
                energies[i] = Kokkos::sqrt(massSq[i] + x2 * momSq[i]);
                f += energies[i];
                df += momSq[i] / energies[i];
            }
            if (Kokkos::fabs(f) <= accuracyGoal) break;
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
        double logWeightMassive = (2.0 * nParticles - 3.0) * Kokkos::log(x) 
                                 + Kokkos::log(weightProduct / weightSum * cmEnergy);
        
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
    
    KOKKOS_FUNCTION PhaseSpaceGenerator(double energy, const double* m) 
        : cmEnergy(energy), masses(m), algorithm() {}
    
    // Generate momenta and return the log of the phase space weight
    // TRng: Kokkos RNG generator type
    template <typename TRng>
    KOKKOS_FUNCTION auto operator()(TRng& rng, double momenta[][4]) const -> double {
        return algorithm.generate(cmEnergy, masses, rng, momenta);
    }
};

// Convenience type alias using RAMBO as the default algorithm
template <int nParticles>
using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

// Backwards compatibility alias
template <int NP>
using RamboDevice = DefaultPhaseSpaceGenerator<NP>;

} // namespace rambo

#endif // RAMBO_PHASE_SPACE_HPP
