#pragma once
#ifndef RAMBO_KOKKOS_PHASE_SPACE_HPP
#define RAMBO_KOKKOS_PHASE_SPACE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rambo {

// =============================================================================
// Random Number Generation (XorShift64) - Device-compatible
// =============================================================================
/**
 * Advance a 64-bit xorshift RNG state in-place and return the new state.
 * @param state RNG state (updated by this call)
 * @return Updated RNG state value
 */
KOKKOS_INLINE_FUNCTION
uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

/**
 * Produce a uniform double in [0, 1) using bits from the RNG state.
 * @param state RNG state (advanced by this call)
 * @return Uniform random double in [0, 1)
 */
KOKKOS_INLINE_FUNCTION
auto uniformRandom(uint64_t& state) -> double {
    return static_cast<double>(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

// =============================================================================
// RAMBO Algorithm Implementation
// =============================================================================
// Reference: R. Kleiss, W.J. Stirling, S.D. Ellis, Comp. Phys. Comm. 40 (1986) 359

template <int nParticles>
struct RamboAlgorithm {
    static constexpr double tolerance = 1e-14;
    static constexpr int maxIterations = 1000;
    static constexpr int nRandomNumbers = 4 * nParticles;

    // Pre-computed mathematical constants
    static constexpr double twoPi = 6.283185307179586476925286766559;      // 2*pi
    static constexpr double logPiOver2 = 0.45158270528945486472619522989488; // log(pi/2)

    // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
    static constexpr double zCoeffTable[10] = {
        0.0,                      // n = 1
        0.0,                      // n = 2
       -0.24156447527049046409,   // n = 3
       -1.58174123920909082130,   // n = 4
       -3.61506518370763618719,   // n = 5
       -6.15921475197217294095,   // n = 6
       -9.10882942834487252526,   // n = 7
      -12.39491634133878683599,   // n = 8
      -15.96868532678448104889,   // n = 9
      -19.79376874051108003982,   // n = 10
    };
    static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

    // Pre-computed quantities from masses
    double totalMass = 0.0;
    double totalMassSq = 0.0;
    int nMassive = 0;
    double massSq[nParticles] = {};

    KOKKOS_FUNCTION
    RamboAlgorithm() = default;

    KOKKOS_FUNCTION
    explicit RamboAlgorithm(const double* masses) {
        initializeMasses(masses);
    }

    KOKKOS_FUNCTION
    void initializeMasses(const double* masses) {
        totalMass = 0.0;
        nMassive = 0;
        for (int i = 0; i < nParticles; ++i) {
            massSq[i] = masses[i] * masses[i];
            if (masses[i] != 0.0) nMassive++;
            totalMass += Kokkos::fabs(masses[i]);
        }
        totalMassSq = totalMass * totalMass;
    }

public:
    /**
     * Generate an n-particle phase-space point.
     * @param cmEnergy Total center-of-mass energy available.
     * @param r Array of `4*nParticles` uniform random numbers in [0,1).
     * @param momenta Output array of shape `[nParticles][4]`.
     * @return Natural logarithm of the phase-space weight. Returns 0.0 on failure.
     */
    KOKKOS_INLINE_FUNCTION
    auto generate(double cmEnergy, const double r[4 * nParticles], 
                  double momenta[][4]) const -> double {
        double q[nParticles][4];
        double p[nParticles][4];
        double totalMom[4];
        double boostVec[3];

        if (nParticles < 1 || nParticles > 10) return 0.0;

        for (int i = 0; i < nParticles; ++i) {
            double cosTheta = 2.0 * r[4 * i] - 1.0;
            double sinTheta = Kokkos::sqrt(1.0 - cosTheta * cosTheta);
            double phi = twoPi * r[4 * i + 1];
            
            q[i][0] = -Kokkos::log(r[4 * i + 2] * r[4 * i + 3]);
            q[i][3] = q[i][0] * cosTheta;
            q[i][2] = q[i][0] * sinTheta * Kokkos::cos(phi);
            q[i][1] = q[i][0] * sinTheta * Kokkos::sin(phi);
        }

        for (int mu = 0; mu < 4; ++mu) totalMom[mu] = 0.0;
        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                totalMom[mu] += q[i][mu];
            }
        }

        double invariantMass = Kokkos::sqrt(totalMom[0]*totalMom[0] - totalMom[1]*totalMom[1] 
                                        - totalMom[2]*totalMom[2] - totalMom[3]*totalMom[3]);
        for (int k = 0; k < 3; ++k) {
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
            logWeight += (2.0 * nParticles - 4.0) * Kokkos::log(cmEnergy) + zCoeffFinal;
        }

        if (nMassive == 0) {
            for (int i = 0; i < nParticles; ++i) {
                for (int mu = 0; mu < 4; ++mu) {
                    momenta[i][mu] = p[i][mu];
                }
            }
            return logWeight;
        }

        // Make a conformal transformation to give particles mass
        double momSq[nParticles];
        double energies[nParticles];
        double virtMom[nParticles];

        double cmEnergySq = cmEnergy * cmEnergy;
        double xMax = Kokkos::sqrt(1.0 - totalMassSq / cmEnergySq);
        for (int i = 0; i < nParticles; ++i) {
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

        for (int i = 0; i < nParticles; ++i) {
            virtMom[i] = x * p[i][0];
            for (int k = 1; k < 4; ++k) {
                p[i][k] *= x;
            }
            p[i][0] = energies[i];
        }

        double weightProduct = 1.0;
        double weightSum = 0.0;
        for (int i = 0; i < nParticles; ++i) {
            weightProduct *= virtMom[i] / energies[i];
            weightSum += (virtMom[i] * virtMom[i]) / energies[i];
        }
        double logWeightMassive = (2.0 * nParticles - 3.0) * Kokkos::log(x) 
                                 + Kokkos::log(weightProduct / weightSum * cmEnergy);

        logWeight += logWeightMassive;

        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                momenta[i][mu] = p[i][mu];
            }
        }

        return logWeight;
    }

    /**
     * RNG-based overload.
     * @param cmEnergy Total center-of-mass energy available.
     * @param rngState Mutable RNG state (advanced by this call).
     * @param momenta Output array `[nParticles][4]` for generated 4-momenta.
     * @return Natural logarithm of the phase-space weight.
     */
    KOKKOS_INLINE_FUNCTION
    auto generate(double cmEnergy, uint64_t& rngState, double momenta[][4]) const -> double {
        double r[4 * nParticles];
        for (int i = 0; i < 4 * nParticles; ++i) {
            r[i] = uniformRandom(rngState);
        }
        return generate(cmEnergy, r, momenta);
    }
};

// =============================================================================
// RAMBO "Diet" variant
// =============================================================================
// Reference: Plätzer, S, ArXiv: 1308.2922 [hep-ph]

template <int nParticles>
struct RamboDietAlgorithm {
    static constexpr double tolerance = 1e-14;
    static constexpr int maxIterations = 1000;
    static constexpr int nRandomNumbers = 3 * nParticles - 4;

    // Pre-computed mathematical constants
    static constexpr double twoPi = 6.283185307179586476925286766559;      // 2*pi
    static constexpr double logPiOver2 = 0.45158270528945486472619522989488; // log(pi/2)

    // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
    static constexpr double zCoeffTable[10] = {
        0.0,                      // n = 1
        0.0,                      // n = 2
       -0.24156447527049046409,   // n = 3
       -1.58174123920909082130,   // n = 4
       -3.61506518370763618719,   // n = 5
       -6.15921475197217294095,   // n = 6
       -9.10882942834487252526,   // n = 7
      -12.39491634133878683599,   // n = 8
      -15.96868532678448104889,   // n = 9
      -19.79376874051108003982,   // n = 10
    };
    static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

    // Pre-computed quantities from masses
    double totalMass = 0.0;
    double totalMassSq = 0.0;
    int nMassive = 0;
    double massSq[nParticles] = {};

    KOKKOS_FUNCTION
    RamboDietAlgorithm() = default;

    KOKKOS_FUNCTION
    explicit RamboDietAlgorithm(const double* masses) {
        initializeMasses(masses);
    }

    KOKKOS_FUNCTION
    void initializeMasses(const double* masses) {
        totalMass = 0.0;
        nMassive = 0;
        for (int i = 0; i < nParticles; ++i) {
            massSq[i] = masses[i] * masses[i];
            if (masses[i] != 0.0) nMassive++;
            totalMass += Kokkos::fabs(masses[i]);
        }
        totalMassSq = totalMass * totalMass;
    }

public:
    /**
     * Apply a Lorentz boost to a 4-vector `p` in-place.
     */
    KOKKOS_INLINE_FUNCTION
    auto boost(double p[4], const double* boostVec) const -> void {
        double b2 = boostVec[0]*boostVec[0] + boostVec[1]*boostVec[1] + boostVec[2]*boostVec[2];
        if (b2 >= 1.0 || b2 <= 0.0) return;
        double gamma = 1.0 / Kokkos::sqrt(1.0 - b2);
        double bDotP = boostVec[0]*p[1] + boostVec[1]*p[2] + boostVec[2]*p[3];
        double factor = (gamma - 1.0) * bDotP / b2 - gamma * p[0];
        
        p[0] = gamma * (p[0] - bDotP);
        for (int k = 1; k < 4; ++k) {
            p[k] += boostVec[k-1] * factor;
        }
    }

    /**
     * Newton solve for the intermediate variable `u`.
     */
    KOKKOS_INLINE_FUNCTION
    auto solveForU(double &u, double r, int index) const -> void {
        int iteration = 0;
        const int m = nParticles - index;
        u = Kokkos::pow(r, 1.0 / static_cast<double>(m - 1));
        if (u <= 0.0) u = 1e-12;
        if (u >= 1.0) u = 1.0 - 1e-12;
        while (true) {
            double f = uEquation(u, r, index);
            double df = dUEquation(u, index);
            if (Kokkos::fabs(f) <= tolerance) break;
            if (++iteration > maxIterations) break;
            if (df == 0.0) break;
            u = u - f / df;
            if (u <= 0.0) u = 1e-12;
            if (u >= 1.0) u = 1.0 - 1e-12;
        }
    }

    KOKKOS_INLINE_FUNCTION
    auto uEquation(double u, double r, int index) const -> double {
        const int m = nParticles - index;
        return r - m * Kokkos::pow(u, m - 1) + (m - 1) * Kokkos::pow(u, m);
    }

    KOKKOS_INLINE_FUNCTION
    auto dUEquation(double u, int index) const -> double {
        const int m = nParticles - index;
        return m * (m - 1) * (-Kokkos::pow(u, m - 2) + Kokkos::pow(u, m - 1));
    }

    /**
     * Diet-variant generate using pre-computed masses.
     */
    KOKKOS_INLINE_FUNCTION
    auto generate(double cmEnergy, const double r[3 * nParticles - 4], 
                  double momenta[nParticles][4]) const -> double {
        double QPrev[4]{cmEnergy, 0.0, 0.0, 0.0};
        double QCurr[4]{cmEnergy, 0.0, 0.0, 0.0};
        double MPrev = cmEnergy;
        double MCurr = MPrev;
        double u[nParticles > 2 ? nParticles - 2 : 1];
        double cosTheta, sinTheta, phi;
        double q;
        double p[nParticles][4];
        double boostVec[3];

        if (totalMass > cmEnergy) return -Kokkos::Experimental::infinity<double>::value;

        // Generate phase space for massless particles first
        for (int i = 1; i < nParticles; ++i) {
            if (nParticles > 2) {
                if (i < nParticles - 1) {
                    const int m = nParticles - i;
                    u[i - 1] = Kokkos::pow(r[i - 1], 1.0 / static_cast<double>(m - 1));
                    solveForU(u[i - 1], r[i - 1], i);
                    MCurr = u[i - 1] * MPrev;
                } else {
                    MCurr = 0.0;
                }
            } else {
                MCurr = 0.0;
            }
            cosTheta = 2.0 * r[nParticles - 4 + 2 * i] - 1.0;
            sinTheta = Kokkos::sqrt(1.0 - cosTheta * cosTheta);
            phi = twoPi * r[nParticles - 3 + 2 * i];
            q = 0.5 * (MPrev * MPrev - MCurr * MCurr) / MPrev;
            p[i - 1][0] = q;
            p[i - 1][1] = q * sinTheta * Kokkos::cos(phi);
            p[i - 1][2] = q * sinTheta * Kokkos::sin(phi);
            p[i - 1][3] = q * cosTheta;
            QCurr[0] = Kokkos::sqrt(p[i - 1][0] * p[i - 1][0] + MCurr * MCurr);
            QCurr[1] = -p[i - 1][1];
            QCurr[2] = -p[i - 1][2];
            QCurr[3] = -p[i - 1][3];

            if (nParticles > 2) {
                if (i > 1) {
                    boostVec[0] = QPrev[1] / QPrev[0];
                    boostVec[1] = QPrev[2] / QPrev[0];
                    boostVec[2] = QPrev[3] / QPrev[0];
                    boost(p[i - 1], boostVec);
                    boost(QCurr, boostVec);
                }

                MPrev = MCurr;
                for (int k = 0; k < 4; ++k) {
                    QPrev[k] = QCurr[k];
                }
            }
        }
        // Last particle
        for (int k = 0; k < 4; ++k) {
            p[nParticles - 1][k] = QCurr[k];
        }

        double logWeight = logPiOver2;
        if (nParticles > 2) {
            logWeight += (2.0 * nParticles - 4.0) * Kokkos::log(cmEnergy) + zCoeffFinal;
        }

        if (nMassive == 0) {
            for (int i = 0; i < nParticles; ++i) {
                for (int mu = 0; mu < 4; ++mu) {
                    momenta[i][mu] = p[i][mu];
                }
            }
            return logWeight;
        }

        // Make a conformal transformation to give particles mass
        double momSq[nParticles];
        double energies[nParticles];
        double virtMom[nParticles];

        double cmEnergySq = cmEnergy * cmEnergy;
        double xMax = Kokkos::sqrt(1.0 - totalMassSq / cmEnergySq);
        for (int i = 0; i < nParticles; ++i) {
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

        for (int i = 0; i < nParticles; ++i) {
            virtMom[i] = x * p[i][0];
            for (int k = 1; k < 4; ++k) {
                p[i][k] *= x;
            }
            p[i][0] = energies[i];
        }

        double weightProduct = 1.0;
        double weightSum = 0.0;
        for (int i = 0; i < nParticles; ++i) {
            weightProduct *= virtMom[i] / energies[i];
            weightSum += (virtMom[i] * virtMom[i]) / energies[i];
        }
        double logWeightMassive = (2.0 * nParticles - 3.0) * Kokkos::log(x) 
                                 + Kokkos::log(weightProduct / weightSum * cmEnergy);

        logWeight += logWeightMassive;

        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                momenta[i][mu] = p[i][mu];
            }
        }

        return logWeight;
    }

    /**
     * RNG-based overload.
     */
    KOKKOS_INLINE_FUNCTION
    auto generate(double cmEnergy, uint64_t& rngState, 
                  double momenta[nParticles][4]) const -> double {
        constexpr int numRandoms = 3 * nParticles - 4;
        double r[numRandoms > 0 ? numRandoms : 1];
        for (int i = 0; i < numRandoms; ++i) {
            r[i] = uniformRandom(rngState);
        }
        return generate(cmEnergy, r, momenta);
    }
};

// =============================================================================
// Phase Space Generator (Wrapper)
// =============================================================================

template <int nParticles, typename Algorithm = RamboAlgorithm<nParticles>>
struct PhaseSpaceGenerator {
    double cmEnergy;
    Algorithm algorithm;

    /// Number of random numbers required by the underlying algorithm.
    static constexpr int nRandomNumbers = Algorithm::nRandomNumbers;

    /**
     * Constructor that pre-computes mass-dependent quantities in the algorithm.
     * @param energy Center-of-mass energy used for generated points.
     * @param masses Pointer to an array of length `nParticles` with particle masses.
     */
    KOKKOS_FUNCTION
    PhaseSpaceGenerator(double energy, const double* masses) 
        : cmEnergy(energy), algorithm(masses) {}

    /**
     * Generate a phase-space point using an RNG state.
     * @param rngState Mutable RNG state (will be advanced).
     * @param momenta Output array `[nParticles][4]` (E, px, py, pz).
     * @return Natural logarithm of the phase-space weight.
     */
    KOKKOS_INLINE_FUNCTION
    auto operator()(uint64_t& rngState, double momenta[][4]) const -> double {
        return algorithm.generate(cmEnergy, rngState, momenta);
    }

    /**
     * Generate a phase-space point using pre-generated random numbers.
     * @param r Array of `nRandomNumbers` uniform random values in [0, 1).
     * @param momenta Output array `[nParticles][4]` (E, px, py, pz).
     * @return Natural logarithm of the phase-space weight.
     */
    KOKKOS_INLINE_FUNCTION
    auto operator()(const double* r, double momenta[][4]) const -> double {
        return algorithm.generate(cmEnergy, r, momenta);
    }
};

template <int nParticles>
using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

} // namespace rambo

#endif // RAMBO_KOKKOS_PHASE_SPACE_HPP
