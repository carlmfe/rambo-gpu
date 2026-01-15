#pragma once
#ifndef RAMBO_CUDA_PHASE_SPACE_CUH
#define RAMBO_CUDA_PHASE_SPACE_CUH

#include <cmath>
#include <cstdint>

namespace rambo
{

    // =============================================================================
    // Random Number Generation (XorShift64 for CUDA)
    // =============================================================================

    __device__ __forceinline__ uint64_t xorshift64(uint64_t &state)
    {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }

    __device__ __forceinline__ auto uniformRandom(uint64_t &state) -> double
    {
        return (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
    }

    // =============================================================================
    // RAMBO Algorithm Implementation
    // =============================================================================

    template <int nParticles>
    struct RamboAlgorithm
    {
        static constexpr double tolerance = 1e-14;
        static constexpr int maxIterations = 1000;
        static constexpr int nRandomNumbers = 4 * nParticles;

        // Pre-computed mathematical constants
        static constexpr double twoPi = 6.283185307179586476925286766559;
        static constexpr double logPiOver2 = 0.45158270528945486472619522989488;

        // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
        static constexpr double zCoeffTable[10] = {
            0.0,                      // n = 1
            0.0,                      // n = 2
            -0.24156447527049046409,  // n = 3
            -1.58174123920909082130,  // n = 4
            -3.61506518370763618719,  // n = 5
            -6.15921475197217294095,  // n = 6
            -9.10882942834487252526,  // n = 7
            -12.39491634133878683599, // n = 8
            -15.96868532678448104889, // n = 9
            -19.79376874051108003982, // n = 10
        };
        static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

        // Pre-computed quantities from masses
        double totalMass = 0.0;
        double totalMassSq = 0.0;
        int nMassive = 0;
        double massSq[nParticles] = {};

        __device__ __host__ RamboAlgorithm() = default;

        __device__ __host__ explicit RamboAlgorithm(const double *masses)
        {
            initializeMasses(masses);
        }

        __device__ __host__ void initializeMasses(const double *masses)
        {
            totalMass = 0.0;
            nMassive = 0;
            for (int i = 0; i < nParticles; ++i)
            {
                massSq[i] = masses[i] * masses[i];
                if (masses[i] != 0.0)
                    nMassive++;
                totalMass += fabs(masses[i]);
            }
            totalMassSq = totalMass * totalMass;
        }

    public:
        __device__ auto generate(double cmEnergy, const double r[4 * nParticles],
                                 double momenta[][4]) const -> double
        {
            double q[nParticles][4];
            double p[nParticles][4];
            double totalMom[4];
            double boostVec[3];

            if (nParticles < 1 || nParticles > 10)
                return 0.0;

            for (int i = 0; i < nParticles; i++)
            {
                double cosTheta = 2.0 * r[4 * i] - 1.0;
                double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                double phi = twoPi * r[4 * i + 1];

                q[i][0] = -log(r[4 * i + 2] * r[4 * i + 3]);
                q[i][3] = q[i][0] * cosTheta;
                q[i][2] = q[i][0] * sinTheta * cos(phi);
                q[i][1] = q[i][0] * sinTheta * sin(phi);
            }

            for (int mu = 0; mu < 4; mu++)
                totalMom[mu] = 0.0;
            for (int i = 0; i < nParticles; i++)
            {
                for (int mu = 0; mu < 4; mu++)
                {
                    totalMom[mu] += q[i][mu];
                }
            }

            double invariantMass = sqrt(totalMom[0] * totalMom[0] - totalMom[1] * totalMom[1] - totalMom[2] * totalMom[2] - totalMom[3] * totalMom[3]);
            for (int k = 0; k < 3; k++)
            {
                boostVec[k] = -totalMom[k + 1] / invariantMass;
            }
            double gamma = totalMom[0] / invariantMass;
            double boostFactor = 1.0 / (1.0 + gamma);
            double scaleFactor = cmEnergy / invariantMass;

            for (int i = 0; i < nParticles; ++i)
            {
                double bDotQ = boostVec[0] * q[i][1] + boostVec[1] * q[i][2] + boostVec[2] * q[i][3];
                for (int k = 1; k < 4; ++k)
                {
                    p[i][k] = scaleFactor * (q[i][k] + boostVec[k - 1] * (q[i][0] + boostFactor * bDotQ));
                }
                p[i][0] = scaleFactor * (gamma * q[i][0] + bDotQ);
            }

            double logWeight = logPiOver2;
            if (nParticles != 2)
            {
                logWeight += (2.0 * nParticles - 4.0) * log(cmEnergy) + zCoeffFinal;
            }

            if (nMassive == 0)
            {
                for (int i = 0; i < nParticles; ++i)
                {
                    for (int mu = 0; mu < 4; ++mu)
                    {
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
            double xMax = sqrt(1.0 - totalMassSq / cmEnergySq);
            for (int i = 0; i < nParticles; ++i)
            {
                momSq[i] = p[i][0] * p[i][0];
            }

            int iteration = 0;
            double x = xMax;
            double accuracyGoal = cmEnergy * tolerance;

            while (true)
            {
                double f = -cmEnergy;
                double df = 0.0;
                double x2 = x * x;
                for (int i = 0; i < nParticles; ++i)
                {
                    energies[i] = sqrt(massSq[i] + x2 * momSq[i]);
                    f += energies[i];
                    df += momSq[i] / energies[i];
                }
                if (fabs(f) <= accuracyGoal)
                    break;
                if (++iteration > maxIterations)
                    break;
                x = x - f / (df * x);
            }

            for (int i = 0; i < nParticles; i++)
            {
                virtMom[i] = x * p[i][0];
                for (int k = 1; k < 4; k++)
                {
                    p[i][k] *= x;
                }
                p[i][0] = energies[i];
            }

            double weightProduct = 1.0;
            double weightSum = 0.0;
            for (int i = 0; i < nParticles; i++)
            {
                weightProduct *= virtMom[i] / energies[i];
                weightSum += (virtMom[i] * virtMom[i]) / energies[i];
            }
            double logWeightMassive = (2.0 * nParticles - 3.0) * log(x) + log(weightProduct / weightSum * cmEnergy);

            logWeight += logWeightMassive;

            for (int i = 0; i < nParticles; ++i)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    momenta[i][mu] = p[i][mu];
                }
            }

            return logWeight;
        }

        __device__ auto generate(double cmEnergy, uint64_t &rngState,
                                 double momenta[][4]) const -> double
        {
            double r[4 * nParticles];
            for (int i = 0; i < 4 * nParticles; ++i)
            {
                r[i] = uniformRandom(rngState);
            }
            return generate(cmEnergy, r, momenta);
        }
    };

    // =============================================================================
    // RAMBO "Diet" variant (Plätzer) - CUDA device implementation
    // =============================================================================

    template <int nParticles>
    struct RamboDietAlgorithm
    {
        static constexpr double tolerance = 1e-14;
        static constexpr int maxIterations = 1000;
        static constexpr int nRandomNumbers = 3 * nParticles - 4;

        // Pre-computed mathematical constants
        static constexpr double twoPi = 6.283185307179586476925286766559;
        static constexpr double logPiOver2 = 0.45158270528945486472619522989488;

        // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
        static constexpr double zCoeffTable[10] = {
            0.0,                      // n = 1
            0.0,                      // n = 2
            -0.24156447527049046409,  // n = 3
            -1.58174123920909082130,  // n = 4
            -3.61506518370763618719,  // n = 5
            -6.15921475197217294095,  // n = 6
            -9.10882942834487252526,  // n = 7
            -12.39491634133878683599, // n = 8
            -15.96868532678448104889, // n = 9
            -19.79376874051108003982  // n = 10
        };
        static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

        // Pre-computed quantities from masses
        double totalMass = 0.0;
        double totalMassSq = 0.0;
        int nMassive = 0;
        double massSq[nParticles] = {};

        __device__ __host__ RamboDietAlgorithm() = default;

        __device__ __host__ explicit RamboDietAlgorithm(const double *masses)
        {
            initializeMasses(masses);
        }

        __device__ __host__ void initializeMasses(const double *masses)
        {
            totalMass = 0.0;
            nMassive = 0;
            for (int i = 0; i < nParticles; ++i)
            {
                massSq[i] = masses[i] * masses[i];
                if (masses[i] != 0.0)
                    nMassive++;
                totalMass += fabs(masses[i]);
            }
            totalMassSq = totalMass * totalMass;
        }

    public:
        __device__ auto uEquation(double u, double r, int index) const -> double
        {
            int m = nParticles - index;
            return r - m * pow(u, m - 1) + (m - 1) * pow(u, m);
        }

        __device__ auto dUEquation(double u, int index) const -> double
        {
            int m = nParticles - index;
            return m * (m - 1) * (-pow(u, m - 2) + pow(u, m - 1));
        }

        __device__ inline void solveForU(double &u, double r, int index) const
        {
            int iteration = 0;
            int m = nParticles - index;
            u = pow(r, 1.0 / static_cast<double>(m - 1));
            if (u <= 0.0)
                u = 1e-12;
            if (u >= 1.0)
                u = 1.0 - 1e-12;
            while (true)
            {
                double f = uEquation(u, r, index);
                double df = dUEquation(u, index);
                if (fabs(f) <= tolerance)
                    break;
                if (++iteration > maxIterations)
                    break;
                if (df == 0.0)
                    break;
                u = u - f / df;
                if (u <= 0.0)
                    u = 1e-12;
                if (u >= 1.0)
                    u = 1.0 - 1e-12;
            }
        }

        __device__ void boost_p(double p[4], const double *boostVec) const
        {
            double b2 = boostVec[0] * boostVec[0] + boostVec[1] * boostVec[1] + boostVec[2] * boostVec[2];
            if (b2 >= 1.0 || b2 <= 0.0)
                return;
            double gamma = 1.0 / sqrt(1.0 - b2);
            double bDotP = boostVec[0] * p[1] + boostVec[1] * p[2] + boostVec[2] * p[3];
            double factor = (gamma - 1.0) * bDotP / b2 - gamma * p[0];
            p[0] = gamma * (p[0] - bDotP);
            for (int k = 1; k < 4; ++k)
                p[k] += boostVec[k - 1] * factor;
        }

        __device__ auto generate(double cmEnergy, const double r[3 * nParticles - 4],
                                 double momenta[nParticles][4]) const -> double
        {
            double QPrev[4];
            QPrev[0] = cmEnergy;
            QPrev[1] = QPrev[2] = QPrev[3] = 0.0;
            double QCurr[4];
            double MPrev = cmEnergy;
            double MCurr = MPrev;

            double p[nParticles][4];
            double boostVec[3];
            double u[(nParticles > 2) ? (nParticles - 2) : 1];

            if (totalMass > cmEnergy)
                return -1e300;

            for (int i = 1; i < nParticles; i++)
            {
                if (nParticles > 2)
                {
                    if (i < nParticles - 1)
                    {
                        solveForU(u[i - 1], r[i - 1], i);
                        MCurr = u[i - 1] * MPrev;
                    }
                    else
                    {
                        MCurr = 0.0;
                    }
                }
                else
                {
                    MCurr = 0.0;
                }

                double cosTheta = 2.0 * r[nParticles - 4 + 2 * i] - 1.0;
                double sinTheta = sqrt(fmax(0.0, 1.0 - cosTheta * cosTheta));
                double phi = twoPi * r[nParticles - 3 + 2 * i];

                double q = 0.5 * (MPrev * MPrev - MCurr * MCurr) / MPrev;
                p[i - 1][0] = q;
                p[i - 1][1] = q * sinTheta * cos(phi);
                p[i - 1][2] = q * sinTheta * sin(phi);
                p[i - 1][3] = q * cosTheta;

                QCurr[0] = sqrt(p[i - 1][0] * p[i - 1][0] + MCurr * MCurr);
                QCurr[1] = -p[i - 1][1];
                QCurr[2] = -p[i - 1][2];
                QCurr[3] = -p[i - 1][3];

                if (nParticles > 2)
                {
                    if (i > 1)
                    {
                        boostVec[0] = QPrev[1] / QPrev[0];
                        boostVec[1] = QPrev[2] / QPrev[0];
                        boostVec[2] = QPrev[3] / QPrev[0];
                        boost_p(p[i - 1], boostVec);
                        boost_p(QCurr, boostVec);
                    }
                    MPrev = MCurr;
                    for (int k = 0; k < 4; k++)
                        QPrev[k] = QCurr[k];
                }
            }

            for (int k = 0; k < 4; k++)
                p[nParticles - 1][k] = QCurr[k];

            double logWeight = logPiOver2;
            if (nParticles > 2)
            {
                logWeight += (2.0 * nParticles - 4.0) * log(cmEnergy) + zCoeffFinal;
            }

            if (nMassive == 0)
            {
                for (int i = 0; i < nParticles; i++)
                    for (int mu = 0; mu < 4; mu++)
                        momenta[i][mu] = p[i][mu];
                return logWeight;
            }

            // Make a conformal transformation to give particles mass
            double momSq[nParticles];
            double energies[nParticles];
            double virtMom[nParticles];

            double cmEnergySq = cmEnergy * cmEnergy;
            double xMax = sqrt(1.0 - totalMassSq / cmEnergySq);
            for (int i = 0; i < nParticles; i++)
            {
                momSq[i] = p[i][0] * p[i][0];
            }

            int iteration = 0;
            double x = xMax;
            double accuracyGoal = cmEnergy * tolerance;

            while (true)
            {
                double f = -cmEnergy;
                double df = 0.0;
                double x2 = x * x;
                for (int i = 0; i < nParticles; i++)
                {
                    energies[i] = sqrt(massSq[i] + x2 * momSq[i]);
                    f += energies[i];
                    df += momSq[i] / energies[i];
                }
                if (fabs(f) <= accuracyGoal)
                    break;
                if (++iteration > maxIterations)
                    break;
                x = x - f / (df * x);
            }

            for (int i = 0; i < nParticles; i++)
            {
                virtMom[i] = x * p[i][0];
                for (int k = 1; k < 4; k++)
                    p[i][k] *= x;
                p[i][0] = energies[i];
            }

            double weightProduct = 1.0;
            double weightSum = 0.0;
            for (int i = 0; i < nParticles; i++)
            {
                weightProduct *= virtMom[i] / energies[i];
                weightSum += (virtMom[i] * virtMom[i]) / energies[i];
            }
            double logWeightMassive = (2.0 * nParticles - 3.0) * log(x) + log(weightProduct / weightSum * cmEnergy);
            logWeight += logWeightMassive;

            for (int i = 0; i < nParticles; i++)
                for (int mu = 0; mu < 4; mu++)
                    momenta[i][mu] = p[i][mu];

            return logWeight;
        }

        __device__ auto generate(double cmEnergy, uint64_t &rngState,
                                 double momenta[nParticles][4]) const -> double
        {
            constexpr int numRandoms = 3 * nParticles - 4;
            double r[numRandoms > 0 ? numRandoms : 1];
            for (int i = 0; i < numRandoms; i++)
            {
                r[i] = uniformRandom(rngState);
            }
            return generate(cmEnergy, r, momenta);
        }
    };

    // =============================================================================
    // Phase Space Generator (Wrapper)
    // =============================================================================

    template <int nParticles, typename Algorithm = RamboAlgorithm<nParticles>>
    struct PhaseSpaceGenerator
    {
        double cmEnergy;
        Algorithm algorithm;

        static constexpr int nRandomNumbers = Algorithm::nRandomNumbers;

        __device__ __host__ PhaseSpaceGenerator(double energy, const double *masses)
            : cmEnergy(energy), algorithm(masses) {}

        __device__ auto operator()(uint64_t &rngState, double momenta[][4]) const -> double
        {
            return algorithm.generate(cmEnergy, rngState, momenta);
        }

        __device__ auto operator()(const double *r, double momenta[][4]) const -> double
        {
            return algorithm.generate(cmEnergy, r, momenta);
        }
    };

    template <int nParticles>
    using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

} // namespace rambo

#endif // RAMBO_CUDA_PHASE_SPACE_CUH
