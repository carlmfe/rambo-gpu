#pragma once

#include <cmath>

// =============================================================================
// Integrand Interface
// =============================================================================
// All integrands must provide an evaluate() method that takes a momenta array
// and returns a double.
// =============================================================================

// -----------------------------------------------------------------------------
// EggholderIntegrand: Physics-inspired test integrand for 3 particles
// -----------------------------------------------------------------------------
// Computes Lorentz-invariant quantities from 4-momenta and applies
// trigonometric functions. Good for testing phase space sampling.
struct EggholderIntegrand {
    double lambdaSquared;  // Scale parameter
    
    EggholderIntegrand(double lambda = 1000000.0) 
        : lambdaSquared(lambda) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // Compute Mandelstam-like invariants s_ij = (p_i - p_j)^2
        double s12 = 0.0, s13 = 0.0, s23 = 0.0;
        
        for (int mu = 0; mu < 4; ++mu) {
            const double d12 = momenta[0][mu] - momenta[1][mu];
            const double d13 = momenta[0][mu] - momenta[2][mu];
            const double d23 = momenta[1][mu] - momenta[2][mu];
            
            // Metric signature (+,-,-,-)
            const double sign = (mu == 0) ? 1.0 : -1.0;
            s12 += sign * d12 * d12;
            s13 += sign * d13 * d13;
            s23 += sign * d23 * d23;
        }
        
        const double arg1 = std::fabs((s12 - s23) / lambdaSquared);
        const double arg2 = std::fabs(s13 / lambdaSquared);
        return std::sin(std::sqrt(arg1)) * std::cos(std::sqrt(arg2));
    }
};

// -----------------------------------------------------------------------------
// ConstantIntegrand: Returns a constant value (for validation)
// -----------------------------------------------------------------------------
// Useful for testing: integral should equal constant × phase_space_volume
struct ConstantIntegrand {
    double value;
    
    ConstantIntegrand(double v = 1.0) : value(v) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        (void)momenta;  // Unused
        return value;
    }
};

// -----------------------------------------------------------------------------
// MandelstamSIntegrand: Total invariant mass squared
// -----------------------------------------------------------------------------
// Computes s = (Σp_i)^2 for N particles, normalized by a scale
template <int nParticles>
struct MandelstamSIntegrand {
    double scale;
    
    MandelstamSIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        double totalMom[4] = {0.0, 0.0, 0.0, 0.0};
        
        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                totalMom[mu] += momenta[i][mu];
            }
        }
        
        // s = E² - px² - py² - pz²
        double sInvariant = totalMom[0]*totalMom[0] 
                          - totalMom[1]*totalMom[1] 
                          - totalMom[2]*totalMom[2] 
                          - totalMom[3]*totalMom[3];
        
        return sInvariant / (scale * scale);
    }
};

// =============================================================================
// Backwards compatibility aliases
// =============================================================================
using Eggholder_3particles = EggholderIntegrand;

template <int NP>
using MandelstamS_Integrand = MandelstamSIntegrand<NP>;
