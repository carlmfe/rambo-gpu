#pragma once

#include <Kokkos_Core.hpp>
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
    
    KOKKOS_FUNCTION EggholderIntegrand(double lambda = 1000000.0) 
        : lambdaSquared(lambda) {}
    
    KOKKOS_INLINE_FUNCTION auto evaluate(const double momenta[][4]) const -> double {
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
        
        const double arg1 = Kokkos::fabs((s12 - s23) / lambdaSquared);
        const double arg2 = Kokkos::fabs(s13 / lambdaSquared);
        return Kokkos::sin(Kokkos::sqrt(arg1)) * Kokkos::cos(Kokkos::sqrt(arg2));
    }
};

// -----------------------------------------------------------------------------
// ConstantIntegrand: Returns a constant value (for validation)
// -----------------------------------------------------------------------------
// Useful for testing: integral should equal constant × phase_space_volume
struct ConstantIntegrand {
    double value;
    
    KOKKOS_FUNCTION ConstantIntegrand(double v = 1.0) : value(v) {}
    
    KOKKOS_INLINE_FUNCTION auto evaluate(const double momenta[][4]) const -> double {
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
    
    KOKKOS_FUNCTION MandelstamSIntegrand(double s = 1.0) : scale(s) {}
    
    KOKKOS_INLINE_FUNCTION auto evaluate(const double momenta[][4]) const -> double {
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

// -----------------------------------------------------------------------------
// DrellYanIntegrand: Parton-level Drell-Yan cross-section (q qbar -> e+ e-)
// -----------------------------------------------------------------------------
// Computes the differential cross-section for quark-antiquark annihilation
// into an electron-positron pair via virtual photon exchange.
//
// Matrix element squared (averaged over initial spins, summed over final):
//   |M|^2 = 2 * e^4 * e_q^2 * (t^2 + u^2) / s^2
//
// The integrated parton-level cross-section is:
//   sigma = 4 * pi * alpha^2 * e_q^2 / (3 * s)
//
// RAMBO Integration:
// The 2-body RAMBO phase space has weight: W_RAMBO = pi/2
// Cross-section: sigma = (1/2s) * int |M|^2 * d(LIPS)
//              = (1/2s) * <|M|^2 * W_RAMBO / (4*pi^2)>
// -----------------------------------------------------------------------------
struct DrellYanIntegrand {
    double quarkCharge;      // Quark charge in units of e (e.g., 2/3 for up)
    double alphaEM;          // Fine structure constant (~1/137)
    
    KOKKOS_FUNCTION DrellYanIntegrand(double eq = 2.0/3.0, double alpha = 1.0/137.035999)
        : quarkCharge(eq), alphaEM(alpha) {}
    
    KOKKOS_INLINE_FUNCTION auto evaluate(const double momenta[][4]) const -> double {
        // Total 4-momentum (= incoming q + qbar)
        double Ptot[4];
        for (int mu = 0; mu < 4; ++mu) {
            Ptot[mu] = momenta[0][mu] + momenta[1][mu];
        }
        
        // Mandelstam s = (k1 + k2)^2
        double s = Ptot[0]*Ptot[0] - Ptot[1]*Ptot[1] 
                 - Ptot[2]*Ptot[2] - Ptot[3]*Ptot[3];
        
        if (s <= 0.0) return 0.0;
        
        double sqrtS = Kokkos::sqrt(s);
        
        // Incoming parton momenta in CM frame
        double p1[4] = {sqrtS/2.0, 0.0, 0.0, +sqrtS/2.0};
        
        // Outgoing momenta
        const double* k1 = momenta[0];
        const double* k2 = momenta[1];
        
        // Mandelstam t and u (using inline dot product)
        double t = -2.0 * (p1[0]*k1[0] - p1[1]*k1[1] - p1[2]*k1[2] - p1[3]*k1[3]);
        double u = -2.0 * (p1[0]*k2[0] - p1[1]*k2[1] - p1[2]*k2[2] - p1[3]*k2[3]);
        
        // |M|^2 = 2 * e^4 * e_q^2 * (t^2 + u^2) / s^2
        constexpr double PI = 3.14159265358979323846;
        double e4 = 16.0 * PI * PI * alphaEM * alphaEM;
        double eq2 = quarkCharge * quarkCharge;
        double Msq = 2.0 * e4 * eq2 * (t*t + u*u) / (s*s);
        
        // Cross-section integrand: |M|^2 / (2s * 4*pi^2)
        double dsigma = Msq / (2.0 * s) / (4.0 * PI * PI);
        
        // Convert to millibarns: hbar^2*c^2 = 0.3894 GeV^2 * mb
        constexpr double hbarc2 = 0.3893793656;
        
        return dsigma * hbarc2;
    }
    
    // Analytic integrated cross-section for validation
    static auto analyticCrossSection(double s, double eq, double alpha) -> double {
        constexpr double PI = 3.14159265358979323846;
        constexpr double hbarc2 = 0.3893793656;
        return 4.0 * PI * alpha * alpha * eq * eq / (3.0 * s) * hbarc2;
    }
};

// =============================================================================
// Backwards compatibility aliases
// =============================================================================
using Eggholder_3particles = EggholderIntegrand;

template <int NP>
using MandelstamS_Integrand = MandelstamSIntegrand<NP>;
