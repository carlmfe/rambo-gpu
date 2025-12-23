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

// -----------------------------------------------------------------------------
// DrellYanIntegrand: Parton-level Drell-Yan cross-section (q qbar -> e+ e-)
// -----------------------------------------------------------------------------
// Computes the differential cross-section for quark-antiquark annihilation
// into an electron-positron pair via virtual photon exchange.
//
// Matrix element squared (averaged over initial spins, summed over final):
//   |M|^2 = 2 * e^4 * e_q^2 * (t^2 + u^2) / s^2
//
// where s, t, u are Mandelstam variables for 2->2 scattering.
//
// The integrated parton-level cross-section is:
//   sigma = 4 * pi * alpha^2 * e_q^2 / (3 * s)
//
// RAMBO Integration:
// The 2-body RAMBO phase space has weight: W_RAMBO = pi/2
// The standard 2-body phase space integral is: int d(LIPS) = 1/(8*pi)
// So: d(LIPS) = W_RAMBO / (4*pi^2)
//
// Cross-section formula:
//   sigma = (1/2s) * int |M|^2 * d(LIPS)
//         = (1/2s) * <|M|^2 * W_RAMBO / (4*pi^2)>
//
// References:
// - Peskin & Schroeder, Section 5.1
// - Drell & Yan, Phys. Rev. Lett. 25, 316 (1970)
// -----------------------------------------------------------------------------
struct DrellYanIntegrand {
    double quarkCharge;      // Quark charge in units of e (e.g., 2/3 for up)
    double alphaEM;          // Fine structure constant (~1/137)
    
    // Constructor with default up-quark charge
    DrellYanIntegrand(double eq = 2.0/3.0, double alpha = 1.0/137.035999)
        : quarkCharge(eq), alphaEM(alpha) {}
    
    // Evaluate the integrand that, when multiplied by exp(log_weight) and averaged,
    // gives the total cross-section.
    //
    // momenta[0] = electron 4-momentum (E, px, py, pz)
    // momenta[1] = positron 4-momentum (E, px, py, pz)
    // 
    // We reconstruct the incoming parton momenta from momentum conservation
    // assuming massless quarks in the CM frame.
    auto evaluate(const double momenta[][4]) const -> double {
        // Total 4-momentum (= incoming q + qbar)
        double Ptot[4];
        for (int mu = 0; mu < 4; ++mu) {
            Ptot[mu] = momenta[0][mu] + momenta[1][mu];
        }
        
        // Mandelstam s = (p1 + p2)^2 = (k1 + k2)^2
        // In CM frame: s = E_cm^2
        double s = Ptot[0]*Ptot[0] - Ptot[1]*Ptot[1] 
                 - Ptot[2]*Ptot[2] - Ptot[3]*Ptot[3];
        
        if (s <= 0.0) return 0.0;
        
        double sqrtS = std::sqrt(s);
        
        // In the CM frame, incoming partons have momenta:
        // p1 = (sqrtS/2, 0, 0, +sqrtS/2)  [quark along +z]
        // p2 = (sqrtS/2, 0, 0, -sqrtS/2)  [antiquark along -z]
        double p1[4] = {sqrtS/2.0, 0.0, 0.0, +sqrtS/2.0};
        double p2[4] = {sqrtS/2.0, 0.0, 0.0, -sqrtS/2.0};
        
        // Outgoing momenta (electron k1, positron k2)
        const double* k1 = momenta[0];
        const double* k2 = momenta[1];
        
        // Compute dot products using Minkowski metric (+,-,-,-)
        auto dot = [](const double a[4], const double b[4]) -> double {
            return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
        };
        
        // Mandelstam t = (p1 - k1)^2 and u = (p1 - k2)^2
        // For massless particles:
        //   t = -2 * p1.k1
        //   u = -2 * p1.k2
        double t = -2.0 * dot(p1, k1);
        double u = -2.0 * dot(p1, k2);
        
        // Matrix element squared (spin-averaged):
        // |M|^2 = 2 * e^4 * e_q^2 * (t^2 + u^2) / s^2
        // 
        // Note: e^2 = 4*pi*alpha, so e^4 = 16*pi^2*alpha^2
        double e4 = 16.0 * M_PI * M_PI * alphaEM * alphaEM;
        double eq2 = quarkCharge * quarkCharge;
        
        double Msq = 2.0 * e4 * eq2 * (t*t + u*u) / (s*s);
        
        // Cross-section integrand:
        // sigma = (1/2s) * int |M|^2 d(LIPS)
        //
        // RAMBO provides exp(log_weight) = W_RAMBO = pi/2 for 2-body
        // The phase space measure: d(LIPS) = W_RAMBO / (4*pi^2)
        //
        // So we return: |M|^2 / (2s * 4*pi^2)
        // and the MC will compute: <f * W_RAMBO> = <|M|^2 / (2s * 4*pi^2) * pi/2>
        //                                       = <|M|^2 / (16*pi*s)>
        //
        // Actually, for 2-body, RAMBO weight is exp(log(pi/2)) = pi/2
        // and d(LIPS) when integrated gives 1/(8*pi)
        // So: integral = <|M|^2> * W_RAMBO / (4*pi^2) should give <|M|^2>/(8*pi)
        //
        // Cross-section: sigma = 1/(2s) * 1/(8*pi) * <|M|^2>
        //
        // With RAMBO: <f * W_RAMBO> = sigma
        // So: f = |M|^2 / (2s) / (4*pi^2)
        
        double dsigma = Msq / (2.0 * s) / (4.0 * M_PI * M_PI);
        
        // Convert to standard units: multiply by hbar^2*c^2 = 0.3894 mb*GeV^2
        // to get cross-section in millibarns when s is in GeV^2
        constexpr double hbarc2 = 0.3893793656;  // GeV^2 * mb
        
        return dsigma * hbarc2;
    }
    
    // Analytic integrated cross-section for validation
    // sigma = 4 * pi * alpha^2 * e_q^2 / (3 * s)
    static auto analyticCrossSection(double s, double eq, double alpha) -> double {
        constexpr double hbarc2 = 0.3893793656;  // GeV^2 * mb
        return 4.0 * M_PI * alpha * alpha * eq * eq / (3.0 * s) * hbarc2;
    }
};

// =============================================================================
// Backwards compatibility aliases
// =============================================================================
using Eggholder_3particles = EggholderIntegrand;

template <int NP>
using MandelstamS_Integrand = MandelstamSIntegrand<NP>;
