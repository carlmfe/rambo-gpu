#pragma once
#ifndef RAMBO_HPP
#define RAMBO_HPP

// =============================================================================
// RAMBO - Phase Space Generator Library for Kokkos
// =============================================================================
// Main include file - includes all library components
//
// Usage:
//   #include <rambo/rambo.hpp>
//
//   // Use with custom integrand
//   rambo::RamboIntegrator<MyIntegrand, 2> integrator(nEvents, integrand);
//   integrator.run(cmEnergy, masses, mean, error, seed);
//
// Requirements:
//   - Kokkos must be initialized before using the library
//   - Kokkos must be finalized after use
// =============================================================================

#include "phase_space.hpp"
#include "integrator.hpp"
#include "integrands.hpp"

namespace rambo {

// Library version
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace rambo

#endif // RAMBO_HPP
