#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// Use KOKKOS_FUNCTION (non-inline) so device symbol is emitted in the TU with the definition
KOKKOS_FUNCTION
void rambo_device(double energy, const double* masses, int n,
                  Kokkos::Random_XorShift64_Pool<>::generator_type &rng,
                  double momenta_out[][4],
                  double &weight);
