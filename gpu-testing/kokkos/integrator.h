#pragma once
 
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

void integrator_kokkos(int64_t nEvents, double energy,
                       Kokkos::View<const double*> masses_d,
                       double &mean, double &error);

void integrator_2particle_kokkos(int64_t nEvents, double energy,
                                 Kokkos::View<const double*> masses_d,
                                 double &mean, double &error);


KOKKOS_INLINE_FUNCTION
double integrand_device(int n, const double momenta_out[][4]);

KOKKOS_INLINE_FUNCTION
double integrand_2particle_device(const double momenta_out[4][4]);
