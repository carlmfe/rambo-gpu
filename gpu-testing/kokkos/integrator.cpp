#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "rambo_kokkos.h"
#include "integrator.h"

const int MAX_PARTICLES = 10; // Define maximum number of particles

struct SumPair {
  double sum;
  double sum2;
  KOKKOS_INLINE_FUNCTION void operator+=(const SumPair &other) {
    sum += other.sum; sum2 += other.sum2;
  }
  KOKKOS_INLINE_FUNCTION static void join(SumPair &l, const SumPair &r) {
    l.sum += r.sum; l.sum2 += r.sum2;
  }
  KOKKOS_INLINE_FUNCTION static void init(SumPair &v) {
    v.sum = 0.0; v.sum2 = 0.0;
  }
};


void integrator_kokkos(int64_t nEvents, double energy, Kokkos::View<const double*> masses_d, double &mean, double &error) {
  Kokkos::Random_XorShift64_Pool<> pool(12345);
  SumPair result;
  const int nParticles = (int)masses_d.extent(0);
  Kokkos::parallel_reduce("RAMBO_MC", nEvents, KOKKOS_LAMBDA(const int64_t ev, SumPair &acc) {
    auto rng = pool.get_state();
    double weight;
    double momenta_local[MAX_PARTICLES][4];
    // Pass the raw data pointer and particle count
    rambo_device(energy, masses_d.data(), nParticles, rng, momenta_local, weight);
    double integrand_value = integrand_device(nParticles, momenta_local);
    double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
    acc.sum += wval;
    acc.sum2 += wval * wval;
    pool.free_state(rng);
  }, result);
  mean = result.sum / nEvents;
  double variance = (result.sum2 / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
}


void integrator_2particle_kokkos(int64_t nEvents, double energy, Kokkos::View<const double*> masses_d, double &mean, double &error) {
  Kokkos::Random_XorShift64_Pool<> pool(12345);
  SumPair result;
  const int nParticles = (int)masses_d.extent(0);
  if (nParticles != 2) {
    Kokkos::abort("integrator_2particle_kokkos called with nParticles != 4");
    return;
  }
  Kokkos::parallel_reduce("RAMBO_MC", nEvents, KOKKOS_LAMBDA(const int64_t ev, SumPair &acc) {
    auto rng = pool.get_state();
    double weight;
    double momenta_local[MAX_PARTICLES][4];
    // Pass the raw data pointer and particle count
    rambo_device(energy, masses_d.data(), nParticles, rng, momenta_local, weight);
    double integrand_value = integrand_2particle_device(momenta_local);
    double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
    acc.sum += wval;
    acc.sum2 += wval * wval;
    pool.free_state(rng);
  }, result);
  mean = result.sum / nEvents;
  double variance = (result.sum2 / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
}




KOKKOS_INLINE_FUNCTION
double integrand_device(int n, const double momenta_out[][4]) {
  double result = 0.0;
  for (int i = 0; i < n; ++i) {
    result += momenta_out[i][0]; // example: sum of energies
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double integrand_2particle_device(const double momenta[2][4]) {
  double tot_momentum[4];
  for (int k = 0; k < 4; k++) {
    tot_momentum[k] = momenta[0][k] + momenta[1][k];
  }
  double s = tot_momentum[0] * tot_momentum[0] - tot_momentum[1] * tot_momentum[1]
              - tot_momentum[2] * tot_momentum[2] - tot_momentum[3] * tot_momentum[3];
  double m12 = momenta[0][0] * momenta[0][0] - momenta[0][1] * momenta[0][1]
              - momenta[0][2] * momenta[0][2] - momenta[0][3] * momenta[0][3];
  double m22 = momenta[1][0] * momenta[1][0] - momenta[1][1] * momenta[1][1]
              - momenta[1][2] * momenta[1][2] - momenta[1][3] * momenta[1][3];
  double t = m12 - sqrt(s) * (momenta[0][0] - momenta[0][3]);
  double u = m22 - sqrt(s) * (momenta[1][0] - momenta[1][3]);

  return (t * t + u * u + 4 * s * m22 - 2 * m22 * m22) / pow(s - 1000.0, 2);


}
