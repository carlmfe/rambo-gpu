#include "integrator.h"

#include <vector>
#include <cmath>
#include "rng.h"
#include "rambo.h"

using namespace std;

const int MAX_PARTICLES = 10;

void integrator(int64_t nEvents, double energy, vector<double> &masses, const int nParticles, double &mean, double &error) {

  double sum_h = 0.0;
  double sum2_h = 0.0;

  const uint64_t base_seed = 123456789ULL;
  for (int i = 0; i < nEvents; ++i) {
  }

  for (int i = 0; i < nEvents; ++i) {
    XorShift64State rng;
    double weight;
    vector<double*> momenta_local = vector<double*>(nParticles);
    for (int j = 0; j < nParticles; ++j) {
      momenta_local[j] = new double[4];
    }

    // Seed the RNG state uniquely per event
    uint64_t s = base_seed + (uint64_t)i * 2ULL + 1ULL;
    // call splitting + seed
    xorshift64_seed(rng, s);

    // Pass the raw data pointer and particle count
    rambo(energy, masses, rng, momenta_local, weight);
    double integrand_value = integrand(nParticles, momenta_local);
    double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
    sum_h += wval;
    sum2_h += wval * wval;
  }
  mean = sum_h / nEvents;
  double variance = (sum2_h / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
}

void integrator_2particle(int64_t nEvents, double energy,  vector<double> &masses, double &mean, double &error) {

  double sum_h = 0.0;
  double sum2_h = 0.0;

  const uint64_t base_seed = 123456789ULL;
  for (int i = 0; i < nEvents; ++i) {
    XorShift64State rng;
    double weight;
    vector<double*> momenta_local = vector<double*>(4);
    for (int j = 0; j < 2; ++j) {
      momenta_local[j] = new double[4];
    }

    // Seed the RNG state uniquely per event
    uint64_t s = base_seed + (uint64_t)i * 2ULL + 1ULL;
    // call splitting + seed
    xorshift64_seed(rng, s);

    // Pass the raw data pointer and particle count
    rambo(energy, masses, rng, momenta_local, weight);
    double integrand_value = integrand_2particle(momenta_local);
    double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
    sum_h += wval;
    sum2_h += wval * wval;
  }
  mean = sum_h / nEvents;
  double variance = (sum2_h / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
}

double integrand(int n, vector<double*> momenta_out) {
  double result = 0.0;
  for (int i = 0; i < n; ++i) {
    result += momenta_out[i][0]; // example: sum of energies
  }
  return result;
}

double integrand_2particle(vector<double*> momenta) {
  // Example specialized integrand for 2-particle case

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
