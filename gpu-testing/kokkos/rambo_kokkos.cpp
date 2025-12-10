#include <iostream>
#include <cmath>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "rambo_kokkos.h"

using RNGPool = Kokkos::Random_XorShift64_Pool<>;
using namespace std;

const int MAX_PARTICLES = 10; // Define maximum number of particles


// Use the same non-inline macro for the definition
KOKKOS_FUNCTION
void rambo_device(double energy, const double* masses, int n,
                  Kokkos::Random_XorShift64_Pool<>::generator_type &rng,
                  double momenta_out[][4],
                  double &weight) {
  /**********************************************************************
   *                       rambo                                         *
   *    ra(ndom)  m(omenta)  b(eautifully)  o(rganized)                  *
   *                                                                     *
   *    a democratic multi-particle phase space generator                *
   *    authors:  s.d. ellis,  r. kleiss,  w.j. stirling                 *
   *    this is version 1.0 -  written by r. kleiss                      *
   *    -- adjusted by hans kuijf, weights are logarithmic (20-08-90)    *
   *                                                                     *
   *    n  = number of particles                                         *
   *    energy = total centre-of-mass energy                             *
   *    masses = particle masses ( dim=nexternal-nincoming )             *
   *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
   *    weight = weight of the event                                     *
   ***********************************************************************/

  // Local (stack) arrays sized to MAX_PARTICLES
  double q[MAX_PARTICLES][4];
  double p[MAX_PARTICLES][4];
  double z[MAX_PARTICLES];
  double r[4];
  double b[3];
  double p2[MAX_PARTICLES];
  double xm2[MAX_PARTICLES];
  double e[MAX_PARTICLES];
  double v[MAX_PARTICLES];

  int iwarn[5] = {0,0,0,0,0};

  const double acc = 1e-14;
  const int itmax = 1000;
  int ibegin = 0;
  const double twopi = 8.0 * atan(1.0);
  const double po2log = log(twopi / 4.0);
  
  // initialization step: factorials for the phase space weight
  if(ibegin == 0) {
    ibegin = 1;
    z[1] = po2log;
    for(int k = 2; k < n; k++)
      z[k] = z[k-1] + po2log - 2.0 * log(double(k-1));
    for(int k = 2; k < n; k++)
      z[k] = (z[k] - log(double(k)));
  }

  // check on the number of particles
  if (n < 1 || n > MAX_PARTICLES) {
    // Kokkos::abort("Too few or many particles: " + std::to_string(n));
    return;
  }

  // check whether total energy is sufficient; count nonzero masses
  double xmt = 0.;
  int nm = 0;
  for(int i = 0; i < n; i++){
    if(masses[i] != 0.) nm = nm + 1;
    xmt = xmt + abs(masses[i]);
  }
  if (xmt > energy){
    // Kokkos::abort("Too low energy: " + std::to_string(energy) + " needed " + std::to_string(xmt));
    return;
  }
  // the parameter values are now accepted

  // generate n massless momenta in infinite phase space
  for(int i = 0; i < n; i++) {
    double r1 = rng.drand();
    double c = 2.0 * r1 - 1.0;
    double s = sqrt(1.0 - c * c);
    double f = twopi * rng.drand();
    r1 = rng.drand();
    double r2 = rng.drand();
    q[i][0] = -log(r1 * r2);
    q[i][3] = q[i][0] * c;
    q[i][2] = q[i][0] * s * cos(f);
    q[i][1] = q[i][0] * s * sin(f);
  }
  // calculate the parameters of the conformal transformation
  for (int i = 0; i < 4; i++) {
    r[i] = 0.0;
  };
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < 4; k++) {
      r[k] += q[i][k];
    }
  }
  double rmas = sqrt(r[0]*r[0] - r[1]*r[1] - r[2]*r[2] - r[3]*r[3]);
  for (int k = 1; k < 4; k++) {
    b[k-1] = -r[k] / rmas;
  }
  double g = r[0] / rmas;
  double a = 1.0 / (1.0 + g);
  double x = energy / rmas;

  // transform the q's conformally into the p's
  for (int i = 0; i < n; ++i) {
    double bq = b[0]*q[i][1] + b[1]*q[i][2] + b[2]*q[i][3];
    for (int k = 1; k < 4; ++k) {
      p[i][k] = x * (q[i][k] + b[k-1] * (q[i][0] + a * bq));
    }
    p[i][0] = x * (g * q[i][0] + bq);
  }

  // calculate weight and possible warnings
  weight = po2log;
  if (n != 2) {
    weight += (2.0 * n - 4.0) * log(energy) + z[n-1];
  }
  if (weight < -180.0) {
    iwarn[0] = iwarn[0] + 1;
  }
  if (weight > 174.0) {
    iwarn[1] = iwarn[1] + 1;
  }

  // return for weighted massless momenta
  if (nm == 0) {
    // return log of weight
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 4; ++k) {
        momenta_out[i][k] = p[i][k];
      }
    }
    return;
  }

  // massive particles: rescale the momenta by a factor x
  double xmax = sqrt(1.0 - (xmt * xmt) / (energy * energy));
  for (int i = 0; i < n; ++i) {
    xm2[i] = masses[i] * masses[i];
    p2[i] = p[i][0] * p[i][0];
  }
  int iter = 0;
  x = xmax;
  while (true) {
    double f0 = -energy;
    double g0 = 0.0;
    double x2 = x * x;
    for (int i = 0; i < n; ++i) {
      e[i] = sqrt(xm2[i] + x2 * p2[i]);
      f0 += e[i];
      g0 += p2[i] / e[i];
    }
    if (fabs(f0) <= energy * acc) break;
    iter += 1;
    if (iter > itmax) {
      iwarn[2] = iwarn[2] + 1;
      break;
    }
    x += x - f0 / (g0 * x);
  }
  for (int i = 0; i < n; i++) {
    v[i] = x * p[i][0];
    for (int k = 1; k < 4; k++) {
      p[i][k] = p[i][k] * x;
    }
    p[i][0] = e[i];
  }

  // calculate the mass-effect weight factor
  double wt2 = 1.0;
  double wt3 = 0.0;
  for (int i = 0; i < n; i++) {
    wt2 = wt2 * v[i] / e[i];
    wt3 = wt3 + (v[i] * v[i]) / e[i];
  }
  double wtm = (2.0 * n - 3.0) * log(x) + log(wt2 / wt3 * energy);

  // return for  weighted massive momenta
  weight += wtm;
  if (weight < -180.0) {
    iwarn[0] = iwarn[0] + 1;
  }
  if (weight > 174.0) {
    iwarn[1] = iwarn[1] + 1;
  }
  // return log of weight
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 4; ++k) {
      momenta_out[i][k] = p[i][k];
    }
  }
  return;
}
