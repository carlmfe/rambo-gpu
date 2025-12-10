#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>


#include "rng.h"
#include "rambo.h"
#include "integrator.h"



int main(int argc, char* argv[]) {
  
  const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 100000;
  const double energy = 1000.0;
  const int nParticles = 2;
  std::vector<double> masses(nParticles, 0.0);

  double mean, error;

  integrator_2particle(nEvents, energy, masses, mean, error);

  std::cout << "Mean: " << mean << ", Error: " << error << std::endl;

  return 0;
}
