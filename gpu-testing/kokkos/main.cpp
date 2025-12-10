#include <iostream>
#include <Kokkos_Core.hpp>

#include "rambo_kokkos.h"
#include "integrator.h"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  // Kokkos::print_configuration(std::cout);
  {
      const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 100000;
      const double energy = 1000.0;
      const int nParticles = 2;
      Kokkos::View<double*> masses_d("masses", nParticles);
      auto masses_h = Kokkos::create_mirror_view(masses_d);
      for (int i = 0; i < nParticles; ++i) masses_h(i) = 0.0;
      Kokkos::deep_copy(masses_d, masses_h);
      Kokkos::View<const double*> masses_const = masses_d;
      double mean, error;

      Kokkos::Timer timer;
      integrator_2particle_kokkos(nEvents, energy, masses_const, mean, error);
      double elapsed = timer.seconds();

      std::cout << "Mean: " << mean << ", Error: " << error << std::endl;
      std::cout << "Elapsed time: " << elapsed << " s" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}