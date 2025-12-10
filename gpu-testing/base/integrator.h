#pragma once
#include <cstdint>
#include <vector>

using namespace std;

void integrator(int64_t nEvents, double energy, vector<double> &masses, const int nParticles, double &mean, double &error);

void integrator_2particle(int64_t nEvents, double energy,  vector<double> &masses, double &mean, double &error);

double integrand(int n, vector<double*> momenta_out);

double integrand_2particle(vector<double*> momenta_out);
