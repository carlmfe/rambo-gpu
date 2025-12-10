#include <iostream>
#include <cmath>
#include <cstdint>

using namespace std;

#define CUDA_MAX_PARTICLES 10

// Use a simpler, smaller RNG that requires fewer registers
__device__ uint64_t xorshift64(uint64_t &state)
{
  uint64_t x = state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  state = x;
  return x;
}

__device__ double uniform_random(uint64_t &state)
{
  // Convert to [0, 1)
  return (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ void rambo(const double et, const double *xm, const int n, double momenta_out[CUDA_MAX_PARTICLES][4], double &wt, uint64_t &rng_state)
{
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
   *    et = total centre-of-mass energy                                 *
   *    xm = particle masses ( dim=nexternal-nincoming )                 *
   *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
   *    wt = weight of the event                                         *
   ***********************************************************************/

  double twopi = 8. * atan(1.);
  double po2log = log(twopi / 4.);
  double acc = 1e-14;
  int itmax = 6;

  // check on the number of particles
  if (n < 1 || n > 101)
    return;

  // check whether total energy is sufficient
  double xmt = 0.;
  int nm = 0;
  for (int i = 0; i < n; i++)
  {
    if (xm[i] != 0.)
      nm = nm + 1;
    xmt = xmt + fabs(xm[i]);
  }
  if (xmt > et)
    return;

  // Initialize weight using only loop-computed factorial terms
  wt = po2log;
  if (n != 2)
  {
    double z_sum = po2log;
    for (int k = 2; k < n; k++)
    {
      z_sum = z_sum + po2log - 2. * log((double)(k - 1));
      z_sum = z_sum - log((double)k);
    }
    wt = (2. * n - 4.) * log(et) + z_sum;
  }

  // generate n massless momenta - compute r[] on-the-fly
  double r[4] = {0., 0., 0., 0.};
  double q_tmp[4];

  for (int i = 0; i < n; i++)
  {
    double r1 = uniform_random(rng_state);
    double c = 2. * r1 - 1.;
    double s = sqrt(1. - c * c);
    double f = twopi * uniform_random(rng_state);
    r1 = uniform_random(rng_state);
    double r2 = uniform_random(rng_state);

    q_tmp[0] = -log(r1 * r2);
    q_tmp[3] = q_tmp[0] * c;
    q_tmp[2] = q_tmp[0] * s * cos(f);
    q_tmp[1] = q_tmp[0] * s * sin(f);

    // accumulate r and directly compute p (conformal transform will be applied next)
    for (int k = 0; k < 4; k++)
    {
      r[k] = r[k] + q_tmp[k];
      momenta_out[i][k] = q_tmp[k];
    }
  }

  // calculate the parameters of the conformal transformation
  double rmas = sqrt(r[0] * r[0] - r[3] * r[3] - r[2] * r[2] - r[1] * r[1]);
  double b[3];
  for (int k = 1; k < 4; k++)
    b[k - 1] = -r[k] / rmas;
  double g = r[0] / rmas;
  double a = 1. / (1. + g);
  double x = et / rmas;

  // transform the q's conformally into the p's
  for (int i = 0; i < n; i++)
  {
    double bq = b[0] * momenta_out[i][1] + b[1] * momenta_out[i][2] + b[2] * momenta_out[i][3];
    for (int k = 1; k < 4; k++)
      momenta_out[i][k] = x * (momenta_out[i][k] + b[k - 1] * (momenta_out[i][0] + a * bq));
    momenta_out[i][0] = x * (g * momenta_out[i][0] + bq);
  }

  if (wt < -180.)
    wt = -180.;
  if (wt > 174.)
    wt = 174.;

  // return for weighted massless momenta
  if (nm == 0)
    return;

  // massive particles: rescale the momenta by a factor x
  double xmax = sqrt(1. - (xmt / et) * (xmt / et));
  x = xmax;
  double accu = et * acc;

  int iter = 0;
  while (true)
  {
    double f0 = -et;
    double g0 = 0.;
    double x2 = x * x;

    for (int i = 0; i < n; i++)
    {
      double p2_i = momenta_out[i][1] * momenta_out[i][1] +
                    momenta_out[i][2] * momenta_out[i][2] +
                    momenta_out[i][3] * momenta_out[i][3];
      double e_i = sqrt(xm[i] * xm[i] + x2 * p2_i);
      f0 = f0 + e_i;
      g0 = g0 + p2_i / e_i;
    }

    if (fabs(f0) <= accu)
      break;
    iter = iter + 1;
    if (iter > itmax)
      break;
    x = x - f0 / (x * g0);
  }

  // apply rescaling and compute mass-effect weight
  double wt2 = 1.;
  double wt3 = 0.;

  for (int i = 0; i < n; i++)
  {
    double p2_i = momenta_out[i][1] * momenta_out[i][1] +
                  momenta_out[i][2] * momenta_out[i][2] +
                  momenta_out[i][3] * momenta_out[i][3];
    double e_i = sqrt(xm[i] * xm[i] + x * x * p2_i);
    double v_i = x * momenta_out[i][0];

    for (int k = 1; k < 4; k++)
      momenta_out[i][k] = x * momenta_out[i][k];
    momenta_out[i][0] = e_i;

    wt2 = wt2 * v_i / e_i;
    wt3 = wt3 + v_i * v_i / e_i;
  }

  double wtm = (2. * n - 3.) * log(x) + log(wt2 / wt3 * et);
  wt = wt + wtm;

  if (wt < -180.)
    wt = -180.;
  if (wt > 174.)
    wt = 174.;
}

__device__ double integrand_func(double momenta[CUDA_MAX_PARTICLES][4], int n)
{
  if (n == 2)
  {
    double tot_momentum[4];
    for (int k = 0; k < 4; k++)
    {
      tot_momentum[k] = momenta[0][k] + momenta[1][k];
    }
    double s = tot_momentum[0] * tot_momentum[0] - tot_momentum[1] * tot_momentum[1] - tot_momentum[2] * tot_momentum[2] - tot_momentum[3] * tot_momentum[3];
    double m12 = momenta[0][0] * momenta[0][0] - momenta[0][1] * momenta[0][1] - momenta[0][2] * momenta[0][2] - momenta[0][3] * momenta[0][3];
    double m22 = momenta[1][0] * momenta[1][0] - momenta[1][1] * momenta[1][1] - momenta[1][2] * momenta[1][2] - momenta[1][3] * momenta[1][3];
    double t = m12 - sqrt(s) * (momenta[0][0] - momenta[0][3]);
    double u = m22 - sqrt(s) * (momenta[1][0] - momenta[1][3]);

    return (t * t + u * u + 4 * s * m22 - 2 * m22 * m22) / pow(s - 1000.0, 2);
  }

  return 0.0;
}

__global__ void integrand_kernel(int nEvents, double et, double *xm, int n, double *sum, double *sum2, uint64_t seed = 12345)
{
  // event number being simulated
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double wt = 0.0;
  double tmp_val;
  double local_sum = 0.0;
  double local_sum2 = 0.0;
  double local_momenta[CUDA_MAX_PARTICLES][4];

  // Initialize random number generator state per thread
  uint64_t rng_state = seed + idx;

  while (idx < nEvents)
  {
    // generate event
    rambo(et, xm, n, local_momenta, wt, rng_state);
    tmp_val = integrand_func(local_momenta, n) * exp(wt);
    local_sum += tmp_val;
    local_sum2 += tmp_val * tmp_val;
    idx += gridDim.x * blockDim.x;
  }

  atomicAdd(sum, local_sum);
  atomicAdd(sum2, local_sum2);

  return;
}

int main(int argc, char **argv)
{
  const int nEvents = (argc > 1) ? atoi(argv[1]) : 1000000;
  const uint64_t seed = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 12345;
  const double et = 1000.0;
  const int nParticles = 2;
  double xm[nParticles] = {0.0, 0.0};

  // Figure out grid and block sizes
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  // Use smaller block size to reduce register pressure per block
  int blockSize = 256; // Reduced from maxThreadsPerBlock
  int numBlocks = min((nEvents + blockSize - 1) / blockSize, prop.maxGridSize[0]);

  double h_sum = 0.0;
  double h_sum2 = 0.0;

  // Allocate device memory
  double *d_xm = nullptr;
  double *d_sum = nullptr;
  double *d_sum2 = nullptr;
  cudaMalloc(&d_xm, sizeof(double) * nParticles);
  cudaMalloc(&d_sum, sizeof(double));
  cudaMalloc(&d_sum2, sizeof(double));

  cudaMemcpy(d_xm, xm, sizeof(double) * (size_t)nParticles, cudaMemcpyHostToDevice);
  cudaMemset(d_sum, 0.0, sizeof(double));
  cudaMemset(d_sum2, 0.0, sizeof(double));

  // Launch kernel
  cout << "Running " << nEvents << " events with " << nParticles << " particles each." << endl;
  cout << "Using " << numBlocks << " blocks of " << blockSize << " threads." << endl;
  integrand_kernel<<<numBlocks, blockSize>>>(nEvents, et, d_xm, nParticles, d_sum, d_sum2, seed);

  cudaError_t err;

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    cout << "Kernel launch error: " << cudaGetErrorString(err) << endl;
  }

  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    cout << "Cuda synchronize error: " << cudaGetErrorString(err) << endl;
  }

  cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum2, d_sum2, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_xm);
  cudaFree(d_sum);
  cudaFree(d_sum2);

  double mean = h_sum / nEvents;
  double error = sqrt((h_sum2 / (double)nEvents - mean * mean) / (double)nEvents);

  cout << "Mean: " << mean << ", Error: " << error << endl;

  return 0;
}
