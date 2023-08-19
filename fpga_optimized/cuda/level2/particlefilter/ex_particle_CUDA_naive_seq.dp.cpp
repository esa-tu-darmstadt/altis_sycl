/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:
// altis\src\cuda\level2\particlefilter\ex_particle_CUDA_naive_seq.cu
//
// summary:	Exception particle cuda float sequence class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "compute_unit.hpp"
#include "fpga_pwr.hpp"
#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

#include <chrono>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#define PI 3.1415926535897932

template <size_t cu> class kernel_cu;
template <size_t cu> class writeback_cu;

bool norand = false;
bool verbose = false;
bool quiet = false;

/// @brief	@var M value for Linear Congruential Generator (LCG); use GCC's
/// value
long M = INT_MAX;
/// @brief	@var A value for LCG
int A = 1103515245;
/// @brief	@var C value for LCG
int C = 12345;

int findIndexSeq(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1)
    return lengthCDF - 1;
  return index;
}

int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
  if (endIndex < beginIndex)
    return -1;
  int middleIndex;
  while (endIndex > beginIndex) {
    middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
    if (CDF[middleIndex] >= value) {
      if (middleIndex == 0)
        return middleIndex;
      else if (CDF[middleIndex - 1] < value)
        return middleIndex;
      else if (CDF[middleIndex - 1] == value) {
        while (CDF[middleIndex] == value && middleIndex >= 0)
          middleIndex--;
        middleIndex++;
        return middleIndex;
      }
    }
    if (CDF[middleIndex] > value)
      endIndex = middleIndex - 1;
    else
      beginIndex = middleIndex + 1;
  }
  return -1;
}

double roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY,
           int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++)
    for (y = 0; y < *dimY; y++)
      for (z = 0; z < *dimZ; z++)
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
}

double randu(int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((double)M));
}

double randn(int *seed, int index) {
  /*Box-Muller algorithm*/
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2 * PI * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++)
    for (y = 0; y < *dimY; y++)
      for (z = 0; z < *dimZ; z++)
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            (int)(5 * randn(seed, 0));
}

void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; x++) {
    for (int y = 0; y < diameter; y++) {
      double distance =
          sqrt((double)(x - radius + 1) * (double)(x - radius + 1) +
               (double)(y - radius + 1) * (double)(y - radius + 1));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance = sqrt((double)(x - posX) * (double)(x - posX) +
                             (double)(y - posY) * (double)(y - posY));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}

void videoSequence(OptionParser &op, int *I, int IszX, int IszY, int Nfr,
                   int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY / 2.0);
  int y0 = (int)roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  int *newMatrix = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  assert(newMatrix);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}

double calcLikelihoodSum(int *I, int *ind, int numOnes) {
  double likelihoodSum = 0.0;
  int y;
  for (y = 0; y < numOnes; y++)
    likelihoodSum += ((double)(I[ind[y]] - 100) * (double)(I[ind[y]] - 100) -
                      (double)(I[ind[y]] - 228) * (double)(I[ind[y]] - 228)) /
                     50.0;
  return likelihoodSum;
}

int findIndex(double *CDF, int lengthCDF, double value) {
  int index = -1;
  for (int x = 0; x < lengthCDF; x++)
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  if (index == -1)
    return lengthCDF - 1;
  return index;
}

using cached_lsu =
    sycl::ext::intel::lsu</*
        sycl::ext::intel::statically_coalesce<false>,
        sycl::ext::intel::burst_coalesce<true>,
        sycl::ext::intel::cache<524288>*/>;

#ifdef _STRATIX10
constexpr int32_t num_cus = 10;
constexpr uint8_t u_vals_size = 50; // #ParticlesToProcessAtOnce
#endif
#ifdef _AGILEX
constexpr int32_t num_cus = 4;
constexpr uint8_t u_vals_size = 24; // #ParticlesToProcessAtOnce
#endif

using idx_pipe =
    fpga_tools::PipeArray<class idx_pipe_id, int32_t, u_vals_size * 2, num_cus>;

void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles, OptionParser &op, ResultDatabase &resultDB,
                    size_t device_idx) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  sycl::queue queue(devices[device_idx],
                    sycl::property::queue::enable_profiling{});

  float kernelTime = 0.0f;
  float transferTime = 0.0f;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  float elapsedTime;

  int max_size = IszX * IszY * Nfr;
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)malloc(diameter * diameter * sizeof(int));
  assert(disk);
  strelDisk(disk, radius);

  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++)
    for (y = 0; y < diameter; y++)
      if (disk[x * diameter + y] == 1)
        countOnes++;
  double *objxy = (double *)malloc(countOnes * 2 * sizeof(double));
  assert(objxy);
  getneighbors(disk, countOnes, objxy, radius);

  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)malloc(sizeof(double) * Nparticles);
  assert(weights);
  for (x = 0; x < Nparticles; x++)
    weights[x] = 1 / ((double)(Nparticles));

  // initial likelihood to 0.0
  double *likelihood = NULL;
  double *arrayX = NULL;
  double *arrayY = NULL;
  double *xj = NULL;
  double *yj = NULL;
  double *CDF = NULL;
  likelihood = (double *)malloc(sizeof(double) * Nparticles);
  assert(likelihood);
  arrayX = (double *)malloc(sizeof(double) * Nparticles);
  assert(arrayX);
  arrayY = (double *)malloc(sizeof(double) * Nparticles);
  assert(arrayY);
  xj = (double *)malloc(sizeof(double) * Nparticles);
  assert(xj);
  yj = (double *)malloc(sizeof(double) * Nparticles);
  assert(yj);
  CDF = (double *)malloc(sizeof(double) * Nparticles);
  assert(CDF);

  // GPU copies of arrays
  int *ind = (int *)malloc(sizeof(int) * countOnes);
  double *u = (double *)malloc(sizeof(double) * Nparticles);
  assert(ind);
  assert(u);

  double *arrayX_GPU = sycl::malloc_device<double>(Nparticles, queue);
  double *arrayY_GPU = sycl::malloc_device<double>(Nparticles, queue);
  double *xj_GPU = sycl::malloc_device<double>(Nparticles, queue);
  double *yj_GPU = sycl::malloc_device<double>(Nparticles, queue);
  double *CDF_GPU = sycl::malloc_device<double>(Nparticles, queue);
  double *u_GPU = sycl::malloc_device<double>(Nparticles, queue);

  for (x = 0; x < Nparticles; x++) {
    arrayX[x] = xe;
    arrayY[x] = ye;
  }

  int k;
  // double * Ik = (double *)malloc(sizeof(double)*IszX*IszY);
  int indX, indY;
  for (k = 1; k < Nfr; k++) {
    // apply motion model
    // draws sample from motion model (random walk). The only prior
    // information is that the object moves 2x as fast as in the y direction

    for (x = 0; x < Nparticles; x++) {
      arrayX[x] = arrayX[x] + 1.0 + 5.0 * randn(seed, x);
      arrayY[x] = arrayY[x] - 2.0 + 2.0 * randn(seed, x);
    }
    // particle filter likelihood
    for (x = 0; x < Nparticles; x++) {

      // compute the likelihood: remember our assumption is that you know
      //  foreground and the background image intensity distribution.
      //  Notice that we consider here a likelihood ratio, instead of
      //  p(z|x). It is possible in this case. why? a hometask for you.
      // calc ind
      for (y = 0; y < countOnes; y++) {
        indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
        indY = roundDouble(arrayY[x]) + objxy[y * 2];
        ind[y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
        if (ind[y] >= max_size)
          ind[y] = 0;
      }
      likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
      likelihood[x] = likelihood[x] / countOnes;
    }
    // update & normalize weights
    // using equation (63) of Arulampalam Tutorial
    for (x = 0; x < Nparticles; x++)
      weights[x] = weights[x] * exp(likelihood[x]);
    double sumWeights = 0;
    for (x = 0; x < Nparticles; x++)
      sumWeights += weights[x];
    for (x = 0; x < Nparticles; x++)
      weights[x] = weights[x] / sumWeights;
    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
      xe += arrayX[x] * weights[x];
      ye += arrayY[x] * weights[x];
    }
    if (verbose && !quiet) {
      printf("XE: %lf\n", xe);
      printf("YE: %lf\n", ye);
      double distance =
          sqrt(sycl::pown((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
               sycl::pown((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
      printf("%lf\n", distance);
    }

    // resampling
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++)
      CDF[x] = weights[x] + CDF[x - 1];
    double u1 = (1 / ((double)(Nparticles))) * randu(seed, 0);
    for (x = 0; x < Nparticles; x++)
      u[x] = u1 + x / ((double)(Nparticles));

    // CUDA memory copying from CPU memory to GPU memory
    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(arrayX_GPU, arrayX, sizeof(double) * Nparticles);
    queue.memcpy(arrayY_GPU, arrayY, sizeof(double) * Nparticles);
    queue.memcpy(xj_GPU, xj, sizeof(double) * Nparticles);
    queue.memcpy(yj_GPU, yj, sizeof(double) * Nparticles);
    queue.memcpy(CDF_GPU, CDF, sizeof(double) * Nparticles);
    queue.memcpy(u_GPU, u, sizeof(double) * Nparticles);
    queue.wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;

    // KERNEL FUNCTION CALL
    // u_vals 8  1 cus: 58us, 200us, 256.382s
    // u_vals 32 4 cus: 52us, 323us, 62.72s
    // u_vals 32 4+4 cus: 148us, 765us, 32.72s
    // FPGA_PWR_MEAS_START
    const int32_t num_particles_per_cu =
        std::ceil(Nparticles / double(num_cus));
    std::array<sycl::event, num_cus> writeback_events;
    SubmitComputeUnits<num_cus, writeback_cu>(
        queue, writeback_events, [=](auto ID) {
          const int32_t part_start = num_particles_per_cu * ID;
          const int32_t part_stop = num_particles_per_cu * (ID + 1);

          for (int32_t particle_idx = part_start; particle_idx < part_stop;
               particle_idx += u_vals_size) {
            [[intel::initiation_interval(1), intel::ivdep,
              intel::speculated_iterations(0)]] //
            for (uint8_t i = 0; i < u_vals_size; i++) {
              if (particle_idx + i < part_stop) {
                const int32_t idx = idx_pipe::PipeAt<ID>::read();
                xj_GPU[particle_idx + i] = arrayX_GPU[idx];
                yj_GPU[particle_idx + i] = arrayY_GPU[idx];
              }
            }
          }
        });
    std::array<sycl::event, num_cus> submit_events;
    SubmitComputeUnits<num_cus, kernel_cu>(queue, submit_events, [=](auto ID) {
      const int32_t part_start = num_particles_per_cu * ID;
      const int32_t part_stop = num_particles_per_cu * (ID + 1);

      // Process u_vals_size particles at once. This means, when iterating
      // through the particle array in the inner loop, we calculate the
      // resulting index of u_vals_size particles at once. Reduces the amount of
      // times we need to read all particle-data by u_vals_size !
      [[intel::initiation_interval(1)]] //
      for (int32_t particle_idx = part_start; particle_idx < part_stop;
           particle_idx += u_vals_size) {
        double u_vals[u_vals_size];
#pragma unroll 8
        for (uint8_t i = 0; i < u_vals_size; i++)
          u_vals[i] =
              (particle_idx + i) < part_stop ? u_GPU[particle_idx + i] : .0;

        constexpr uint8_t cdf_vals_size = 8; // 8xdouble=512bit
        int indices[u_vals_size] = {-1, -1, -1, -1, -1, -1, -1, -1};
        [[intel::initiation_interval(1),
          intel::speculated_iterations(0)]] //
        for (int32_t x = 0; x < Nparticles; x += cdf_vals_size) {
          double cdf_vals[cdf_vals_size];
#pragma unroll
          for (uint8_t i = 0; i < cdf_vals_size; i++)
            cdf_vals[i] = (x + i) < Nparticles ? CDF_GPU[x + i] : .0;

#pragma unroll
          for (uint8_t u = 0; u < u_vals_size; u++) {
            const double u_val = u_vals[u];

#pragma unroll
            for (uint8_t c = 0; c < cdf_vals_size; c++)
              if (cdf_vals[c] < u_val)
                if (indices[u] == -1)
                  indices[u] = x + c;
          }
        }

        [[intel::initiation_interval(1), intel::ivdep,
          intel::speculated_iterations(0)]] for (uint8_t i = 0; i < u_vals_size;
                                                 i++)
          if (particle_idx + i < part_stop)
            idx_pipe::PipeAt<ID>::write(indices[i] == -1 ? Nparticles - 1
                                                         : indices[i]);
      }
    });
    queue.wait();
    // FPGA_PWR_MEAS_END
    float elapsed = .0f;
    for (auto &e : writeback_events) {
      float t =
          e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      if (t > elapsed)
        elapsed = t;
    }
    kernelTime += elapsed * 1.e-9;

    // CUDA memory copying back from GPU to CPU memory
    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(yj, yj_GPU, sizeof(double) * Nparticles);
    queue.memcpy(xj, xj_GPU, sizeof(double) * Nparticles);
    queue.wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;

    for (x = 0; x < Nparticles; x++) {
      // reassign arrayX and arrayY
      arrayX[x] = xj[x];
      arrayY[x] = yj[x];
      weights[x] = 1 / ((double)(Nparticles));
    }
  }

  char atts[1024];
  sprintf(atts, "dimx:%d, dimy:%d, numframes:%d, numparticles:%d", IszX, IszY,
          Nfr, Nparticles);
  resultDB.AddResult("particlefilter_naive_kernel_time", atts, "sec",
                     kernelTime);
  resultDB.AddResult("particlefilter_naive_transfer_time", atts, "sec",
                     transferTime);
  resultDB.AddResult("particlefilter_naive_total_time", atts, "sec",
                     kernelTime + transferTime);
  resultDB.AddResult("particlefilter_naive_parity", atts, "N",
                     transferTime / kernelTime);
  resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

  // CUDA freeing of memory
  sycl::free(u_GPU, queue);
  sycl::free(CDF_GPU, queue);
  sycl::free(yj_GPU, queue);
  sycl::free(xj_GPU, queue);
  sycl::free(arrayY_GPU, queue);
  sycl::free(arrayX_GPU, queue);

  // free memory
  free(disk);
  free(objxy);
  free(weights);
  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(u);
  free(ind);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void addBenchmarkSpecOptions(OptionParser &op)
///
/// @brief	Adds a benchmark specifier options
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	op	The operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("dimx", OPT_INT, "0", "grid x dimension", 'x');
  op.addOption("dimy", OPT_INT, "0", "grid y dimension", 'y');
  op.addOption("framecount", OPT_INT, "0", "number of frames to track across",
               'f');
  op.addOption("np", OPT_INT, "0", "number of particles to use");
  op.addOption("norand", OPT_BOOL, "0",
               "do not use random input for vertification", 'r');
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void particlefilter_naive(ResultDatabase &resultDB, int args[]);
///
/// @brief	Particlefilter naive
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database.
/// @param 		   	args		The arguments.
////////////////////////////////////////////////////////////////////////////////////////////////////

void particlefilter_naive(ResultDatabase &resultDB, OptionParser &op,
                          int args[], size_t device_idx);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
///
/// @brief	Executes the benchmark operation
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database.
/// @param [in,out]	op			The operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op,
                  size_t device_idx) {
  printf("Running ParticleFilter (naive)\n");
  int args[4];
  args[0] = op.getOptionInt("dimx");
  args[1] = op.getOptionInt("dimy");
  args[2] = op.getOptionInt("framecount");
  args[3] = op.getOptionInt("np");
  bool preset = false;
  verbose = op.getOptionBool("verbose");
  quiet = op.getOptionBool("quiet");
  norand = op.getOptionBool("norand");

  for (int i = 0; i < 4; i++)
    if (args[i] <= 0)
      preset = true;

  if (preset) {
    int probSizes[4][4] = {{10, 10, 2, 100},
                           {40, 40, 5, 500},
                           {200, 200, 8, 500000},
                           {500, 500, 15, 1000000}};
    int size = op.getOptionInt("size") - 1;
    for (int i = 0; i < 4; i++)
      args[i] = probSizes[size][i];
  }

  if (!quiet)
    printf("Using dimx=%d, dimy=%d, framecount=%d, numparticles=%d\n", args[0],
           args[1], args[2], args[3]);

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    if (!quiet)
      printf("Pass %d: ", i);

    particlefilter_naive(resultDB, op, args, device_idx);
    if (!quiet)
      printf("Done.\n");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void particlefilter_naive(ResultDatabase &resultDB, int args[])
///
/// @brief	Particlefilter naive
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database.
/// @param 		   	args		The arguments.
////////////////////////////////////////////////////////////////////////////////////////////////////

void particlefilter_naive(ResultDatabase &resultDB, OptionParser &op,
                          int args[], size_t device_idx) {
  int IszX = args[0];
  int IszY = args[1];
  int Nfr = args[2];
  int Nparticles = args[3];

  // establish seed
  int *seed = (int *)malloc(sizeof(int) * Nparticles);
  assert(seed);
  for (int i = 0; i < Nparticles; i++)
    seed[i] = norand ? i * i : time(0) * i;

  int *I = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  assert(I);

  videoSequence(op, I, IszX, IszY, Nfr, seed);
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, op, resultDB,
                 device_idx);

  free(seed);
  free(I);
}