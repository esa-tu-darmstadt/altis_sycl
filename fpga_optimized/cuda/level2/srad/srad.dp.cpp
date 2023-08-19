////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\srad\srad.cu
//
// summary:	Srad class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "compute_unit.hpp"
#include "cudacommon.h"
#include "fpga_pwr.hpp"
#include "srad.h"
#include "srad_kernel.dp.cpp"

float kernelTime = 0.0f;
float transferTime = 0.0f;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float elapsed;
float *check;

template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

#define SEED 7

void random_matrix(float *I, int rows, int cols);

float srad(ResultDatabase &resultDB, OptionParser &op, float *matrix,
           int imageSize, int speckleSize, int iters, size_t device_idx);

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("imageSize", OPT_INT, "0", "image height and width");
  op.addOption("speckleSize", OPT_INT, "0", "speckle height and width");
  op.addOption("iterations", OPT_INT, "0", "iterations of algorithm");
  op.addOption("gen_input", OPT_BOOL, "0", "create input file for given size");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op,
                  size_t device_idx) {
  printf("Running SRAD\n");

  srand(SEED);
  bool quiet = op.getOptionBool("quiet");

  // set parameters
  int imageSize = op.getOptionInt("imageSize");
  int speckleSize = op.getOptionInt("speckleSize");
  int iters = op.getOptionInt("iterations");
  if (imageSize == 0 || speckleSize == 0 || iters == 0) {
    int imageSizes[5] = {128, 512, 4096, 8192, 16384};
    int iterSizes[5] = {5, 1, 15, 20, 40};
    imageSize = imageSizes[op.getOptionInt("size") - 1];
    speckleSize = imageSize / 2;
    iters = iterSizes[op.getOptionInt("size") - 1];
  }

  if (!quiet) {
    printf("WG size of kernel = %d X %d\n", g_block_size, g_block_size);
    printf("Image Size: %d x %d\n", imageSize, imageSize);
    printf("Speckle size: %d x %d\n", speckleSize, speckleSize);
    printf("Num Iterations: %d\n\n", iters);
  }

  bool gen_input = op.getOptionBool("gen_input");
  int rows = imageSize; // number of rows in the domain
  int cols = imageSize; // number of cols in the domain
  if (gen_input) {
    std::ofstream ostrm("input.txt");
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        ostrm << rand() / (float)RAND_MAX << '\n';
  }

  // run workload
  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    float *matrix = (float *)malloc(imageSize * imageSize * sizeof(float));
    assert(matrix);

    string inputFile = op.getOptionString("inputFile");
    std::ifstream file(inputFile.c_str());

    if (inputFile != "") {
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
          float val;
          file >> val;
          matrix[i * cols + j] = val;
        }
    } else {
      random_matrix(matrix, imageSize, imageSize);
    }

    if (!quiet)
      printf("Pass %d:\n", i);

    float time =
        srad(resultDB, op, matrix, imageSize, speckleSize, iters, device_idx);
    if (!quiet)
      printf("Running SRAD...Done.\n");

    free(matrix);
  }
}

constexpr unsigned bits_needed_for(unsigned x) {
  return x < 2 ? x : 1 + bits_needed_for(x >> 1);
}

constexpr int32_t srad_size_1 = 128 * 128;
constexpr int32_t srad_size_2 = 512 * 512;
constexpr int32_t srad_size_3 = 4096 * 4096;
using idx_t = ac_int<bits_needed_for(srad_size_3 + 1), false>;

constexpr int32_t num_cus = 4;
template <std::size_t ID> class srad1_kernel_cu;
template <std::size_t ID> class srad2_kernel_cu;

float srad(ResultDatabase &resultDB, OptionParser &op, float *matrix,
           int imageSize, int speckleSize, int iters, size_t device_idx) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  sycl::queue queue(devices[device_idx],
                    sycl::property::queue::enable_profiling{});

  kernelTime = 0.0f;
  transferTime = 0.0f;

  int rows = imageSize; // number of rows in the domain
  int cols = imageSize; // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }

  unsigned int r1 = 0;           // y1 position of the speckle
  unsigned int r2 = speckleSize; // y2 position of the speckle
  unsigned int c1 = 0;           // x1 position of the speckle
  unsigned int c2 = speckleSize; // x2 position of the speckle
  constexpr float lambda = 0.5;  // Lambda value
  int niter = iters;             // number of iterations

  int size_I = cols * rows;
  int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  float *I = (float *)malloc(size_I * sizeof(float));
  assert(I);
  float *J = (float *)malloc(size_I * sizeof(float));
  assert(J);
  float *c = (float *)malloc(sizeof(float) * size_I);
  assert(c);

  // Allocate device memory
  float *J_cuda = sycl::malloc_device<float>(size_I, queue);
  float *C_cuda = sycl::malloc_device<float>(size_I, queue);
  float *E_C = sycl::malloc_device<float>(size_I, queue);
  float *W_C = sycl::malloc_device<float>(size_I, queue);
  float *S_C = sycl::malloc_device<float>(size_I, queue);
  float *N_C = sycl::malloc_device<float>(size_I, queue);

  // copy random matrix
  memcpy(I, matrix, rows * cols * sizeof(float));

  for (int k = 0; k < size_I; k++)
    J[k] = (float)exp(I[k]);

  float sum, sum2, tmp;
  for (int iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++)
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }

    float meanROI = sum / size_R;
    float varROI = (sum2 / size_R) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

    // Currently the input size must be divided by 16 - the block size
    int block_x = cols / g_block_size;
    int block_y = rows / g_block_size;

    // Copy data from main memory to device memory
    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(J_cuda, J, sizeof(float) * size_I).wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;

    // Run kernels
    std::array<sycl::event, num_cus> events;
    const int32_t bx_per_cu =
        max(int32_t(std::ceil(block_x / float(num_cus))), 1);
    const int32_t by_per_cu =
        max(int32_t(std::ceil(block_y / float(num_cus))), 1);
    constexpr int32_t bs = g_block_size;
    constexpr float one_by_sixteen = 1.0f / 16.0f;
    queue.wait_and_throw();

    FPGA_PWR_MEAS_START
    constexpr int cci = 128;
    SubmitComputeUnits<num_cus, srad1_kernel_cu>(queue, events, [=](auto ID) {
      const int32_t by_start = ID * by_per_cu;
      const int32_t by_end =
          ::min(int32_t(ID * by_per_cu + by_per_cu), block_y);

      [[intel::initiation_interval(1), intel::ivdep, intel::loop_coalesce(2),
        intel::max_concurrency(cci)]] for (int32_t by = by_start; by < by_end;
                                           by++)
          [[intel::ivdep, intel::max_concurrency(cci)]]
        for (int32_t bx = 0; bx < block_x; bx++) {
          const int32_t base = cols * bs * by + bs * bx;

          // shared memory allocation
          [[intel::private_copies(cci)]] float temp[bs * bs];
          [[intel::private_copies(cci)]] float north[bs * bs];
          [[intel::private_copies(cci)]] float south[bs * bs];
          [[intel::private_copies(cci)]] float east[bs * bs];
          [[intel::private_copies(cci)]] float west[bs * bs];

          [[intel::loop_coalesce(2), intel::initiation_interval(1),
            intel::ivdep,
            intel::speculated_iterations(0)]] //
          for (uint8_t ty = 0; ty < g_block_size; ty++) {
            [[intel::ivdep, intel::initiation_interval(1),
              intel::speculated_iterations(0)]] //
            for (uint8_t tx = 0; tx < g_block_size; tx++) {
              // indices
              const idx_t index = base + cols * ty + tx;
              const idx_t index_n = base + tx - cols;
              const idx_t index_s = base + cols * bs + tx;
              const idx_t index_w = base + cols * ty - 1;
              const idx_t index_e = base + cols * ty + bs;
              if (index_n < rows * cols && index_s < rows * cols &&
                  index_e < rows * cols && index_w < rows * cols &&
                  index_n >= 0 && index_s >= 0 && index_e >= 0 &&
                  index_w >= 0) {
                // load data to shared memory
                temp[tx + ty * bs] = J_cuda[index.to_int()];
                north[tx + ty * bs] =
                    J_cuda[(by == 0) ? (bs * bx + tx) : (index_n.to_int())];
                south[tx + ty * bs] =
                    J_cuda[(by == block_y - 1)
                               ? (cols * bs * (block_y - 1) + bs * bx +
                                  cols * (bs - 1) + tx)
                               : index_s.to_int()];
                west[tx + ty * bs] =
                    J_cuda[(bx == 0) ? (cols * bs * by + cols * ty)
                                     : index_w.to_int()];
                east[tx + ty * bs] =
                    J_cuda[(bx == block_x - 1)
                               ? (cols * bs * by + bs * (block_x - 1) +
                                  cols * ty + bs - 1)
                               : index_e.to_int()];
              }
            }
          }
          [[intel::loop_coalesce(2), intel::initiation_interval(1),
            intel::ivdep,
            intel::speculated_iterations(0)]] //
          for (uint8_t ty = 0; ty < g_block_size; ty++) {
            [[intel::ivdep, intel::initiation_interval(1),
              intel::speculated_iterations(0)]] //
            for (uint8_t tx = 0; tx < g_block_size; tx++) {
              // indices
              const idx_t index = base + cols * ty + tx;
              const idx_t index_n = base + tx - cols;
              const idx_t index_s = base + cols * bs + tx;
              const idx_t index_w = base + cols * ty - 1;
              const idx_t index_e = base + cols * ty + bs;

              if (index_n < rows * cols && index_s < rows * cols &&
                  index_e < rows * cols && index_w < rows * cols &&
                  index_n >= 0 && index_s >= 0 && index_e >= 0 &&
                  index_w >= 0) {
                const float jc = temp[tx + ty * bs];
                const float n =
                    ((ty == 0) ? north[tx + ty * bs] : temp[tx - 1 + ty * bs]) -
                    jc;
                const float w = ((tx == 0) ? west[tx + ty * bs]
                                           : temp[tx + (ty - 1) * bs]) -
                                jc;
                const float e = ((tx == bs - 1) ? east[tx + ty * bs]
                                                : temp[tx + (ty + 1) * bs]) -
                                jc;
                const float s = ((ty == bs - 1) ? south[tx + ty * bs]
                                                : temp[tx + 1 + ty * bs]) -
                                jc;

                const float g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
                const float l = (n + s + w + e) / jc;

                const float num = (0.5f * g2) - (one_by_sixteen * (l * l));
                float den = 1 + (.25f * l);
                const float qsqr = num / (den * den);

                // diffusion coefficent (equ 33)
                den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
                const float c = 1.0f / (1.0f + den);

                // saturate diffusion coefficent
                C_cuda[index] = (c < 0) ? 0 : ((c > 1) ? 1 : c);
                E_C[index] = e;
                W_C[index] = w;
                S_C[index] = s;
                N_C[index] = n;
              }
            }
          }
        }
    });
    float max_elapsed = 0;
    for (auto &e : events) {
      e.wait();
      float elapsed =
          e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      max_elapsed = max(max_elapsed, elapsed * 1.e-9f);
    }
    kernelTime += max_elapsed;

    std::array<sycl::event, num_cus> events2;
    SubmitComputeUnits<num_cus, srad2_kernel_cu>(queue, events2, [=](auto ID) {
      const int32_t by_start = ID * by_per_cu;
      const int32_t by_end =
          ::min(int32_t(ID * by_per_cu + by_per_cu), block_y);

      [[intel::loop_coalesce(2), intel::ivdep(J_cuda),
        intel::max_concurrency(cci)]] for (int32_t by = by_start; by < by_end;
                                           by++)
          [[intel::ivdep(C_cuda), intel::max_concurrency(cci)]]
        for (int32_t bx = 0; bx < block_x; bx++) {
          const int32_t base = cols * bs * by + bs * bx;

          // shared memory allocation
          [[intel::private_copies(cci)]] float temp[bs * bs];
          [[intel::private_copies(cci)]] float south_c[bs * bs];
          [[intel::private_copies(cci)]] float east_c[bs * bs];

          [[intel::loop_coalesce(2), intel::initiation_interval(1),
            intel::speculated_iterations(0)]] //
          for (uint8_t ty = 0; ty < g_block_size; ty++) {
            [[intel::initiation_interval(1),
              intel::speculated_iterations(0)]] //
            for (uint8_t tx = 0; tx < g_block_size; tx++) {
              // indices
              const idx_t index = base + cols * ty + tx;
              const idx_t index_s = base + cols * bs + tx;
              const idx_t index_e = base + cols * ty + bs;

              // load data to shared memory
              temp[tx + ty * bs] = C_cuda[index];
              south_c[tx + ty * bs] =
                  C_cuda[(by == block_y - 1) ? (cols * bs * (block_y - 1) +
                                                bs * bx + cols * (bs - 1) + tx)
                                             : index_s.to_int()];
              east_c[tx + ty * bs] =
                  C_cuda[(bx == block_x - 1)
                             ? (cols * bs * by + bs * (block_x - 1) +
                                cols * ty + bs - 1)
                             : index_e.to_int()];
            }
          }
          [[intel::ivdep(J_cuda), intel::loop_coalesce(2),
            intel::initiation_interval(1),
            intel::speculated_iterations(0)]] //
          for (uint8_t ty = 0; ty < g_block_size; ty++) {
            [[intel::ivdep(J_cuda), intel::initiation_interval(1),
              intel::speculated_iterations(0)]] //
            for (uint8_t tx = 0; tx < g_block_size; tx++) {
              // indices
              const idx_t index = base + cols * ty + tx;

              const float cc = temp[tx + ty * bs];
              const float cs = (ty == bs - 1) ? south_c[tx + ty * bs]
                                              : temp[tx + 1 + ty * bs];
              const float ce = (tx == bs - 1) ? east_c[tx + ty * bs]
                                              : temp[tx + (ty + 1) * bs];
              const float cn = cc;
              const float cw = cc;

              // divergence (equ 58)
              const float d_sum = cn * N_C[index] + cs * S_C[index] +
                                  cw * W_C[index] + ce * E_C[index];

              // image update (equ 61)
              J_cuda[index] += 0.25f * lambda * d_sum;
            }
          }
        }
    });
    max_elapsed = 0;
    for (auto &e : events2) {
      e.wait();
      float elapsed =
          e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      max_elapsed = max(max_elapsed, elapsed * 1.e-9f);
    }
    kernelTime += max_elapsed;
    FPGA_PWR_MEAS_END

    // Copy data from device memory to main memory
    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(J, J_cuda, sizeof(float) * size_I).wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;
  }

  char atts[1024];
  sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
  resultDB.AddResult("srad_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("srad_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("srad_total_time", atts, "sec", kernelTime + transferTime);
  resultDB.AddResult("srad_parity", atts, "N", transferTime / kernelTime);
  resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

  string outfile = op.getOptionString("outputFile");
  if (!outfile.empty()) {
    // Printing output
    if (!op.getOptionBool("quiet"))
      printf("Writing output to %s\n", outfile.c_str());

    FILE *fp = NULL;
    fp = fopen(outfile.c_str(), "w");
    if (!fp) {
      printf("Error: Unable to write to file %s\n", outfile.c_str());
    } else {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
          fprintf(fp, "%.5f ", J[i * cols + j]);
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }
  // write results to validate with srad_gridsync
  check = (float *)malloc(sizeof(float) * size_I);
  assert(check);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      check[i * cols + j] = J[i * cols + j];

  free(I);
  free(J);
  free(c);
  sycl::free(C_cuda, queue);
  sycl::free(J_cuda, queue);
  sycl::free(E_C, queue);
  sycl::free(W_C, queue);
  sycl::free(N_C, queue);
  sycl::free(S_C, queue);

  return kernelTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random matrix. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="I">   	[in,out] If non-null, zero-based index of the. </param>
/// <param name="rows">	The rows. </param>
/// <param name="cols">	The cols. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void random_matrix(float *I, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }
}
