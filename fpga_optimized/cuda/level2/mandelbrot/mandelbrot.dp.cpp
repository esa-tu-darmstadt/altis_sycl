////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\mandelbrot\mandelbrot.cu
//
// summary:	Mandelbrot class
//
//  @file histo-global.cu histogram with global memory atomics
//
//  origin:
//  (http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "compute_unit.hpp"
#include "constexpr_math.hpp"
#include "fpga_pwr.hpp"
#include "memory_utils.hpp"
#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

// Maximum input-size is 3 for us.
//
using coordinate_t = ac_int<15, false>;
using dwell_t = ac_int<12, true>; // Max 512.

#ifdef _STRATIX10
constexpr int32_t num_cus = 8; // Size1 32 | Size2 8 | Size3 2
#endif
#ifdef _AGILEX
constexpr int32_t num_cus = 5; // Size1 14 | Size2 5 | Size3 1
#endif
constexpr int32_t max_iterations = 128; // Size1 32 | Size2 128 | Size3 512

float kernelTime, transferTime;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float elapsed;

struct complex_t {
  constexpr complex_t() {}
  constexpr complex_t(float x, float y) : re(x), im(y) {}
  float re;
  float im;
};

constexpr SYCL_EXTERNAL inline complex_t operator-(const complex_t &a,
                                                   const complex_t &b) {
  return complex_t(a.re - b.re, a.im - b.im);
}

constexpr SYCL_EXTERNAL inline complex_t operator+(const complex_t &a,
                                                   const complex_t &b) {
  return complex_t(a.re + b.re, a.im + b.im);
}

constexpr SYCL_EXTERNAL inline complex_t operator*(const complex_t &a,
                                                   const complex_t &b) {
  return complex_t(a.re * b.re + a.im * b.im, a.re * b.im + a.im * b.re);
}

template <size_t cu> class mandel_submit_cu;
template <size_t cu> class mandel_calc_cu;
template <size_t cu> class mandel_writeback_cu;

constexpr complex_t cmin = complex_t(-1.5, -1);
constexpr complex_t cmax = complex_t(0.5, 1);
using submit_pa =
    fpga_tools::PipeArray<class submit_pipe_array_id, complex_t, 1, num_cus>;
using final_pa =
    fpga_tools::PipeArray<class final_pipe_array_id, dwell_t, 1, num_cus>;

int pixel_dwell_cpu(int w, int h, int x, int y, int MAX_DWELL, float one_div_w,
                    float one_div_h) {
  complex_t dc = cmax - cmin;
  float fx = (float)x * one_div_w, fy = (float)y * one_div_h;
  complex_t c = cmin + complex_t(float(fx) * dc.re, float(fy) * dc.im);
  int dwell = 0;
  complex_t z = c;
  while (dwell < MAX_DWELL && (float(z.re * z.re + z.im * z.im)) < 4 * 4) {
    z = {z.re * z.re + z.im * z.im + c.re, z.re * z.im + z.re * z.im + c.im};
    dwell++;
  }
  return dwell;
} // pixel_dwell_cpu

void mandelbrot(ResultDatabase &resultDB, OptionParser &op, int size,
                int MAX_DWELL, size_t device_idx, bool verify) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  sycl::queue queue(devices[device_idx]);

  // Allocate memory.
  //
  int w = size, h = size;
  size_t dwell_sz = w * h * sizeof(int);
  int *d_dwells = (int *)sycl::malloc_device(dwell_sz, queue);
  int *h_dwells = (int *)malloc(dwell_sz);
  assert(h_dwells);

  // Calculate Mandelbrot on device.
  //
  constexpr complex_t dc = cmax - cmin;
  const coordinate_t rows_per_cu = h / num_cus;
  const float one_div_w = 1.0f / float(w);
  const float one_div_h = 1.0f / float(h);

  FPGA_PWR_MEAS_START
  start_ct1 = std::chrono::steady_clock::now();

  std::array<sycl::event, num_cus> submit_events;
  SubmitComputeUnits<num_cus, mandel_submit_cu>(
      queue, submit_events, [=](auto ID) {
        const coordinate_t height = h;
        const coordinate_t width = w;
        const coordinate_t start_y = ID * rows_per_cu;
        coordinate_t end_y;
        if constexpr (ID == num_cus)
          end_y = height;
        else
          end_y = (ID + 1) * rows_per_cu;

        [[intel::loop_coalesce(2),
          intel::initiation_interval(1)]] //
        for (coordinate_t y = start_y; y < end_y; y++)
          for (coordinate_t x = 0; x < width; x++) {
            const float fx = float(x) * one_div_w;
            const float fy = float(y) * one_div_h;
            const complex_t c = cmin + complex_t(fx * dc.re, fy * dc.im);
            submit_pa::PipeAt<ID>::write(c);
          }
      });
  std::array<sycl::event, num_cus> calc_events;
  SubmitComputeUnits<num_cus, mandel_calc_cu>(queue, calc_events, [=](auto ID) {
    const coordinate_t height = h;
    const coordinate_t width = w;
    const coordinate_t max_dwell = MAX_DWELL;
    const coordinate_t start_y = ID * rows_per_cu;
    coordinate_t end_y;
    if constexpr (ID == num_cus)
      end_y = height;
    else
      end_y = (ID + 1) * rows_per_cu;

    [[intel::loop_coalesce(2),
      intel::initiation_interval(1)]] //
    for (coordinate_t y = start_y; y < end_y; y++)
      for (coordinate_t x = 0; x < width; x++) {
        auto in = submit_pa::PipeAt<ID>::read();
        complex_t z = in;
        complex_t c = in;

        // Unroll with 32 (Size1), 128 (Size2) or 512 (Size3)
        // -> Currently manual, resulting in a seperate binary for each size.
        ac_int<1, false> flags[max_iterations];
#pragma unroll
        for (int16_t d = 0; d < max_iterations; d++) {
          flags[d] = (z.re * z.re + z.im * z.im < 16);
          z = z * z + c;
        }

        bool done = false;
        dwell_t res = 0;
#pragma unroll
        for (int16_t d = 1; d < max_iterations; d++)
          if (flags[d] & !done)
            res = d + 1;
          else
            done = true;
        final_pa::PipeAt<ID>::write(res <= max_dwell ? res
                                                     : dwell_t(max_dwell));
      }
  });
  std::array<sycl::event, num_cus> writeback_events;
  SubmitComputeUnits<num_cus, mandel_writeback_cu>(
      queue, calc_events, [=](auto ID) {
        const coordinate_t height = h;
        const coordinate_t width = w;
        const coordinate_t start_y = ID * rows_per_cu;
        coordinate_t end_y;
        if constexpr (ID == num_cus)
          end_y = height;
        else
          end_y = (ID + 1) * rows_per_cu;

        [[intel::loop_coalesce(2),
          intel::initiation_interval(1)]] //
        for (coordinate_t y = start_y; y < end_y; y++)
          for (coordinate_t x = 0; x < width; x++)
            d_dwells[x + y * h] = final_pa::PipeAt<ID>::read();
      });
  queue.wait();

  stop_ct1 = std::chrono::steady_clock::now();
  elapsed =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  kernelTime += elapsed * 1.e-3;
  FPGA_PWR_MEAS_END

  // Transfer Mandelbrot from device to host.
  //
  start_ct1 = std::chrono::steady_clock::now();
  queue.memcpy(h_dwells, d_dwells, dwell_sz).wait();
  stop_ct1 = std::chrono::steady_clock::now();
  elapsed =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsed * 1.e-3;

  if (verify) {
    // Calculate Mandelbrot on CPU, for validation.
    //
    int *cpu_dwells = (int *)malloc(dwell_sz);
    for (int64_t x = 0; x < w; x++)
      for (int64_t y = 0; y < h; y++)
        cpu_dwells[x + y * h] =
            pixel_dwell_cpu(w, h, x, y, MAX_DWELL, one_div_w, one_div_h);

    int64_t diff = 0;
    for (int64_t x = 0; x < w; ++x) {
      for (int64_t y = 0; y < h; ++y) {
        if (cpu_dwells[x + y * h] != h_dwells[x + h * y]) {
          std::cout << "diff at " << x << " " << y << ": "
                    << cpu_dwells[x + y * h] << " vs " << h_dwells[x + y * h]
                    << std::endl;
          diff++;
        }
      }
    }

    double tolerance = 0.05;
    double ratio = (double)diff / (double)(dwell_sz);
    if (ratio > tolerance)
      std::cout << "Fail verification - diff larger than tolerance"
                << std::endl;
    else
      std::cout << "Vertification successfull" << std::endl;

    free(cpu_dwells);
  }

  // free data
  sycl::free(d_dwells, queue);
  free(h_dwells);
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
  op.addOption("imageSize", OPT_INT, "0", "image height and width");
  op.addOption("iterations", OPT_INT, "0",
               "iterations of algorithm (the more iterations, the greater "
               "speedup from dynamic parallelism)");
  op.addOption("verify", OPT_BOOL, "0", "verify the results computed on host");
}

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
  printf("Running Mandelbrot\n");

  bool quiet = op.getOptionBool("quiet");
  int imageSize = op.getOptionInt("imageSize");
  int iters = op.getOptionInt("iterations");
  bool verify = op.getOptionBool("verify");
  if (imageSize == 0 || iters == 0) {
    int imageSizes[5] = {2 << 11, 2 << 12, 2 << 13, 2 << 14, 2 << 14};
    int iterSizes[5] = {32, 128, 512, 1024, 8192 * 16};
    imageSize = imageSizes[op.getOptionInt("size") - 1];
    iters = iterSizes[op.getOptionInt("size") - 1];
  }

  if (!quiet) {
    printf("Image Size: %d by %d\n", imageSize, imageSize);
    printf("Num Iterations: %d\n", iters);
    printf("Not using dynamic parallelism\n");
  }

  char atts[1024];
  sprintf(atts, "img:%d,iter:%d", imageSize, iters);

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    if (!quiet)
      printf("Pass %d:\n", i);

    kernelTime = 0.0f;
    transferTime = 0.0f;
    mandelbrot(resultDB, op, imageSize, iters, device_idx, verify);
    resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("mandelbrot_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("mandelbrot_total_time", atts, "sec",
                       transferTime + kernelTime);
    resultDB.AddResult("mandelbrot_parity", atts, "N",
                       transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

    if (!quiet)
      printf("Done.\n");
  }
}
