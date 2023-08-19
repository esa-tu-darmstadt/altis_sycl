////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\where\where.cu
//
// summary:	Where class
//
// origin:
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#include "fpga_pwr.hpp"
#include "unrolled_loop.hpp"

#include <chrono>
#include <stdio.h>

constexpr uint16_t g_thread_cnt = 1024;
#ifdef _STRATIX10
constexpr int32_t num_k1_cu = 2;
constexpr int32_t num_k3_cu = 20;
#endif
#ifdef _AGILEX
constexpr int32_t num_k1_cu = 4;
constexpr int32_t num_k3_cu = 25;
#endif

float kernelTime = 0.0f;
float transferTime = 0.0f;

std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float elapsedTime;

void seedArr(int *arr, int size) {
  for (int i = 0; i < size; i++)
    arr[i] = rand() % 100;
}

template <std::size_t ID> class mark_matches_cu;
class exclusive_scan_id;
template <std::size_t ID> class map_matches_cu;

void where(ResultDatabase &resultDB, OptionParser &op, int size, int coverage,
           size_t device_idx) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  sycl::queue queue(devices[device_idx],
                    sycl::property::queue::enable_profiling{});

  int *d_arr = sycl::malloc_device<int>(size, queue);
  int *arr = (int *)malloc(sizeof(int) * size);
  assert(arr);
  seedArr(arr, size);

  start_ct1 = std::chrono::steady_clock::now();
  queue.memcpy(d_arr, arr, sizeof(int) * size).wait();
  stop_ct1 = std::chrono::steady_clock::now();
  elapsedTime =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsedTime * 1.e-3;

  sycl::buffer<int> results_buff{sycl::range(size)};
  sycl::buffer<int> prefix_buff{sycl::range(size)};

  int32_t grid_range = size / g_thread_cnt + 1;
  sycl::range<1> grid(grid_range);
  sycl::range<1> threads(g_thread_cnt);

  int32_t grid_per_cu_k3[num_k3_cu];
  for (int32_t i = 0; i < num_k3_cu; i++)
    grid_per_cu_k3[i] = 0;

  sycl::range<1> grid_dim = grid * threads;
  sycl::range<1> block_dim = threads;
  int32_t grid_per_cu_idx = 0;
  int32_t combined_grid_size = 0;
  for (int32_t i = 0; i < std::numeric_limits<int32_t>::max(); i++) {
    grid_per_cu_k3[grid_per_cu_idx] += block_dim[0];

    combined_grid_size += block_dim[0];
    if (combined_grid_size >= grid_dim[0])
      break;

    grid_per_cu_idx = (grid_per_cu_idx + 1) % num_k3_cu;
  }

  FPGA_PWR_MEAS_START
  std::array<sycl::event, num_k1_cu> k1_events;
  fpga_tools::UnrolledLoop<num_k1_cu>([&](auto CU) {
    if (grid_per_cu_k3[CU] > 0) {
      k1_events[CU] = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor results{results_buff, cgh, sycl::write_only,
                               sycl::noinit};

        const int32_t offset = grid_per_cu_k3[CU];

        cgh.parallel_for<mark_matches_cu<CU>>(
            sycl::nd_range<1>(grid_per_cu_k3[CU], block_dim),
            [=](sycl::nd_item<1> item_ct1)
                [[intel::kernel_args_restrict, intel::num_simd_work_items(16),
                  intel::no_global_work_offset(1),
                  sycl::reqd_work_group_size(1, 1, g_thread_cnt),
                  intel::max_work_group_size(1, 1, g_thread_cnt)]] {
                  const int tid = item_ct1.get_global_id(0) + offset;

                  for (int i = tid; i < size; i += g_thread_cnt * grid_range)
                    results[tid] = (d_arr[tid] < coverage) ? 1 : 0;
                });
      });
    }
  });
  double max_k1_duration = 0.0;
  for (auto idx = 0; idx < num_k1_cu; idx++) {
    auto &e = k1_events[idx];
    if (grid_per_cu_k3[idx] > 0) {
      e.wait();
      double elapsedTime =
          e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      if (elapsedTime > max_k1_duration)
        max_k1_duration = elapsedTime;
    }
  }
  kernelTime += max_k1_duration * 1.e-9;
  queue.wait_and_throw();

  sycl::event k2_event = queue.submit([&](sycl::handler &cgh) {
    const sycl::accessor results{results_buff, cgh, sycl::read_only};
    sycl::accessor prefix{prefix_buff, cgh, sycl::write_only, sycl::noinit};
    cgh.single_task<class exclusive_scan_id>(
        [=]() [[intel::kernel_args_restrict, intel::max_global_work_dim(0),
                intel::no_global_work_offset(1)]] {
          prefix[0] = 0;
#pragma unroll 2
          for (int i = 1; i < size; i++)
            prefix[i] = prefix[i - 1] + results[i];
        });
  });
  k2_event.wait();
  elapsedTime =
      k2_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k2_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsedTime * 1.e-9;

  int matchSize;
  {
    auto d_prefix = prefix_buff.template get_access<sycl::access::mode::read>();
    matchSize = d_prefix[size - 1];
  }
  matchSize++;

  int *d_final = sycl::malloc_device<int>(matchSize, queue);
  int *final = (int *)malloc(sizeof(int) * matchSize);
  assert(final);

  std::array<sycl::event, num_k3_cu> k3_events;
  fpga_tools::UnrolledLoop<num_k3_cu>([&](auto CU) {
    if (grid_per_cu_k3[CU] > 0) {
      k3_events[CU] = queue.submit([&](sycl::handler &cgh) {
        const sycl::accessor results{results_buff, cgh, sycl::read_only};
        const sycl::accessor prefix{prefix_buff, cgh, sycl::read_only};

        const int32_t offset = grid_per_cu_k3[CU];

        cgh.parallel_for<map_matches_cu<CU>>(
            sycl::nd_range<1>(grid_per_cu_k3[CU], block_dim),
            [=](sycl::nd_item<1> item_ct1)
                [[intel::kernel_args_restrict, intel::no_global_work_offset(1),
                  sycl::reqd_work_group_size(1, 1, g_thread_cnt),
                  intel::max_work_group_size(1, 1, g_thread_cnt)]] {
                  const int tid = item_ct1.get_global_id(0) + offset;

                  for (int i = tid; i < size; i += g_thread_cnt * grid_range)
                    if (results[tid])
                      d_final[prefix[tid]] = d_arr[tid];
                });
      });
    }
  });
  double max_k3_duration = 0.0;
  for (auto idx = 0; idx < num_k3_cu; idx++) {
    auto &e = k3_events[idx];
    if (grid_per_cu_k3[idx] > 0) {
      e.wait();
      double elapsedTime =
          e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      if (elapsedTime > max_k3_duration)
        max_k3_duration = elapsedTime;
    }
  }
  kernelTime += max_k3_duration * 1.e-9;
  FPGA_PWR_MEAS_END

  queue.wait_and_throw();
  start_ct1 = std::chrono::steady_clock::now();
  queue.memcpy(final, d_final, sizeof(int) * matchSize).wait();
  stop_ct1 = std::chrono::steady_clock::now();
  elapsedTime =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsedTime * 1.e-3;
  queue.wait_and_throw();

  free(arr);
  free(final);
  sycl::free(d_arr, queue);
  sycl::free(d_final, queue);

  char atts[1024];
  sprintf(atts, "size:%d, coverage:%d", size, coverage);
  resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("where_total_time", atts, "sec",
                     kernelTime + transferTime);
  resultDB.AddResult("where_parity", atts, "N", transferTime / kernelTime);
  resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("length", OPT_INT, "0", "number of elements in input");
  op.addOption("coverage", OPT_INT, "-1",
               "0 to 100 percentage of elements to allow through where filter");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op,
                  size_t device_idx) {
  printf("Running Where\n");

  srand(7);

  bool quiet = op.getOptionBool("quiet");
  int size = op.getOptionInt("length");
  int coverage = op.getOptionInt("coverage");
  if (size == 0 || coverage == -1) {
    int sizes[5] = {1000, 10000, 500000000, 1000000000, 1050000000};
    int coverages[5] = {20, 30, 40, 80, 240};
    size = sizes[op.getOptionInt("size") - 1];
    coverage = coverages[op.getOptionInt("size") - 1];
  }

  if (!quiet)
    printf("Using size=%d, coverage=%d\n", size, coverage);

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    kernelTime = 0.0f;
    transferTime = 0.0f;
    if (!quiet)
      printf("Pass %d: ", i);

    where(resultDB, op, size, coverage, device_idx);
    if (!quiet)
      printf("Done.\n");
  }
}
