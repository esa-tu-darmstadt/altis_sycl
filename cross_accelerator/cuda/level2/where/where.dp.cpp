////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\where\where.cu
//
// summary:	Where class
//
// origin:
////////////////////////////////////////////////////////////////////////////////////////////////////

// These headers must stay 1st, for tbb-namespace weirdness
#ifdef _FPGA
// The following macros are to enable make_fpga_policy and the usage of
// FPGA-suited algorithms.
#define ONEDPL_USE_DPCPP_BACKEND 1
#define ONEDPL_FPGA_DEVICE       1
//#define ONEDPL_FPGA_EMULATOR 1
#endif
#include <oneapi/dpl/execution>

#include <CL/sycl.hpp>
#include <oneapi/dpl/algorithm>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#include <chrono>
#include <stdio.h>

// Original is 1024, however on P630 use 256 - the maximum allowed.
constexpr uint16_t g_thread_cnt = 256;

float kernelTime   = 0.0f;
float transferTime = 0.0f;

std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float                                              elapsedTime;

bool
check(int val, int bound)
{
    return (val < bound);
}

void
markMatches(
    int *arr, int *results, int size, int bound, sycl::nd_item<3> item_ct1)
{

    // Block index
    int bx = item_ct1.get_group(2);

    // Thread index
    int tx = item_ct1.get_local_id(2);

    int tid = (item_ct1.get_local_range(2) * bx) + tx;

    for (; tid < size;
         tid += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
    {
        if (check(arr[tid], bound))
            results[tid] = 1;
        else
            results[tid] = 0;
    }
}

void
mapMatches(int             *arr,
           int             *results,
           int             *prefix,
           int             *final,
           int              size,
           sycl::nd_item<3> item_ct1)
{

    // Block index
    int bx = item_ct1.get_group(2);

    // Thread index
    int tx = item_ct1.get_local_id(2);

    int tid = (item_ct1.get_local_range(2) * bx) + tx;

    for (; tid < size;
         tid += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
        if (results[tid])
            final[prefix[tid]] = arr[tid];
}

void
seedArr(int *arr, int size)
{
    for (int i = 0; i < size; i++)
        arr[i] = rand() % 100;
}

void
where(ResultDatabase &resultDB,
      OptionParser   &op,
      int             size,
      int             coverage,
      size_t          device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    int *d_arr = sycl::malloc_device<int>(size, queue);
    int *arr   = (int *)malloc(sizeof(int) * size);
    assert(arr);
    seedArr(arr, size);

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(d_arr, arr, sizeof(int) * size).wait();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;

    int *d_results = sycl::malloc_device<int>(size, queue);
    int *d_prefix  = sycl::malloc_device<int>(size, queue);

    sycl::range<3> grid(1, 1, size / g_thread_cnt + 1);
    sycl::range<3> threads(1, 1, g_thread_cnt);

    sycl::event k1_event = queue.parallel_for<class markMatches_id>(
        sycl::nd_range<3>(grid * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            markMatches(d_arr, d_results, size, coverage, item_ct1);
        });
    k1_event.wait();
    elapsedTime
        = k1_event
              .get_profiling_info<sycl::info::event_profiling::command_end>()
          - k1_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsedTime * 1.e-9;

#ifdef _FPGA
    constexpr auto unroll_factor = 8;
    auto           policy
        = oneapi::dpl::execution::make_fpga_policy<unroll_factor,
                                                   class exclusive_scan_id>(
            queue);
#else
    auto policy = oneapi::dpl::execution::make_device_policy(queue);
#endif

    start_ct1 = std::chrono::steady_clock::now();
    oneapi::dpl::exclusive_scan(
        policy, d_results, d_results + size, d_prefix, 0);
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    kernelTime += elapsedTime * 1.e-3;

    int matchSize;
    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(&matchSize, d_prefix + size - 1, sizeof(int)).wait();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;
    matchSize++;

    int *d_final = sycl::malloc_device<int>(matchSize, queue);
    int *final   = (int *)malloc(sizeof(int) * matchSize);
    assert(final);

    start_ct1            = std::chrono::steady_clock::now();
    sycl::event k3_event = queue.parallel_for<class mapMatches_id>(
        sycl::nd_range<3>(grid * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            mapMatches(d_arr, d_results, d_prefix, d_final, size, item_ct1);
        });
    k3_event.wait();
    elapsedTime
        = k3_event
              .get_profiling_info<sycl::info::event_profiling::command_end>()
          - k3_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsedTime * 1.e-9;

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(final, d_final, sizeof(int) * matchSize).wait();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;
    queue.wait_and_throw();

    free(arr);
    free(final);
    sycl::free(d_arr, queue);
    sycl::free(d_results, queue);
    sycl::free(d_prefix, queue);
    sycl::free(d_final, queue);

    char atts[1024];
    sprintf(atts, "size:%d, coverage:%d", size, coverage);
    resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult(
        "where_total_time", atts, "sec", kernelTime + transferTime);
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

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("length", OPT_INT, "0", "number of elements in input");
    op.addOption(
        "coverage",
        OPT_INT,
        "-1",
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

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running Where\n");

    srand(7);

    bool quiet    = op.getOptionBool("quiet");
    int  size     = op.getOptionInt("length");
    int  coverage = op.getOptionInt("coverage");
    if (size == 0 || coverage == -1)
    {
        int sizes[5]     = { 1000, 10000, 500000000, 1000000000, 1050000000 };
        int coverages[5] = { 20, 30, 40, 80, 240 };
        size             = sizes[op.getOptionInt("size") - 1];
        coverage         = coverages[op.getOptionInt("size") - 1];
    }

    if (!quiet)
        printf("Using size=%d, coverage=%d\n", size, coverage);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        kernelTime   = 0.0f;
        transferTime = 0.0f;
        if (!quiet)
            printf("Pass %d: ", i);

        where(resultDB, op, size, coverage, device_idx);
        if (!quiet)
            printf("Done.\n");
    }
}
