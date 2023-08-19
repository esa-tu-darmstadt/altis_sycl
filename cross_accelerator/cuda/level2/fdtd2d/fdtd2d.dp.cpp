/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Modfified by Bodun Hu <bodunhu@utexas.edu>
 * Added: UVM and coop support
 *
 */

#include <CL/sycl.hpp>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "polybenchUtilFuncts.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float, int, and double */
typedef float DATA_TYPE;

void
init_arrays(size_t     NX,
            size_t     NY,
            size_t     tmax,
            DATA_TYPE *_fict_,
            DATA_TYPE *ex,
            DATA_TYPE *ey,
            DATA_TYPE *hz)
{
    for (int i = 0; i < tmax; i++)
        _fict_[i] = (DATA_TYPE)i;

    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
        {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
}

void
runFdtd(size_t     NX,
        size_t     NY,
        size_t     tmax,
        DATA_TYPE *_fict_,
        DATA_TYPE *ex,
        DATA_TYPE *ey,
        DATA_TYPE *hz)
{
    for (int t = 0; t < tmax; t++)
    {
        for (int j = 0; j < NY; j++)
            ey[0 * NY + j] = _fict_[t];

        for (int i = 1; i < NX; i++)
            for (int j = 0; j < NY; j++)
                ey[i * NY + j]
                    = ey[i * NY + j]
                      - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);

        for (int i = 0; i < NX; i++)
            for (int j = 1; j < NY; j++)
                ex[i * (NY + 1) + j]
                    = ex[i * (NY + 1) + j]
                      - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);

        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                hz[i * NY + j]
                    = hz[i * NY + j]
                      - 0.7
                            * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j]
                               + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}

void
compareResults(size_t NX, size_t NY, DATA_TYPE *hz1, DATA_TYPE *hz2)
{
    int fail = 0;
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            if (percentDiff(hz1[i * NY + j], hz2[i * NY + j])
                > PERCENT_DIFF_ERROR_THRESHOLD)
                fail++;

    // Print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD,
        fail);
}

void
fdtd_step1_kernel(size_t           NX,
                  size_t           NY,
                  const DATA_TYPE *_fict_,
                  DATA_TYPE       *ey,
                  const DATA_TYPE *hz,
                  const int        t,
                  sycl::nd_item<3> item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2)
            + item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1)
            + item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY))
    {
        if (i == 0)
            ey[i * NY + j] = _fict_[t];
        else
            ey[i * NY + j] = ey[i * NY + j]
                             - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
    }
}

void
fdtd_step2_kernel(size_t           NX,
                  size_t           NY,
                  DATA_TYPE       *ex,
                  const DATA_TYPE *hz,
                  sycl::nd_item<3> item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2)
            + item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1)
            + item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY) && (j > 0))
        ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j]
                               - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
}

void
fdtd_step3_kernel(size_t           NX,
                  size_t           NY,
                  const DATA_TYPE *ex,
                  const DATA_TYPE *ey,
                  DATA_TYPE       *hz,
                  sycl::nd_item<3> item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2)
            + item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1)
            + item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY))
        hz[i * NY + j]
            = hz[i * NY + j]
              - 0.7f
                    * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j]
                       + ey[(i + 1) * NY + j] - ey[i * NY + j]);
}

void
fdtdCuda(size_t          NX,
         size_t          NY,
         size_t          tmax,
         DATA_TYPE      *_fict_,
         DATA_TYPE      *ex,
         DATA_TYPE      *ey,
         DATA_TYPE      *hz,
         DATA_TYPE      *hz_outputFromGpu,
         ResultDatabase &DB,
         OptionParser   &op,
         size_t          device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx]);

    // Allocate memory on device.
    //
    DATA_TYPE *_fict_gpu = sycl::malloc_device<DATA_TYPE>(tmax, queue);
    DATA_TYPE *ex_gpu    = (DATA_TYPE *)sycl::malloc_device(
        sizeof(DATA_TYPE) * NX * (NY + 1), queue);
    DATA_TYPE *ey_gpu = (DATA_TYPE *)sycl::malloc_device(
        sizeof(DATA_TYPE) * (NX + 1) * NY, queue);
    DATA_TYPE *hz_gpu
        = (DATA_TYPE *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY, queue);

    sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
    sycl::range<3> grid(1,
                        (size_t)ceil(((float)NX) / ((float)block[1])),
                        (size_t)ceil(((float)NY) / ((float)block[2])));

    // Transfer the inout data to the device.
    //
    queue.memcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax).wait();
    queue.memcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1)).wait();
    queue.memcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY).wait();
    queue.memcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY).wait();

    // Start the calculation.
    //
    double t_start = rtclock();

    for (int t = 0; t < tmax; t++)
    {
        queue
            .parallel_for<class fdtd_step1>(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item_ct1) {
                    fdtd_step1_kernel(
                        NX, NY, _fict_gpu, ey_gpu, hz_gpu, t, item_ct1);
                });
        queue
            .parallel_for<class fdtd_step2>(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item_ct1) {
                    fdtd_step2_kernel(NX, NY, ex_gpu, hz_gpu, item_ct1);
                });
        queue.wait_and_throw();
        queue
            .parallel_for<class fdtd_step3>(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item_ct1) {
                    fdtd_step3_kernel(NX, NY, ex_gpu, ey_gpu, hz_gpu, item_ct1);
                }).wait();
    }

    double t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

    // Transfer back the result from the device.
    //
    queue.memcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY).wait();

    // Free memory on device.
    //
    sycl::free(_fict_gpu, queue);
    sycl::free(ex_gpu, queue);
    sycl::free(ey_gpu, queue);
    sycl::free(hz_gpu, queue);
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption(
        "compare", OPT_BOOL, "0", "compare GPU output with CPU output");
}

void
RunBenchmark(ResultDatabase &DB, OptionParser &op, size_t device_idx)
{
    const bool compare = op.getOptionBool("compare");

    const size_t s             = 5;
    size_t       NX_sizes[s]   = { 100, 1000, 2000, 8000, 16000 };
    size_t       NY_sizes[s]   = { 200, 1200, 2600, 9600, 20000 };
    size_t       tmax_sizes[s] = { 240, 500, 1000, 4000, 8000 };

    size_t NX   = NX_sizes[op.getOptionInt("size") - 1];
    size_t NY   = NY_sizes[op.getOptionInt("size") - 1];
    size_t tmax = tmax_sizes[op.getOptionInt("size") - 1];

    DATA_TYPE *_fict_ = (DATA_TYPE *)malloc(tmax * sizeof(DATA_TYPE));
    assert(_fict_);
    DATA_TYPE *ex = (DATA_TYPE *)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
    assert(ex);
    DATA_TYPE *ey = (DATA_TYPE *)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
    assert(ey);
    DATA_TYPE *hz = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    assert(hz);
    DATA_TYPE *hz_outputFromGpu
        = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    assert(hz_outputFromGpu);

    // Warmup
    fdtdCuda(
        NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op, device_idx);

    init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
    fdtdCuda(
        NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op, device_idx);

    if (compare)
    {
        double t_start = rtclock();
        runFdtd(NX, NY, tmax, _fict_, ex, ey, hz);
        double t_end = rtclock();
        fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
        compareResults(NX, NY, hz, hz_outputFromGpu);
    }

    free(_fict_);
    free(ex);
    free(ey);
    free(hz);
    free(hz_outputFromGpu);
}
