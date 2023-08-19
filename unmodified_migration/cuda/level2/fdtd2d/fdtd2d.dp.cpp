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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "polybenchUtilFuncts.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float, int, and double */
typedef float DATA_TYPE;

struct fdtd_params {
    int NX;
    int NY;
    DATA_TYPE *_fict_;
    DATA_TYPE *ex;
    DATA_TYPE *ey;
    DATA_TYPE *hz;
    int t;
};


void init_arrays(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int i, j;

    for (i = 0; i < tmax; i++)
    {
        _fict_[i] = (DATA_TYPE) i;
    }
    
    for (i = 0; i < NX; i++)
    {
        for (j = 0; j < NY; j++)
        {
            ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
            ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
            hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
        }
    }
}


void runFdtd(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int t, i, j;
    
    for (t=0; t < tmax; t++)  
    {
        for (j=0; j < NY; j++)
        {
            ey[0*NY + j] = _fict_[t];
        }
    
        for (i = 1; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 1; j < NY; j++)
            {
                ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
            }
        }
    }
}


void compareResults(size_t NX, size_t NY, DATA_TYPE* hz1, DATA_TYPE* hz2)
{
    int i, j, fail;
    fail = 0;
    
    for (i=0; i < NX; i++) 
    {
        for (j=0; j < NY; j++) 
        {
            if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }
    
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void fdtd_step1_kernel(size_t NX, size_t NY, DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       const sycl::nd_item<3> &item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY))
    {
        if (i == 0) 
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        { 
            ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
        }
    }
}



void fdtd_step2_kernel(size_t NX, size_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       const sycl::nd_item<3> &item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
    }
}


void fdtd_step3_kernel(size_t NX, size_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       const sycl::nd_item<3> &item_ct1)
{
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if ((i < NX) && (j < NY))
    {	
        hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}

void fdtd_coop_kernel(fdtd_params params, const sycl::nd_item<3> &item_ct1)
{
    int NX = params.NX;
    int NY = params.NY;
    DATA_TYPE *_fict_ = params._fict_;
    DATA_TYPE *ex = params.ex;
    DATA_TYPE *ey = params.ey;
    DATA_TYPE *hz = params.hz;
    int t = params.t;

    /*
    DPCT1087:1: SYCL currently does not support cross group synchronization. You
    can specify "--use-experimental-features=nd_range_barrier" to use the dpct
    helper function nd_range_barrier to migrate this_grid().
    */
    grid_group g = this_grid();
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    // kernel 1
    if ((i < NX) && (j < NY))
    {
        if (i == 0) 
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        { 
            ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
        }
    }

    // kernel 2
    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
    }

    /*
    DPCT1087:0: SYCL currently does not support cross group synchronization. You
    can specify "--use-experimental-features=nd_range_barrier" to use the dpct
    helper function nd_range_barrier to migrate g.sync().
    */
    g.sync();

    // kernel 3
    if ((i < NX) && (j < NY))
    {
        hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}


void fdtdCuda(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu,
            ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    double t_start, t_end;

    DATA_TYPE *_fict_gpu;
    DATA_TYPE *ex_gpu;
    DATA_TYPE *ey_gpu;
    DATA_TYPE *hz_gpu;

    /*
    DPCT1064:266: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(_fict_gpu = sycl::malloc_device<DATA_TYPE>(
                                         tmax, dpct::get_default_queue())));
    /*
    DPCT1064:267: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        ex_gpu = (DATA_TYPE *)sycl::malloc_device(
            sizeof(DATA_TYPE) * NX * (NY + 1), dpct::get_default_queue())));
    /*
    DPCT1064:268: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        ey_gpu = (DATA_TYPE *)sycl::malloc_device(
            sizeof(DATA_TYPE) * (NX + 1) * NY, dpct::get_default_queue())));
    /*
    DPCT1064:269: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        hz_gpu = (DATA_TYPE *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                                  dpct::get_default_queue())));

    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax)
            .wait()));
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1))
            .wait()));
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY)
            .wait()));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY)
                             .wait()));

    sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
    sycl::range<3> grid(1, (size_t)ceil(((float)NX) / ((float)block[1])),
                        (size_t)ceil(((float)NY) / ((float)block[2])));

    t_start = rtclock();

    if (op.getOptionBool("coop"))
    {
        fdtd_params params;
        params.NX = NX;
        params.NY = NY;
        params._fict_ = _fict_gpu;
        params.ex = ex_gpu;
        params.ey = ey_gpu;
        params.hz = hz_gpu;
        void *p_params = {&params};
        for (int t = 0; t < tmax; t++)
        {
            params.t = t;
            /*
            DPCT1049:2: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            /*
            DPCT1007:10: Migration of cudaLaunchCooperativeKernel is not
            supported.
            */
                  checkCudaErrors(
                      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                            auto params_ct0 = *(fdtd_params *)(&p_params)[0];

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * block, block),
                                [=](sycl::nd_item<3> item_ct1) {
                                      fdtd_coop_kernel(params_ct0, item_ct1);
                                });
                      }););
        }
    }
    else
    {
        dpct::queue_ptr stream1, stream2;
        checkCudaErrors(DPCT_CHECK_ERROR(
            stream1 = dpct::get_current_device().create_queue()));
        checkCudaErrors(DPCT_CHECK_ERROR(
            stream2 = dpct::get_current_device().create_queue()));
        for (int t = 0; t < tmax; t++)
        {
            /*
            DPCT1049:3: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  stream1->parallel_for(sycl::nd_range<3>(grid * block, block),
                                        [=](sycl::nd_item<3> item_ct1) {
                                              fdtd_step1_kernel(
                                                  NX, NY, _fict_gpu, ex_gpu,
                                                  ey_gpu, hz_gpu, t, item_ct1);
                                        });
            /*
            DPCT1049:4: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  stream2->parallel_for(sycl::nd_range<3>(grid * block, block),
                                        [=](sycl::nd_item<3> item_ct1) {
                                              fdtd_step2_kernel(NX, NY, ex_gpu,
                                                                ey_gpu, hz_gpu,
                                                                t, item_ct1);
                                        });
            /*
            DPCT1049:5: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3> item_ct1) {
                            fdtd_step3_kernel(NX, NY, ex_gpu, ey_gpu, hz_gpu, t,
                                              item_ct1);
                      });
        }
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().destroy_queue(stream1)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().destroy_queue(stream2)));
    }
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY)
            .wait()));

    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(_fict_gpu, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(ex_gpu, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(ey_gpu, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(hz_gpu, dpct::get_default_queue())));
}

void fdtdCudaUnifiedMemory(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
    ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    double t_start, t_end;

    DATA_TYPE *_fict_gpu;
    DATA_TYPE *ex_gpu;
    DATA_TYPE *ey_gpu;
    DATA_TYPE *hz_gpu;

    _fict_gpu = _fict_;
    ex_gpu = ex;
    ey_gpu = ey;
    hz_gpu = hz;

    if (uvm)
    {
        // Do nothing
    }
    else if (uvm_advise)
    {
        /*
        DPCT1063:245: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));
        /*
        DPCT1063:246: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));
        /*
        DPCT1063:247: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));

        /*
        DPCT1063:248: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0)));
        /*
        DPCT1063:249: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0)));

        /*
        DPCT1063:250: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 0)));
        /*
        DPCT1063:251: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 0)));

        /*
        DPCT1063:252: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
        /*
        DPCT1063:253: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
    }
    else if (uvm_prefetch)
    {
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(_fict_gpu, sizeof(DATA_TYPE) * tmax)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1))));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(hz_gpu, sizeof(DATA_TYPE) * NX * NY)));
    }
    else if (uvm_prefetch_advise)
    {
        /*
        DPCT1063:254: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));
        /*
        DPCT1063:255: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));
        /*
        DPCT1063:256: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                _fict_gpu, sizeof(DATA_TYPE) * tmax, 0)));

        /*
        DPCT1063:257: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0)));
        /*
        DPCT1063:258: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0)));

        /*
        DPCT1063:259: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 0)));
        /*
        DPCT1063:260: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 0)));

        /*
        DPCT1063:261: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
        /*
        DPCT1063:262: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));

        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(_fict_gpu, sizeof(DATA_TYPE) * tmax)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1))));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dev_mgr::instance()
                .get_device(device)
                .default_queue()
                .prefetch(hz_gpu, sizeof(DATA_TYPE) * NX * NY)));
    }
    else
    {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
    sycl::range<3> grid(1, (size_t)ceil(((float)NX) / ((float)block[1])),
                        (size_t)ceil(((float)NY) / ((float)block[2])));

    // cudaStream_t stream1, stream2;
    // checkCudaErrors(cudaStreamCreate(&stream1));
    // checkCudaErrors(cudaStreamCreate(&stream2));
    t_start = rtclock();

    if (op.getOptionBool("coop"))
    {
        fdtd_params params;
        params.NX = NX;
        params.NY = NY;
        params._fict_ = _fict_gpu;
        params.ex = ex_gpu;
        params.ey = ey_gpu;
        params.hz = hz_gpu;
        void *p_params = {&params};
        for (int t = 0; t < tmax; t++)
        {
            params.t = t;
            /*
            DPCT1049:6: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            /*
            DPCT1007:11: Migration of cudaLaunchCooperativeKernel is not
            supported.
            */
                  checkCudaErrors(
                      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                            auto params_ct0 = *(fdtd_params *)(&p_params)[0];

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * block, block),
                                [=](sycl::nd_item<3> item_ct1) {
                                      fdtd_coop_kernel(params_ct0, item_ct1);
                                });
                      }););
        }
    }
    else
    {
        dpct::queue_ptr stream1, stream2;
        checkCudaErrors(DPCT_CHECK_ERROR(
            stream1 = dpct::get_current_device().create_queue()));
        checkCudaErrors(DPCT_CHECK_ERROR(
            stream2 = dpct::get_current_device().create_queue()));
        for (int t = 0; t < tmax; t++)
        {
            /*
            DPCT1049:7: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  stream1->parallel_for(sycl::nd_range<3>(grid * block, block),
                                        [=](sycl::nd_item<3> item_ct1) {
                                              fdtd_step1_kernel(
                                                  NX, NY, _fict_gpu, ex_gpu,
                                                  ey_gpu, hz_gpu, t, item_ct1);
                                        });
            /*
            DPCT1049:8: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  stream1->parallel_for(sycl::nd_range<3>(grid * block, block),
                                        [=](sycl::nd_item<3> item_ct1) {
                                              fdtd_step2_kernel(NX, NY, ex_gpu,
                                                                ey_gpu, hz_gpu,
                                                                t, item_ct1);
                                        });
            /*
            DPCT1049:9: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3> item_ct1) {
                            fdtd_step3_kernel(NX, NY, ex_gpu, ey_gpu, hz_gpu, t,
                                              item_ct1);
                      });
        }
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().destroy_queue(stream1)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().destroy_queue(stream2)));
    }

    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
    // Do nothing
    }

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        /*
        DPCT1063:263: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
        /*
        DPCT1063:264: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
        /*
        DPCT1063:265: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0)));
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
                hz_gpu, sizeof(DATA_TYPE) * NX * NY)));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    }
}

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
    op.addOption("uvm-advise", OPT_BOOL, "0", "guide the driver about memory usage patterns");
    op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
    op.addOption("uvm-prefetch-advise", OPT_BOOL, "0", "prefetch memory the specified destination device with memory guidance on");
    op.addOption("coop", OPT_BOOL, "0", "use cooperative kernel instead normal kernels");
    op.addOption("compare", OPT_BOOL, "0", "compare GPU output with CPU output");
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool compare = op.getOptionBool("compare");

    const size_t s = 5;
    size_t NX_sizes[s] = {100, 1000, 2000, 8000, 16000};
    size_t NY_sizes[s] = {200, 1200, 2600, 9600, 20000};
    size_t tmax_sizes[s] =  {240, 500, 1000, 4000, 8000};

    size_t NX = NX_sizes[op.getOptionInt("size") - 1];
    size_t NY = NY_sizes[op.getOptionInt("size") - 1];
    size_t tmax = tmax_sizes[op.getOptionInt("size") - 1];

    double t_start, t_end;

    DATA_TYPE* _fict_;
    DATA_TYPE* ex;
    DATA_TYPE* ey;
    DATA_TYPE* hz;
    DATA_TYPE* hz_outputFromGpu;

    if (compare)
    {
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        {
            DATA_TYPE* _fict_gpu;
            DATA_TYPE* ex_gpu;
            DATA_TYPE* ey_gpu;
            DATA_TYPE* hz_gpu;
            /*
            DPCT1064:270: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(
                DPCT_CHECK_ERROR(_fict_gpu = sycl::malloc_shared<DATA_TYPE>(
                                     tmax, dpct::get_default_queue())));
            /*
            DPCT1064:271: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(DPCT_CHECK_ERROR(
                ex_gpu = sycl::malloc_shared<DATA_TYPE>(
                    NX * (NY + 1), dpct::get_default_queue())));
            /*
            DPCT1064:272: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(DPCT_CHECK_ERROR(
                ey_gpu = sycl::malloc_shared<DATA_TYPE>(
                    (NX + 1) * NY, dpct::get_default_queue())));
            /*
            DPCT1064:273: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(
                DPCT_CHECK_ERROR(hz_gpu = sycl::malloc_shared<DATA_TYPE>(
                                     NX * NY, dpct::get_default_queue())));

            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);

            init_arrays(NX, NY, tmax, _fict_gpu, ex_gpu, ey_gpu, hz_gpu);
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(_fict_, _fict_gpu, tmax * sizeof(DATA_TYPE))
                    .wait()));
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(ex, ex_gpu, NX * (NY + 1) * sizeof(DATA_TYPE))
                    .wait()));
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(ey, ey_gpu, (NX + 1) * NY * sizeof(DATA_TYPE))
                    .wait()));
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(hz, hz_gpu, NX * NY * sizeof(DATA_TYPE))
                    .wait()));

            fdtdCudaUnifiedMemory(NX, NY, tmax, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, DB, op);
            t_start = rtclock();
            runFdtd(NX, NY, tmax, _fict_, ex, ey, hz);
            t_end = rtclock();
            fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
            compareResults(NX, NY, hz, hz_gpu);

            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free(_fict_gpu, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free(ex_gpu, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free(ey_gpu, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free(hz_gpu, dpct::get_default_queue())));
            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
        }
        else
        {
            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);
            hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz_outputFromGpu);

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCuda(NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op);
            t_start = rtclock();
            runFdtd(NX, NY, tmax, _fict_, ex, ey, hz);
            t_end = rtclock();
            fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
            compareResults(NX, NY, hz, hz_outputFromGpu);

            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
            free(hz_outputFromGpu);
        }
    }
    else
    {
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        {
            /*
            DPCT1064:274: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(
                DPCT_CHECK_ERROR(_fict_ = sycl::malloc_shared<DATA_TYPE>(
                                     tmax, dpct::get_default_queue())));
            /*
            DPCT1064:275: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(DPCT_CHECK_ERROR(
                ex = sycl::malloc_shared<DATA_TYPE>(
                    NX * (NY + 1), dpct::get_default_queue())));
            /*
            DPCT1064:276: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(DPCT_CHECK_ERROR(
                ey = sycl::malloc_shared<DATA_TYPE>(
                    (NX + 1) * NY, dpct::get_default_queue())));
            /*
            DPCT1064:277: Migrated cudaMallocManaged call is used in a
            macro/template definition and may not be valid for all
            macro/template uses. Adjust the code.
            */
            checkCudaErrors(
                DPCT_CHECK_ERROR(hz = sycl::malloc_shared<DATA_TYPE>(
                                     NX * NY, dpct::get_default_queue())));

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCudaUnifiedMemory(NX, NY, tmax, _fict_, ex, ey, hz, DB, op);

            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free(_fict_, dpct::get_default_queue())));
            checkCudaErrors(
                DPCT_CHECK_ERROR(sycl::free(ex, dpct::get_default_queue())));
            checkCudaErrors(
                DPCT_CHECK_ERROR(sycl::free(ey, dpct::get_default_queue())));
            checkCudaErrors(
                DPCT_CHECK_ERROR(sycl::free(hz, dpct::get_default_queue())));
        }
        else
        {
            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);
            hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz_outputFromGpu);

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCuda(NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op);

            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
            free(hz_outputFromGpu);
        }
    }
}