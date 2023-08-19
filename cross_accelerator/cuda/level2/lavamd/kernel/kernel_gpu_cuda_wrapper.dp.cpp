#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "cudacommon.h"
#include "fpga_pwr.hpp"

#include "./../lavaMD.h"
#include "./kernel_gpu_cuda_wrapper.h"
#include "./kernel_gpu_cuda.dp.cpp"

#include <chrono>

/// <summary>	An enum constant representing the void option. </summary>
void
kernel_gpu_cuda_wrapper(par_str         par_cpu,
                        dim_str         dim_cpu,
                        box_str        *box_cpu,
                        FOUR_VECTOR    *rv_cpu,
                        fp             *qv_cpu,
                        FOUR_VECTOR    *fv_cpu,
                        ResultDatabase &resultDB,
                        OptionParser   &op,
                        sycl::queue    &queue)
{
    float                                              kernelTime   = 0.0f;
    float                                              transferTime = 0.0f;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float                                              elapsedTime;

    //====================================================================================================100
    //	VARIABLES
    //====================================================================================================100

    sycl::range<3> threads(1, 1, NUMBER_THREADS);
    sycl::range<3> blocks(1, 1, dim_cpu.number_boxes);

    box_str     *d_box_gpu = (box_str *)sycl::malloc_device(dim_cpu.box_mem, queue);
    FOUR_VECTOR *d_rv_gpu  = (FOUR_VECTOR *)sycl::malloc_device(dim_cpu.space_mem, queue);
    fp          *d_qv_gpu  = (double *)sycl::malloc_device(dim_cpu.space_mem2, queue);
    FOUR_VECTOR *d_fv_gpu  = (FOUR_VECTOR *)sycl::malloc_device(dim_cpu.space_mem, queue);

    start_ct1 = std::chrono::steady_clock::now();

    queue.memcpy(d_box_gpu, box_cpu, dim_cpu.box_mem);
    queue.memcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem);
    queue.memcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2);
    queue.memcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem);
    queue.wait_and_throw();

    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;

    //======================================================================================================================================================150
    //	KERNEL
    //======================================================================================================================================================150
    FPGA_PWR_MEAS_START
    sycl::event k_event = queue
        .submit([&](sycl::handler &cgh) {
            sycl::accessor<FOUR_VECTOR,
                           1,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                rA_shared_acc_ct1(sycl::range<1>(100), cgh);
            sycl::accessor<FOUR_VECTOR,
                           1,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                rB_shared_acc_ct1(sycl::range<1>(100), cgh);
            sycl::accessor<double,
                           1,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                qB_shared_acc_ct1(sycl::range<1>(100), cgh);

            cgh.parallel_for<class lavamd_kernel>(
                sycl::nd_range<3>(blocks * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_gpu_cuda(par_cpu,
                                    dim_cpu,
                                    d_box_gpu,
                                    d_rv_gpu,
                                    d_qv_gpu,
                                    d_fv_gpu,
                                    item_ct1,
                                    rA_shared_acc_ct1.get_pointer(),
                                    rB_shared_acc_ct1.get_pointer(),
                                    qB_shared_acc_ct1.get_pointer());
                });
        });
    k_event.wait();
    FPGA_PWR_MEAS_END
    elapsedTime = k_event.get_profiling_info<
                    sycl::info::event_profiling::command_end>()
                - k_event.get_profiling_info<
                    sycl::info::event_profiling::command_start>();
    kernelTime += elapsedTime * 1.e-9;

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem).wait();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;

    char atts[1024];
    sprintf(atts, "boxes1d:%d", dim_cpu.boxes1d_arg);
    resultDB.AddResult("lavamd_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("lavamd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("lavamd_parity", atts, "N", transferTime / kernelTime);

    //======================================================================================================================================================150
    //	GPU MEMORY DEALLOCATION
    //======================================================================================================================================================150

    sycl::free(d_rv_gpu, queue);
    sycl::free(d_qv_gpu, queue);
    sycl::free(d_fv_gpu, queue);
    sycl::free(d_box_gpu, queue);
}
