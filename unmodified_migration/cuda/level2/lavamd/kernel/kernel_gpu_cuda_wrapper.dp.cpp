//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "./../lavaMD.h" // (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer
#include "cudacommon.h"

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.dp.cpp"
#include <chrono>
                                                // (in the current directory)
                                                // GPU kernel, cannot include
                                                // with header file because of
                                                // complications with passing of
                                                // constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

/// <summary>	An enum constant representing the void option. </summary>
void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu,
                        ResultDatabase &resultDB,
						OptionParser &op)
{
	bool uvm = op.getOptionBool("uvm");

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));
    float elapsedTime;

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));

        //====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

        sycl::range<3> threads(1, 1, 1);
        sycl::range<3> blocks(1, 1, 1);

        //====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

        blocks[2] = dim_cpu.number_boxes;
        blocks[1] = 1;
        threads[2] = NUMBER_THREADS; // define the number of threads in the block
        threads[1] = 1;

        //======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	if (uvm) {
		d_box_gpu = box_cpu;
	} else {
                /*
                DPCT1064:292: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    d_box_gpu = (box_str *)sycl::malloc_device(
                        dim_cpu.box_mem, dpct::get_default_queue())));
        }

	//==================================================50
	//	rv
	//==================================================50

	if (uvm) {
		d_rv_gpu = rv_cpu;
	} else {
                /*
                DPCT1064:293: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    d_rv_gpu = (FOUR_VECTOR *)sycl::malloc_device(
                        dim_cpu.space_mem, dpct::get_default_queue())));
        }

	//==================================================50
	//	qv
	//==================================================50

	if (uvm) {
		d_qv_gpu = qv_cpu;
	} else {
                /*
                DPCT1064:294: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    d_qv_gpu = (double *)sycl::malloc_device(
                        dim_cpu.space_mem2, dpct::get_default_queue())));
        }

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm) {
		d_fv_gpu = fv_cpu;
	} else {
                /*
                DPCT1064:295: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    d_fv_gpu = (FOUR_VECTOR *)sycl::malloc_device(
                        dim_cpu.space_mem, dpct::get_default_queue())));
        }

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

    /*
    DPCT1012:278: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:279: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);

        if (uvm) {
		// Demand paging
	} else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(d_box_gpu, box_cpu, dim_cpu.box_mem)
                        .wait()));
        }

	//==================================================50
	//	rv
	//==================================================50
	
	if (uvm) {
		// Demand paging
	} else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem)
                        .wait()));
        }

	//==================================================50
	//	qv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2)
                        .wait()));
        }

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem)
                        .wait()));
        }

        /*
        DPCT1012:280: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:281: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3;

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// launch kernel - all boxes
    /*
    DPCT1012:282: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:283: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      {
            dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                         {sycl::aspect::fp64});
            *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::local_accessor<FOUR_VECTOR, 1> rA_shared_acc_ct1(
                      sycl::range<1>(100), cgh);
                  sycl::local_accessor<FOUR_VECTOR, 1> rB_shared_acc_ct1(
                      sycl::range<1>(100), cgh);
                  sycl::local_accessor<double, 1> qB_shared_acc_ct1(
                      sycl::range<1>(100), cgh);

                  cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         kernel_gpu_cuda(
                                             par_cpu, dim_cpu, d_box_gpu,
                                             d_rv_gpu, d_qv_gpu, d_fv_gpu,
                                             item_ct1,
                                             rA_shared_acc_ct1.get_pointer(),
                                             rB_shared_acc_ct1.get_pointer(),
                                             qB_shared_acc_ct1.get_pointer());
                                   });
            });
      }
        /*
        DPCT1012:284: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:285: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop->wait();
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    kernelTime += elapsedTime * 1.e-3;

    CHECK_CUDA_ERROR();
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));

        //======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)kernel
	//======================================================================================================================================================150

    /*
    DPCT1012:288: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:289: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);

        if (uvm) {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().prefetch(
                        d_fv_gpu, dim_cpu.space_mem)));
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_default_queue().wait()));
        } else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem)
                        .wait()));
        }

        /*
        DPCT1012:290: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:291: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3;

    char atts[1024];
    sprintf(atts, "boxes1d:%d", dim_cpu.boxes1d_arg);
    resultDB.AddResult("lavamd_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("lavamd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("lavamd_parity", atts, "N", transferTime / kernelTime);

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	if (uvm) {
		// Demand paging, no need to free
	} else {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(d_rv_gpu, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(d_qv_gpu, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(d_fv_gpu, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(d_box_gpu, dpct::get_default_queue())));
        }
}
