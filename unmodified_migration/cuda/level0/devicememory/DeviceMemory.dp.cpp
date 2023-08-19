////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level0\devicememory\DeviceMemory.cu
//
// summary:	Device memory class
// 
// origin: SHOC (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"
// #include "Timer.h"
#include "Utility.h"
#include "cudacommon.h"
#include "support.h"
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	 Forward declarations for texture memory test and benchmark kernels.
/// 			 Tests texture memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation specified by the user in command line. </param>
/// <param name="scalet">  	The scalet. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void TestTextureMem(ResultDatabase &resultDB, OptionParser &op, double scalet);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readGlobalMemoryCoalesced(float *data, float *output, int globalWorkSize, int size, int repeat,
                               sycl::nd_item<3> item_ct1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readGlobalMemoryUnit(float *data, float *output, int maxGroupSize, int size, int repeat,
                          sycl::nd_item<3> item_ct1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	The data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size pf input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readLocalMemory(const float *data, float *output, int size, int repeat,
                     sycl::nd_item<3> item_ct1, float *lbuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times for specified operations. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeGlobalMemoryCoalesced(float *output, int globalWorkSize, int size, int repeat,
                                sycl::nd_item<3> item_ct1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of data. </param>
/// <param name="repeat">	The repeat times of global write. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeGlobalMemoryUnit(float *output, int maxGroupSize, int size, int repeat,
                           sycl::nd_item<3> item_ct1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat times of write operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeLocalMemory(float *output, int size, int repeat,
                      sycl::nd_item<3> item_ct1, float *lbuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets a random int (GPU). </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="seed">	The seed. </param>
/// <param name="mod"> 	The modifier. </param>
///
/// <returns>	The randomly generated int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int getRand(int seed, int mod);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads the texels. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
/// <param name="width">	The width. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexels(int n, float *d_out, int width, sycl::nd_item<3> item_ct1,
                dpct::image_accessor_ext<sycl::float4, 2> texA);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels in cache. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexelsInCache(int n, float *d_out, sycl::nd_item<3> item_ct1,
                       dpct::image_accessor_ext<sycl::float4, 2> texA);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels random. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">	 	An int to process. </param>
/// <param name="d_out"> 	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexelsRandom(int n, float *d_out, int width, int height,
                      sycl::nd_item<3> item_ct1,
                      dpct::image_accessor_ext<sycl::float4, 2> texA);
// Texture to use for the benchmarks
/// <summary>	The tex a. </summary>
dpct::image_wrapper<sycl::float4, 2> texA;

// ****************************************************************************
// Function: addBenchmarkSpecOptions (From SHOC)
//
// Purpose:
//   Add benchmark specific options parsing.  Note that device memory has no
//   benchmark specific options, so this is just a stub.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 11, 2009
//
// Modifications:
//
// ****************************************************************************

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only use demand paging");
}

// ****************************************************************************
// Function: runBenchmark (From SHOC)
//
// Purpose:
//   This benchmark measures the device memory bandwidth for several areas
//   of memory including global, shared, and texture memories for several
//   types of access patterns.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: September 08, 2009
//
// Modifications:
//   Gabriel Marin, 06/09/2010: Change memory access patterns to eliminate
//   data reuse. Add auto-scaling factor.
//
//   Jeremy Meredith, 10/09/2012: Ignore errors at large thread counts
//   in case only smaller thread counts succeed on some devices.
//
//   Jeremy Meredith, Wed Oct 10 11:54:32 EDT 2012
//   Auto-scaling factor could be less than 1 on some problems.  This would
//   make some iteration counts zero and actually skip tests.  I enforced
//   that the factor ber at least 1.
//   
//   Bodun Hu, May 29 2020: added safe call wrappers
//   Update deprecated calls
//   Update threads and blocks dim
//
// ****************************************************************************

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
    cout << "Running DeviceMemory" << endl;
    // Enable quiet output
    bool quiet = op.getOptionBool("quiet");
    // Number of times to repeat each test
    const unsigned int passes = op.getOptionInt("passes");
    const bool uvm = op.getOptionBool("uvm");

    // threads dim per block
    size_t minGroupSize = warp_size(op.getOptionInt("device"));
    size_t maxGroupSize = max_threads_per_block(op.getOptionInt("device"));

    size_t globalWorkSize = 64 * maxGroupSize;  // 64 * 1024=65536, 64 blocks, may need tuning
    unsigned long memSize = 1024 * 1024 * 1024; // 64MB buffer
    // unsigned long memSize = ((unsigned long)-1) / 2;
    void *testmem = NULL;
    testmem =
        (void *)sycl::malloc_device(memSize * 2, dpct::get_default_queue());
    /*
    DPCT1010:548: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    while (0 != 0 && memSize != 0) {
        memSize >>= 1; // keept it a power of 2
        testmem =
            (void *)sycl::malloc_device(memSize * 2, dpct::get_default_queue());
    }
    /*
    DPCT1003:549: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:627: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(testmem, dpct::get_default_queue()), 0));
    if (memSize == 0) {
        printf("Not able to allocate device memory. Exiting!\n");
        safe_exit(-1);
    }

    const unsigned int numWordsFloat = memSize / sizeof(float);

    // Initialize host memory
    float *h_in = new float[numWordsFloat];
    float *h_out = new float[numWordsFloat];
    srand48(8650341L);
    for (int i = 0; i < numWordsFloat; ++i) {
        h_in[i] = (float)(drand48() * numWordsFloat);
    }

    // Allocate two chunks of device memory
    float *d_mem1, *d_mem2;
    char sizeStr[128];

    if (uvm) {
        /*
        DPCT1003:550: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:628: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((d_mem1 = sycl::malloc_shared<float>(
                             (numWordsFloat), dpct::get_default_queue()),
                         0));
        /*
        DPCT1003:629: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((d_mem2 = sycl::malloc_shared<float>(
                             (numWordsFloat), dpct::get_default_queue()),
                         0));
    } else {
        checkCudaErrors((d_mem1 = sycl::malloc_device<float>(
                             (numWordsFloat), dpct::get_default_queue()),
                         0));
        /*
        DPCT1003:551: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((d_mem2 = sycl::malloc_device<float>(
                             (numWordsFloat), dpct::get_default_queue()),
                         0));
    }

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1027:552: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:630: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:553: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:631: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);

    /*
    DPCT1012:554: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:555: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    /*
    DPCT1012:632: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
      stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                sycl::range<3>(1, 1, 64),
                            sycl::range<3>(1, 1, 64)),
          [=](sycl::nd_item<3> item_ct1) {
                readGlobalMemoryCoalesced(d_mem1, d_mem2, 512 * 64,
                                          numWordsFloat, 256, item_ct1);
          });
    /*
    DPCT1012:556: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:557: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    /*
    DPCT1012:633: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop.wait();
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    float t = 0.0f;
    /*
    DPCT1003:634: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count(),
         0));
    t /= 1.e3;
    double scalet = 0.1 / t;
    if (scalet < 1)
        scalet = 1;

    const unsigned int maxRepeatsCoal = 256 * scalet;
    const unsigned int maxRepeatsUnit = 16 * scalet;
    const unsigned int maxRepeatsLocal = 300 * scalet;

    for (int p = 0; p < passes; p++) {
        // Run the kernel for each group size
        if (!quiet) {
            cout << "Pass: " << p << "\n";
        }
        for (int threads = minGroupSize; threads <= maxGroupSize; threads *= 2) {
            const unsigned int blocks = globalWorkSize / threads;
            assert(globalWorkSize % threads == 0);     // Just to make sure no outstanding thread
            double bdwth;
            sprintf(sizeStr, "blockSize:%03d", threads);

            // Test 1
            /*
            DPCT1012:558: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:559: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:635: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:560: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                            sycl::range<3>(1, 1, threads),
                                        sycl::range<3>(1, 1, threads)),
                      [=](sycl::nd_item<3> item_ct1) {
                            readGlobalMemoryCoalesced(
                                d_mem1, d_mem2, globalWorkSize, numWordsFloat,
                                maxRepeatsCoal, item_ct1);
                      });

            /*
            DPCT1012:561: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:562: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:636: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);

            // We can run out of resources at larger thread counts on
            // some devices.  If we made a successful run at smaller
            // thread counts, just ignore errors at this size.
            if (threads > minGroupSize) {
                /*
                DPCT1010:563: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                if (0 != 0)
                    break;
            } else {
                CHECK_CUDA_ERROR();
            }
            t = 0.0f;
            /*
            DPCT1003:564: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:637: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsCoal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readGlobalMemoryCoalesced", sizeStr, "GB/s", bdwth);

            // Test 2
            /*
            DPCT1012:565: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:566: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:638: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:567: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                            sycl::range<3>(1, 1, threads),
                                        sycl::range<3>(1, 1, threads)),
                      [=](sycl::nd_item<3> item_ct1) {
                            readGlobalMemoryUnit(d_mem1, d_mem2, maxGroupSize,
                                                 numWordsFloat, maxRepeatsUnit,
                                                 item_ct1);
                      });

            /*
            DPCT1012:568: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:569: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:639: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:640: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsUnit * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readGlobalMemoryUnit", sizeStr, "GB/s", bdwth);
            resultDB.AddOverall("Bandwidth", "GB/s", bdwth);

            // Test 3
            /*
            DPCT1012:570: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:571: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:641: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:572: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor<float, 1, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            lbuf_acc_ct1(sycl::range<1>(2048), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                  sycl::range<3>(1, 1, threads),
                                              sycl::range<3>(1, 1, threads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  readLocalMemory(d_mem1, d_mem2, numWordsFloat,
                                                  maxRepeatsLocal, item_ct1,
                                                  lbuf_acc_ct1.get_pointer());
                            });
                  });

            /*
            DPCT1012:573: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:574: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:642: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:643: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsLocal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readLocalMemory", sizeStr, "GB/s", bdwth);

            // Test 4
            /*
            DPCT1012:644: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:575: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                            sycl::range<3>(1, 1, threads),
                                        sycl::range<3>(1, 1, threads)),
                      [=](sycl::nd_item<3> item_ct1) {
                            writeGlobalMemoryCoalesced(
                                d_mem2, globalWorkSize, numWordsFloat,
                                maxRepeatsCoal, item_ct1);
                      });

            /*
            DPCT1012:576: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:577: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:645: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:578: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:646: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsCoal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeGlobalMemoryCoalesced", sizeStr, "GB/s", bdwth);

            // Test 5
            /*
            DPCT1012:579: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:580: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:647: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:581: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                            sycl::range<3>(1, 1, threads),
                                        sycl::range<3>(1, 1, threads)),
                      [=](sycl::nd_item<3> item_ct1) {
                            writeGlobalMemoryUnit(d_mem2, maxGroupSize,
                                                  numWordsFloat, maxRepeatsUnit,
                                                  item_ct1);
                      });

            /*
            DPCT1012:582: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:583: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:648: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:584: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:649: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsUnit * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeGlobalMemoryUnit", sizeStr, "GB/s", bdwth);
            resultDB.AddOverall("Bandwidth", "GB/s", bdwth);

            // Test 6
            /*
            DPCT1012:585: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:586: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:650: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            /*
            DPCT1049:587: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor<float, 1, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            lbuf_acc_ct1(sycl::range<1>(2048), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                  sycl::range<3>(1, 1, threads),
                                              sycl::range<3>(1, 1, threads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  writeLocalMemory(d_mem2, numWordsFloat,
                                                   maxRepeatsLocal, item_ct1,
                                                   lbuf_acc_ct1.get_pointer());
                            });
                  });

            /*
            DPCT1012:588: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:589: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:651: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:590: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:652: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsLocal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeLocalMemory", sizeStr, "GB/s", bdwth);
        }
    }
    /*
    DPCT1003:591: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:653: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mem1, dpct::get_default_queue()), 0));
    /*
    DPCT1003:592: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:654: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mem2, dpct::get_default_queue()), 0));
    delete[] h_in;
    delete[] h_out;
    /*
    DPCT1027:593: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:655: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:594: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:656: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    TestTextureMem(resultDB, op, scalet);
}

// ****************************************************************************
// Function: TestTextureMem (from SHOC)
//
// Purpose:
//   Measures the bandwidth of texture memory for several access patterns
//   using a 2D texture including sequential, "random", and repeated access to
//   texture cache.  Texture memory is often a viable alternative to global
//   memory, especially when data access patterns prevent good coalescing.
//
// Arguments:
//   resultDB: results from the benchmark are stored to this resultd database
//   op: the options parser / parameter database
//   scalet: auto-scaling factor for the number of repetitions
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 11, 2009
//
// Modifications:
//   Gabriel Marin 06/09/2010: add auto-scaling factor
//
//   Jeremy Meredith, Tue Nov 23 13:45:54 EST 2010
//   Change data sizes to be larger, and textures to be 2D to match OpenCL
//   variant.  Dropped #iterations to compensate.  Had to remove validation
//   for now, which also matches the current OpenCL variant's behavior.
//
//   Jeremy Meredith, Wed Oct 10 11:54:32 EDT 2012
//   Kernel rep factor of 1024 on the last texture test on the biggest
//   texture size caused Windows to time out (the more-than-five-seconds-long
//   kernel problem).  I made kernel rep factor problem-size dependent.
//
// ****************************************************************************

void TestTextureMem(ResultDatabase &resultDB, OptionParser &op, double scalet) {
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
    // Enable quiet output
    bool quiet = op.getOptionBool("quiet");
    // Number of times to repeat each test
    const unsigned int passes = op.getOptionInt("passes");
    const bool uvm = op.getOptionBool("uvm");
    // Sizes of textures tested (in kb)
    const unsigned int nsizes = 5;
    const unsigned int sizes[] = {256, 512, 1024, 2048, 4096};
    // Number of texel accesses by each kernel
    const unsigned int kernelRepFactors[] = {2048, 2048, 2048, 2048, 512};
    // Number of times to repeat each kernel per test
    const unsigned int iterations = 1 * scalet;

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    /*
    DPCT1027:595: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:657: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:596: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:658: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);

    // make sure our texture behaves like we want....
    texA.set(sycl::addressing_mode::clamp_to_edge,
             sycl::filtering_mode::nearest,
             sycl::coordinate_normalization_mode::unnormalized);

    for (int j = 0; j < nsizes; j++) {
        if(!quiet) {
            cout << "Benchmarking Texture Memory, Test Size: " << j + 1 << " / 5\n";
        }
        const unsigned int size = 1024 * sizes[j];
        const unsigned int numFloat = size / sizeof(float);
        const unsigned int numFloat4 = size / sizeof(sycl::float4);
        size_t width, height;

        const unsigned int kernelRepFactor = kernelRepFactors[j];

        // Image memory sizes should be power of 2.
        size_t sizeLog = lround(log2(double(numFloat4)));
        height = 1 << (sizeLog >> 1); // height is the smaller size
        width = numFloat4 / height;

        const sycl::range<3> blockSize(1, 8, 16);
        const sycl::range<3> gridSize(1, height / blockSize[1],
                                      width / blockSize[2]);

        float *h_in = new float[numFloat];
        float *h_out = new float[numFloat4];
        float *d_out = NULL;
        if (uvm) {
            /*
            DPCT1003:597: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:659: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((d_out = sycl::malloc_shared<float>(
                                 numFloat4, dpct::get_default_queue()),
                             0));
        } else {
            /*
            DPCT1003:598: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((d_out = sycl::malloc_device<float>(
                                 numFloat4, dpct::get_default_queue()),
                             0));
        }

        // Fill input data with some pattern
        for (unsigned int i = 0; i < numFloat; i++) {
            h_in[i] = (float)i;
            if (i < numFloat4) {
                h_out[i] = 0.0f;
            }
        }

        // Allocate a cuda array
        dpct::image_matrix *cuArray;
        /*
        DPCT1003:599: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:660: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((cuArray = new dpct::image_matrix(
                             texA.get_channel(), sycl::range<2>(width, height)),
                         0));

        // Copy in source data
        //checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, h_in, size, cudaMemcpyHostToDevice));

        /*
        DPCT1003:600: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:661: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::dpct_memcpy(cuArray->to_pitched_data(), sycl::id<3>(0, 0, 0),
                               dpct::pitched_data(h_in, width, width, 1),
                               sycl::id<3>(0, 0, 0),
                               sycl::range<3>(width, height, 1)),
             0));

        // Bind texture to the array
        /*
        DPCT1003:662: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((texA.attach(cuArray), 0));

        for (int p = 0; p < passes; p++) {
            // Test 1: Repeated Linear Access
            float t = 0.0f;

            /*
            DPCT1012:602: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:603: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:663: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            // read texels from texture
            for (int iter = 0; iter < iterations; iter++) {
                /*
                DPCT1049:604: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  auto texA_acc = texA.get_access(cgh);

                                  auto texA_smpl = texA.get_sampler();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(gridSize * blockSize,
                                                        blockSize),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            readTexels(
                                                kernelRepFactor, d_out, width,
                                                item_ct1,
                                                dpct::image_accessor_ext<
                                                    sycl::float4, 2>(texA_smpl,
                                                                     texA_acc));
                                      });
                            });
            }
            /*
            DPCT1012:605: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:606: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:664: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:607: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:665: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;

            // Calculate speed in GB/s
            double speed = (double)kernelRepFactor * (double)iterations *
                                         (double)(size / (1000. * 1000. * 1000.)) / (t);

            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedLinearAccess", sizeStr, "GB/s",
                                                 speed);

            // Verify results
            /*
            DPCT1003:608: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:666: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_default_queue()
                     .memcpy(h_out, d_out, numFloat4 * sizeof(float))
                     .wait(),
                 0));

            // Test 2 Repeated Cache Access
            /*
            DPCT1012:609: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:610: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:667: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            for (int iter = 0; iter < iterations; iter++) {
                /*
                DPCT1049:611: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  auto texA_acc = texA.get_access(cgh);

                                  auto texA_smpl = texA.get_sampler();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(gridSize * blockSize,
                                                        blockSize),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            readTexelsInCache(
                                                kernelRepFactor, d_out,
                                                item_ct1,
                                                dpct::image_accessor_ext<
                                                    sycl::float4, 2>(texA_smpl,
                                                                     texA_acc));
                                      });
                            });
            }
            /*
            DPCT1012:601: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            stop = dpct::get_default_queue().ext_oneapi_submit_barrier();
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:612: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:668: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)iterations *
                            ((double)size / (1000. * 1000. * 1000.)) / (t);

            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedCacheHit", sizeStr, "GB/s", speed);

            // Verify results
            /*
            DPCT1003:613: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:669: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_default_queue()
                     .memcpy(h_out, d_out, numFloat4 * sizeof(float))
                     .wait(),
                 0));

            // Test 3 Repeated "Random" Access
            /*
            DPCT1012:614: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:615: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:670: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (start = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));

            // read texels from texture
            for (int iter = 0; iter < iterations; iter++) {
                /*
                DPCT1049:616: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  auto texA_acc = texA.get_access(cgh);

                                  auto texA_smpl = texA.get_sampler();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(gridSize * blockSize,
                                                        blockSize),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            readTexelsRandom(
                                                kernelRepFactor, d_out, width,
                                                height, item_ct1,
                                                dpct::image_accessor_ext<
                                                    sycl::float4, 2>(texA_smpl,
                                                                     texA_acc));
                                      });
                            });
            }

            /*
            DPCT1012:617: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:618: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:671: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            dpct::get_current_device().queues_wait_and_throw();
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(
                (stop = dpct::get_default_queue().ext_oneapi_submit_barrier(),
                 0));
            checkCudaErrors(0);
            CHECK_CUDA_ERROR();
            /*
            DPCT1003:619: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:672: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            t /= 1.e3;

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)iterations *
                            ((double)size / (1000. * 1000. * 1000.)) / (t);

            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedRandomAccess", sizeStr, "GB/s",
                                                 speed);
        }
        delete[] h_in;
        delete[] h_out;
        /*
        DPCT1003:620: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:673: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((sycl::free(d_out, dpct::get_default_queue()), 0));
        /*
        DPCT1003:621: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:674: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((delete cuArray, 0));
        /*
        DPCT1003:622: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:675: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((texA.detach(), 0));
    }
    /*
    DPCT1027:623: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:676: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:624: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:677: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
}

// Begin benchmark kernels

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. 
///             Add total threads into parameter for coalescing </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readGlobalMemoryCoalesced(float *data, float *output, int globalWorkSize, int size, int repeat,
                               sycl::nd_item<3> item_ct1) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    float sum = 0;
    int s = gid;
    for (j = 0; j < repeat; ++j) {
        float a0 = data[(s + 0) & (size - 1)];
        float a1 = data[(s + globalWorkSize) & (size - 1)];
        float a2 = data[(s + globalWorkSize * 2) & (size - 1)];
        float a3 = data[(s + globalWorkSize * 3) & (size - 1)];
        float a4 = data[(s + globalWorkSize * 4) & (size - 1)];
        float a5 = data[(s + globalWorkSize * 5) & (size - 1)];
        float a6 = data[(s + globalWorkSize * 6) & (size - 1)];
        float a7 = data[(s + globalWorkSize * 7) & (size - 1)];
        float a8 = data[(s + globalWorkSize * 8) & (size - 1)];
        float a9 = data[(s + globalWorkSize * 9) & (size - 1)];
        float a10 = data[(s + globalWorkSize * 10) & (size - 1)];
        float a11 = data[(s + globalWorkSize * 11) & (size - 1)];
        float a12 = data[(s + globalWorkSize * 12) & (size - 1)];
        float a13 = data[(s + globalWorkSize * 13) & (size - 1)];
        float a14 = data[(s + globalWorkSize * 14) & (size - 1)];
        float a15 = data[(s + globalWorkSize * 15) & (size - 1)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + globalWorkSize * 16) & (size - 1);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readGlobalMemoryUnit(float *data, float *output, int maxGroupSize, int size, int repeat,
                          sycl::nd_item<3> item_ct1) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    float sum = 0;
    int s = gid * maxGroupSize;
    for (j = 0; j < repeat; ++j) {
        float a0 = data[(s + 0) & (size - 1)];
        float a1 = data[(s + 1) & (size - 1)];
        float a2 = data[(s + 2) & (size - 1)];
        float a3 = data[(s + 3) & (size - 1)];
        float a4 = data[(s + 4) & (size - 1)];
        float a5 = data[(s + 5) & (size - 1)];
        float a6 = data[(s + 6) & (size - 1)];
        float a7 = data[(s + 7) & (size - 1)];
        float a8 = data[(s + 8) & (size - 1)];
        float a9 = data[(s + 9) & (size - 1)];
        float a10 = data[(s + 10) & (size - 1)];
        float a11 = data[(s + 11) & (size - 1)];
        float a12 = data[(s + 12) & (size - 1)];
        float a13 = data[(s + 13) & (size - 1)];
        float a14 = data[(s + 14) & (size - 1)];
        float a15 = data[(s + 15) & (size - 1)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + 16) & (size - 1);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	The data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readLocalMemory(const float *data, float *output, int size, int repeat,
                     sycl::nd_item<3> item_ct1, float *lbuf) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    float sum = 0;
    int tid = item_ct1.get_local_id(2), localSize = item_ct1.get_local_range(2),
        grpid = item_ct1.get_group(2), litems = 2048 / localSize,
        goffset = localSize * grpid + tid * litems;
    int s = tid;

    for (; j < litems && j < (size - goffset); ++j)
        lbuf[tid * litems + j] = data[goffset + j];
    for (int i = 0; j < litems; ++j, ++i)
        lbuf[tid * litems + j] = data[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (j = 0; j < repeat; ++j) {
        float a0 = lbuf[(s + 0) & (2047)];
        float a1 = lbuf[(s + 1) & (2047)];
        float a2 = lbuf[(s + 2) & (2047)];
        float a3 = lbuf[(s + 3) & (2047)];
        float a4 = lbuf[(s + 4) & (2047)];
        float a5 = lbuf[(s + 5) & (2047)];
        float a6 = lbuf[(s + 6) & (2047)];
        float a7 = lbuf[(s + 7) & (2047)];
        float a8 = lbuf[(s + 8) & (2047)];
        float a9 = lbuf[(s + 9) & (2047)];
        float a10 = lbuf[(s + 10) & (2047)];
        float a11 = lbuf[(s + 11) & (2047)];
        float a12 = lbuf[(s + 12) & (2047)];
        float a13 = lbuf[(s + 13) & (2047)];
        float a14 = lbuf[(s + 14) & (2047)];
        float a15 = lbuf[(s + 15) & (2047)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + 16) & (2047);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeGlobalMemoryCoalesced(float *output, int globalWorkSize, int size, int repeat,
                                sycl::nd_item<3> item_ct1) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    int s = gid;
    for (j = 0; j < repeat; ++j) {
        output[(s + 0) & (size - 1)] = gid;
        output[(s + globalWorkSize) & (size - 1)] = gid;
        output[(s + globalWorkSize * 2) & (size - 1)] = gid;
        output[(s + globalWorkSize * 3) & (size - 1)] = gid;
        output[(s + globalWorkSize * 4) & (size - 1)] = gid;
        output[(s + globalWorkSize * 5) & (size - 1)] = gid;
        output[(s + globalWorkSize * 6) & (size - 1)] = gid;
        output[(s + globalWorkSize * 7) & (size - 1)] = gid;
        output[(s + globalWorkSize * 8) & (size - 1)] = gid;
        output[(s + globalWorkSize * 9) & (size - 1)] = gid;
        output[(s + globalWorkSize * 10) & (size - 1)] = gid;
        output[(s + globalWorkSize * 11) & (size - 1)] = gid;
        output[(s + globalWorkSize * 12) & (size - 1)] = gid;
        output[(s + globalWorkSize * 13) & (size - 1)] = gid;
        output[(s + globalWorkSize * 14) & (size - 1)] = gid;
        output[(s + globalWorkSize * 15) & (size - 1)] = gid;
        s = (s + globalWorkSize * 16) & (size - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat times of writing op. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeGlobalMemoryUnit(float *output, int maxGroupSize, int size, int repeat,
                           sycl::nd_item<3> item_ct1) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    int s = gid * maxGroupSize;
    for (j = 0; j < repeat; ++j) {
        output[(s + 0) & (size - 1)] = gid;
        output[(s + 1) & (size - 1)] = gid;
        output[(s + 2) & (size - 1)] = gid;
        output[(s + 3) & (size - 1)] = gid;
        output[(s + 4) & (size - 1)] = gid;
        output[(s + 5) & (size - 1)] = gid;
        output[(s + 6) & (size - 1)] = gid;
        output[(s + 7) & (size - 1)] = gid;
        output[(s + 8) & (size - 1)] = gid;
        output[(s + 9) & (size - 1)] = gid;
        output[(s + 10) & (size - 1)] = gid;
        output[(s + 11) & (size - 1)] = gid;
        output[(s + 12) & (size - 1)] = gid;
        output[(s + 13) & (size - 1)] = gid;
        output[(s + 14) & (size - 1)] = gid;
        output[(s + 15) & (size - 1)] = gid;
        s = (s + 16) & (size - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The number of writes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void writeLocalMemory(float *output, int size, int repeat,
                      sycl::nd_item<3> item_ct1, float *lbuf) {
    int gid = item_ct1.get_local_id(2) +
              (item_ct1.get_local_range(2) * item_ct1.get_group(2)),
        j = 0;
    int tid = item_ct1.get_local_id(2), localSize = item_ct1.get_local_range(2),
        litems = 2048 / localSize;
    int s = tid;

    for (j = 0; j < repeat; ++j) {
        lbuf[(s + 0) & (2047)] = gid;
        lbuf[(s + 1) & (2047)] = gid;
        lbuf[(s + 2) & (2047)] = gid;
        lbuf[(s + 3) & (2047)] = gid;
        lbuf[(s + 4) & (2047)] = gid;
        lbuf[(s + 5) & (2047)] = gid;
        lbuf[(s + 6) & (2047)] = gid;
        lbuf[(s + 7) & (2047)] = gid;
        lbuf[(s + 8) & (2047)] = gid;
        lbuf[(s + 9) & (2047)] = gid;
        lbuf[(s + 10) & (2047)] = gid;
        lbuf[(s + 11) & (2047)] = gid;
        lbuf[(s + 12) & (2047)] = gid;
        lbuf[(s + 13) & (2047)] = gid;
        lbuf[(s + 14) & (2047)] = gid;
        lbuf[(s + 15) & (2047)] = gid;
        s = (s + 16) & (2047);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (j = 0; j < litems; ++j)
        output[gid] = lbuf[tid];
}

// Simple Repeated Linear Read from texture memory

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads the texels. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
/// <param name="width">	The width. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexels(int n, float *d_out, int width, sycl::nd_item<3> item_ct1,
                dpct::image_accessor_ext<sycl::float4, 2> texA) {
    int idx_x = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    int idx_y = (item_ct1.get_group(1) * item_ct1.get_local_range(1)) +
                item_ct1.get_local_id(1);
    int out_idx = idx_y * item_ct1.get_group_range(2) + idx_x;
    float sum = 0.0f;
    int width_bits = width - 1;
    for (int i = 0; i < n; i++) {
        sycl::float4 v = texA.read(float(idx_x), float(idx_y));
        idx_x = (idx_x + 1) & width_bits;
        sum += v.x();
    }
    d_out[out_idx] = sum;
}

// Repeated read of only 4kb of texels (should fit in texture cache)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels in cache. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexelsInCache(int n, float *d_out, sycl::nd_item<3> item_ct1,
                       dpct::image_accessor_ext<sycl::float4, 2> texA) {
    int idx_x = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    int idx_y = (item_ct1.get_group(1) * item_ct1.get_local_range(1)) +
                item_ct1.get_local_id(1);
    int out_idx = idx_y * item_ct1.get_group_range(2) + idx_x;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sycl::float4 v = texA.read(float(idx_x), float(idx_y));
        sum += v.x();
    }
    d_out[out_idx] = sum;
}

// Read "random" texels

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels random. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">	 	An int to process. </param>
/// <param name="d_out"> 	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void readTexelsRandom(int n, float *d_out, int width, int height,
                      sycl::nd_item<3> item_ct1,
                      dpct::image_accessor_ext<sycl::float4, 2> texA) {
    int idx_x = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    int idx_y = (item_ct1.get_group(1) * item_ct1.get_local_range(1)) +
                item_ct1.get_local_id(1);
    int out_idx = idx_y * item_ct1.get_group_range(2) + idx_x;
    float sum = 0.0f;
    int width_bits = width - 1;
    int height_bits = height - 1;
    for (int i = 0; i < n; i++) {
        sycl::float4 v = texA.read(float(idx_x), float(idx_y));
        idx_x = (idx_x * 3 + 29) & (width_bits);
        idx_y = (idx_y * 5 + 11) & (height_bits);
        sum += v.x();
    }
    d_out[out_idx] = sum;
}
