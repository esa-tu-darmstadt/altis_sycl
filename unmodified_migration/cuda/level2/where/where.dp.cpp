////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\where\where.cu
//
// summary:	Where class
// 
// origin: 
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <dpct/dpl_utils.hpp>

#include <chrono>

/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
/// <summary>	The elapsed time. </summary>
float elapsedTime;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Checks. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">  	The value. </param>
/// <param name="bound">	The bound. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

bool check(int val, int bound) {
    return (val < bound);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Mark matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="size">   	The size. </param>
/// <param name="bound">  	The bound. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void markMatches(int *arr, int *results, int size, int bound,
                 const sycl::nd_item<3> &item_ct1) {

    // Block index
    int bx = item_ct1.get_group(2);

    // Thread index
    int tx = item_ct1.get_local_id(2);

    int tid = (item_ct1.get_local_range(2) * bx) + tx;

    for (; tid < size;
         tid += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        if(check(arr[tid], bound)) {
            results[tid] = 1;
        } else {
            results[tid] = 0;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Map matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="prefix"> 	[in,out] If non-null, the prefix. </param>
/// <param name="final">  	[in,out] If non-null, the final. </param>
/// <param name="size">   	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void mapMatches(int *arr, int *results, int *prefix, int *final, int size,
                const sycl::nd_item<3> &item_ct1) {

    // Block index
    int bx = item_ct1.get_group(2);

    // Thread index
    int tx = item_ct1.get_local_id(2);

    int tid = (item_ct1.get_local_range(2) * bx) + tx;

    for (; tid < size;
         tid += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        if(results[tid]) {
            final[prefix[tid]] = arr[tid];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Seed array. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr"> 	[in,out] If non-null, the array. </param>
/// <param name="size">	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void seedArr(int *arr, int size) {
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Wheres. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="size">	   	The size. </param>
/// <param name="coverage">	The coverage. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void where(ResultDatabase &resultDB, OptionParser &op, int size, int coverage) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    int *arr = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:845: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            arr = sycl::malloc_shared<int>(size, dpct::get_default_queue())));
    } else {
        arr = (int*)malloc(sizeof(int) * size);
        assert(arr);
    }
    int *final;
    seedArr(arr, size);

    int *d_arr;
    int *d_results;
    int *d_prefix;
    int *d_final;
    
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        d_arr = arr;
        /*
        DPCT1064:846: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(d_results = sycl::malloc_shared<int>(
                                             size, dpct::get_default_queue())));
        /*
        DPCT1064:847: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(d_prefix = sycl::malloc_shared<int>(
                                             size, dpct::get_default_queue())));
    } else {
        /*
        DPCT1064:848: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_arr = sycl::malloc_device<int>(size, dpct::get_default_queue())));
        /*
        DPCT1064:849: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(d_results = sycl::malloc_device<int>(
                                             size, dpct::get_default_queue())));
        /*
        DPCT1064:850: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(d_prefix = sycl::malloc_device<int>(
                                             size, dpct::get_default_queue())));
    }

    /*
    DPCT1012:813: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:814: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    if (uvm) {
        // do nothing
    } else if (uvm_advise) {
        /*
        DPCT1063:815: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                d_arr, sizeof(int) * size, 0)));
        /*
        DPCT1063:816: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                d_arr, sizeof(int) * size, 0)));
    } else if (uvm_prefetch) {
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                 .get_device(device)
                                 .default_queue()
                                 .prefetch(d_arr, sizeof(int) * size)));
    } else if (uvm_prefetch_advise) {
        /*
        DPCT1063:817: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                d_arr, sizeof(int) * size, 0)));
        /*
        DPCT1063:818: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                d_arr, sizeof(int) * size, 0)));
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                 .get_device(device)
                                 .default_queue()
                                 .prefetch(d_arr, sizeof(int) * size)));
    } else {
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::get_default_queue()
                                 .memcpy(d_arr, arr, sizeof(int) * size)
                                 .wait()));
    }
    /*
    DPCT1012:819: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:820: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3;

    sycl::range<3> grid(1, 1, size / 1024 + 1);
    sycl::range<3> threads(1, 1, 1024);
    /*
    DPCT1012:821: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:822: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    /*
    DPCT1049:168: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
      *stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(grid * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
                markMatches(d_arr, d_results, size, coverage, item_ct1);
          });
    /*
    DPCT1012:823: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:824: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
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

    /*
    DPCT1012:825: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:826: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    std::exclusive_scan(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d_results, d_results + size, d_prefix,
        (decltype(d_prefix)::value_type)0);
    /*
    DPCT1012:827: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:828: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    int matchSize;
    /*
    DPCT1012:829: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:830: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        matchSize = (int)*(d_prefix + size - 1);
    } else {
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue()
                .memcpy(&matchSize, d_prefix + size - 1, sizeof(int))
                .wait()));
    }
    /*
    DPCT1012:831: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:832: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3;
    matchSize++;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:851: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_final = sycl::malloc_shared<int>(
                                 matchSize, dpct::get_default_queue())));
        final = d_final;
    } else {
        /*
        DPCT1064:852: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_final = sycl::malloc_device<int>(
                                 matchSize, dpct::get_default_queue())));
        final = (int*)malloc(sizeof(int) * matchSize);
        assert(final);
    }

    /*
    DPCT1012:833: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:834: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    /*
    DPCT1049:169: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
      *stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(grid * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
                mapMatches(d_arr, d_results, d_prefix, d_final, size, item_ct1);
          });
    /*
    DPCT1012:835: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:836: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
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

    /*
    DPCT1012:837: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:838: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    // No cpy just demand paging
    if (uvm) {
        // Do nothing
    } else if (uvm_advise) {
        /*
        DPCT1063:839: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                final, sizeof(int) * matchSize, 0)));
        /*
        DPCT1063:840: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                final, sizeof(int) * matchSize, 0)));
    } else if (uvm_prefetch) {
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
                final, sizeof(int) * matchSize)));
    } else if (uvm_prefetch_advise) {
        /*
        DPCT1063:841: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                final, sizeof(int) * matchSize, 0)));
        /*
        DPCT1063:842: Advice parameter is device-defined and was set to 0. You
        may need to adjust it.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_device(device).default_queue().mem_advise(
                final, sizeof(int) * matchSize, 0)));
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
                final, sizeof(int) * matchSize)));
    } else {
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue()
                .memcpy(final, d_final, sizeof(int) * matchSize)
                .wait()));
    }
    /*
    DPCT1012:843: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:844: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_arr, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_results, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_prefix, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_final, dpct::get_default_queue())));
    } else {
        free(arr);
        free(final);
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_arr, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_results, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_prefix, dpct::get_default_queue())));
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_final, dpct::get_default_queue())));
    }
    
    char atts[1024];
    sprintf(atts, "size:%d, coverage:%d", size, coverage);
    resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("where_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("where_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
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
  op.addOption("coverage", OPT_INT, "-1", "0 to 100 percentage of elements to allow through where filter");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
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

    if (!quiet) {
        printf("Using size=%d, coverage=%d\n", size, coverage);
    }

    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        if(!quiet) {
            printf("Pass %d: ", i);
        }
        where(resultDB, op, size, coverage);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}
