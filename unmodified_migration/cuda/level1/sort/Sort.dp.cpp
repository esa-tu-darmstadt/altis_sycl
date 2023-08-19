////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level1\sort\Sort.cu
//
// summary:	Sort class
// 
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include "Sort.h"
#include "sort_kernel.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>
#include <chrono>

#define SEED 7

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parsefilePathr / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications: Bodun Hu
// Add UVM support
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    cout << "Running Sort" << endl;
  srand(SEED);
  bool quiet = op.getOptionBool("quiet");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  int device = 0;
  checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

  // Determine size of the array to sort
  int size;
  long long bytes;
  string filePath = op.getOptionString("inputFile");
  ifstream inputFile(filePath.c_str());
  if (filePath == "") {
    if(!quiet) {
        printf("Using problem size %d\n", (int)op.getOptionInt("size"));
    }
    int probSizes[5] = {32, 64, 256, 512, 1024};
    size = probSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
  } else {
    inputFile >> size;
  }
  bytes = size * sizeof(uint);
  if(!quiet) {
    printf("Size: %d items, Bytes: %lld\n", size, bytes);
  }

  // If input file given, populate array
  uint *sourceInput = (uint *)malloc(bytes);
  if (filePath != "") {
      for (int i = 0; i < size; i++) {
          inputFile >> sourceInput[i];
      }
  }

  // create input data on CPU
  uint *hKeys = NULL;
  uint *hVals = NULL;

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// <summary>	allocate using UVM API. </summary>
  ///
  /// <remarks>	Ed, 5/20/2020. </remarks>
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
      /*
      DPCT1064:408: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(hKeys = (uint *)sycl::malloc_shared(
                                           bytes, dpct::get_default_queue())));
      /*
      DPCT1064:409: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(hVals = (uint *)sycl::malloc_shared(
                                           bytes, dpct::get_default_queue())));
  } else {
      /*
      DPCT1064:410: Migrated cudaMallocHost call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          hKeys = (uint *)sycl::malloc_host(bytes, dpct::get_default_queue())));
      /*
      DPCT1064:411: Migrated cudaMallocHost call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          hVals = (uint *)sycl::malloc_host(bytes, dpct::get_default_queue())));
  }

  // Allocate space for block sums in the scan kernel.
  uint numLevelsAllocated = 0;
  uint maxNumScanElements = size;
  uint numScanElts = maxNumScanElements;
  uint level = 0;

  do {
    uint numBlocks =
        std::max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
    if (numBlocks > 1) {
      level++;
    }
    numScanElts = numBlocks;
  } while (numScanElts > 1);

  uint **scanBlockSums = NULL;
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
      /*
      DPCT1064:412: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(scanBlockSums = sycl::malloc_shared<uint *>(
                               (level + 1), dpct::get_default_queue())));
  } else {
      scanBlockSums = (uint **)malloc((level + 1) * sizeof(uint *));
      assert(scanBlockSums);
  }

  numLevelsAllocated = level + 1;
  numScanElts = maxNumScanElements;
  level = 0;

  do {
    uint numBlocks =
        std::max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
    if (numBlocks > 1) {
      // Malloc device mem for block sums
      if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
          /*
          DPCT1064:413: Migrated cudaMallocManaged call is used in a
          macro/template definition and may not be valid for all macro/template
          uses. Adjust the code.
          */
          checkCudaErrors(
              DPCT_CHECK_ERROR(scanBlockSums[level] = sycl::malloc_shared<uint>(
                                   numBlocks, dpct::get_default_queue())));
      } else {
          /*
          DPCT1064:414: Migrated cudaMalloc call is used in a macro/template
          definition and may not be valid for all macro/template uses. Adjust
          the code.
          */
          checkCudaErrors(
              DPCT_CHECK_ERROR(scanBlockSums[level] = sycl::malloc_device<uint>(
                                   numBlocks, dpct::get_default_queue())));
      }
      level++;
    }
    numScanElts = numBlocks;
  } while (numScanElts > 1);

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    /*
    DPCT1064:415: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        scanBlockSums[level] =
            sycl::malloc_shared<uint>(1, dpct::get_default_queue())));
  } else {
    /*
    DPCT1064:416: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        scanBlockSums[level] =
            sycl::malloc_device<uint>(1, dpct::get_default_queue())));
  }

  // Allocate device mem for sorting kernels
  uint *dKeys, *dVals, *dTempKeys, *dTempVals;

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    dKeys = hKeys;
    dVals = hVals;
    /*
    DPCT1064:417: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dTempKeys = (uint *)sycl::malloc_shared(
                                         bytes, dpct::get_default_queue())));
    /*
    DPCT1064:418: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dTempVals = (uint *)sycl::malloc_shared(
                                         bytes, dpct::get_default_queue())));
  } else {
    /*
    DPCT1064:419: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dKeys = (uint *)sycl::malloc_device(bytes, dpct::get_default_queue())));
    /*
    DPCT1064:420: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dVals = (uint *)sycl::malloc_device(bytes, dpct::get_default_queue())));
    /*
    DPCT1064:421: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dTempKeys = (uint *)sycl::malloc_device(
                                         bytes, dpct::get_default_queue())));
    /*
    DPCT1064:422: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dTempVals = (uint *)sycl::malloc_device(
                                         bytes, dpct::get_default_queue())));
  }

  // Each thread in the sort kernel handles 4 elements
  size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);

  uint *dCounters, *dCounterSums, *dBlockOffsets;
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    /*
    DPCT1064:423: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dCounters = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                              dpct::get_default_queue())));
    /*
    DPCT1064:424: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dCounterSums = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                                 dpct::get_default_queue())));
    /*
    DPCT1064:425: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dBlockOffsets = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                                  dpct::get_default_queue())));
  }
  else {
    /*
    DPCT1064:426: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dCounters = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                              dpct::get_default_queue())));
    /*
    DPCT1064:427: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dCounterSums = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                                 dpct::get_default_queue())));
    /*
    DPCT1064:428: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dBlockOffsets = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                                  dpct::get_default_queue())));
  }

  int iterations = op.getOptionInt("passes");
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

  for (int it = 0; it < iterations; it++) {
    if(!quiet) {
        printf("Pass %d: ", it);
    }
/// <summary>	Initialize host memory to some pattern. </summary>
    for (uint i = 0; i < size; i++) {
      hKeys[i] = i % 1024;
      if (filePath == "") {
        hVals[i] = rand() % 1024;
      } else {
        hVals[i] = sourceInput[i];
      }
    }

    // Copy inputs to GPU
    double transferTime = 0.;
    /*
    DPCT1012:388: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:389: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:390: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              dKeys, bytes, 0)));
      /*
      DPCT1063:391: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              dVals, bytes, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dKeys, bytes)));
      dpct::queue_ptr s1;
      checkCudaErrors(
          DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
      checkCudaErrors(DPCT_CHECK_ERROR(s1->prefetch(dVals, bytes)));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:392: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              dKeys, bytes, 0)));
      /*
      DPCT1063:393: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              dVals, bytes, 0)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dKeys, bytes)));
      dpct::queue_ptr s1;
      checkCudaErrors(
          DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dVals, bytes)));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
    } else {
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue().memcpy(dKeys, hKeys, bytes).wait()));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue().memcpy(dVals, hVals, bytes).wait()));
    }
    /*
    DPCT1012:394: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:395: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    float elapsedTime;
    checkCudaErrors(DPCT_CHECK_ERROR(
        (elapsedTime =
             std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                 .count())));
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    /*
    DPCT1012:396: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:397: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    // Perform Radix Sort (4 bits at a time)
    for (int i = 0; i < SORT_BITS; i += 4) {
      radixSortStep(4, i, (sycl::uint4 *)dKeys, (sycl::uint4 *)dVals,
                    (sycl::uint4 *)dTempKeys, (sycl::uint4 *)dTempVals,
                    dCounters, dCounterSums, dBlockOffsets, scanBlockSums,
                    size);
    }
    /*
    DPCT1012:398: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:399: The original code returned the error code that was further
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
    double kernelTime = elapsedTime * 1.e-3;
    // Readback data from device
    /*
    DPCT1012:400: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:401: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);

    // prefetch or demand paging
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:402: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().mem_advise(dKeys, bytes, 0)));
      /*
      DPCT1063:403: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().mem_advise(dVals, bytes, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().prefetch(dKeys, bytes)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().prefetch(dVals, bytes)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:404: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().mem_advise(dKeys, bytes, 0)));
      /*
      DPCT1063:405: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().mem_advise(dVals, bytes, 0)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().prefetch(dKeys, bytes)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::cpu_device().default_queue().prefetch(dVals, bytes)));
    } else {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_default_queue().memcpy(hKeys, dKeys, bytes).wait()));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_default_queue().memcpy(hVals, dVals, bytes).wait()));
    }

    /*
    DPCT1012:406: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:407: The original code returned the error code that was further
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

    // Test to make sure data was sorted properly, if not, return
    if (!verifySort(hKeys, hVals, size, op.getOptionBool("verbose"), op.getOptionBool("quiet"))) {
      return;
    }

    char atts[1024];
    sprintf(atts, "%ditems", size);
    double gb = (bytes * 2.) / (1000. * 1000. * 1000.);
    resultDB.AddResult("Sort-KernelTime", atts, "sec", kernelTime);
    resultDB.AddResult("Sort-TransferTime", atts, "sec", transferTime);
    resultDB.AddResult("Sort-TotalTime", atts, "sec", transferTime + kernelTime);
    resultDB.AddResult("Sort-Rate", atts, "GB/s", gb / kernelTime);
    resultDB.AddResult("Sort-Rate_PCIe", atts, "GB/s",
                       gb / (kernelTime + transferTime));
    resultDB.AddResult("Sort-Rate_Parity", atts, "N",
                       transferTime / kernelTime);
    resultDB.AddOverall("Rate", "GB/s", gb/kernelTime);
  }
  // Clean up
  for (int i = 0; i < numLevelsAllocated; i++) {
    checkCudaErrors(DPCT_CHECK_ERROR(
        sycl::free(scanBlockSums[i], dpct::get_default_queue())));
  }
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dKeys, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dVals, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dTempKeys, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dTempVals, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dCounters, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dCounterSums, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(dBlockOffsets, dpct::get_default_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(scanBlockSums, dpct::get_default_queue())));
  } else {
    free(scanBlockSums);
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(hKeys, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(hVals, dpct::get_default_queue())));
  }
  free(sourceInput);
}

// ****************************************************************************
// Function: radixSortStep
//
// Purpose:
//   This function performs a radix sort, using bits startbit to
//   (startbit + nbits).  It is designed to sort by 4 bits at a time.
//   It also reorders the data in the values array based on the sort.
//
// Arguments:
//      nbits: the number of key bits to use
//      startbit: the bit to start on, 0 = lsb
//      keys: the input array of keys
//      values: the input array of values
//      tempKeys: temporary storage, same size as keys
//      tempValues: temporary storage, same size as values
//      counters: storage for the index counters, used in sort
//      countersSum: storage for the sum of the counters
//      blockOffsets: storage used in sort
//      scanBlockSums: input to Scan, see below
//      numElements: the number of elements to sort
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// origin: SHOC (https://github.com/vetter/shoc)
//
// ****************************************************************************
void radixSortStep(uint nbits, uint startbit, sycl::uint4 *keys,
                   sycl::uint4 *values, sycl::uint4 *tempKeys,
                   sycl::uint4 *tempValues, uint *counters, uint *countersSum,
                   uint *blockOffsets, uint **scanBlockSums, uint numElements) {
  // Threads handle either 4 or two elements each
  const size_t radixGlobalWorkSize = numElements / 4;
  const size_t findGlobalWorkSize = numElements / 2;
  const size_t reorderGlobalWorkSize = numElements / 2;

  // Radix kernel uses block size of 128, others use 256 (same as scan)
  const size_t radixBlocks = radixGlobalWorkSize / SORT_BLOCK_SIZE;
  const size_t findBlocks = findGlobalWorkSize / SCAN_BLOCK_SIZE;
  const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint, 1> sMem_acc_ct1(sycl::range<1>(512),
                                                       cgh);
            sycl::local_accessor<uint, 0> numtrue_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, radixBlocks) *
                                      sycl::range<3>(1, 1, SORT_BLOCK_SIZE),
                                  sycl::range<3>(1, 1, SORT_BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                      radixSortBlocks(nbits, startbit, tempKeys, tempValues,
                                      keys, values, item_ct1,
                                      sMem_acc_ct1.get_pointer(),
                                      numtrue_acc_ct1);
                });
      });

      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1083:1071: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(2 * SCAN_BLOCK_SIZE * sizeof(uint)), cgh);
            sycl::local_accessor<uint, 1> sStartPointers_acc_ct1(
                sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, findBlocks) *
                                      sycl::range<3>(1, 1, SCAN_BLOCK_SIZE),
                                  sycl::range<3>(1, 1, SCAN_BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                      findRadixOffsets((sycl::uint2 *)tempKeys, counters,
                                       blockOffsets, startbit, numElements,
                                       findBlocks, item_ct1,
                                       dpct_local_acc_ct1.get_pointer(),
                                       sStartPointers_acc_ct1.get_pointer());
                });
      });

  scanArrayRecursive(countersSum, counters, 16 * reorderBlocks, 0,
                     scanBlockSums);

      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::uint2, 1> sKeys2_acc_ct1(
                sycl::range<1>(256), cgh);
            sycl::local_accessor<sycl::uint2, 1> sValues2_acc_ct1(
                sycl::range<1>(256), cgh);
            sycl::local_accessor<uint, 1> sOffsets_acc_ct1(sycl::range<1>(16),
                                                           cgh);
            sycl::local_accessor<uint, 1> sBlockOffsets_acc_ct1(
                sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, reorderBlocks) *
                                      sycl::range<3>(1, 1, SCAN_BLOCK_SIZE),
                                  sycl::range<3>(1, 1, SCAN_BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                      reorderData(startbit, (uint *)keys, (uint *)values,
                                  (sycl::uint2 *)tempKeys,
                                  (sycl::uint2 *)tempValues, blockOffsets,
                                  countersSum, counters, reorderBlocks,
                                  item_ct1, sKeys2_acc_ct1.get_pointer(),
                                  sValues2_acc_ct1.get_pointer(),
                                  sOffsets_acc_ct1.get_pointer(),
                                  sBlockOffsets_acc_ct1.get_pointer());
                });
      });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Perform scan op on input array recursively. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="outArray">   	[in,out] If non-null, array of outs. </param>
/// <param name="inArray">	  	[in,out] If non-null, array of INS. </param>
/// <param name="numElements">	Number of elements. </param>
/// <param name="level">	  	The num of levels. </param>
/// <param name="blockSums">  	[in,out] The block sum array. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void scanArrayRecursive(uint *outArray, uint *inArray, int numElements,
                        int level, uint **blockSums) {
  // Kernels handle 8 elems per thread
  unsigned int numBlocks = dpct::max(
      1, (unsigned int)ceil((float)numElements / (4.f * SCAN_BLOCK_SIZE)));
  unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
  /*
  DPCT1083:54: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

  bool fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE);

  sycl::range<3> grid(1, 1, numBlocks);
  sycl::range<3> threads(1, 1, SCAN_BLOCK_SIZE);

  // execute the scan
  if (numBlocks > 1) {
    /*
    DPCT1049:53: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::local_accessor<uint, 1> s_data_acc_ct1(
                      sycl::range<1>(512), cgh);

                  auto blockSums_level_ct2 = blockSums[level];

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         scan(outArray, inArray,
                                              blockSums_level_ct2, numElements,
                                              fullBlock, true, item_ct1,
                                              s_data_acc_ct1.get_pointer());
                                   });
            });
  } else {
    /*
    DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::local_accessor<uint, 1> s_data_acc_ct1(
                      sycl::range<1>(512), cgh);

                  auto blockSums_level_ct2 = blockSums[level];

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         scan(outArray, inArray,
                                              blockSums_level_ct2, numElements,
                                              fullBlock, false, item_ct1,
                                              s_data_acc_ct1.get_pointer());
                                   });
            });
  }
  if (numBlocks > 1) {
    scanArrayRecursive(blockSums[level], blockSums[level], numBlocks, level + 1,
                       blockSums);
    /*
    DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::local_accessor<uint, 1> uni_acc_ct1(sycl::range<1>(1),
                                                            cgh);

                  auto blockSums_level_ct1 = blockSums[level];

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         vectorAddUniform4(
                                             outArray, blockSums_level_ct1,
                                             numElements, item_ct1,
                                             uni_acc_ct1.get_pointer());
                                   });
            });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Verify the correctness of sort on cpu. </summary>
///
/// <remarks>	Kyle Spafford, 8/13/2009
/// 			Ed, 5/19/2020. </remarks>
///
/// <param name="keys">   	[in,out] If non-null, the keys. </param>
/// <param name="vals">   	[in,out] If non-null, the vals. </param>
/// <param name="size">   	The size. </param>
/// <param name="verbose">	True to verbose. </param>
/// <param name="quiet">  	True to quiet. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

bool verifySort(uint *keys, uint *vals, const size_t size, bool verbose, bool quiet) {
  bool passed = true;
  for (unsigned int i = 0; i < size - 1; i++) {
    if (keys[i] > keys[i + 1]) {
      passed = false;
      if(verbose && !quiet)  {
          cout << "Failure: at idx: " << i << endl;
          cout << "Key: " << keys[i] << " Val: " << vals[i] << endl;
          cout << "Idx: " << i + 1 << " Key: " << keys[i + 1]
              << " Val: " << vals[i + 1] << endl;
      }
    }
  }
  if (!quiet) {
      cout << "Test ";
      if (passed) {
          cout << "Passed" << endl;
      } else {
          cout << "Failed" << endl;
      }
  }
  return passed;
}
