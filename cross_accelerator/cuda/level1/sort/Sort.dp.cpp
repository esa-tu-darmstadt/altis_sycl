////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level1\sort\Sort.cu
//
// summary:	Sort class
//
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include "Sort.h"
#include "sort_kernel.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>

#ifdef _FPGA
#define SORT_ATTRIBUTE                                        \
    [[sycl::reqd_work_group_size(1, 1, SORT_BLOCK_SIZE), \
      intel::max_work_group_size(1, 1, SORT_BLOCK_SIZE)]]
#define SCAN_ATTRIBUTE                                        \
    [[sycl::reqd_work_group_size(1, 1, SCAN_BLOCK_SIZE), \
      intel::max_work_group_size(1, 1, SCAN_BLOCK_SIZE)]]
#else
#define SORT_ATTRIBUTE
#define SCAN_ATTRIBUTE
#endif

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
void
addBenchmarkSpecOptions(OptionParser &op)
{
}

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
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    cout << "Running Sort" << endl;
    srand(SEED);
    bool       quiet               = op.getOptionBool("quiet");
    const bool uvm                 = op.getOptionBool("uvm");
    const bool uvm_prefetch        = op.getOptionBool("uvm-prefetch");
    const bool uvm_advise          = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    // Determine size of the array to sort
    int       size;
    long long bytes;
    string    filePath = op.getOptionString("inputFile");
    ifstream  inputFile(filePath.c_str());
    if (filePath == "")
    {
        if (!quiet)
            printf("Using problem size %d\n", (int)op.getOptionInt("size"));
        int probSizes[5] = { 32, 64, 256, 512, 1024 };
        size             = probSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
    }
    else
    {
        inputFile >> size;
    }
    bytes = size * sizeof(uint);
    if (!quiet)
        printf("Size: %d items, Bytes: %lld\n", size, bytes);

    // If input file given, populate array
    uint *sourceInput = (uint *)malloc(bytes);
    if (filePath != "")
        for (int i = 0; i < size; i++)
            inputFile >> sourceInput[i];

    // create input data on CPU
    uint *hKeys = NULL;
    uint *hVals = NULL;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	allocate using UVM API. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        hKeys = (uint *)sycl::malloc_shared(bytes, queue);
        hVals = (uint *)sycl::malloc_shared(bytes, queue);
    }
    else
    {
        hKeys = (uint *)malloc(bytes);
        hVals = (uint *)malloc(bytes);
    }
    if (hKeys == nullptr || hVals == nullptr)
    {
        std::cerr << "Error allocating memory." << std::endl;
        return;
    }

    // Allocate space for block sums in the scan kernel.
    uint numLevelsAllocated = 0;
    uint maxNumScanElements = size;
    uint numScanElts        = maxNumScanElements;
    uint level              = 0;

    do
    {
        uint numBlocks
            = max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
            level++;
        numScanElts = numBlocks;
    } while (numScanElts > 1);

    uint **scanBlockSums = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        scanBlockSums = sycl::malloc_shared<uint *>((level + 1),
                                                    queue);
    }
    else
    {
        scanBlockSums = (uint **)malloc((level + 1) * sizeof(uint *));
        assert(scanBlockSums);
    }
    
    numLevelsAllocated = level + 1;
    numScanElts        = maxNumScanElements;
    level              = 0;

    do
    {
        uint numBlocks
            = max(1, (int)ceil((float)numScanElts / (4 * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
        {
            // Malloc device mem for block sums
            if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
                scanBlockSums[level] = sycl::malloc_shared<uint>(
                    numBlocks, queue);
            else
                scanBlockSums[level] = sycl::malloc_device<uint>(
                    numBlocks, queue);
            level++;
        }
        numScanElts = numBlocks;
    } while (numScanElts > 1);

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        scanBlockSums[level]
            = sycl::malloc_shared<uint>(1, queue);
    else
        scanBlockSums[level]
            = sycl::malloc_device<uint>(1, queue);

    // Allocate device mem for sorting kernels
    uint *dKeys, *dVals, *dTempKeys, *dTempVals;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        dKeys = hKeys;
        dVals = hVals;
        dTempKeys = (uint *)sycl::malloc_shared(
                             bytes, queue);
        dTempVals = (uint *)sycl::malloc_shared(
                             bytes, queue);
    }
    else
    {
        dKeys = (uint *)sycl::malloc_device(
                             bytes, queue);
        dVals = (uint *)sycl::malloc_device(
                             bytes, queue);
        dTempKeys = (uint *)sycl::malloc_device(
                             bytes, queue);
        dTempVals = (uint *)sycl::malloc_device(
                             bytes, queue);
    }

    // Each thread in the sort kernel handles 4 elements
    size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);

    uint *dCounters, *dCounterSums, *dBlockOffsets;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        dCounters = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                                   queue);
        dCounterSums
                         = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                                     queue);
        dBlockOffsets
                         = sycl::malloc_shared<uint>(WARP_SIZE * numSortGroups,
                                                     queue);
    }
    else
    {
        dCounters = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                                   queue);
        dCounterSums
                         = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                                     queue);
        dBlockOffsets
                         = sycl::malloc_device<uint>(WARP_SIZE * numSortGroups,
                                                     queue);
    }

    int         iterations = op.getOptionInt("passes");
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    for (int it = 0; it < iterations; it++)
    {
        if (!quiet)
            printf("Pass %d: ", it);

        /// <summary>	Initialize host memory to some pattern. </summary>
        for (uint i = 0; i < size; i++)
        {
            hKeys[i] = i % 1024;
            if (filePath == "")
                hVals[i] = rand() % 1024;
            else
                hVals[i] = sourceInput[i];
        }

        // Copy inputs to GPU
        double transferTime = 0.;
        start_ct1           = std::chrono::steady_clock::now();
        if (uvm)
        {
            // do nothing
        }
        else if (uvm_advise)
        {
            queue.mem_advise(dKeys, bytes, 0);
            queue.mem_advise(dVals, bytes, 0);
        }
        else if (uvm_prefetch)
        {
            queue.prefetch(dKeys, bytes);
            queue.prefetch(dVals, bytes);
        }
        else if (uvm_prefetch_advise)
        {
            queue.mem_advise(dKeys, bytes, 0);
            queue.mem_advise(dVals, bytes, 0);
            queue.prefetch(dKeys, bytes);
            queue.prefetch(dVals, bytes);
        }
        else
        {
            queue.memcpy(dKeys, hKeys, bytes).wait();
            queue.memcpy(dVals, hVals, bytes).wait();
        }
        stop_ct1 = std::chrono::steady_clock::now();
        float elapsedTime;
        elapsedTime
            = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
        transferTime += elapsedTime * 1.e-3; // convert to seconds
 
        start_ct1 = std::chrono::steady_clock::now();
        // Perform Radix Sort (4 bits at a time)
        for (int i = 0; i < SORT_BITS; i += 4)
            radixSortStep(4,
                          i,
                          (sycl::uint4 *)dKeys,
                          (sycl::uint4 *)dVals,
                          (sycl::uint4 *)dTempKeys,
                          (sycl::uint4 *)dTempVals,
                          dCounters,
                          dCounterSums,
                          dBlockOffsets,
                          scanBlockSums,
                          size,
                          queue);
        stop_ct1 = std::chrono::steady_clock::now();
        elapsedTime = std::chrono::duration<float, std::milli>(
                                           stop_ct1 - start_ct1)
                                           .count();
        double kernelTime = elapsedTime * 1.e-3;

        // Readback data from device
        start_ct1 = std::chrono::steady_clock::now();

        // prefetch or demand paging
        if (uvm)
        {
            // do nothing
        }
        else if (uvm_advise)
        {
            queue.mem_advise(dKeys, bytes, 0);
            queue.mem_advise(dVals, bytes, 0);
        }
        else if (uvm_prefetch)
        {
            queue.prefetch(dKeys, bytes);
            queue.prefetch(dVals, bytes);
        }
        else if (uvm_prefetch_advise)
        {
            queue.mem_advise(dKeys, bytes, 0);
            queue.mem_advise(dVals, bytes, 0);
            queue.prefetch(dKeys, bytes);
            queue.prefetch(dVals, bytes);
        }
        else
        {
            queue.memcpy(hKeys, dKeys, bytes).wait();
            queue.memcpy(hVals, dVals, bytes).wait();
        }

        stop_ct1 = std::chrono::steady_clock::now();
        elapsedTime
            = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
        transferTime += elapsedTime * 1.e-3;

        // Test to make sure data was sorted properly, if not, return
        if (!verifySort(hKeys,
                        hVals,
                        size,
                        op.getOptionBool("verbose"),
                        op.getOptionBool("quiet")))
            return;

        char atts[1024];
        sprintf(atts, "%ditems", size);
        double gb = (bytes * 2.) / (1000. * 1000. * 1000.);
        resultDB.AddResult("Sort-KernelTime", atts, "sec", kernelTime);
        resultDB.AddResult("Sort-TransferTime", atts, "sec", transferTime);
        resultDB.AddResult(
            "Sort-TotalTime", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult("Sort-Rate", atts, "GB/s", gb / kernelTime);
        resultDB.AddResult(
            "Sort-Rate_PCIe", atts, "GB/s", gb / (kernelTime + transferTime));
        resultDB.AddResult(
            "Sort-Rate_Parity", atts, "N", transferTime / kernelTime);
        resultDB.AddOverall("Rate", "GB/s", gb / kernelTime);
    }

    // Clean up
    for (int i = 0; i < numLevelsAllocated; i++)
        sycl::free(scanBlockSums[i], queue);
    sycl::free(dKeys, queue);
    sycl::free(dVals, queue);
    sycl::free(dTempKeys, queue);
    sycl::free(dTempVals, queue);
    sycl::free(dCounters, queue);
    sycl::free(dCounterSums, queue);
    sycl::free(dBlockOffsets, queue);

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        sycl::free(scanBlockSums, queue);
    }
    else
    {
        free(scanBlockSums);
        free(hKeys);
        free(hVals);
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

class radix_sort_blocks_kernel_id;
class find_radix_offsets_kernel_id;
class reorder_data_kernel_id;

void
radixSortStep(uint         nbits,
              uint         startbit,
              sycl::uint4 *keys,
              sycl::uint4 *values,
              sycl::uint4 *tempKeys,
              sycl::uint4 *tempValues,
              uint        *counters,
              uint        *countersSum,
              uint        *blockOffsets,
              uint       **scanBlockSums,
              uint         numElements,
              sycl::queue &queue)
{
    // Threads handle either 4 or two elements each
    const size_t radixGlobalWorkSize   = numElements / 4;
    const size_t findGlobalWorkSize    = numElements / 2;
    const size_t reorderGlobalWorkSize = numElements / 2;

    // Radix kernel uses block size of 128, others use 256 (same as scan)
    const size_t radixBlocks   = radixGlobalWorkSize / SORT_BLOCK_SIZE;
    const size_t findBlocks    = findGlobalWorkSize / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<uint,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sMem_acc_ct1(sycl::range<1>(512), cgh);
        sycl::accessor<uint,
                       0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            numtrue_acc_ct1(cgh);

        cgh.parallel_for<radix_sort_blocks_kernel_id>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, radixBlocks)
                                  * sycl::range<3>(1, 1, SORT_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SORT_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) SORT_ATTRIBUTE {
                radixSortBlocks(nbits,
                                startbit,
                                tempKeys,
                                tempValues,
                                keys,
                                values,
                                item_ct1,
                                sMem_acc_ct1.get_pointer(),
                                numtrue_acc_ct1.get_pointer());
            });
    }).wait();

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(
                sycl::range<1>(2 * SCAN_BLOCK_SIZE * sizeof(uint)), cgh);
        sycl::accessor<uint,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sStartPointers_acc_ct1(sycl::range<1>(16), cgh);

        cgh.parallel_for<find_radix_offsets_kernel_id>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, findBlocks)
                                  * sycl::range<3>(1, 1, SCAN_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SCAN_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) SCAN_ATTRIBUTE {
                findRadixOffsets((sycl::uint2 *)tempKeys,
                                 counters,
                                 blockOffsets,
                                 startbit,
                                 findBlocks,
                                 item_ct1,
                                 dpct_local_acc_ct1.get_pointer(),
                                 sStartPointers_acc_ct1.get_pointer());
            });
    }).wait();

    scanArrayRecursive(
        countersSum, counters, 16 * reorderBlocks, 0, scanBlockSums, queue);

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<sycl::uint2,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sKeys2_acc_ct1(sycl::range<1>(256), cgh);
        sycl::accessor<sycl::uint2,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sValues2_acc_ct1(sycl::range<1>(256), cgh);
        sycl::accessor<uint,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sOffsets_acc_ct1(sycl::range<1>(16), cgh);
        sycl::accessor<uint,
                       1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sBlockOffsets_acc_ct1(sycl::range<1>(16), cgh);

        cgh.parallel_for<reorder_data_kernel_id>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, reorderBlocks)
                                  * sycl::range<3>(1, 1, SCAN_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SCAN_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) SCAN_ATTRIBUTE {
                reorderData(startbit,
                            (uint *)keys,
                            (uint *)values,
                            (sycl::uint2 *)tempKeys,
                            (sycl::uint2 *)tempValues,
                            blockOffsets,
                            countersSum,
                            reorderBlocks,
                            item_ct1,
                            sKeys2_acc_ct1.get_pointer(),
                            sValues2_acc_ct1.get_pointer(),
                            sOffsets_acc_ct1.get_pointer(),
                            sBlockOffsets_acc_ct1.get_pointer());
            });
    }).wait();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Perform scan op on input array recursively. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="outArray">   	[in,out] If non-null, array of outs. </param>
/// <param name="inArray">	  	[in,out] If non-null, array of INS.
/// </param> <param name="numElements">	Number of elements. </param> <param
/// name="level">	  	The num of levels. </param> <param
/// name="blockSums">  	[in,out] The block sum array. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

class scan_kernel_ID;
class vec_add_kernel_ID;

void
scanArrayRecursive(
    uint *outArray, uint *inArray, int numElements, int level, uint **blockSums,
              sycl::queue &queue)
{
    // Kernels handle 8 elems per thread
    unsigned int numBlocks = std::max<unsigned int>(
        1, (unsigned int)ceil((float)numElements / (4.f * SCAN_BLOCK_SIZE)));
    unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
    /*
    DPCT1083:2435: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

    bool fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE);

    sycl::range<3> grid(1, 1, numBlocks);
    sycl::range<3> threads(1, 1, SCAN_BLOCK_SIZE);

    // execute the scan
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<uint,
                        1,
                        sycl::access_mode::read_write,
                        sycl::access::target::local>
            s_data_acc_ct1(sycl::range<1>(512), cgh);

        auto blockSums_level_ct2 = blockSums[level];

        cgh.parallel_for<scan_kernel_ID>(sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) SCAN_ATTRIBUTE {
                                scan(outArray,
                                    inArray,
                                    blockSums_level_ct2,
                                    numElements,
                                    fullBlock,
                                    numBlocks > 1,
                                    item_ct1,
                                    s_data_acc_ct1.get_pointer());
                            });
    }).wait();
    if (numBlocks > 1)
    {
        scanArrayRecursive(blockSums[level],
                           blockSums[level],
                           numBlocks,
                           level + 1,
                           blockSums,
                           queue);
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<uint,
                           1,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                uni_acc_ct1(sycl::range<1>(1), cgh);

            auto blockSums_level_ct1 = blockSums[level];

            cgh.parallel_for<vec_add_kernel_ID>(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) SCAN_ATTRIBUTE {
                                 vectorAddUniform4(outArray,
                                                   blockSums_level_ct1,
                                                   numElements,
                                                   item_ct1,
                                                   uni_acc_ct1.get_pointer());
                             });
        }).wait();
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

bool
verifySort(uint *keys, uint *vals, const size_t size, bool verbose, bool quiet)
{
    bool passed = true;
    for (unsigned int i = 0; i < size - 1; i++)
    {
        if (keys[i] > keys[i + 1])
        {
            passed = false;
            if (verbose && !quiet)
            {
                cout << "Failure: at idx: " << i << endl;
                cout << "Key: " << keys[i] << " Val: " << vals[i] << endl;
                cout << "Idx: " << i + 1 << " Key: " << keys[i + 1]
                     << " Val: " << vals[i + 1] << endl;
            }
        }
    }
    if (!quiet)
    {
        cout << "Test ";
        if (passed)
            cout << "Passed" << endl;
        else
            cout << "Failed" << endl;
    }
    return passed;
}
