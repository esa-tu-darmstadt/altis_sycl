////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level0\busspeeddownload\BusSpeedDownload.cu
//
// summary:	Bus speed download test.
// 
// modified from: SHOC Benchmark Suite (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <chrono>

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
// 
// Modifications: Ed, 5/19/2020.
//
// ****************************************************************************

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
    op.addOption("pinned", OPT_BOOL, "0", "use pinned (pagelocked) memory");
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the OpenCL device, and calculates the bandwidth.
//
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//  
//    Bodun Hu (bodunhu@utexas.edu), Jan 3 2021
//    Added UVM prefetch.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    cout << "Running BusSpeedDownload" << endl;
    const bool verbose = op.getOptionBool("verbose");
    const bool quiet = op.getOptionBool("quiet");
    const bool pinned = op.getOptionBool("pinned");

    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");

    // Sizes are in kb
    int nSizes = 21;
    int sizes[21] = {1,     2,     4,     8,      16,     32,    64,
                    128,   256,   512,   1024,   2048,   4096,  8192,
                    16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    long long numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;

    // Create some host memory pattern
    float *hostMem = NULL;
    if (uvm_prefetch) {
        hostMem = (float *)sycl::malloc_shared(sizeof(float) * numMaxFloats,
                                               dpct::get_default_queue());
        /*
        DPCT1010:1935: SYCL uses exceptions to report errors and does not use
        the error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        while (0 != 0) {
            // drop the size and try again
            if (verbose && !quiet) {
                cout << " - dropping size allocating unified mem\n";
            }
            --nSizes;
            if (nSizes < 1) {
                cerr << "Error: Couldn't allocated any unified buffer\n";
                return;
            }
            numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;
            hostMem = (float *)sycl::malloc_shared(sizeof(float) * numMaxFloats,
                                                   dpct::get_default_queue());
        }
    } else {
        if (pinned) {
            hostMem = (float *)sycl::malloc_host(sizeof(float) * numMaxFloats,
                                                 dpct::get_default_queue());
            /*
            DPCT1010:1936: SYCL uses exceptions to report errors and does not
            use the error codes. The call was replaced with 0. You need to
            rewrite this code.
            */
            while (0 != 0) {
                // drop the size and try again
                if (verbose && !quiet) {
                    cout << " - dropping size allocating pinned mem\n";
                }
                --nSizes;
                if (nSizes < 1) {
                    cerr << "Error: Couldn't allocated any pinned buffer\n";
                    return;
                }
                numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;
                hostMem = (float *)sycl::malloc_host(
                    sizeof(float) * numMaxFloats, dpct::get_default_queue());
            }
        } else {
            hostMem = new float[numMaxFloats];
        }
    }

    // Initialize host memory
    for (int i = 0; i < numMaxFloats; i++) {
        hostMem[i] = i % 77;
    }

    float *device = NULL;
    if (uvm_prefetch) {
        device = hostMem;
    } else {
        device = (float *)sycl::malloc_device(sizeof(float) * numMaxFloats,
                                              dpct::get_default_queue());
        /*
        DPCT1010:1937: SYCL uses exceptions to report errors and does not use
        the error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        while (0 != 0) {
            // drop the size and try again
            if (verbose && !quiet) {
                cout << " - dropping size allocating device mem\n";
            }
            --nSizes;
            if (nSizes < 1) {
                cerr << "Error: Couldn't allocated any device buffer\n";
                return;
            }
            numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;
            device = (float *)sycl::malloc_device(sizeof(float) * numMaxFloats,
                                                  dpct::get_default_queue());
        }
    }

    const unsigned int passes = op.getOptionInt("passes");

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1027:1938: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:1950: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:1939: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:1951: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    int deviceID = 0;
    checkCudaErrors(deviceID = dpct::dev_mgr::instance().current_device_id());

    // Three passes, forward and backward both
    for (int pass = 0; pass < passes; pass++) {
        // store the times temporarily to estimate latency
        // float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++) {
            int sizeIndex;
            if ((pass % 2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            /*
            DPCT1012:1940: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            if (uvm_prefetch) {
                // Use default stream
                /*
                DPCT1003:1952: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::dev_mgr::instance()
                                     .get_device(deviceID)
                                     .default_queue()
                                     .prefetch(device, nbytes),
                                 0));
                checkCudaErrors((dpct::get_default_queue().wait(), 0));
            } else {
                /*
                DPCT1003:1953: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::get_default_queue()
                                     .memcpy(device, hostMem, nbytes)
                                     .wait(),
                                 0));
            }
            /*
            DPCT1012:1941: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            stop_ct1 = std::chrono::steady_clock::now();
            float t = 0;
            t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
            // times[sizeIndex] = t;

            // Convert to GB/sec
            if (verbose && !quiet) {
                cout << "size " << sizes[sizeIndex] << "k took " << t << " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000 * 1000)) / t;
            resultDB.AddResult("DownloadSpeed", "---", "GB/sec", speed);
            resultDB.AddOverall("DownloadSpeed", "GB/sec", speed);

            // Move data back to host if it's already prefetched to device
            if (uvm_prefetch) {
                /*
                DPCT1003:1942: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                /*
                DPCT1003:1954: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::dev_mgr::instance()
                                     .get_device(cudaCpuDeviceId)
                                     .default_queue()
                                     .prefetch(device, nbytes),
                                 0));
                /*
                DPCT1003:1943: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::get_default_queue().wait(), 0));
            }
        }
    }

    // Cleanup
    if (uvm_prefetch) {
        /*
        DPCT1003:1955: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (sycl::free((void *)device, dpct::get_default_queue()), 0));
    } else {
        /*
        DPCT1003:1944: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:1956: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (sycl::free((void *)device, dpct::get_default_queue()), 0));
        if (pinned) {
            /*
            DPCT1003:1945: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:1957: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (sycl::free((void *)hostMem, dpct::get_default_queue()), 0));
        } else {
            delete[] hostMem;
        }
    }
    /*
    DPCT1027:1946: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:1958: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:1947: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:1959: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
}
