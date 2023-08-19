////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level0\busspeedreadback\BusSpeedReadback.cu
//
// summary:	Bus speed readback class
// 
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
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
// Modifications: Bodun Hu (bodunhu@utexas.edu)
//   added safe call wrapper to CUDA call
// 
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
//   sizes across the bus to the host from the device and calculates the
//   bandwidth for each chunk size.
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

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) try {
    cout << "Running BusSpeedReadback" << endl;

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
    float *hostMem1 = NULL;
    float *hostMem2 = NULL;
    if (uvm_prefetch) {
        hostMem1 = (float *)sycl::malloc_shared(sizeof(float) * numMaxFloats,
                                                dpct::get_default_queue());
        /*
        DPCT1010:2029: SYCL uses exceptions to report errors and does not use
        the error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        while (0 != 0) {
            // free the first buffer if only the second failed
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
            hostMem1 = (float *)sycl::malloc_shared(
                sizeof(float) * numMaxFloats, dpct::get_default_queue());
        }
        hostMem2 = hostMem1;
    } else {
        if (pinned) {
            hostMem1 = (float *)sycl::malloc_host(sizeof(float) * numMaxFloats,
                                                  dpct::get_default_queue());
            /*
            DPCT1010:2030: SYCL uses exceptions to report errors and does not
            use the error codes. The call was replaced with 0. You need to
            rewrite this code.
            */
            int err1 = 0;
            hostMem2 = (float *)sycl::malloc_host(sizeof(float) * numMaxFloats,
                                                  dpct::get_default_queue());
            /*
            DPCT1010:2031: SYCL uses exceptions to report errors and does not
            use the error codes. The call was replaced with 0. You need to
            rewrite this code.
            */
            int err2 = 0;
            while (err1 != 0 || err2 != 0) {
                // free the first buffer if only the second failed
                if (err1 == 0) {
                    /*
                    DPCT1003:2032: Migrated API does not return error code. (*,
                    0) is inserted. You may need to rewrite this code.
                    */
                    /*
                    DPCT1003:2051: Migrated API does not return error code. (*,
                    0) is inserted. You may need to rewrite this code.
                    */
                    CUDA_SAFE_CALL((
                        sycl::free((void *)hostMem1, dpct::get_default_queue()),
                        0));
                }

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
                hostMem1 = (float *)sycl::malloc_host(
                    sizeof(float) * numMaxFloats, dpct::get_default_queue());
                /*
                DPCT1010:2033: SYCL uses exceptions to report errors and does
                not use the error codes. The call was replaced with 0. You need
                to rewrite this code.
                */
                err1 = 0;
                hostMem2 = (float *)sycl::malloc_host(
                    sizeof(float) * numMaxFloats, dpct::get_default_queue());
                /*
                DPCT1010:2034: SYCL uses exceptions to report errors and does
                not use the error codes. The call was replaced with 0. You need
                to rewrite this code.
                */
                err2 = 0;
            }
        } else {
            hostMem1 = new float[numMaxFloats];
            hostMem2 = new float[numMaxFloats];
        }
    }

    // fillup allocated region
    for (int i = 0; i < numMaxFloats; i++) {
        hostMem1[i] = i % 77;
    }

    float *device = NULL;
    if (uvm_prefetch) {
        device = hostMem1;
    } else {
        device = (float *)sycl::malloc_device(sizeof(float) * numMaxFloats,
                                              dpct::get_default_queue());
        /*
        DPCT1010:2035: SYCL uses exceptions to report errors and does not use
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

    int deviceID = 0;
    checkCudaErrors(deviceID = dpct::dev_mgr::instance().current_device_id());
    if (uvm_prefetch) {
        /*
        DPCT1003:2052: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::dev_mgr::instance()
                             .get_device(deviceID)
                             .default_queue()
                             .prefetch(device, numMaxFloats * sizeof(float)),
                         0));
        checkCudaErrors((dpct::get_default_queue().wait(), 0));
    } else {
        /*
        DPCT1003:2053: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_default_queue()
                 .memcpy(device, hostMem1, numMaxFloats * sizeof(float))
                 .wait(),
             0));
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }
    const unsigned int passes = op.getOptionInt("passes");

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1027:2036: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:2054: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:2037: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:2055: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);

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
            DPCT1012:2038: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:2039: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:2056: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(0);
            if (uvm_prefetch) {
                /*
                DPCT1003:2040: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                /*
                DPCT1003:2057: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::dev_mgr::instance()
                                     .get_device(cudaCpuDeviceId)
                                     .default_queue()
                                     .prefetch(device, nbytes),
                                 0));
                /*
                DPCT1003:2041: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::get_default_queue().wait(), 0));
            } else {
                /*
                DPCT1003:2058: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::get_default_queue()
                                     .memcpy(hostMem2, device, nbytes)
                                     .wait(),
                                 0));
            }
            /*
            DPCT1012:2042: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:2043: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            /*
            DPCT1012:2059: Detected kernel execution time measurement pattern
            and generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            stop_ct1 = std::chrono::steady_clock::now();
            checkCudaErrors(0);
            checkCudaErrors(0);
            float t = 0;
            /*
            DPCT1003:2060: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors((t = std::chrono::duration<float, std::milli>(
                                     stop_ct1 - start_ct1)
                                     .count(),
                             0));
            // times[sizeIndex] = t;

            // Convert to GB/sec
            if (verbose && !quiet) {
                cout << "size " << sizes[sizeIndex] << "k took " << t << " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000 * 1000)) / t;
            resultDB.AddResult("ReadbackSpeed", "---", "GB/sec", speed);
            resultDB.AddOverall("ReadbackSpeed", "GB/sec", speed);

            // Move data back to device if it's already prefetched to host
            if (uvm_prefetch) {
                /*
                DPCT1003:2044: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                /*
                DPCT1003:2061: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::dev_mgr::instance()
                                     .get_device(deviceID)
                                     .default_queue()
                                     .prefetch(device, nbytes),
                                 0));
                /*
                DPCT1003:2045: Migrated API does not return error code. (*, 0)
                is inserted. You may need to rewrite this code.
                */
                checkCudaErrors((dpct::get_default_queue().wait(), 0));
            }
        }
    }

    // Cleanup
    /*
    DPCT1003:2062: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free((void *)device, dpct::get_default_queue()), 0));
    if (!uvm_prefetch) {
        if (pinned) {
            /*
            DPCT1003:2063: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (sycl::free((void *)hostMem1, dpct::get_default_queue()), 0));
            /*
            DPCT1003:2046: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            /*
            DPCT1003:2064: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (sycl::free((void *)hostMem2, dpct::get_default_queue()), 0));
        } else {
            delete[] hostMem1;
            delete[] hostMem2;
        }
    }
    /*
    DPCT1027:2047: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:2065: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
    /*
    DPCT1027:2048: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    /*
    DPCT1027:2066: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    checkCudaErrors(0);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
