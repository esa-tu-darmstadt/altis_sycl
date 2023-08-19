////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\common\support.h
//
// summary:	Declares the support class
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SUPPORT_H
#define SUPPORT_H

#include <CL/sycl.hpp>
#include "cudacommon.h"
#include <iostream>
/// <summary>	The standard cin. </summary>
using std::cin;
/// <summary>	The standard cout. </summary>
using std::cout;

// ****************************************************************************
// Method:  findAvailBytes
//
// Purpose: returns maximum number of bytes *allocatable* (likely less than
//          device memory size) on the device.
//
// Arguments: None.
//
// Programmer:  Collin McCurdy
// Creation:    June 8, 2010
//
// ****************************************************************************

// inline unsigned long
// findAvailBytes(void)
// {
//         dpct::device_ext &dev_ct1 = dpct::get_current_device();
//         sycl::queue &q_ct1 = dev_ct1.default_queue();
//     int device;
//     device = dpct::dev_mgr::instance().current_device_id();
//     CHECK_CUDA_ERROR();
//     dpct::device_info deviceProp;
//     dpct::dev_mgr::instance().get_device(device).get_device_info(deviceProp);
//     CHECK_CUDA_ERROR();
//     unsigned long total_bytes = deviceProp.get_global_mem_size();
//     unsigned long avail_bytes = total_bytes;
//     void* work;

//     while (1) {
//         work =
//             (void *)sycl::malloc_device(avail_bytes, dpct::get_default_queue());
//         /*
//         DPCT1010:547: SYCL uses exceptions to report errors and does not use the
//         error codes. The call was replaced with 0. You need to rewrite this
//         code.
//         */
//         if (0 == 0) {
//             break;
//         }
//         avail_bytes -= (1024*1024);
//     }
//     sycl::free(work, dpct::get_default_queue());
//     CHECK_CUDA_ERROR();

//     return avail_bytes;
// }



#endif
