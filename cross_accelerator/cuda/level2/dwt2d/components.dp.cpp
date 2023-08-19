/*
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\dwt2d\dwt_cuda\components.cu
//
// summary:	Sort class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <assert.h>
#include <chrono>
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cudacommon.h"
#include "components.h"
#include "common.h"
#include "dwt.h"

#define THREADS 64

#ifdef _FPGA
#define COMP_ATTRIBUTE                           \
    [[sycl::reqd_work_group_size(1, 1, THREADS), \
      intel::max_work_group_size(1, 1, THREADS)]]
#else
#define COMP_ATTRIBUTE
#endif

void
storeComponents(sycl::device_ptr<float> d_r,
                sycl::device_ptr<float> d_g,
                sycl::device_ptr<float> d_b,
                float                   r,
                float                   g,
                float                   b,
                int                     pos)
{
    d_r[pos] = (r / 255.0f) - 0.5f;
    d_g[pos] = (g / 255.0f) - 0.5f;
    d_b[pos] = (b / 255.0f) - 0.5f;
}

void
storeComponents(sycl::device_ptr<int> d_r,
                sycl::device_ptr<int> d_g,
                sycl::device_ptr<int> d_b,
                int                   r,
                int                   g,
                int                   b,
                int                   pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
}

void
storeComponent(sycl::device_ptr<float> d_c, float c, int pos)
{
    d_c[pos] = (c / 255.0f) - 0.5f;
}

void
storeComponent(sycl::device_ptr<int> d_c, int c, int pos)
{
    d_c[pos] = c - 128;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies the source to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">   	[in,out] If non-null, the r. </param>
/// <param name="d_g">   	[in,out] If non-null, the g. </param>
/// <param name="d_b">   	[in,out] If non-null, the b. </param>
/// <param name="d_src"> 	[in,out] If non-null, source for the. </param>
/// <param name="pixels">	The pixels. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void
c_CopySrcToComponents(T                *d_r,
                      T                *d_g,
                      T                *d_b,
                      unsigned char    *d_src,
                      int               pixels,
                      sycl::nd_item<3>  item_ct1,
                      unsigned char    *sData)
{
    int x  = item_ct1.get_local_id(2);
    int gX = item_ct1.get_local_range(2) * item_ct1.get_group(2);

    /* Copy data to shared mem by 4bytes
       other checks are not necessary, since
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS*3 ) {
        sycl::device_ptr<float> s = (float *)d_src;
        sycl::local_ptr<float>  d = (float *)sData;
        d[x] = s[((gX*3)>>2) + x];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int offset = x * 3;
    T   r      = (T)(sData[offset]);
    T   g      = (T)(sData[offset + 1]);
    T   b      = (T)(sData[offset + 2]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels)
        storeComponents(sycl::device_ptr<T>(d_r),
                        sycl::device_ptr<T>(d_g),
                        sycl::device_ptr<T>(d_b),
                        r,
                        g,
                        b,
                        globalOutputPosition);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies the source to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">   	[in,out] If non-null, the c. </param>
/// <param name="d_src"> 	[in,out] If non-null, source for the. </param>
/// <param name="pixels">	The pixels. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void
c_CopySrcToComponent(sycl::device_ptr<T>             d_c,
                     sycl::device_ptr<unsigned char> d_src,
                     int                             pixels,
                     sycl::nd_item<3>                item_ct1,
                     sycl::local_ptr<unsigned char>  sData)
{
    int x  = item_ct1.get_local_id(2);
    int gX = item_ct1.get_local_range(2) * item_ct1.get_group(2);

    /* Copy data to shared mem by 4bytes
       other checks are not necessary, since
       d_src buffer is aligned to sharedDataSize */
    if ((x * 4) < THREADS)
        sData[x] = d_src[(gX >> 2) + x];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    T c = (T)(sData[x]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels)
        storeComponent(d_c, c, globalOutputPosition);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

class RgbToComponentKernelID;

template<typename T>
void
rgbToComponents(T             *d_r,
                T             *d_g,
                T             *d_b,
                unsigned char *src,
                int            width,
                int            height,
                float         &transferTime,
                float         &kernelTime,
                OptionParser  &op,
                sycl::queue   &queue)
{
    // Using this prevents usage of kernel -> Reduce FPGA area
    if constexpr (g_components == 3)
    {
        int pixels      = width * height;
        int alignedSize = DIVANDRND(width * height, THREADS) * THREADS
                          * 3; // aligned to thread block size -- THREADS

        unsigned char *d_src
            = (unsigned char *)sycl::malloc_device(alignedSize, queue);
        if (d_src == nullptr)
        {
            std::cerr << "Error allocating memory on device." << std::endl;
            std::terminate();
        }
        queue.memset(d_src, 0, alignedSize).wait();

        /* Copy data to device */
        sycl::event t_event = queue.memcpy(d_src, src, pixels * 3);
        t_event.wait();
        float elapsed
            = t_event.get_profiling_info<sycl::info::event_profiling::command_end>()
            - t_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        transferTime += elapsed * 1.e-9;

        /* Kernel */
        sycl::range<3> threads(1, 1, THREADS);
        sycl::range<3> grid(1, 1, alignedSize / (THREADS * 3));
        assert(alignedSize % (THREADS * 3) == 0);

        sycl::event k_event = queue
            .submit([&](sycl::handler &cgh) {
                sycl::accessor<unsigned char,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sData_acc_ct1(sycl::range<1>(THREADS * 3), cgh);

                cgh.parallel_for<RgbToComponentKernelID>(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) COMP_ATTRIBUTE {
                        c_CopySrcToComponents(d_r,
                                              d_g,
                                              d_b,
                                              d_src,
                                              pixels,
                                              item_ct1,
                                              sData_acc_ct1.get_pointer());
                    });
        });
        k_event.wait();
        elapsed
            = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
            - k_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsed * 1.e-9;

        /* Free Memory */
        sycl::free(d_src, queue);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_FLOAT
template void rgbToComponents<float>(float         *d_r,
                                     float         *d_g,
                                     float         *d_b,
                                     unsigned char *src,
                                     int            width,
                                     int            height,
                                     float         &transferTime,
                                     float         &kernelTime,
                                     OptionParser  &op,
                                     sycl::queue   &queue);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_INT
template void rgbToComponents<int>(int           *d_r,
                                   int           *d_g,
                                   int           *d_b,
                                   unsigned char *src,
                                   int            width,
                                   int            height,
                                   float         &transferTime,
                                   float         &kernelTime,
                                   OptionParser  &op,
                                   sycl::queue   &queue);
#endif

/* Copy a 8bit source image data into a color compoment of type T */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

class BwToComponentKernelID;

template<typename T>
void
bwToComponent(T             *d_c,
              unsigned char *src,
              int            width,
              int            height,
              float         &transferTime,
              float         &kernelTime,
              sycl::queue   &queue)
{
    // Using this prevents usage of kernel -> Reduce FPGA area
    if constexpr (g_components == 1)
    {
        int pixels      = width * height;
        int alignedSize = DIVANDRND(pixels, THREADS)
                          * THREADS; // aligned to thread block size -- THREADS

        /* Alloc d_src buffer */
        unsigned char *d_src
            = (unsigned char *)sycl::malloc_device(alignedSize, queue);
        if (d_src == nullptr)
        {
            std::cerr << "Error allocating memory on device." << std::endl;
            std::terminate();
        }
        queue.memset(d_src, 0, alignedSize).wait();

        /* Copy data to device */
        sycl::event t_event = queue.memcpy(d_src, src, pixels);
        t_event.wait();
        float elapsed
            = t_event.get_profiling_info<sycl::info::event_profiling::command_end>()
            - t_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        transferTime += elapsed * 1.e-9;

        /* Kernel */
        sycl::range<3> threads(1, 1, THREADS);
        sycl::range<3> grid(1, 1, alignedSize / (THREADS));
        assert(alignedSize % (THREADS) == 0);

        sycl::event k_event = queue
            .submit([&](sycl::handler &cgh) COMP_ATTRIBUTE {
                sycl::accessor<unsigned char,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sData_acc_ct1(sycl::range<1>(THREADS), cgh);

                cgh.parallel_for<BwToComponentKernelID>(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        c_CopySrcToComponent(d_c,
                                             d_src,
                                             pixels,
                                             item_ct1,
                                             sData_acc_ct1.get_pointer());
                    });
            });
        k_event.wait();
        elapsed
            = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
            - k_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsed * 1.e-9;

        /* Free Memory */
        sycl::free(d_src, queue);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_FLOAT
template void bwToComponent<float>(float         *d_c,
                                   unsigned char *src,
                                   int            width,
                                   int            height,
                                   float         &transferTime,
                                   float         &kernelTime,
                                   sycl::queue   &queue);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the.
/// </param> <param name="width">	   	The width. </param> <param
/// name="height">	   	The height. </param> <param name="transferTime">
/// [in,out] The transfer time. </param> <param name="kernelTime">
/// [in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_INT
template void bwToComponent<int>(int           *d_c,
                                 unsigned char *src,
                                 int            width,
                                 int            height,
                                 float         &transferTime,
                                 float         &kernelTime,
                                 sycl::queue   &queue);
#endif
