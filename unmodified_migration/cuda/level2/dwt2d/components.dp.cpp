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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "cudacommon.h"
#include "components.h"
#include "common.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines cuda threads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define THREADS 256

/* Store 3 RGB float components */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores the components in rgb format. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_r">	[in,out] If non-null, the r. </param>
/// <param name="d_g">	[in,out] If non-null, the g. </param>
/// <param name="d_b">	[in,out] If non-null, the b. </param>
/// <param name="r">  	A float to process. </param>
/// <param name="g">  	A float to process. </param>
/// <param name="b">  	A float to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void storeComponents(float *d_r, float *d_g, float *d_b, float r, float g, float b, int pos)
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}

/* Store 3 RGB intege components */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores the components in rgb. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_r">	[in,out] If non-null, the r. </param>
/// <param name="d_g">	[in,out] If non-null, the g. </param>
/// <param name="d_b">	[in,out] If non-null, the b. </param>
/// <param name="r">  	An int to process. </param>
/// <param name="g">  	An int to process. </param>
/// <param name="b">  	An int to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void storeComponents(int *d_r, int *d_g, int *d_b, int r, int g, int b, int pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
} 

/* Store float component */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores a component. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_c">	[in,out] If non-null, the c. </param>
/// <param name="c">  	A float to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void storeComponent(float *d_c, float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}

/* Store integer component */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores a component. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_c">	[in,out] If non-null, the c. </param>
/// <param name="c">  	An int to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void storeComponent(int *d_c, int c, int pos)
{
    d_c[pos] = c - 128;
}

/* Copy img src data into three separated component buffers */

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
void c_CopySrcToComponents(T *d_r, T *d_g, T *d_b, 
                                  unsigned char * d_src, 
                                  int pixels, const sycl::nd_item<3> &item_ct1,
                                  unsigned char *sData)
{
    int x = item_ct1.get_local_id(2);
    int gX = item_ct1.get_local_range(2) * item_ct1.get_group(2);

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS*3 ) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[((gX*3)>>2) + x];
    }
    /*
    DPCT1065:34: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    T r, g, b;

    int offset = x*3;
    r = (T)(sData[offset]);
    g = (T)(sData[offset+1]);
    b = (T)(sData[offset+2]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
    }
}

/* Copy img src data into three separated component buffers */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies the source to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">   	[in,out] If non-null, the c. </param>
/// <param name="d_src"> 	[in,out] If non-null, source for the. </param>
/// <param name="pixels">	The pixels. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void c_CopySrcToComponent(T *d_c, unsigned char * d_src, int pixels,
                          const sycl::nd_item<3> &item_ct1, unsigned char *sData)
{
    int x = item_ct1.get_local_id(2);
    int gX = item_ct1.get_local_range(2) * item_ct1.get_group(2);

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[(gX>>2) + x];
    }
    /*
    DPCT1065:35: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    T c;

    c = (T)(sData[x]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponent(d_c, c, globalOutputPosition);
    }
}


/* Separate compoents of 8bit RGB source image */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void rgbToComponents(T *d_r, T *d_g, T *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op)
{
    bool uvm = op.getOptionBool("uvm");
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(width*height, THREADS) * THREADS * 3; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    if (uvm) {
        // do nothing
    } else {
        /*
        DPCT1064:334: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_src = (unsigned char *)sycl::malloc_device(
                                 alignedSize, dpct::get_default_queue())));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue().memset(d_src, 0, alignedSize).wait()));
    }
    /* timing events */
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    float elapsed;

    /* Copy data to device */
    /*
    DPCT1012:326: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    if (uvm) {
        d_src = src;
    } else {
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue().memcpy(d_src, src, pixels * 3).wait()));
    }

    // TODO time needs to be change
    /*
    DPCT1012:327: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;

    /* Kernel */
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> grid(1, 1, alignedSize / (THREADS * 3));
    assert(alignedSize%(THREADS*3) == 0);

    /*
    DPCT1012:328: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    /*
    DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
      *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:1072: 'THREADS*3' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<unsigned char, 1> sData_acc_ct1(
                sycl::range<1>(768 /*THREADS*3*/), cgh);

            cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                   c_CopySrcToComponents(
                                       d_r, d_g, d_b, d_src, pixels, item_ct1,
                                       sData_acc_ct1.get_pointer());
                             });
      });
    /*
    DPCT1012:329: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    /* Free Memory */
    if (uvm) {
        // do nothing
    } else {
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_src, dpct::get_default_queue())));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void rgbToComponents<float>(float *d_r, float *d_g, float *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void rgbToComponents<int>(int *d_r, int *d_g, int *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op);


/* Copy a 8bit source image data into a color compoment of type T */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void bwToComponent(T *d_c, unsigned char * src, int width, int height, float &transferTime, float &kernelTime)
{
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(pixels, THREADS) * THREADS; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    /*
    DPCT1064:335: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(d_src = (unsigned char *)sycl::malloc_device(
                             alignedSize, dpct::get_default_queue())));
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue().memset(d_src, 0, alignedSize).wait()));

    /* timing events */
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    float elapsed;

    /* Copy data to device */
    /*
    DPCT1012:330: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    dpct::get_default_queue().memcpy(d_src, src, pixels).wait();
    /*
    DPCT1012:331: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;

    /* Kernel */
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> grid(1, 1, alignedSize / (THREADS));
    assert(alignedSize%(THREADS) == 0);

    /*
    DPCT1012:332: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    /*
    DPCT1049:37: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
      *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:1073: 'THREADS' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<unsigned char, 1> sData_acc_ct1(
                sycl::range<1>(256 /*THREADS*/), cgh);

            cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                   c_CopySrcToComponent(
                                       d_c, d_src, pixels, item_ct1,
                                       sData_acc_ct1.get_pointer());
                             });
      });
    /*
    DPCT1012:333: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    /* Free Memory */
    sycl::free(d_src, dpct::get_default_queue());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void bwToComponent<float>(float *d_c, unsigned char *src, int width, int height, float &transferTime, float &kernelTime);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void bwToComponent<int>(int *d_c, unsigned char *src, int width, int height, float &transferTime, float &kernelTime);
