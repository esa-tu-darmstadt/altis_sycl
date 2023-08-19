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
// file:	altis\src\cuda\level2\dwt2d\dwt_cuda\dwt.cu
//
// summary:	Sort class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <error.h>

#include "cudacommon.h"
#include "dwt_cuda/dwt.h"
#include "dwt.h"
#include "common.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Fdwts97 kernel wrapper. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="in">	  	[in,out] If non-null, the in. </param>
/// <param name="out">	  	[in,out] If non-null, the out. </param>
/// <param name="width">  	The width. </param>
/// <param name="height"> 	The height. </param>
/// <param name="levels"> 	The levels. </param>
/// <param name="verbose">	True to verbose. </param>
/// <param name="quiet">  	True to quiet. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline float
fdwt(float       *in,
     float       *out,
     int          width,
     int          height,
     int          levels,
     bool         verbose,
     bool         quiet,
     sycl::queue &queue)
{
    if constexpr (g_forward && dwt97)
        return dwt_cuda::fdwt97(in, out, width, height, levels, queue);
    else
        return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Fdwts53 kernel wrapper. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="in">	  	[in,out] If non-null, the in. </param>
/// <param name="out">	  	[in,out] If non-null, the out. </param>
/// <param name="width">  	The width. </param>
/// <param name="height"> 	The height. </param>
/// <param name="levels"> 	The levels. </param>
/// <param name="verbose">	True to verbose. </param>
/// <param name="quiet">  	True to quiet. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline float
fdwt(int         *in,
     int         *out,
     int          width,
     int          height,
     int          levels,
     bool         verbose,
     bool         quiet,
     sycl::queue &queue)
{
    if constexpr (g_forward && !dwt97)
        return dwt_cuda::fdwt53(
            in, out, width, height, levels, verbose, quiet, queue);
    else 
        return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Rdwts97 kernel wrapper. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="in">	 	[in,out] If non-null, the in. </param>
/// <param name="out">   	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
/// <param name="levels">	The levels. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline float
rdwt(float       *in,
     float       *out,
     int          width,
     int          height,
     int          levels,
     sycl::queue &queue)
{
    if constexpr (!g_forward && dwt97)
        return dwt_cuda::rdwt97(in, out, width, height, levels, queue);
    else
        return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Rdwts53 kernel wrapper with int. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="in">	 	[in,out] If non-null, the in. </param>
/// <param name="out">   	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
/// <param name="levels">	The levels. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline float
rdwt(int *in, int *out, int width, int height, int levels, sycl::queue &queue)
{
    if constexpr (!g_forward && !dwt97)
        return dwt_cuda::rdwt53(in, out, width, height, levels, queue);
    else
        return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	2D dwt with stages. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="in">		   	[in,out] If non-null, the in. </param>
/// <param name="out">		   	[in,out] If non-null, the out. </param>
/// <param name="backup">	   	[in,out] If non-null, the backup.
/// </param> <param name="pixWidth">	   	Width of the pix. </param>
/// <param name="pixHeight">   	Height of the pix. </param>
/// <param name="stages">	   	The stages. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
/// <param name="verbose">	   	True to verbose. </param>
/// <param name="quiet">	   	True to quiet. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
int
nStage2dDWT(T           *in,
            T           *out,
            T           *backup,
            int          pixWidth,
            int          pixHeight,
            int          stages,
            float       &transferTime,
            float       &kernelTime,
            bool         verbose,
            bool         quiet,
            sycl::queue &queue)
{
    if (verbose && !quiet)
        printf("%d stages of 2D DWT:\n", stages);

    /* create backup of input, because each test iteration overwrites it */
    const int size = pixHeight * pixWidth * sizeof(T);

    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float elapsed;

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(backup, in, size).wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed  = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
    transferTime += elapsed * 1.e-3;

    /* Measure time of individual levels. */
    if constexpr (g_forward)
        kernelTime
            += fdwt(in, out, pixWidth, pixHeight, stages, verbose, quiet, queue);
    else
        kernelTime += rdwt(in, out, pixWidth, pixHeight, stages, queue);

    return 0;
}

template int nStage2dDWT<float>(float *,
                                float *,
                                float *,
                                int,
                                int,
                                int,
                                float &,
                                float &,
                                bool,
                                bool,
                                sycl::queue &);

template int nStage2dDWT<int>(int *,
                              int *,
                              int *,
                              int,
                              int,
                              int,
                              float &,
                              float &,
                              bool,
                              bool,
                              sycl::queue &);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Samples to character. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="dst">		 	[in,out] If non-null, destination for
/// the.
/// </param>
/// <param name="src">		 	[in,out] If non-null, source for the.
/// </param> <param name="samplesNum">	The samples number. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
samplesToChar(unsigned char *dst, float *src, int samplesNum)
{
    int i;

    for (i = 0; i < samplesNum; i++)
    {
        float r = (src[i] + 0.5f) * 255;
        if (r > 255)
            r = 255;
        if (r < 0)
            r = 0;
        dst[i] = (unsigned char)r;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Samples to character. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="dst">		 	[in,out] If non-null, destination for
/// the.
/// </param>
/// <param name="src">		 	[in,out] If non-null, source for the.
/// </param> <param name="samplesNum">	The samples number. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
samplesToChar(unsigned char *dst, int *src, int samplesNum)
{
    int i;

    for (i = 0; i < samplesNum; i++)
    {
        int r = src[i] + 128;
        if (r > 255)
            r = 255;
        if (r < 0)
            r = 0;
        dst[i] = (unsigned char)r;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	* Write output linear orderd. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
int
writeLinear(T           *component_cuda,
            int          pixWidth,
            int          pixHeight,
            const char  *filename,
            const char  *suffix,
            sycl::queue &queue)
{
    int samplesNum = pixWidth * pixHeight;
    int size       = samplesNum * sizeof(T);

    T *gpu_output = (T*)malloc(size);
    if (nullptr == gpu_output)
    {
        std::cerr << "Error allocating memory on host." << std::endl;
        std::terminate();
    }
    memset(gpu_output, 0, size);

    unsigned char *result = (unsigned char *)malloc(samplesNum);
    queue.memcpy(gpu_output, component_cuda, size).wait();

    /* T to char */
    samplesToChar(result, gpu_output, samplesNum);

    /* Write component */
    char outfile[strlen(filename) + strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile + strlen(filename), suffix);
    int i = open(outfile, O_CREAT | O_WRONLY, 0644);
    if (i == -1)
    {
        error(0, errno, "cannot access %s", outfile);
        return -1;
    }
    std::cout << "Writing " << samplesNum << " bytes to file..." << std::endl;
    ssize_t x;
    x = write(i, result, samplesNum);
    close(i);

    /* Clean up */
    free(gpu_output);
    free(result);

    if (x == 0)
        return 1;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a linear. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template int writeLinear<float>(float       *component_cuda,
                                int          pixWidth,
                                int          pixHeight,
                                const char  *filename,
                                const char  *suffix,
                                sycl::queue &queue);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a linear. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template int writeLinear<int>(int         *component_cuda,
                              int          pixWidth,
                              int          pixHeight,
                              const char  *filename,
                              const char  *suffix,
                              sycl::queue &queue);

/* Write output visual ordered */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a n stage 2D dwt. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="stages">		 	The stages. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
int
writeNStage2DDWT(T           *component_cuda,
                 int          pixWidth,
                 int          pixHeight,
                 int          stages,
                 const char  *filename,
                 const char  *suffix,
                 sycl::queue &queue)
try
{
    struct band
    {
        int dimX;
        int dimY;
    };
    struct dimensions
    {
        struct band LL;
        struct band HL;
        struct band LH;
        struct band HH;
    };

    struct dimensions *bandDims
        = (struct dimensions *)malloc(stages * sizeof(struct dimensions));
    bandDims[0].LL.dimX = DIVANDRND(pixWidth, 2);
    bandDims[0].LL.dimY = DIVANDRND(pixHeight, 2);
    bandDims[0].HL.dimX = pixWidth - bandDims[0].LL.dimX;
    bandDims[0].HL.dimY = bandDims[0].LL.dimY;
    bandDims[0].LH.dimX = bandDims[0].LL.dimX;
    bandDims[0].LH.dimY = pixHeight - bandDims[0].LL.dimY;
    bandDims[0].HH.dimX = bandDims[0].HL.dimX;
    bandDims[0].HH.dimY = bandDims[0].LH.dimY;

    for (int i = 1; i < stages; i++)
    {
        bandDims[i].LL.dimX = DIVANDRND(bandDims[i - 1].LL.dimX, 2);
        bandDims[i].LL.dimY = DIVANDRND(bandDims[i - 1].LL.dimY, 2);
        bandDims[i].HL.dimX = bandDims[i - 1].LL.dimX - bandDims[i].LL.dimX;
        bandDims[i].HL.dimY = bandDims[i].LL.dimY;
        bandDims[i].LH.dimX = bandDims[i].LL.dimX;
        bandDims[i].LH.dimY = bandDims[i - 1].LL.dimY - bandDims[i].LL.dimY;
        bandDims[i].HH.dimX = bandDims[i].HL.dimX;
        bandDims[i].HH.dimY = bandDims[i].LH.dimY;
    }

    int samplesNum = pixWidth * pixHeight;
    int size       = samplesNum * sizeof(T);
    T  *src        = (T *)malloc(size);
    T  *dst        = (T *)malloc(size);
    memset(src, 0, size);
    memset(dst, 0, size);

    unsigned char *result = (unsigned char *)malloc(samplesNum);
    queue.memcpy(src, component_cuda, size).wait();

    // LL Band
    size = bandDims[stages - 1].LL.dimX * sizeof(T);
    for (int i = 0; i < bandDims[stages - 1].LL.dimY; i++)
        memcpy(
            dst + i * pixWidth, src + i * bandDims[stages - 1].LL.dimX, size);

    for (int s = stages - 1; s >= 0; s--)
    {
        // HL Band
        size       = bandDims[s].HL.dimX * sizeof(T);
        int offset = bandDims[s].LL.dimX * bandDims[s].LL.dimY;
        for (int i = 0; i < bandDims[s].HL.dimY; i++)
            memcpy(dst + i * pixWidth + bandDims[s].LL.dimX,
                   src + offset + i * bandDims[s].HL.dimX,
                   size);

        // LH band
        size = bandDims[s].LH.dimX * sizeof(T);
        offset += bandDims[s].HL.dimX * bandDims[s].HL.dimY;
        int yOffset = bandDims[s].LL.dimY;
        for (int i = 0; i < bandDims[s].HL.dimY; i++)
            memcpy(dst + (yOffset + i) * pixWidth,
                   src + offset + i * bandDims[s].LH.dimX,
                   size);

        // HH band
        size = bandDims[s].HH.dimX * sizeof(T);
        offset += bandDims[s].LH.dimX * bandDims[s].LH.dimY;
        yOffset = bandDims[s].HL.dimY;
        for (int i = 0; i < bandDims[s].HH.dimY; i++)
            memcpy(dst + (yOffset + i) * pixWidth + bandDims[s].LH.dimX,
                   src + offset + i * bandDims[s].HH.dimX,
                   size);
    }

    /* Write component */
    samplesToChar(result, dst, samplesNum);

    char outfile[strlen(filename) + strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile + strlen(filename), suffix);
    int i = open(outfile, O_CREAT | O_WRONLY, 0644);
    if (i == -1)
    {
        error(0, errno, "cannot access %s", outfile);
        return -1;
    }
    ssize_t x;
    x = write(i, result, samplesNum);
    close(i);

    free(src);
    free(dst);
    free(result);
    free(bandDims);
    if (x == 0)
        return 1;
    return 0;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a n stage 2D dwt. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="stages">		 	The stages. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template int writeNStage2DDWT<float>(float       *component_cuda,
                                     int          pixWidth,
                                     int          pixHeight,
                                     int          stages,
                                     const char  *filename,
                                     const char  *suffix,
                                     sycl::queue &queue);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a n stage 2D dwt. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component
/// cuda.
/// </param> <param name="pixWidth">		 	Width of the pix.
/// </param> <param name="pixHeight">	 	Height of the pix. </param>
/// <param name="stages">		 	The stages. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template int writeNStage2DDWT<int>(int         *component_cuda,
                                   int          pixWidth,
                                   int          pixHeight,
                                   int          stages,
                                   const char  *filename,
                                   const char  *suffix,
                                   sycl::queue &queue);
