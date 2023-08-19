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
// file:	altis\src\cuda\level2\dwt2d\dwt_cuda\dwt_main.cu
//
// summary:	Sort class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <assert.h>
#include <chrono>
#include <error.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "cudacommon.h"
#include "common.h"
#include "components.h"
#include "dwt.h"
#include "data/create.cpp"

struct dwt
{
    char          *srcFilename;
    char          *outFilename;
    unsigned char *srcImg;
    int            pixWidth;
    int            pixHeight;
    int            dwtLvls;
};

int
getImg(char *srcFilename, unsigned char *srcImg, int inputSize, bool quiet)
{
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1)
    {
        error(0, errno, "Error: cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    close(i);

    if (!quiet)
        printf("precteno %d, inputsize %d\n", ret, inputSize);

    return 0;
}

template<typename T>
void
processDWT(struct dwt     *d,
           ResultDatabase &resultDB,
           OptionParser   &op,
           bool            lastPass,
           sycl::queue    &queue)
{
    // times
    float transferTime = 0;
    float kernelTime   = 0;
    bool  verbose      = op.getOptionBool("verbose");
    bool  quiet        = op.getOptionBool("quiet");

    int componentSize = d->pixWidth * d->pixHeight * sizeof(T);
    T  *c_r_out       = (T *)sycl::malloc_device(componentSize, queue);
    T  *backup_r      = (T *)sycl::malloc_device(componentSize, queue);
    T  *backup_g      = (T *)sycl::malloc_device(componentSize, queue);
    T  *backup_b      = (T *)sycl::malloc_device(componentSize, queue);
    if (c_r_out == nullptr || backup_r == nullptr || backup_g == nullptr || backup_b == nullptr)
    {
        std::cerr << "Error allocating memory on device." << std::endl;
        std::terminate();
    }

    queue.memset(c_r_out, 0, componentSize);
    queue.memset(backup_r, 0, componentSize);
    queue.memset(backup_g, 0, componentSize);
    queue.memset(backup_b, 0, componentSize);
    queue.wait_and_throw();

    if constexpr (g_components == 3)
    {
        T *c_g_out = (T *)sycl::malloc_device(componentSize, queue);
        T *c_b_out = (T *)sycl::malloc_device(componentSize, queue);
        T *c_r = (T *)sycl::malloc_device(componentSize, queue);
        T *c_g = (T *)sycl::malloc_device(componentSize, queue);
        T *c_b = (T *)sycl::malloc_device(componentSize, queue);
        if (c_r == nullptr || c_g == nullptr || c_b == nullptr || c_g_out == nullptr || c_b_out == nullptr)
        {
            std::cerr << "Error allocating memory on device." << std::endl;
            std::terminate();
        }

        queue.memset(c_g_out, 0, componentSize);
        queue.memset(c_b_out, 0, componentSize);
        queue.memset(c_r, 0, componentSize);
        queue.memset(c_g, 0, componentSize);
        queue.memset(c_b, 0, componentSize);
        queue.wait_and_throw();

        rgbToComponents(c_r,
                        c_g,
                        c_b,
                        d->srcImg,
                        d->pixWidth,
                        d->pixHeight,
                        transferTime,
                        kernelTime,
                        op,
                        queue);

        // Compute DWT and always store into file
        const auto start_ct1 = std::chrono::steady_clock::now();
        nStage2dDWT(c_r,
                    c_r_out,
                    backup_r,
                    d->pixWidth,
                    d->pixHeight,
                    d->dwtLvls,
                    transferTime,
                    kernelTime,
                    verbose,
                    quiet,
                    queue);
        nStage2dDWT(c_g,
                    c_g_out,
                    backup_g,
                    d->pixWidth,
                    d->pixHeight,
                    d->dwtLvls,
                    transferTime,
                    kernelTime,
                    verbose,
                    quiet,
                    queue);
        nStage2dDWT(c_b,
                    c_b_out,
                    backup_b,
                    d->pixWidth,
                    d->pixHeight,
                    d->dwtLvls,
                    transferTime,
                    kernelTime,
                    verbose,
                    quiet,
                    queue);
        const auto stop_ct1 = std::chrono::steady_clock::now();
        const auto time
            = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
        printf("Time to generate:  %3.1f ms \n", time);

        queue.wait_and_throw();

        // Store DWT to file
        if constexpr (g_write_visual)
        {
            writeNStage2DDWT(c_r_out,
                             d->pixWidth,
                             d->pixHeight,
                             d->dwtLvls,
                             d->outFilename,
                             ".r",
                             queue);
            writeNStage2DDWT(c_g_out,
                             d->pixWidth,
                             d->pixHeight,
                             d->dwtLvls,
                             d->outFilename,
                             ".g",
                             queue);
            writeNStage2DDWT(c_b_out,
                             d->pixWidth,
                             d->pixHeight,
                             d->dwtLvls,
                             d->outFilename,
                             ".b",
                             queue);
        }
        else
        {
            writeLinear(c_r_out,
                        d->pixWidth,
                        d->pixHeight,
                        d->outFilename,
                        ".r",
                        queue);
            writeLinear(c_g_out,
                        d->pixWidth,
                        d->pixHeight,
                        d->outFilename,
                        ".g",
                        queue);
            writeLinear(c_b_out,
                        d->pixWidth,
                        d->pixHeight,
                        d->outFilename,
                        ".b",
                        queue);
        }

        if (lastPass && !quiet)
        {
            printf("Writing to %s.r (%d x %d)\n",
                   d->outFilename,
                   d->pixWidth,
                   d->pixHeight);
            printf("Writing to %s.g (%d x %d)\n",
                   d->outFilename,
                   d->pixWidth,
                   d->pixHeight);
            printf("Writing to %s.b (%d x %d)\n",
                   d->outFilename,
                   d->pixWidth,
                   d->pixHeight);
        }

        sycl::free(c_r, queue);
        sycl::free(c_g, queue);
        sycl::free(c_b, queue);
        sycl::free(c_g_out, queue);
        sycl::free(c_b_out, queue);
    }
    else if constexpr (g_components == 1)
    {
        // Load component
        T *c_r = (T *)sycl::malloc_device(componentSize, queue);
        if (c_r == nullptr)
        {
            std::cerr << "Error allocating memory on device." << std::endl;
            std::terminate();
        }
        queue.memset(c_r, 0, componentSize).wait();

        bwToComponent(c_r,
                      d->srcImg,
                      d->pixWidth,
                      d->pixHeight,
                      transferTime,
                      kernelTime,
                      queue);

        // Compute DWT
        nStage2dDWT(c_r,
                    c_r_out,
                    backup_r,
                    d->pixWidth,
                    d->pixHeight,
                    d->dwtLvls,
                    transferTime,
                    kernelTime,
                    verbose,
                    quiet,
                    queue);

        // Store DWT to file
        if constexpr (g_write_visual)
        {
            writeNStage2DDWT(c_r_out,
                             d->pixWidth,
                             d->pixHeight,
                             d->dwtLvls,
                             d->outFilename,
                             ".out",
                             queue);
            if (lastPass && !quiet)
                printf("Writing to %s.out (%d x %d)\n",
                       d->outFilename,
                       d->pixWidth,
                       d->pixHeight);
        }
        else
        {
            writeLinear(c_r_out,
                        d->pixWidth,
                        d->pixHeight,
                        d->outFilename,
                        ".lin.out",
                        queue);
            if (lastPass && !quiet)
                printf("Writing to %s.lin.out (%d x %d)\n",
                       d->outFilename,
                       d->pixWidth,
                       d->pixHeight);
        }
        sycl::free(c_r, queue);
    }

    sycl::free(c_r_out, queue);
    sycl::free(backup_r, queue);
    sycl::free(backup_g, queue);
    sycl::free(backup_b, queue);

    char atts[16];
    sprintf(atts, "%dx%d", d->pixWidth, d->pixHeight);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_kernel_time", atts, "sec", kernelTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_transfer_time", atts, "sec", transferTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult(
        "dwt_total_time", atts, "sec", kernelTime + transferTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_parity", atts, "N", transferTime / kernelTime);
    /// <summary>	The result db. add overall. </summary>
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("pixWidth", OPT_INT, "1", "real pixel width");
    op.addOption("pixHeight", OPT_INT, "1", "real pixel height");
    op.addOption("bitDepth", OPT_INT, "8", "bit depth of src img");
    op.addOption("levels", OPT_INT, "3", "number of DWT levels");
    op.addOption(
        "reverse", OPT_BOOL, "0", "reverse transform -> CURRENTLY COMPILE-TIME DEFINED");
    op.addOption("53", OPT_BOOL, "0", "5/3 transform -> CURRENTLY COMPILE-TIME DEFINED");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    printf("Running DWT2D\n");

    bool quiet     = op.getOptionBool("quiet");
    bool verbose   = op.getOptionBool("verbose");
    int  pixWidth  = op.getOptionInt("pixWidth");  //<real pixWidth
    int  pixHeight = op.getOptionInt("pixHeight"); //<real pixHeight
    int bitDepth = op.getOptionInt("bitDepth");
    int  dwtLvls = op.getOptionInt("levels"); // default numuber of DWT levels

    string inputFile = op.getOptionString("inputFile");
    if (inputFile.empty())
    {
        int probSizes[4] = { 48, 192, 8192, 2 << 13 };
        int pix          = probSizes[op.getOptionInt("size") - 1];
        inputFile        = datagen(pix);
        pixWidth         = pix;
        pixHeight        = pix;
    }

    if (pixWidth <= 0 || pixHeight <= 0)
    {
        printf("Wrong or missing dimensions\n");
        return;
    }

    struct dwt *d;
    d             = (struct dwt *)malloc(sizeof(struct dwt));
    d->srcImg     = NULL;
    d->pixWidth   = pixWidth;
    d->pixHeight  = pixHeight;
    d->dwtLvls    = dwtLvls;

    // file names
    d->srcFilename = (char *)malloc(strlen(inputFile.c_str()));
    strcpy(d->srcFilename, inputFile.c_str());
    d->outFilename = (char *)malloc(strlen(d->srcFilename) + 4);
    strcpy(d->outFilename, d->srcFilename);
    strcpy(d->outFilename + strlen(d->srcFilename), ".dwt");

    // Input review
    if (!quiet)
    {
        printf("Source file:\t\t%s\n", d->srcFilename);
        printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
        printf(" Components count:\t%d\n", g_components);
        printf(" Bit depth:\t\t%d\n", bitDepth);
        printf(" DWT levels:\t\t%d\n", dwtLvls);
        printf(" Forward transform:\t%d\n", g_forward);
        printf(" 9/7 transform:\t\t%d\n", dwt97);
        printf(" Write visual:\t\t%d\n", g_write_visual);
    }

    // data sizes
    int inputSize = pixWidth * pixHeight
                    * g_components; //<amount of data (in bytes) to proccess

    d->srcImg = (unsigned char*)malloc(inputSize);
    if (nullptr == d->srcImg) std::cerr << "Error allocating image!" << std::endl;

    if (getImg(d->srcFilename, d->srcImg, inputSize, quiet) == -1)
        return;

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        bool lastPass = i + 1 == passes;
        if (!quiet)
            printf("Pass %d:\n", i);

        // Made all of this constexpr to limit amount of active kernels in
        // build to 1. This way it fits in a single FPGA image.
        //
        if constexpr (g_forward == 1)
        {
            if constexpr (dwt97 == 1)
                processDWT<float>(
                    d, resultDB, op, lastPass, queue);
            else // 5/3
                processDWT<int>(
                    d, resultDB, op, lastPass, queue);
        }
        else
        { // reverse
            if constexpr (dwt97 == 1) 
                processDWT<float>(
                    d, resultDB, op, lastPass, queue);
            else // 5/3
                processDWT<int>(
                    d, resultDB, op, lastPass, queue);
        }

        if (!quiet)
            printf("Done.\n");
    }

    free(d->srcImg);
}
