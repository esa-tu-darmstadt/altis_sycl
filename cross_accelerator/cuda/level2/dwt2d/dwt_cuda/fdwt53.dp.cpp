/// @file    fdwt53.cu
/// @brief   CUDA implementation of forward 5/3 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-04 13:23
///
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///

#include <CL/sycl.hpp>

#include <chrono>

#include "common.h"
#include "cudacommon.h"
#include "io.h"
#include "transform_buffer.h"
#include "../dwt.h"

namespace dwt_cuda {

/// Wraps buffer and methods needed for computing one level of 5/3 FDWT
/// using sliding window approach.
/// @tparam WIN_SIZE_X  width of sliding window
/// @tparam WIN_SIZE_Y  height of sliding window
template<int WIN_SIZE_X, int WIN_SIZE_Y>
class FDWT53
{
private:
    /// Info needed for processing of one input column.
    /// @tparam CHECKED_LOADER  true if column's loader should check boundaries
    ///                         false if there are no near boudnaries to check
    template<bool CHECKED_LOADER>
    struct FDWT53Column
    {
        /// loader for the column
        VerticalDWTPixelLoader<int, CHECKED_LOADER> loader;

        /// offset of the column in shared buffer
        int offset;

        // backup of first 3 loaded pixels (not transformed)
        int pixel0, pixel1, pixel2;

        /// Sets all fields to anything to prevent 'uninitialized' warnings.
        void clear()
        {
            offset = pixel0 = pixel1 = pixel2 = 0;
            loader.clear();
        }
    };

    /// Type of shared memory buffer for 5/3 FDWT transforms.
    typedef TransformBuffer<int, WIN_SIZE_X, WIN_SIZE_Y + 3, 2> FDWT53Buffer;

    /// Actual shared buffer used for forward 5/3 DWT.
    FDWT53Buffer buffer;

    /// Difference between indices of two vertical neighbors in buffer.
    enum
    {
        STRIDE = FDWT53Buffer::VERTICAL_STRIDE
    };

    /// Forward 5/3 DWT predict operation.
    struct Forward53Predict
    {
        void operator()(const int p, int &c, const int n) const
        {
            // c = n;
            c -= (p + n) / 2; // F.8, page 126, ITU-T Rec. T.800 final draft the
                              // real one
        }
    };

    /// Forward 5/3 DWT update operation.
    struct Forward53Update
    {
        void operator()(const int p, int &c, const int n) const
        {
            c += (p + n + 2) / 4; // F.9, page 126, ITU-T Rec. T.800 final draft
        }
    };

    /// Initializes one column: computes offset of the column in shared memory
    /// buffer, initializes loader and finally uses it to load first 3 pixels.
    /// @tparam CHECKED  true if loader of the column checks boundaries
    /// @param column    (uninitialized) column info to be initialized
    /// @param input     input image
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param colIndex  x-axis coordinate of the column (relative to the left
    ///                  side of this threadblock's block of input pixels)
    /// @param firstY    y-axis coordinate of first image row to be transformed

    template<bool CHECKED> SYCL_EXTERNAL
    void initColumn(FDWT53Column<CHECKED> &column,
                    const int *const       input,
                    const int              sizeX,
                    const int              sizeY,
                    const int              colIndex,
                    const int              firstY,
                    sycl::nd_item<3>       item_ct1)
    {
        // get offset of the column with index 'cId'
        column.offset = buffer.getColumnOffset(colIndex);

        // coordinates of the first pixel to be loaded
        const int firstX = item_ct1.get_group(2) * WIN_SIZE_X + colIndex;

        if (item_ct1.get_group(1) == 0)
        {
            // topmost block - apply mirroring rules when loading first 3 rows
            column.loader.init(sizeX, sizeY, firstX, firstY);

            // load pixels in mirrored way
            column.pixel2 = column.loader.loadFrom(input); // loaded pixel #0
            column.pixel1 = column.loader.loadFrom(input); // loaded pixel #1
            column.pixel0 = column.loader.loadFrom(input); // loaded pixel #2

            // reinitialize loader to start with pixel #1 again
            column.loader.init(sizeX, sizeY, firstX, firstY + 1);
        }
        else
        {
            // non-topmost row - regular loading:
            column.loader.init(sizeX, sizeY, firstX, firstY - 2);

            // load 3 rows into the column
            column.pixel0 = column.loader.loadFrom(input);
            column.pixel1 = column.loader.loadFrom(input);
            column.pixel2 = column.loader.loadFrom(input);
            // Now, the next pixel, which will be loaded by loader, is pixel #1.
        }
    }

    /// Loads and vertically transforms given column. Assumes that first 3
    /// pixels are already loaded in column fields pixel0 ... pixel2.
    /// @tparam CHECKED  true if loader of the column checks boundaries
    /// @param column    column to be loaded and vertically transformed
    /// @param input     pointer to input image data
    template<bool CHECKED> SYCL_EXTERNAL
    void loadAndVerticallyTransform(FDWT53Column<CHECKED> &column,
                                    const int *const       input)
    {
        // take 3 loaded pixels and put them into shared memory transform buffer
        buffer[column.offset + 0 * STRIDE] = column.pixel0;
        buffer[column.offset + 1 * STRIDE] = column.pixel1;
        buffer[column.offset + 2 * STRIDE] = column.pixel2;

        // load remaining pixels to be able to vertically transform the window

        for (int i = 3; i < (3 + WIN_SIZE_Y); i++)
            buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);

        // remember last 3 pixels for use in next iteration
        column.pixel0 = buffer[column.offset + (WIN_SIZE_Y + 0) * STRIDE];
        column.pixel1 = buffer[column.offset + (WIN_SIZE_Y + 1) * STRIDE];
        column.pixel2 = buffer[column.offset + (WIN_SIZE_Y + 2) * STRIDE];

        // vertically transform the column in transform buffer
        buffer.forEachVerticalOdd(column.offset, Forward53Predict());
        buffer.forEachVerticalEven(column.offset, Forward53Update());
    }

    /// Actual implementation of 5/3 FDWT.
    /// @tparam CHECK_LOADS   true if input loader must check boundaries
    /// @tparam CHECK_WRITES  true if output writer must check boundaries
    /// @param in        input image
    /// @param out       output buffer
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param winSteps  number of sliding window steps
    template<bool CHECK_LOADS, bool CHECK_WRITES> SYCL_EXTERNAL
    void transform(const int *const in,
                   int *const       out,
                   const int        sizeX,
                   const int        sizeY,
                   const int        winSteps,
                   sycl::nd_item<3> item_ct1)
    {
        // info about one main and one boundary columns processed by this thread
        FDWT53Column<CHECK_LOADS> column;
        FDWT53Column<CHECK_LOADS> boundaryColumn; // only few threads use this

        // Initialize all column info: initialize loaders, compute offset of
        // column in shared buffer and initialize loader of column.
        const int firstY = item_ct1.get_group(1) * WIN_SIZE_Y * winSteps;
        initColumn(column,
                   in,
                   sizeX,
                   sizeY,
                   item_ct1.get_local_id(2),
                   firstY,
                   item_ct1); // has been checked Mar 9th

        // first 3 threads initialize boundary columns, others do not use them
        boundaryColumn.clear();
        if (item_ct1.get_local_id(2) < 3)
        {
            // index of boundary column (relative x-axis coordinate of the
            // column)
            const int colId
                = item_ct1.get_local_id(2)
                  + ((item_ct1.get_local_id(2) == 0) ? WIN_SIZE_X : -3);

            // initialize the column
            initColumn(
                boundaryColumn, in, sizeX, sizeY, colId, firstY, item_ct1);
        }

        // index of column which will be written into output by this thread
        const int outColumnIndex = parityIdx<WIN_SIZE_X>(item_ct1);

        // offset of column which will be written by this thread into output
        const int outColumnOffset = buffer.getColumnOffset(outColumnIndex);

        // initialize output writer for this thread
        const int outputFirstX
            = item_ct1.get_group(2) * WIN_SIZE_X + outColumnIndex;
        VerticalDWTBandWriter<int, CHECK_WRITES> writer;
        writer.init(sizeX, sizeY, outputFirstX, firstY);

        // Sliding window iterations:
        // Each iteration assumes that first 3 pixels of each column are loaded.
        for (int w = 0; w < winSteps; w++)
        {
            // For each column (including boundary columns): load and vertically
            // transform another WIN_SIZE_Y lines.
            loadAndVerticallyTransform(column, in);
            if (item_ct1.get_local_id(2) < 3)
                loadAndVerticallyTransform(boundaryColumn, in);

            // wait for all columns to be vertically transformed and transform
            // all output rows horizontally
            /*
            DPCT1065:2116: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            buffer.forEachHorizontalOdd(
                2, WIN_SIZE_Y, Forward53Predict(), item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);

            buffer.forEachHorizontalEven(
                2, WIN_SIZE_Y, Forward53Update(), item_ct1);

            // wait for all output rows to be transformed horizontally and write
            // them into output buffe
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int r = 2; r < (2 + WIN_SIZE_Y); r += 2)
            {
                // Write low coefficients from output column into low band ...
                writer.writeLowInto(out, buffer[outColumnOffset + r * STRIDE]);
                // ... and high coeficients into the high band.
                writer.writeHighInto(
                    out, buffer[outColumnOffset + (r + 1) * STRIDE]);
            }

            // before proceeding to next iteration, wait for all output columns
            // to be written into the output
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

public:
    /// Determines, whether this block's pixels touch boundary and selects
    /// right version of algorithm according to it - for many threadblocks, it
    /// selects version which does not deal with boundary mirroring and thus is
    /// slightly faster.
    /// @param in     input image
    /// @param out    output buffer
    /// @param sx     width of the input image
    /// @param sy     height of the input image
    /// @param steps  number of sliding window steps
    static SYCL_EXTERNAL void run(const int *const                in,
                    int *const                      out,
                    const int                       sx,
                    const int                       sy,
                    const int                       steps,
                    sycl::nd_item<3>                item_ct1,
                    FDWT53<WIN_SIZE_X, WIN_SIZE_Y> *fdwt53)
    {
        // if(blockIdx.x==0 && blockIdx.y ==11 && threadIdx.x >=0&&threadIdx.x
        // <64){
        // object with transform buffer in shared memory

        // Compute limits of this threadblock's block of pixels and use them to
        // determine, whether this threadblock will have to deal with boundary.
        // (1 in next expressions is for radius of impulse response of 9/7
        // FDWT.)
        const int  maxX = (item_ct1.get_group(2) + 1) * WIN_SIZE_X + 1;
        const int  maxY = (item_ct1.get_group(1) + 1) * WIN_SIZE_Y * steps + 1;
        const bool atRightBoudary  = maxX >= sx;
        const bool atBottomBoudary = maxY >= sy;

        // Select specialized version of code according to distance of this
        // threadblock's pixels from image boundary.
        // NOTE: On FPGA skip this, too much kernels!
#ifndef _FPGA
        if (atBottomBoudary)
        {
#endif
            // near bottom boundary => check both writing and reading
            fdwt53->transform<true, true>(in, out, sx, sy, steps, item_ct1);
#ifndef _FPGA
        }
        else if (atRightBoudary)
        {
            // near right boundary only => check writing only
            fdwt53->transform<false, true>(in, out, sx, sy, steps, item_ct1);
        }
        else
        {
            // no nearby boundary => check nothing
            fdwt53->transform<false, false>(in, out, sx, sy, steps, item_ct1);
        }
#endif
    }
    // }

}; // end of class FDWT53

/// Main GPU 5/3 FDWT entry point.
/// @tparam WIN_SX   width of sliding window to be used
/// @tparam WIN_SY   height of sliding window to be used
/// @param input     input image
/// @param output    output buffer
/// @param sizeX     width of the input image
/// @param sizeY     height of the input image
/// @param winSteps  number of sliding window steps
template<int WIN_SX, int WIN_SY> SYCL_EXTERNAL
void
fdwt53Kernel(const int *const        input,
             int *const              output,
             const int               sizeX,
             const int               sizeY,
             const int               winSteps,
             sycl::nd_item<3>        item_ct1,
             FDWT53<WIN_SX, WIN_SY> *fdwt53)
{
    FDWT53<WIN_SX, WIN_SY>::run(
        input, output, sizeX, sizeY, winSteps, item_ct1, fdwt53);
}

class FDWT53KernelID;

/// Only computes optimal number of sliding window steps,
/// number of threadblocks and then lanches the 5/3 FDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param in       input image
/// @param out      output buffer
/// @param sx       width of the input image
/// @param sy       height of the input image
template<int WIN_SX, int WIN_SY>
void
launchFDWT53Kernel(int         *in,
                   int         *out,
                   int          sx,
                   int          sy,
                   float       &kernelTime,
                   bool         verbose,
                   bool         quiet,
                   sycl::queue &queue)
{
    if constexpr (g_forward && !dwt97)
    {
        // compute optimal number of steps of each sliding window
        const int steps = divRndUp(sy, 15 * WIN_SY);

        int gx = divRndUp(sx, WIN_SX);
        int gy = divRndUp(sy, WIN_SY * steps);

        // prepare grid size
        sycl::range<3> gSize(1, divRndUp(sy, WIN_SY * steps), divRndUp(sx, WIN_SX));
        if (verbose && !quiet)
        {
            printf("sliding steps = %d , gx = %d , gy = %d \n", steps, gx, gy);
            printf("\n globalx=%d, globaly=%d, blocksize=%d\n",
                gSize[2],
                gSize[1],
                WIN_SX);
        }

        // run kernel, possibly measure time and finally check the call
        sycl::event compute_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<FDWT53<WIN_SX, WIN_SY>,
                        0,
                        sycl::access_mode::read_write,
                        sycl::access::target::local>
                fdwt53_acc_ct1(cgh);

            cgh.parallel_for<FDWT53KernelID>(
                sycl::nd_range<3>(gSize * sycl::range<3>(1, 1, WIN_SX),
                                sycl::range<3>(1, 1, WIN_SX)),
                [=](sycl::nd_item<3> item_ct1) {
                    fdwt53Kernel<WIN_SX, WIN_SY>(in,
                                                out,
                                                sx,
                                                sy,
                                                steps,
                                                item_ct1,
                                                fdwt53_acc_ct1.get_pointer());
                });
        });
        compute_event.wait();
        float elapsedTime
            = compute_event
                .get_profiling_info<sycl::info::event_profiling::command_end>()
            - compute_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;
    }
}

/// Forward 5/3 2D DWT. See common rules (above) for more details.
/// @param in      Expected to be normalized into range [-128, 127].
///                Will not be preserved (will be overwritten).
/// @param out     output buffer on GPU
/// @param sizeX   width of input image (in pixels)
/// @param sizeY   height of input image (in pixels)
/// @param levels  number of recursive DWT levels
float
fdwt53(int         *in,
       int         *out,
       int          sizeX,
       int          sizeY,
       int          levels,
       bool         verbose,
       bool         quiet,
       sycl::queue &queue)
{
    float kernelTime = 0;

    // Select right width of kernel for the size of the image. Made this
    // constexpr to limit amount of kernels to one for FPGA build.
    //
    if constexpr (g_sizeX >= 960)
        launchFDWT53Kernel<192, 8>(
            in, out, sizeX, sizeY, kernelTime, verbose, quiet, queue);
    else if constexpr (g_sizeX >= 480)
        launchFDWT53Kernel<128, 8>(
            in, out, sizeX, sizeY, kernelTime, verbose, quiet, queue);
    else
        launchFDWT53Kernel<64, 8>(
            in, out, sizeX, sizeY, kernelTime, verbose, quiet, queue);

    // if this was not the last level, continue recursively with other levels
    if (levels > 1)
    {
        // copy output's LL band back into input buffer
        const int llSizeX = divRndUp(sizeX, 2);
        const int llSizeY = divRndUp(sizeY, 2);

        // run remaining levels of FDWT
        kernelTime += fdwt53(
            in, out, llSizeX, llSizeY, levels - 1, verbose, quiet, queue);
    }
    return kernelTime;
}

} // end of namespace dwt_cuda
