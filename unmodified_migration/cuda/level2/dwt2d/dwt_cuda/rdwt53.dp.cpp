/// 
/// @file    rdwt53.cu
/// @brief   CUDA implementation of reverse 5/3 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-04 14:19
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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudacommon.h"
#include "common.h"
#include "transform_buffer.h"
#include "io.h"
#include <chrono>

namespace dwt_cuda {

  

  /// Wraps shared momory buffer and algorithms needed for computing 5/3 RDWT
  /// using sliding window and lifting schema.
  /// @tparam WIN_SIZE_X  width of sliding window
  /// @tparam WIN_SIZE_Y  height of sliding window
  template <int WIN_SIZE_X, int WIN_SIZE_Y>
  class RDWT53 {
  private: 
    
    /// Shared memory buffer used for 5/3 DWT transforms.
    typedef TransformBuffer<int, WIN_SIZE_X, WIN_SIZE_Y + 3, 2> RDWT53Buffer;

    /// Shared buffer used for reverse 5/3 DWT.
    RDWT53Buffer buffer;

    /// Difference between indices of two vertically neighboring items in buffer.
    enum { STRIDE = RDWT53Buffer::VERTICAL_STRIDE };


    /// Info needed for loading of one input column from input image.
    /// @tparam CHECKED  true if loader should check boundaries
    template <bool CHECKED>
    struct RDWT53Column {
      /// loader of pixels from column in input image
      VerticalDWTBandLoader<int, CHECKED> loader;
      
      /// Offset of corresponding column in shared buffer.
      int offset;
      
      /// Sets all fields to some values to avoid 'uninitialized' warnings.
      void clear() {
        offset = 0;
        loader.clear();
      }
    };


    /// 5/3 DWT reverse update operation.
    struct Reverse53Update {
      SYCL_EXTERNAL void operator()(const int p, int &c, const int n) const {
        c -= (p + n + 2) / 4;  // F.3, page 118, ITU-T Rec. T.800 final draft
      }
    };


    /// 5/3 DWT reverse predict operation.
    struct Reverse53Predict {
      SYCL_EXTERNAL void operator()(const int p, int &c, const int n) const {
        c += (p + n) / 2;      // F.4, page 118, ITU-T Rec. T.800 final draft
      }
    };


    /// Horizontal 5/3 RDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    void horizontalTransform(const int lines, const int firstLine,
                             const sycl::nd_item<3> &item_ct1) {
      /*
      DPCT1065:170: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      buffer.forEachHorizontalEven(firstLine, lines, Reverse53Update(),
                                   item_ct1);
      /*
      DPCT1065:171: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      buffer.forEachHorizontalOdd(firstLine, lines, Reverse53Predict(),
                                  item_ct1);
      /*
      DPCT1065:172: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }


    /// Using given loader, it loads another WIN_SIZE_Y coefficients
    /// into specified column.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param input     input coefficients to load from
    /// @param col       info about loaded column
    template <bool CHECKED>
    inline void loadWindowIntoColumn(const int * const input,
                                                RDWT53Column<CHECKED> & col) {
      for(int i = 3; i < (3 + WIN_SIZE_Y); i += 2) {
        buffer[col.offset + i * STRIDE] = col.loader.loadLowFrom(input);
        buffer[col.offset + (i + 1) * STRIDE] = col.loader.loadHighFrom(input);
      }
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param columnX   x coordinate of column in shared transform buffer
    /// @param input     input image
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param loader    (uninitialized) info about loaded column
    template <bool CHECKED>
    void initColumn(const int columnX, const int * const input, 
                               const int sizeX, const int sizeY,
                               RDWT53Column<CHECKED> & column,
                               const int firstY,
                               const sycl::nd_item<3> &item_ct1) {
      // coordinates of the first coefficient to be loaded
      const int firstX = item_ct1.get_group(2) * WIN_SIZE_X + columnX;

      // offset of the column with index 'colIndex' in the transform buffer
      column.offset = buffer.getColumnOffset(columnX);

      if (item_ct1.get_group(1) == 0) {
        // topmost block - apply mirroring rules when loading first 3 rows
        column.loader.init(sizeX, sizeY, firstX, firstY);

        // load pixels in mirrored way
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 0 * STRIDE] =
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
      } else {
        // non-topmost row - regular loading:
        column.loader.init(sizeX, sizeY, firstX, firstY - 1);
        buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
      }
      // Now, the next coefficient, which will be loaded by loader, is #2.
    }


    /// Actual GPU 5/3 RDWT implementation.
    /// @tparam CHECKED_LOADS   true if boundaries must be checked when reading
    /// @tparam CHECKED_WRITES  true if boundaries must be checked when writing
    /// @param in        input image (5/3 transformed coefficients)
    /// @param out       output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image 
    /// @param sizeY     height of the output image
    /// @param winSteps  number of sliding window steps
    template<bool CHECKED_LOADS, bool CHECKED_WRITES>
    void transform(const int * const in, int * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps,
                              const sycl::nd_item<3> &item_ct1) {
      // info about one main and one boundary column
      RDWT53Column<CHECKED_LOADS> column, boundaryColumn;

      // index of first row to be transformed
      const int firstY = item_ct1.get_group(1) * WIN_SIZE_Y * winSteps;

      // some threads initialize boundary columns
      boundaryColumn.clear();
      if (item_ct1.get_local_id(2) < 3) {
        // First 3 threads also handle boundary columns. Thread #0 gets right
        // column #0, thread #1 get right column #1 and thread #2 left column.
        const int colId = item_ct1.get_local_id(2) +
                          ((item_ct1.get_local_id(2) != 2) ? WIN_SIZE_X : -3);

        // Thread initializes offset of the boundary column (in shared 
        // buffer), first 3 pixels of the column and a loader for this column.
        initColumn(colId, in, sizeX, sizeY, boundaryColumn, firstY, item_ct1);
      }

      // All threads initialize central columns.
      initColumn(parityIdx<WIN_SIZE_X>(item_ct1), in, sizeX, sizeY, column,
                 firstY, item_ct1);

      // horizontally transform first 3 rows
      horizontalTransform(3, 0, item_ct1);

      // writer of output pixels - initialize it
      const int outX =
          item_ct1.get_group(2) * WIN_SIZE_X + item_ct1.get_local_id(2);
      VerticalDWTPixelWriter<int, CHECKED_WRITES> writer;
      writer.init(sizeX, sizeY, outX, firstY);

      // offset of column (in transform buffer) saved by this thread
      const int outputColumnOffset =
          buffer.getColumnOffset(item_ct1.get_local_id(2));

      // (Each iteration assumes that first 3 rows of transform buffer are
      // already loaded with horizontally transformed pixels.)
      for(int w = 0; w < winSteps; w++) {
        // Load another WIN_SIZE_Y lines of this thread's column
        // into the transform buffer.
        loadWindowIntoColumn(in, column);

        // possibly load boundary columns
        if (item_ct1.get_local_id(2) < 3) {
          loadWindowIntoColumn(in, boundaryColumn);
        }

        // horizontally transform all newly loaded lines
        horizontalTransform(WIN_SIZE_Y, 3, item_ct1);

        // Using 3 registers, remember current values of last 3 rows 
        // of transform buffer. These rows are transformed horizontally 
        // only and will be used in next iteration.
        int last3Lines[3];
        last3Lines[0] = buffer[outputColumnOffset + (WIN_SIZE_Y + 0) * STRIDE];
        last3Lines[1] = buffer[outputColumnOffset + (WIN_SIZE_Y + 1) * STRIDE];
        last3Lines[2] = buffer[outputColumnOffset + (WIN_SIZE_Y + 2) * STRIDE];

        // vertically transform all central columns
        buffer.forEachVerticalOdd(outputColumnOffset, Reverse53Update());
        buffer.forEachVerticalEven(outputColumnOffset, Reverse53Predict());

        // Save all results of current window. Results are in transform buffer
        // at rows from #1 to #(1 + WIN_SIZE_Y). Other rows are invalid now.
        // (They only served as a boundary for vertical RDWT.)
        for(int i = 1; i < (1 + WIN_SIZE_Y); i++) {
          writer.writeInto(out, buffer[outputColumnOffset + i * STRIDE]);
        }

        // Use last 3 remembered lines as first 3 lines for next iteration.
        // As expected, these lines are already horizontally transformed.
        buffer[outputColumnOffset + 0 * STRIDE] = last3Lines[0];
        buffer[outputColumnOffset + 1 * STRIDE] = last3Lines[1];
        buffer[outputColumnOffset + 2 * STRIDE] = last3Lines[2];

        // Wait for all writing threads before proceeding to loading new
        // coeficients in next iteration. (Not to overwrite those which
        // are not written yet.)
        /*
        DPCT1065:173: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
      }
    }


  public:
    /// Main GPU 5/3 RDWT entry point.
    /// @param in     input image (5/3 transformed coefficients)
    /// @param out    output buffer (for reverse transformed image)
    /// @param sizeX  width of the output image 
    /// @param sizeY  height of the output image
    /// @param winSteps  number of sliding window steps
    static void run(const int * const input, int * const output,
                               const int sx, const int sy, const int steps,
                               const sycl::nd_item<3> &item_ct1,
                               RDWT53<WIN_SIZE_X, WIN_SIZE_Y> &rdwt53) {
      // prepare instance with buffer in shared memory

      // Compute limits of this threadblock's block of pixels and use them to
      // determine, whether this threadblock will have to deal with boundary.
      // (1 in next expressions is for radius of impulse response of 5/3 RDWT.)
      const int maxX = (item_ct1.get_group(2) + 1) * WIN_SIZE_X + 1;
      const int maxY = (item_ct1.get_group(1) + 1) * WIN_SIZE_Y * steps + 1;
      const bool atRightBoudary = maxX >= sx;
      const bool atBottomBoudary = maxY >= sy;

      // Select specialized version of code according to distance of this
      // threadblock's pixels from image boundary.
      if(atBottomBoudary) {
        // near bottom boundary => check both writing and reading
        rdwt53.transform<true, true>(input, output, sx, sy, steps, item_ct1);
      } else if(atRightBoudary) {
        // near right boundary only => check writing only
        rdwt53.transform<false, true>(input, output, sx, sy, steps, item_ct1);
      } else {
        // no nearby boundary => check nothing
        rdwt53.transform<false, false>(input, output, sx, sy, steps, item_ct1);
      }
    }

  }; // end of class RDWT53
  
  
  
  /// Main GPU 5/3 RDWT entry point.
  /// @param in     input image (5/3 transformed coefficients)
  /// @param out    output buffer (for reverse transformed image)
  /// @param sizeX  width of the output image 
  /// @param sizeY  height of the output image
  /// @param winSteps  number of sliding window steps
  template <int WIN_SX, int WIN_SY>
  
  void rdwt53Kernel(const int * const in, int * const out,
                               const int sx, const int sy, const int steps,
                               const sycl::nd_item<3> &item_ct1,
                               RDWT53<WIN_SX, WIN_SY> &rdwt53) {
    RDWT53<WIN_SX, WIN_SY>::run(in, out, sx, sy, steps, item_ct1, rdwt53);
  }
  
  
  
  /// Only computes optimal number of sliding window steps, 
  /// number of threadblocks and then lanches the 5/3 RDWT kernel.
  /// @tparam WIN_SX  width of sliding window
  /// @tparam WIN_SY  height of sliding window
  /// @param in       input image
  /// @param out      output buffer
  /// @param sx       width of the input image 
  /// @param sy       height of the input image
  template <int WIN_SX, int WIN_SY>
  void launchRDWT53Kernel (int * in, int * out, const int sx, const int sy, float& kernelTime) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);
    
    // prepare grid size
    sycl::range<3> gSize(1, divRndUp(sy, WIN_SY * steps), divRndUp(sx, WIN_SX));

    // timing events
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    float elapsedTime;

    // finally transform this level
    /*
    DPCT1012:853: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    /*
    DPCT1049:174: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
      *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<RDWT53<WIN_SX, WIN_SY>, 0> rdwt53_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(gSize * sycl::range<3>(1, 1, WIN_SX),
                                  sycl::range<3>(1, 1, WIN_SX)),
                [=](sycl::nd_item<3> item_ct1) {
                      rdwt53Kernel<WIN_SX, WIN_SY>(in, out, sx, sy, steps,
                                                   item_ct1, rdwt53_acc_ct1);
                });
      });
    /*
    DPCT1012:854: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }

#ifdef HYPERQ
  template <int WIN_SX, int WIN_SY>
  void launchRDWT53Kernel (int * in, int * out, const int sx, const int sy, float& kernelTime, cudaStream_t stream) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);
    
    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));
    
    // timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // finally transform this level
    cudaEventRecord(start, 0);
    rdwt53Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX, 0, stream>>>(in, out, sx, sy, steps);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }
#endif
    
  
  
  /// Reverse 5/3 2D DWT. See common rules (above) for more details.
  /// @param in      Input DWT coefficients. Format described in common rules.
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - will contain original image
  ///                in normalized range [-128, 127].
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  float rdwt53(int * in, int * out, int sizeX, int sizeY, int levels) {
    float kernelTime = 0;

    if(levels > 1) {
      // let this function recursively reverse transform deeper levels first
      const int llSizeX = divRndUp(sizeX, 2);
      const int llSizeY = divRndUp(sizeY, 2);
      kernelTime += rdwt53(in, out, llSizeX, llSizeY, levels - 1);
      
      // copy reverse transformed LL band from output back into the input
      //memCopy(in, out, llSizeX, llSizeY);
    }
    
    // select right width of kernel for the size of the image
    if(sizeX >= 960) {
      launchRDWT53Kernel<192, 8>(in, out, sizeX, sizeY, kernelTime);
    } else if (sizeX >= 480) {
      launchRDWT53Kernel<128, 8>(in, out, sizeX, sizeY, kernelTime);
    } else {
      launchRDWT53Kernel<64, 8>(in, out, sizeX, sizeY, kernelTime);
    }
      return kernelTime;
  }

#ifdef HYPERQ
  float rdwt53(int * in, int * out, int sizeX, int sizeY, int levels, cudaStream_t stream) {
    float kernelTime = 0;

    if(levels > 1) {
      // let this function recursively reverse transform deeper levels first
      const int llSizeX = divRndUp(sizeX, 2);
      const int llSizeY = divRndUp(sizeY, 2);
      kernelTime += rdwt53(in, out, llSizeX, llSizeY, levels - 1, stream);
      
      // copy reverse transformed LL band from output back into the input
      //memCopy(in, out, llSizeX, llSizeY);
    }
    
    // select right width of kernel for the size of the image
    if(sizeX >= 960) {
      launchRDWT53Kernel<192, 8>(in, out, sizeX, sizeY, kernelTime, stream);
    } else if (sizeX >= 480) {
      launchRDWT53Kernel<128, 8>(in, out, sizeX, sizeY, kernelTime, stream);
    } else {
      launchRDWT53Kernel<64, 8>(in, out, sizeX, sizeY, kernelTime, stream);
    }
      return kernelTime;
  }
#endif
  

} // end of namespace dwt_cuda
