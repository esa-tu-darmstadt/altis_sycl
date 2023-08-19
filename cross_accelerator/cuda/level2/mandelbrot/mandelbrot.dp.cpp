////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\mandelbrot\mandelbrot.cu
//
// summary:	Mandelbrot class
//
//  @file histo-global.cu histogram with global memory atomics
//
//  origin:
//  (http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BSX
///
/// @brief	block size along
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BSX 64

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BSY
///
/// @brief	A macro that defines bsy
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BSY 1

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	MAX_DEPTH
///
/// @brief	maximum recursion depth
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_DEPTH 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	MIN_SIZE
///
/// @brief	region below which do per-pixel
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define MIN_SIZE 32

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	SUBDIV
///
/// @brief	subdivision factor along each axis
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SUBDIV 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	INIT_SUBDIV
///
/// @brief	subdivision when launched from host
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define INIT_SUBDIV 32
/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
                /// @brief	.
                element, -1 = dwells are different */
#define DIFF_DWELL (-1)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @property	float kernelTime, transferTime
///
/// @brief	Gets the transfer time
///
/// @returns	The transfer time.
////////////////////////////////////////////////////////////////////////////////////////////////////

float                                              kernelTime, transferTime;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float                                              elapsed;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__host__ __device__ int divup(int x, int y)
///
/// @brief	a useful function to compute the number of threads
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	x	The x coordinate.
/// @param 	y	The y coordinate.
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

int
divup(int x, int y)
{
    return x / y + (x % y ? 1 : 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @struct	complex
///
/// @brief	a simple complex type
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

struct complex
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @fn	__host__ __device__ complex(float re, float im = 0)
    ///
    /// @brief	Constructor
    ///
    /// @author	Ed
    /// @date	5/20/2020
    ///
    /// @param 	re	The re.
    /// @param 	im	(Optional) The im.
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    complex(float re, float im = 0)
    {
        this->re = re;
        this->im = im;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @property	float re, im
    ///
    /// @brief	/** real and imaginary part
    ///
    /// @returns	The im.
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    float re, im;
}; // struct complex

// operator overloads for complex numbers

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator+ (const complex &a, const
/// complex &b)
///
/// @brief	Addition operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value.
/// @param 	b	A value to add to it.
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline ::complex
operator+(const ::complex &a, const ::complex &b)
{
    return ::complex(a.re + b.re, a.im + b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator- (const complex &a)
///
/// @brief	Subtraction operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	A complex to process.
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline ::complex
operator-(const ::complex &a)
{
    return ::complex(-a.re, -a.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator- (const complex &a, const
/// complex &b)
///
/// @brief	Subtraction operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value.
/// @param 	b	A value to subtract from it.
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline ::complex
operator-(const ::complex &a, const ::complex &b)
{
    return ::complex(a.re - b.re, a.im - b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator* (const complex &a, const
/// complex &b)
///
/// @brief	Multiplication operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value to multiply.
/// @param 	b	The second value to multiply.
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline ::complex
operator*(const ::complex &a, const ::complex &b)
{
    return ::complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ float abs2(const complex &a)
///
/// @brief	Abs 2
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	A complex to process.
///
/// @returns	A float.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline float
abs2(const ::complex &a)
{
    return a.re * a.re + a.im * a.im;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator/ (const complex &a, const
/// complex &b)
///
/// @brief	Division operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The numerator.
/// @param 	b	The denominator.
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline ::complex
operator/(const ::complex &a, const ::complex &b)
{
    float invabs2 = 1 / abs2(b);
    return ::complex((a.re * b.re + a.im * b.im) * invabs2,
                     (a.im * b.re - b.im * a.re) * invabs2);
} // operator/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BS
///
/// @brief	A macro that defines bs
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BS 256

int
pixel_dwell(
    int w, int h, ::complex cmin, ::complex cmax, int x, int y, int MAX_DWELL)
{
    ::complex dc = cmax - cmin;
    float     fx = (float)x / w, fy = (float)y / h;
    ::complex c     = cmin + ::complex(fx * dc.re, fy * dc.im);
    int       dwell = 0;
    ::complex z     = c;
    while (dwell < MAX_DWELL && abs2(z) < 2 * 2)
    {
        /*
        DPCT1084:1584: The function call has multiple migration results
        in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        z = z * z + c;
        dwell++;
    }
    return dwell;
} // pixel_dwell

void
mandelbrot_k(sycl::device_ptr<int> dwells,
             int                   w,
             int                   h,
             ::complex             cmin,
             ::complex             cmax,
             int                   MAX_DWELL,
             sycl::nd_item<3>      item_ct1)
{
    // complex value to start iteration (c)
    int x = item_ct1.get_local_id(2)
            + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int y = item_ct1.get_local_id(1)
            + item_ct1.get_group(1) * item_ct1.get_local_range(1);
    int dwell         = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
    dwells[y * w + x] = dwell;
} // mandelbrot_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void mandelbrot(int size, int MAX_DWELL)
///
/// @brief	Mandelbrots
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	size	 	The size.
/// @param 	MAX_DWELL	The maximum dwell.
////////////////////////////////////////////////////////////////////////////////////////////////////

void
mandelbrot(ResultDatabase &resultDB,
           OptionParser   &op,
           int             size,
           int             MAX_DWELL,
           size_t          device_idx,
           bool            verify)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    // allocate memory
    int    w = size, h = size;
    size_t dwell_sz = w * h * sizeof(int);
    int   *d_dwells = (int *)sycl::malloc_device(dwell_sz, queue);
    int   *h_dwells = (int *)malloc(dwell_sz);
    if (d_dwells == nullptr) std::cerr << "Err alloc dev-mem" << std::endl;
    assert(h_dwells);

    // compute the dwells, copy them back
    sycl::range<3> bs(1, BSY, BSX), grid(1, divup(h, bs[1]), divup(w, bs[2]));
    sycl::event    k_event = queue.parallel_for<class mandelbrot_kernel>(
        sycl::nd_range<3>(grid * bs, bs), [=](sycl::nd_item<3> item_ct1) {
            mandelbrot_k(d_dwells,
                         w,
                         h,
                         ::complex(-1.5, -1),
                         ::complex(0.5, 1),
                         MAX_DWELL,
                         item_ct1);
        });
    k_event.wait();
    elapsed
        = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
          - k_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsed * 1.e-9;

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(h_dwells, d_dwells, dwell_sz).wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed  = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
    transferTime += elapsed * 1.e-3;

    if (verify)
    {
        // Calculate Mandelbrot on CPU, for validation.
        //
        int   *cpu_dwells = (int *)malloc(dwell_sz);
        for (int64_t x = 0; x < w; x++)
            for (int64_t y = 0; y < h; y++)
                cpu_dwells[x + y * h] = 
                    pixel_dwell(w, h, ::complex(-1.5, -1), ::complex(0.5, 1), x, y, MAX_DWELL);

        int64_t diff = 0;
        for (int64_t x = 0; x < w; ++x) {
        for (int64_t y = 0; y < h; ++y) {
            if (cpu_dwells[x + y * h] != h_dwells[x + h * y]) 
            {
                // std::cout << "diff at " << x << " " << y << ": " 
                //     << cpu_dwells[x + y * h] << " vs " 
                //     << h_dwells[x + y * h] << std::endl;
                diff++;
            }
        }
        }

        double tolerance = 0.05;
        double ratio = (double)diff / (double)(dwell_sz);
        if (ratio > tolerance) 
            std::cout << "Fail verification - diff larger than tolerance" << std::endl;
        else
            std::cout << "Vertification successfull" << std::endl;

        free(cpu_dwells);
    }

    // free data
    sycl::free(d_dwells, queue);
    free(h_dwells);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void addBenchmarkSpecOptions(OptionParser &op)
///
/// @brief	Adds a benchmark specifier options
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	op	The operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("iterations",
                 OPT_INT,
                 "0",
                 "iterations of algorithm (the more iterations, the greater "
                 "speedup from dynamic parallelism)");
    op.addOption(
        "verify", OPT_BOOL, "0", "verify the results computed on host");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
///
/// @brief	Executes the benchmark operation
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database.
/// @param [in,out]	op			The operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running Mandelbrot\n");

    bool quiet     = op.getOptionBool("quiet");
    int  imageSize = op.getOptionInt("imageSize");
    int  iters     = op.getOptionInt("iterations");
    bool verify    = op.getOptionBool("verify");
    if (imageSize == 0 || iters == 0)
    {
        int imageSizes[5] = { 2 << 11, 2 << 12, 2 << 13, 2 << 14, 2 << 14 };
        int iterSizes[5]  = { 32, 128, 512, 1024, 8192 * 16 };
        imageSize         = imageSizes[op.getOptionInt("size") - 1];
        iters             = iterSizes[op.getOptionInt("size") - 1];
    }

    if (!quiet)
    {
        printf("Image Size: %d by %d\n", imageSize, imageSize);
        printf("Num Iterations: %d\n", iters);
        printf("Not using dynamic parallelism\n");
    }

    char atts[1024];
    sprintf(atts, "img:%d,iter:%d", imageSize, iters);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        if (!quiet)
            printf("Pass %d:\n", i);

        kernelTime   = 0.0f;
        transferTime = 0.0f;
        mandelbrot(resultDB, op, imageSize, iters, device_idx, verify);
        resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult(
            "mandelbrot_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult(
            "mandelbrot_total_time", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult(
            "mandelbrot_parity", atts, "N", transferTime / kernelTime);
        resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

        if (!quiet)
            printf("Done.\n");
    }
}
