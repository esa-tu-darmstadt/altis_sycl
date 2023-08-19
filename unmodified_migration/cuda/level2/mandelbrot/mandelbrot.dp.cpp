////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\mandelbrot\mandelbrot.cu
//
// summary:	Mandelbrot class
// 
//  @file histo-global.cu histogram with global memory atomics
//  
//  origin: (http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include <chrono>

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

#define BSY 4

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

float kernelTime, transferTime;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @property	cudaEvent_t start, stop
///
/// @brief	Gets the stop
///
/// @returns	The stop.
////////////////////////////////////////////////////////////////////////////////////////////////////

dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
/// @brief	The elapsed
float elapsed;

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

int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @struct	complex
///
/// @brief	a simple complex type
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

struct complex {

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

	complex(float re, float im = 0) {
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
/// @fn	inline __host__ __device__ complex operator+ (const complex &a, const complex &b)
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

inline complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
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

inline complex operator-
(const complex &a) { return complex(-a.re, -a.im); }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator- (const complex &a, const complex &b)
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

inline complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator* (const complex &a, const complex &b)
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

inline complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
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

inline float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator/ (const complex &a, const complex &b)
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

inline complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
}  // operator/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BS
///
/// @brief	A macro that defines bs
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BS 256

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int pixel_dwell (int w, int h, complex cmin, complex cmax, int x, int y, int MAX_DWELL)
///
/// @brief	computes the dwell for a single pixel
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	w		 	The width. 
/// @param 	h		 	The height. 
/// @param 	cmin	 	The cmin. 
/// @param 	cmax	 	The cmax. 
/// @param 	x		 	The x coordinate. 
/// @param 	y		 	The y coordinate. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

int pixel_dwell
(int w, int h, complex cmin, complex cmax, int x, int y, int MAX_DWELL) {
	complex dc = cmax - cmin;
	float fx = (float)x / w, fy = (float)y / h;
	complex c = cmin + complex(fx * dc.re, fy * dc.im);
	int dwell = 0;
	complex z = c;
	while(dwell < MAX_DWELL && abs2(z) < 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}  // pixel_dwell

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int same_dwell(int d1, int d2, int MAX_DWELL)
///
/// @brief	Same dwell
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	d1		 	The first int. 
/// @param 	d2		 	The second int. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

int same_dwell(int d1, int d2, int MAX_DWELL) {
    int NEUT_DWELL = MAX_DWELL + 1;
	if(d1 == d2)
		return d1;
	else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
                return sycl::min(d1, d2);
        else
		return DIFF_DWELL;
}  // same_dwell

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int border_dwell (int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL)
///
/// @brief	evaluates the common border dwell, if it exists
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	w		 	The width. 
/// @param 	h		 	The height. 
/// @param 	cmin	 	The cmin. 
/// @param 	cmax	 	The cmax. 
/// @param 	x0		 	The x coordinate 0. 
/// @param 	y0		 	The y coordinate 0. 
/// @param 	d		 	An int to process. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

int border_dwell
(int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL,
 const sycl::nd_item<3> &item_ct1, int *ldwells) {
	// check whether all boundary pixels have the same dwell
        int tid = item_ct1.get_local_id(1) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        int bs = item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        int comm_dwell = MAX_DWELL + 1;
	// for all boundary pixels, distributed across threads
	for(int r = tid; r < d; r += bs) {
		// for each boundary: b = 0 is east, then counter-clockwise
		for(int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
			comm_dwell = same_dwell(comm_dwell, dwell, MAX_DWELL);
		}
	}  // for all boundary pixels
	// reduce across threads in the block

        int nt = sycl::min(d, BSX * BSY);
        if(tid < nt)
		ldwells[tid] = comm_dwell;
        /*
        DPCT1065:159: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        for(; nt > 1; nt /= 2) {
		if(tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2], MAX_DWELL);
                /*
                DPCT1065:160: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
	return ldwells[0];
}  // border_dwell

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void dwell_fill_k (int *dwells, int w, int x0, int y0, int d, int dwell)
///
/// @brief	the kernel to fill the image region with a specific dwell value
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells	If non-null, the dwells. 
/// @param 		   	w	  	The width. 
/// @param 		   	x0	  	The x coordinate 0. 
/// @param 		   	y0	  	The y coordinate 0. 
/// @param 		   	d	  	An int to process. 
/// @param 		   	dwell 	The dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell,
 const sycl::nd_item<3> &item_ct1) {
        int x = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
        int y = item_ct1.get_local_id(1) +
                item_ct1.get_group(1) * item_ct1.get_local_range(1);
        if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = dwell;
	}
}  // dwell_fill_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_pixel_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL)
///
/// @brief	/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	If non-null, the dwells. 
/// @param 		   	w		 	The width. 
/// @param 		   	h		 	The height. 
/// @param 		   	cmin	 	The cmin. 
/// @param 		   	cmax	 	The cmax. 
/// @param 		   	x0		 	The x coordinate 0. 
/// @param 		   	y0		 	The y coordinate 0. 
/// @param 		   	d		 	An int to process. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void mandelbrot_pixel_k
(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL,
 const sycl::nd_item<3> &item_ct1) {
        int x = item_ct1.get_local_id(2) +
                item_ct1.get_local_range(2) * item_ct1.get_group(2);
        int y = item_ct1.get_local_id(1) +
                item_ct1.get_local_range(1) * item_ct1.get_group(1);
        if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	}
}  // mandelbrot_pixel_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_k (int *dwells, int w, int h, complex cmin, complex cmax, int MAX_DWELL)
///
/// @brief	computes the dwells for Mandelbrot image
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	the output array. 
/// @param 		   	w		 	the width of the output image. 
/// @param 		   	h		 	the height of the output image. 
/// @param 		   	cmin	 	the complex value associated with the left-bottom corner of the
/// 							image. 
/// @param 		   	cmax	 	the complex value associated with the right-top corner of the
/// 							image. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void mandelbrot_k
(int *dwells, int w, int h, complex cmin, complex cmax, int MAX_DWELL,
 const sycl::nd_item<3> &item_ct1) {
	// complex value to start iteration (c)
        int x = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
        int y = item_ct1.get_local_id(1) +
                item_ct1.get_group(1) * item_ct1.get_local_range(1);
        int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	dwells[y * w + x] = dwell;
}  // mandelbrot_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_block_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int depth, int MAX_DWELL)
///
/// @brief	computes the dwells for Mandelbrot image using dynamic parallelism; one block is
/// 		launched per pixel
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	the output array. 
/// @param 		   	w		 	the width of the output image. 
/// @param 		   	h		 	the height of the output image. 
/// @param 		   	cmin	 	the complex value associated with the left-bottom corner of the
/// 							image. 
/// @param 		   	cmax	 	the complex value associated with the right-top corner of the
/// 							image. 
/// @param 		   	x0		 	the starting x coordinate of the portion to compute. 
/// @param 		   	y0		 	the starting y coordinate of the portion to compute. 
/// @param 		   	d		 	the size of the portion to compute (the portion is always a
/// 							square) 
/// @param 		   	depth	 	kernel invocation depth. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
///
/// ### remarks	the algorithm reverts to per-pixel Mandelbrot evaluation once either maximum
/// 			depth or minimum size is reached.
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1109:161: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
void mandelbrot_block_k(int *dwells, int w, int h, complex cmin, complex cmax,
                        int x0, int y0, int d, int depth, int MAX_DWELL,
                        const sycl::nd_item<3> &item_ct1, int *ldwells) {
        x0 += d * item_ct1.get_group(2), y0 += d * item_ct1.get_group(1);
        int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d, MAX_DWELL,
                                      item_ct1, ldwells);
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
                if(comm_dwell != DIFF_DWELL) {
			// uniform dwell, just fill
                        sycl::range<3> bs(1, BSY, BSX),
                            grid(1, divup(d, BSY), divup(d, BSX));
                        /*
                        DPCT1049:162: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(grid * bs, bs),
                      [=](sycl::nd_item<3> item_ct1) {
                            dwell_fill_k(dwells, w, x0, y0, d, comm_dwell,
                                         item_ct1);
                      });
                } else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
			// subdivide recursively
                        sycl::range<3> bs(1, item_ct1.get_local_range(1),
                                          item_ct1.get_local_range(2)),
                            grid(1, SUBDIV, SUBDIV);
                        /*
                        DPCT1049:163: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
                        /*
                        DPCT1109:164: Recursive functions cannot be called in
                        SYCL device code. You need to adjust the code.
                        */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:1124: 'BSX * BSY' expression was replaced
                        with a value. Modify the code to use the original
                        expression, provided in comments, if it is correct.
                        */
                        sycl::local_accessor<int, 1> ldwells_acc_ct1(
                            sycl::range<1>(256 /*BSX * BSY*/), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * bs, bs),
                            [=](sycl::nd_item<3> item_ct1) {
                                  mandelbrot_block_k(
                                      dwells, w, h, cmin, cmax, x0, y0,
                                      d / SUBDIV, depth + 1, MAX_DWELL,
                                      item_ct1, ldwells_acc_ct1.get_pointer());
                            });
                  });
                } else {
			// leaf, per-pixel kernel
                        sycl::range<3> bs(1, BSY, BSX),
                            grid(1, divup(d, BSY), divup(d, BSX));
                        /*
                        DPCT1049:165: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(grid * bs, bs),
                      [=](sycl::nd_item<3> item_ct1) {
                            mandelbrot_pixel_k(dwells, w, h, cmin, cmax, x0, y0,
                                               d, MAX_DWELL, item_ct1);
                      });
                }
		//CUDA_SAFE_CALL(cudaGetLastError());
	}
}  // mandelbrot_block_k

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

void mandelbrot(ResultDatabase &resultDB, OptionParser &op, int size, int MAX_DWELL) {
	const bool uvm = op.getOptionBool("uvm");
	const bool uvm_advise = op.getOptionBool("uvm-advise");
	const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
	const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
	int device = 0;
        checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

        // allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:809: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(d_dwells = (int *)sycl::malloc_shared(
                                         dwell_sz, dpct::get_default_queue())));
        } else {
                /*
                DPCT1064:810: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(d_dwells = (int *)sycl::malloc_device(
                                         dwell_sz, dpct::get_default_queue())));
                h_dwells = (int *)malloc(dwell_sz);
		assert(h_dwells);
	}

	// compute the dwells, copy them back
        sycl::range<3> bs(1, 4, 64), grid(1, divup(h, bs[1]), divup(w, bs[2]));
    /*
    DPCT1012:785: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:786: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
        /*
        DPCT1049:166: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      *stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(grid * bs, bs), [=](sycl::nd_item<3> item_ct1) {
                mandelbrot_k(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1),
                             MAX_DWELL, item_ct1);
          });
        /*
        DPCT1012:787: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:788: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop->wait();
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
        checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    /*
    DPCT1012:789: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:790: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
        if (uvm) {
		h_dwells = d_dwells;
	} else if (uvm_advise) {
		h_dwells = d_dwells;
                /*
                DPCT1063:791: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                /*
                DPCT1063:792: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
        } else if (uvm_prefetch) {
		h_dwells = d_dwells;
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                         .get_device(device)
                                         .default_queue()
                                         .prefetch(h_dwells, dwell_sz)));
        } else if (uvm_prefetch_advise) {
		h_dwells = d_dwells;
                /*
                DPCT1063:793: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                /*
                DPCT1063:794: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                         .get_device(device)
                                         .default_queue()
                                         .prefetch(h_dwells, dwell_sz)));
        } else {
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::get_default_queue()
                                         .memcpy(h_dwells, d_dwells, dwell_sz)
                                         .wait()));
        }
    /*
    DPCT1012:795: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:796: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    transferTime += elapsed * 1.e-3;

	// free data
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_dwells, dpct::get_default_queue())));
        if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
		free(h_dwells);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void mandelbrot_dyn(int size, int MAX_DWELL)
///
/// @brief	Mandelbrot dynamic
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	size	 	The size. 
/// @param 	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void mandelbrot_dyn(ResultDatabase &resultDB, OptionParser &op, int size, int MAX_DWELL) {
	const bool uvm = op.getOptionBool("uvm");
	const bool uvm_advise = op.getOptionBool("uvm-advise");
	const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
	const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
	int device = 0;
        checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

        // allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:811: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(d_dwells = (int *)sycl::malloc_shared(
                                         dwell_sz, dpct::get_default_queue())));
        } else {
                /*
                DPCT1064:812: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(d_dwells = (int *)sycl::malloc_device(
                                         dwell_sz, dpct::get_default_queue())));
                h_dwells = (int *)malloc(dwell_sz);
		assert(h_dwells);
	}

	// compute the dwells, copy them back
        sycl::range<3> bs(1, BSY, BSX), grid(1, INIT_SUBDIV, INIT_SUBDIV);
    /*
    DPCT1012:797: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:798: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
        /*
        DPCT1049:167: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:1125: 'BSX * BSY' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<int, 1> ldwells_acc_ct1(
                sycl::range<1>(256 /*BSX * BSY*/), cgh);

            cgh.parallel_for(sycl::nd_range<3>(grid * bs, bs),
                             [=](sycl::nd_item<3> item_ct1) {
                                   mandelbrot_block_k(
                                       d_dwells, w, h, complex(-1.5, -1),
                                       complex(0.5, 1), 0, 0, w / INIT_SUBDIV,
                                       1, MAX_DWELL, item_ct1,
                                       ldwells_acc_ct1.get_pointer());
                             });
      });
        /*
        DPCT1012:799: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:800: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop->wait();
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    /*
    DPCT1012:801: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:802: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);

        if (uvm) {
		h_dwells = d_dwells;
	} else if (uvm_advise) {
		h_dwells = d_dwells;
                /*
                DPCT1063:803: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                /*
                DPCT1063:804: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
        } else if (uvm_prefetch) {
		h_dwells = d_dwells;
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                         .get_device(device)
                                         .default_queue()
                                         .prefetch(h_dwells, dwell_sz)));
        } else if (uvm_prefetch_advise) {
		h_dwells = d_dwells;
                /*
                DPCT1063:805: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                /*
                DPCT1063:806: Advice parameter is device-defined and was set to
                0. You may need to adjust it.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    dpct::cpu_device().default_queue().mem_advise(
                        h_dwells, dwell_sz, 0)));
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                         .get_device(device)
                                         .default_queue()
                                         .prefetch(h_dwells, dwell_sz)));
        } else {
                checkCudaErrors(
                    DPCT_CHECK_ERROR(dpct::get_default_queue()
                                         .memcpy(h_dwells, d_dwells, dwell_sz)
                                         .wait()));
        }

    /*
    DPCT1012:807: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:808: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    transferTime += elapsed * 1.e-3;

	// free data
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(d_dwells, dpct::get_default_queue())));
        if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
		free(h_dwells);
	}
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

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm (the more iterations, the greater speedup from dynamic parallelism)");
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

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running Mandelbrot\n");

    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

    bool quiet = op.getOptionBool("quiet");
    int imageSize = op.getOptionInt("imageSize");
    int iters = op.getOptionInt("iterations");
	bool dyn = op.getOptionBool("dyn");
    if (imageSize == 0 || iters == 0) {
        int imageSizes[5] = {2 << 11, 2 << 12, 2 << 13, 2 << 14, 2 << 14};
        int iterSizes[5] = {32, 128, 512, 1024, 8192*16};
        imageSize = imageSizes[op.getOptionInt("size") - 1];
        iters = iterSizes[op.getOptionInt("size") - 1];
    }
    
    if (!quiet) {
        printf("Image Size: %d by %d\n", imageSize, imageSize);
        printf("Num Iterations: %d\n", iters);
		if (dyn) printf("Using dynamic parallelism\n");
        else printf("Not using dynamic parallelism\n");
    }
    
    char atts[1024];
    sprintf(atts, "img:%d,iter:%d", imageSize, iters);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }

        kernelTime = 0.0f;
        transferTime = 0.0f;
        mandelbrot(resultDB, op, imageSize, iters);
        resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult("mandelbrot_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult("mandelbrot_total_time", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult("mandelbrot_parity", atts, "N", transferTime / kernelTime);
        resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
		if (dyn) {
			float totalTime = kernelTime;
			kernelTime = 0.0f;
			transferTime = 0.0f;
			mandelbrot_dyn(resultDB, op, imageSize, iters);
			resultDB.AddResult("mandelbrot_dynpar_kernel_time", atts, "sec", kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_transfer_time", atts, "sec", transferTime);
			resultDB.AddResult("mandelbrot_dynpar_total_time", atts, "sec", transferTime + kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_parity", atts, "N", transferTime / kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_speedup", atts, "N", totalTime/kernelTime);
		}

        if(!quiet) {
            printf("Done.\n");
        }
    }

}
