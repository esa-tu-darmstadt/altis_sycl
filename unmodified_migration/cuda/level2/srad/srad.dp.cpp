////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\srad\srad.cu
//
// summary:	Srad class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "srad.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

// includes, project

// includes, kernels
#include "srad_kernel.dp.cpp"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
/// <summary>	The elapsed. </summary>
float elapsed;
/// <summary>	The check. </summary>
float *check;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random matrix. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="I">   	[in,out] If non-null, zero-based index of the. </param>
/// <param name="rows">	The rows. </param>
/// <param name="cols">	The cols. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void random_matrix(float *I, int rows, int cols);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="argc">	The argc. </param>
/// <param name="argv">	[in,out] If non-null, the argv. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize, int speckleSize, int iters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad gridsync. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad_gridsync(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize, int speckleSize, int iters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("imageSize", OPT_INT, "0", "image height and width");
  op.addOption("speckleSize", OPT_INT, "0", "speckle height and width");
  op.addOption("iterations", OPT_INT, "0", "iterations of algorithm");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  printf("Running SRAD\n");

  srand(SEED);
  bool quiet = op.getOptionBool("quiet");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advice");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  const bool coop = op.getOptionBool("coop");
  int device = 0;
  checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

  // set parameters
  int imageSize = op.getOptionInt("imageSize");
  int speckleSize = op.getOptionInt("speckleSize");
  int iters = op.getOptionInt("iterations");
  if (imageSize == 0 || speckleSize == 0 || iters == 0) {
    int imageSizes[5] = {128, 512, 4096, 8192, 16384};
    int iterSizes[5] = {5, 1, 15, 20, 40};
    imageSize = imageSizes[op.getOptionInt("size") - 1];
    speckleSize = imageSize / 2;
    iters = iterSizes[op.getOptionInt("size") - 1];
  }

  // create timing events
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

  if (!quiet) {
      printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
      printf("Image Size: %d x %d\n", imageSize, imageSize);
      printf("Speckle size: %d x %d\n", speckleSize, speckleSize);
      printf("Num Iterations: %d\n\n", iters);
  }

  // run workload
  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    float *matrix = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:583: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            matrix = sycl::malloc_shared<float>(imageSize * imageSize,
                                                dpct::get_default_queue())));
    } else {
        matrix = (float*)malloc(imageSize * imageSize * sizeof(float));
        assert(matrix);
    }
    random_matrix(matrix, imageSize, imageSize);
    if (!quiet) {
        printf("Pass %d:\n", i);
    }
    float time = srad(resultDB, op, matrix, imageSize, speckleSize, iters);
    if (!quiet) {
        printf("Running SRAD...Done.\n");
    }
    if (coop) {
        // if using cooperative groups, add result to compare the 2 times
        char atts[1024];
        sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
        float time_gridsync = srad_gridsync(resultDB, op, matrix, imageSize, speckleSize, iters);
        if(!quiet) {
            if(time_gridsync == FLT_MAX) {
                printf("Running SRAD with cooperative groups...Failed.\n");
            } else {
                printf("Running SRAD with cooperative groups...Done.\n");
            }
        }
        if(time_gridsync == FLT_MAX) {
            resultDB.AddResult("srad_gridsync_speedup", atts, "N", time/time_gridsync);
        }
    }
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(
            DPCT_CHECK_ERROR(sycl::free(matrix, dpct::get_default_queue())));
    } else {
        free(matrix);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize,
          int speckleSize, int iters) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool coop = op.getOptionBool("coop");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    kernelTime = 0.0f;
    transferTime = 0.0f;
    int rows, cols, size_I, size_R, niter, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;

  rows = imageSize;  // number of rows in the domain
  cols = imageSize;  // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  r1 = 0;            // y1 position of the speckle
  r2 = speckleSize;  // y2 position of the speckle
  c1 = 0;            // x1 position of the speckle
  c2 = speckleSize;  // x2 position of the speckle
  lambda = 0.5;      // Lambda value
  niter = iters;     // number of iterations

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    /*
    DPCT1064:584: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        J = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:585: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        c = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
  } else {
    I = (float *)malloc(size_I * sizeof(float));
    assert(I);
    J = (float *)malloc(size_I * sizeof(float));
    assert(J);
    c = (float *)malloc(sizeof(float) * size_I);
    assert(c);
  }

  // Allocate device memory
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    J_cuda = J;
    C_cuda = c;
    /*
    DPCT1064:586: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        E_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:587: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        W_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:588: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        S_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:589: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        N_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
  } else {
    /*
    DPCT1064:590: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(J_cuda = sycl::malloc_device<float>(
                                         size_I, dpct::get_default_queue())));
    /*
    DPCT1064:591: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(C_cuda = sycl::malloc_device<float>(
                                         size_I, dpct::get_default_queue())));
    /*
    DPCT1064:592: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        E_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:593: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        W_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:594: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        S_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:595: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        N_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
  }

  // copy random matrix
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    I = matrix;
  } else {
    memcpy(I, matrix, rows*cols*sizeof(float));
  }

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }
  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    // Currently the input size must be divided by 16 - the block size
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;

    sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, block_y, block_x);

    // Copy data from main memory to device memory
    /*
    DPCT1012:542: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:543: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    if (uvm) {
        // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:544: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:545: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_default_queue()
                               .memcpy(J_cuda, J, sizeof(float) * size_I)
                               .wait()));
    }
    /*
    DPCT1012:546: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:547: The original code returned the error code that was further
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

    // Run kernels
    /*
    DPCT1012:548: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:549: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(
            *start = dpct::get_default_queue().ext_oneapi_submit_barrier()));
    /*
    DPCT1049:115: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  /*
                  DPCT1101:1082: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1083: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> temp_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1084: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1085: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> temp_result_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1086: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1087: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> north_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1088: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1089: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> south_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1090: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1091: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> east_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1092: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1093: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> west_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                      [=](sycl::nd_item<3> item_ct1) {
                            srad_cuda_1(E_C, W_C, N_C, S_C, J_cuda, C_cuda,
                                        cols, rows, q0sqr, item_ct1,
                                        temp_acc_ct1, temp_result_acc_ct1,
                                        north_acc_ct1, south_acc_ct1,
                                        east_acc_ct1, west_acc_ct1);
                      });
            });
    /*
    DPCT1012:550: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:551: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(*stop = dpct::get_default_queue()
                                                 .ext_oneapi_submit_barrier()));
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    /*
    DPCT1012:552: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:553: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(
            *start = dpct::get_default_queue().ext_oneapi_submit_barrier()));
    /*
    DPCT1049:116: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  /*
                  DPCT1101:1094: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1095: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> south_c_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1096: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1097: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> east_c_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1098: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1099: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> c_cuda_temp_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1100: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1101: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> c_cuda_result_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);
                  /*
                  DPCT1101:1102: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  /*
                  DPCT1101:1103: 'BLOCK_SIZE' expression was replaced with a
                  value. Modify the code to use the original expression,
                  provided in comments, if it is correct.
                  */
                  sycl::local_accessor<float, 2> temp_acc_ct1(
                      sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                      cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                      [=](sycl::nd_item<3> item_ct1) {
                            srad_cuda_2(E_C, W_C, N_C, S_C, J_cuda, C_cuda,
                                        cols, rows, lambda, q0sqr, item_ct1,
                                        south_c_acc_ct1, east_c_acc_ct1,
                                        c_cuda_temp_acc_ct1,
                                        c_cuda_result_acc_ct1, temp_acc_ct1);
                      });
            });
    /*
    DPCT1012:554: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:555: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(*stop = dpct::get_default_queue()
                                                 .ext_oneapi_submit_barrier()));
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    // Copy data from device memory to main memory
    /*
    DPCT1012:556: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:557: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);

    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:558: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:559: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
      /*
      DPCT1063:560: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_default_queue()
                               .memcpy(J, J_cuda, sizeof(float) * size_I)
                               .wait()));
    }
    /*
    DPCT1012:561: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:562: The original code returned the error code that was further
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
  }

    char atts[1024];
    sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
    resultDB.AddResult("srad_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("srad_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("srad_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult("srad_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);

  string outfile = op.getOptionString("outputFile");
  if (!outfile.empty()) {
      // Printing output
      if (!op.getOptionBool("quiet")) {
        printf("Writing output to %s\n", outfile.c_str());
      }
      FILE *fp = NULL;
      fp = fopen(outfile.c_str(), "w");
      if (!fp) {
          printf("Error: Unable to write to file %s\n", outfile.c_str());
      } else {
          for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                  fprintf(fp, "%.5f ", J[i * cols + j]);
              }
              fprintf(fp, "\n");
          }
          fclose(fp);
      }
  }
  // write results to validate with srad_gridsync
  check = (float*) malloc(sizeof(float) * size_I);
  assert(check);
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          check[i*cols+j] = J[i*cols+j];
      }
  }

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(C_cuda, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(J_cuda, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(E_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(W_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(N_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(S_C, dpct::get_default_queue())));
  } else {
    free(I);
    free(J);
    free(c);
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(C_cuda, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(J_cuda, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(E_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(W_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(N_C, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(S_C, dpct::get_default_queue())));
  }
  return kernelTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad gridsync with UVM and gridsync. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad_gridsync(ResultDatabase &resultDB, OptionParser &op, float *matrix,
                    int imageSize, int speckleSize, int iters) try {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool coop = op.getOptionBool("coop");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    kernelTime = 0.0f;
    transferTime = 0.0f;
    int rows, cols, size_I, size_R, niter, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;

  rows = imageSize;  // number of rows in the domain
  cols = imageSize;  // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  r1 = 0;            // y1 position of the speckle
  r2 = speckleSize;  // y2 position of the speckle
  c1 = 0;            // x1 position of the speckle
  c2 = speckleSize;  // x2 position of the speckle
  lambda = 0.5;      // Lambda value
  niter = iters;     // number of iterations

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    /*
    DPCT1064:596: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        J = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:597: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        c = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
  } else {
    I = (float *)malloc(size_I * sizeof(float));
    assert(I);
    J = (float *)malloc(size_I * sizeof(float));
    assert(J);
    c = (float *)malloc(sizeof(float) * size_I);
    assert(c);
  }

  // Allocate device memory
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    J_cuda = J;
    C_cuda = c;
    /*
    DPCT1064:598: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        E_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:599: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        W_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:600: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        S_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:601: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        N_C = sycl::malloc_shared<float>(size_I, dpct::get_default_queue())));
  } else {
    /*
    DPCT1064:602: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(J_cuda = sycl::malloc_device<float>(
                                         size_I, dpct::get_default_queue())));
    /*
    DPCT1064:603: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(C_cuda = sycl::malloc_device<float>(
                                         size_I, dpct::get_default_queue())));
    /*
    DPCT1064:604: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        E_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:605: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        W_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:606: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        S_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
    /*
    DPCT1064:607: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        N_C = sycl::malloc_device<float>(size_I, dpct::get_default_queue())));
  }

  // Generate a random matrix
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    I = matrix;
  } else {
    memcpy(I, matrix, rows*cols*sizeof(float));
  }

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }
  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    // Currently the input size must be divided by 16 - the block size
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;

    sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, block_y, block_x);

    // Copy data from main memory to device memory
    /*
    DPCT1012:563: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:564: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    // timing incorrect for page fault
    // J_cuda = J;
    // C_cuda = c;
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:565: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:566: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_device(device).default_queue().mem_advise(
              J_cuda, sizeof(float) * size_I, 0)));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              J_cuda, sizeof(float) * size_I)));
    } else {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_default_queue()
                               .memcpy(J_cuda, J, sizeof(float) * size_I)
                               .wait()));
    }
    /*
    DPCT1012:567: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:568: The original code returned the error code that was further
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

    // Create srad_params struct
    srad_params params;
    params.E_C = E_C;
    params.W_C = W_C;
    params.N_C = N_C;
    params.S_C = S_C;
    params.J_cuda = J_cuda;
    params.C_cuda = C_cuda;
    params.cols = cols;
    params.rows = rows;
    params.lambda = lambda;
    params.q0sqr = q0sqr;
    void* p_params = {&params};

    // Run kernels
    /*
    DPCT1012:569: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:570: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    /*
    DPCT1049:117: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    /*
    DPCT1007:118: Migration of cudaLaunchCooperativeKernel is not supported.
    */
            checkCudaErrors(
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                      /*
                      DPCT1101:1104: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1105: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> temp_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1106: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1107: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> temp_result_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1108: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1109: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> north_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1110: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1111: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> south_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1112: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1113: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> east_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1114: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1115: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> west_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1116: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1117: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> south_c_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1118: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1119: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> east_c_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1120: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1121: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> c_cuda_temp_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);
                      /*
                      DPCT1101:1122: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      /*
                      DPCT1101:1123: 'BLOCK_SIZE' expression was replaced with a
                      value. Modify the code to use the original expression,
                      provided in comments, if it is correct.
                      */
                      sycl::local_accessor<float, 2> c_cuda_result_acc_ct1(
                          sycl::range<2>(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/),
                          cgh);

                      auto params_ct0 = *(srad_params *)(&p_params)[0];

                      cgh.parallel_for(
                          sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                          [=](sycl::nd_item<3> item_ct1) {
                                srad_cuda_3(params_ct0, item_ct1, temp_acc_ct1,
                                            temp_result_acc_ct1, north_acc_ct1,
                                            south_acc_ct1, east_acc_ct1,
                                            west_acc_ct1, south_c_acc_ct1,
                                            east_c_acc_ct1, c_cuda_temp_acc_ct1,
                                            c_cuda_result_acc_ct1);
                          });
                }););
    //srad_cuda_3<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols,
                                       //rows, lambda, q0sqr);
    /*
    DPCT1012:571: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:572: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR((
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
    kernelTime += elapsed * 1.e-3;
    /*
    DPCT1010:573: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 err = 0;

    // Copy data from device memory to main memory
    /*
    DPCT1012:575: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:576: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(0);
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      /*
      DPCT1063:577: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J, sizeof(float) * size_I, 0)));
      /*
      DPCT1063:578: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J, sizeof(float) * size_I, 0)));
    } else if (uvm_prefetch) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
              J, sizeof(float) * size_I)));
    } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:579: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J, sizeof(float) * size_I, 0)));
      /*
      DPCT1063:580: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
              J, sizeof(float) * size_I, 0)));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
              J, sizeof(float) * size_I)));
    } else {
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_default_queue()
                               .memcpy(J, J_cuda, sizeof(float) * size_I)
                               .wait()));
    }
    /*
    DPCT1012:581: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:582: The original code returned the error code that was further
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
  }

    char atts[1024];
    sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
    resultDB.AddResult("srad_gridsync_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("srad_gridsync_transer_time", atts, "sec", transferTime);
    resultDB.AddResult("srad_gridsync_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult("srad_gridsync_parity", atts, "N", transferTime / kernelTime);

  // validate result with result obtained by gridsync
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          if(check[i*cols+j] - J[i*cols+j] > 0.0001) {
              // known bug: with and without gridsync have 10e-5 difference in row 16
              //printf("Error: Validation failed at row %d, col %d\n", i, j);
              //return FLT_MAX;
          }
      }
  }
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(C_cuda, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(J_cuda, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(E_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(W_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(N_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(S_C, dpct::get_default_queue())));
  } else {
    free(I);
    free(J);
    free(c);
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(C_cuda, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(J_cuda, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(E_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(W_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(N_C, dpct::get_default_queue())));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(sycl::free(S_C, dpct::get_default_queue())));
  }
  return kernelTime;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random matrix. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="I">   	[in,out] If non-null, zero-based index of the. </param>
/// <param name="rows">	The rows. </param>
/// <param name="cols">	The cols. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void random_matrix(float *I, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }
}

