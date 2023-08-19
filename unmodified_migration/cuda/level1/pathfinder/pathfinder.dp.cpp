////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	\altis\src\cuda\level1\pathfinder\pathfinder.cu
//
// summary:	Pathfinder class
// 
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"
#include <chrono>

#define BLOCK_SIZE 512
#define STR_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration
#define SEED 7

void run(int borderCols, int smallBlockCol, int blockCols,
         ResultDatabase &resultDB, OptionParser &op);

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;

int device_id;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("rows", OPT_INT, "0", "number of rows");
  op.addOption("cols", OPT_INT, "0", "number of cols");
  op.addOption("pyramidHeight", OPT_INT, "0", "pyramid height");
  op.addOption("instances", OPT_INT, "32", "number of pathfinder instances to run");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the pathfinder benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications: Bodun Hu
// add support for UVM
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  printf("Running Pathfinder\n");
  int device;
  device = dpct::dev_mgr::instance().current_device_id();
  dpct::device_info deviceProp;
  dpct::dev_mgr::instance().get_device(device).get_device_info(deviceProp);

  device_id = device;

  bool quiet = op.getOptionInt("quiet");
  int rowLen = op.getOptionInt("rows");
  int colLen = op.getOptionInt("cols");
  int pyramidHeight = op.getOptionInt("pyramidHeight");

  if (rowLen == 0 || colLen == 0 || pyramidHeight == 0) {
    printf("Using preset problem size %d\n", (int)op.getOptionInt("size"));
    int rowSizes[5] = {8, 16, 32, 40, 48};
    int colSizes[5] = {8, 16, 32, 40, 48};
    int pyramidSizes[5] = {4, 8, 16, 32, 36};
    rows = rowSizes[op.getOptionInt("size") - 1] * 1024;
    cols = colSizes[op.getOptionInt("size") - 1] * 1024;
    pyramid_height = pyramidSizes[op.getOptionInt("size") - 1];
  } else {
    rows = rowLen;
    cols = colLen;
    pyramid_height = pyramidHeight;
  }

  if(!quiet) {
      printf("Row length: %d\n", rows);
      printf("Column length: %d\n", cols);
      printf("Pyramid height: %d\n", pyramid_height);
  }

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols =
      cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  if(!quiet) {
      printf("gridSize: [%d],border:[%d],blockSize:[%d],blockGrid:[%d],targetBlock:[%d]\n",
              cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
  }

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    if(!quiet) {
        printf("Pass %d: ", i);
    }
    run(borderCols, smallBlockCol, blockCols, resultDB, op);
    if(!quiet) {
        printf("Done.\n");
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes parameters for computing.  </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="op">	[in,out] The option specified by the user. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void init(OptionParser &op) {
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    /*
    DPCT1064:891: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        data = (int *)sycl::malloc_shared(sizeof(int) * rows * cols,
                                          dpct::get_default_queue())));
    /*
    DPCT1064:892: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        wall = sycl::malloc_shared<int *>(rows, dpct::get_default_queue())));
    for (int n = 0; n < rows; n++) wall[n] = data + (int)cols * n;
    // checkCudaErrors(cudaMallocManaged(&result, sizeof(int) * cols));
  } else {
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++) wall[n] = data + (int)cols * n;
    result = new int[cols];
  }

  srand(SEED);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }
  string outfile = op.getOptionString("outputFile");
  if (outfile != "") {
    std::fstream fs;
    fs.open(outfile.c_str(), std::fstream::in);
    fs.close();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines whether given point is in range. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="x">  	data point. </param>
/// <param name="min">	The minimum. </param>
/// <param name="max">	The maximum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that specifies how to clamp range. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="x">  	A void to process. </param>
/// <param name="min">	The minimum. </param>
/// <param name="max">	The maximum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Dynproc kernel. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 1/7/2021. </remarks>
///
/// <param name="iteration"> 	The iteration. </param>
/// <param name="gpuWall">   	[in,out] If non-null, the GPU wall. </param>
/// <param name="gpuSrc">	 	[in,out] If non-null, the GPU source. </param>
/// <param name="gpuResults">	[in,out] If non-null, the GPU results. </param>
/// <param name="cols">		 	The cols. </param>
/// <param name="rows">		 	The rows. </param>
/// <param name="startStep"> 	The start step. </param>
/// <param name="border">	 	The border. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void dynproc_kernel(int iteration, int *gpuWall,
                               int *gpuSrc, int *gpuResults,
                               int cols, int rows,
                               int startStep, int border,
                               const sycl::nd_item<3> &item_ct1, int *prev,
                               int *result) {

  int bx = item_ct1.get_group(2);
  int tx = item_ct1.get_local_id(2);

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX =
      (int)small_block_cols * (int)bx - (int)border;
  int blkXmax = blkX + (int)BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + (int)tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax =
      (blkXmax > (int)cols - 1)
          ? (int)BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1)
          : (int)BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, (int)cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  item_ct1.barrier(
      sycl::access::fence_space::local_space); // [Ronny] Added sync to avoid
                                               // race on prev Aug. 14 2012
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (i == iteration - 1) break;
    if (computed)  // Assign the computation range
      prev[tx] = result[tx];
    item_ct1.barrier(
        sycl::access::fence_space::local_space); // [Ronny] Added sync to avoid
                                                 // race on prev Aug. 14 2012
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the path. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="gpuWall">		 	[in,out] If non-null, the GPU wall. </param>
/// <param name="gpuResult">	 	[in,out] If non-null, the GPU result. </param>
/// <param name="rows">			 	The rows. </param>
/// <param name="cols">			 	The cols. </param>
/// <param name="pyramid_height">	Height of the pyramid. </param>
/// <param name="blockCols">	 	The block cols. </param>
/// <param name="borderCols">	 	The border cols. </param>
/// <param name="kernelTime">	 	[in,out] The kernel time. </param>
/// <param name="hyperq">		 	True to hyperq. </param>
/// <param name="numStreams">	 	Number of streams. </param>
///
/// <returns>	The calculated path. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int calc_path(int *gpuWall, int *gpuResult[2], int rows,
                    int cols, int pyramid_height,
                    int blockCols, int borderCols,
                    double &kernelTime, bool hyperq, int numStreams) {
  sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
  sycl::range<3> dimGrid(1, 1, blockCols);

  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();
  float elapsedTime;

  dpct::queue_ptr streams[numStreams];
  for (int s = 0; s < numStreams; s++) {
    streams[s] = dpct::get_current_device().create_queue();
  }
  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    for (int s = 0; s < numStreams; s++) {
      int temp = src;
      src = dst;
      dst = temp;

    if(hyperq) {
      if (t == 0 && s == 0) {
        /*
        DPCT1012:869: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        start_ct1 = std::chrono::steady_clock::now();
      }
      /*
      DPCT1049:175: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
                        *stop = streams[s]->submit([&](sycl::handler &cgh) {
                              /*
                              DPCT1101:1067: 'BLOCK_SIZE' expression was
                              replaced with a value. Modify the code to use the
                              original expression, provided in comments, if it
                              is correct.
                              */
                              sycl::local_accessor<int, 1> prev_acc_ct1(
                                  sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);
                              /*
                              DPCT1101:1068: 'BLOCK_SIZE' expression was
                              replaced with a value. Modify the code to use the
                              original expression, provided in comments, if it
                              is correct.
                              */
                              sycl::local_accessor<int, 1> result_acc_ct1(
                                  sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

                              auto gpuResult_src_ct2 = gpuResult[src];
                              auto gpuResult_dst_ct3 = gpuResult[dst];

                              cgh.parallel_for(
                                  sycl::nd_range<3>(dimGrid * dimBlock,
                                                    dimBlock),
                                  [=](sycl::nd_item<3> item_ct1) {
                                        dynproc_kernel(
                                            MIN(pyramid_height, rows - t - 1),
                                            gpuWall, gpuResult_src_ct2,
                                            gpuResult_dst_ct3, cols, rows, t,
                                            borderCols, item_ct1,
                                            prev_acc_ct1.get_pointer(),
                                            result_acc_ct1.get_pointer());
                                  });
                        });
      if (t + pyramid_height >= rows - 1 && s == numStreams - 1) {
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1012:870: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        stop->wait();
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_ERROR();
        elapsedTime =
            std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                .count();
        kernelTime += elapsedTime * 1.e-3;
      }
    } else {
      /*
      DPCT1012:871: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      start_ct1 = std::chrono::steady_clock::now();
      /*
      DPCT1049:176: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
                        *stop = dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  /*
                                  DPCT1101:1069: 'BLOCK_SIZE' expression was
                                  replaced with a value. Modify the code to use
                                  the original expression, provided in comments,
                                  if it is correct.
                                  */
                                  sycl::local_accessor<int, 1> prev_acc_ct1(
                                      sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);
                                  /*
                                  DPCT1101:1070: 'BLOCK_SIZE' expression was
                                  replaced with a value. Modify the code to use
                                  the original expression, provided in comments,
                                  if it is correct.
                                  */
                                  sycl::local_accessor<int, 1> result_acc_ct1(
                                      sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

                                  auto gpuResult_src_ct2 = gpuResult[src];
                                  auto gpuResult_dst_ct3 = gpuResult[dst];

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(dimGrid * dimBlock,
                                                        dimBlock),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            dynproc_kernel(
                                                MIN(pyramid_height,
                                                    rows - t - 1),
                                                gpuWall, gpuResult_src_ct2,
                                                gpuResult_dst_ct3, cols, rows,
                                                t, borderCols, item_ct1,
                                                prev_acc_ct1.get_pointer(),
                                                result_acc_ct1.get_pointer());
                                      });
                            });
      dpct::get_current_device().queues_wait_and_throw();
      /*
      DPCT1012:872: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      stop->wait();
      stop_ct1 = std::chrono::steady_clock::now();
      CHECK_CUDA_ERROR();
      elapsedTime =
          std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
              .count();
      kernelTime += elapsedTime * 1.e-3;
    }
    }
  }
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Runs the calc_path kernel.
/// 			added UVM and hyperQ support </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="borderCols">   	The border cols. </param>
/// <param name="smallBlockCol">	The small block cols. </param>
/// <param name="blockCols">		The block cols. </param>
/// <param name="resultDB">			[in,out] The result database. </param>
/// <param name="op">				[in,out] The options. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(int borderCols, int smallBlockCol, int blockCols,
         ResultDatabase &resultDB, OptionParser &op) {
  // initialize data
  init(op);
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

  int *gpuWall, *gpuResult[2];
  int size = rows * cols;

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    gpuResult[0] = data;
    /*
    DPCT1064:893: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(gpuResult[1] = sycl::malloc_shared<int>(
                                         cols, dpct::get_default_queue())));
    /*
    DPCT1064:894: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(gpuWall = sycl::malloc_shared<int>(
                             (size - cols), dpct::get_default_queue())));
    // gpuWall = data + cols;
  } else {
    /*
    DPCT1064:895: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(gpuResult[0] = sycl::malloc_device<int>(
                                         cols, dpct::get_default_queue())));
    /*
    DPCT1064:896: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(gpuResult[1] = sycl::malloc_device<int>(
                                         cols, dpct::get_default_queue())));
    /*
    DPCT1064:897: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(gpuWall = sycl::malloc_device<int>(
                             (size - cols), dpct::get_default_queue())));
  }

  // Cuda events and times
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;

  /*
  DPCT1012:873: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:874: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  start_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);

  if (uvm) {
    // do nothing
  } else if (uvm_advise) {
    /*
    DPCT1063:875: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuResult[0], sizeof(int) * cols, 0)));
    /*
    DPCT1063:876: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuWall, sizeof(int) * (size - cols), 0)));
    /*
    DPCT1063:877: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuWall, sizeof(int) * (size - cols), 0)));
  } else if (uvm_prefetch) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                             .get_device(device_id)
                             .default_queue()
                             .prefetch(gpuResult[0], sizeof(int) * cols)));
    dpct::queue_ptr s1;
    checkCudaErrors(
        DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
    checkCudaErrors(
        DPCT_CHECK_ERROR(s1->prefetch(gpuWall, sizeof(int) * (size - cols))));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
  } else if (uvm_prefetch_advise) {
    /*
    DPCT1063:878: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuResult[0], sizeof(int) * cols, 0)));
    /*
    DPCT1063:879: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuWall, sizeof(int) * (size - cols), 0)));
    /*
    DPCT1063:880: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_device(device_id).default_queue().mem_advise(
            gpuWall, sizeof(int) * (size - cols), 0)));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                             .get_device(device_id)
                             .default_queue()
                             .prefetch(gpuResult[0], sizeof(int) * cols)));
    dpct::queue_ptr s1;
    checkCudaErrors(
        DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
    checkCudaErrors(
        DPCT_CHECK_ERROR(s1->prefetch(gpuWall, sizeof(int) * (size - cols))));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
  } else {
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(gpuResult[0], data, sizeof(int) * cols)
                             .wait()));

    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(gpuWall, data + cols, sizeof(int) * (size - cols))
            .wait()));
  }

  /*
  DPCT1012:881: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:882: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  stop_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);
  checkCudaErrors(0);
  checkCudaErrors(DPCT_CHECK_ERROR(
      (elapsedTime =
           std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
               .count())));
  transferTime += elapsedTime * 1.e-3;  // convert to seconds

  int instances = op.getOptionInt("instances");

#ifdef HYPERQ
  double hyperqKernelTime = 0;
  /// <summary>	Calc the path with hyperQ enabled. </summary>
  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols,
            borderCols, hyperqKernelTime, true, instances);
#else
  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols,
                borderCols, kernelTime, false, instances);
#endif

  /*
  DPCT1012:883: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:884: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  start_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);

  if (uvm) {
    result = gpuResult[final_ret];
  } else if (uvm_advise) {
    result = gpuResult[final_ret];
    /*
    DPCT1063:885: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
            gpuResult[final_ret], sizeof(int) * cols, 0)));
    /*
    DPCT1063:886: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
            gpuResult[final_ret], sizeof(int) * cols, 0)));
  } else if (uvm_prefetch) {
    result = gpuResult[final_ret];
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
            result, sizeof(int) * cols)));
  } else if (uvm_prefetch_advise) {
    result = gpuResult[final_ret];
    /*
    DPCT1063:887: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
            gpuResult[final_ret], sizeof(int) * cols, 0)));
    /*
    DPCT1063:888: Advice parameter is device-defined and was set to 0. You may
    need to adjust it.
    */
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().mem_advise(
            gpuResult[final_ret], sizeof(int) * cols, 0)));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
            result, sizeof(int) * cols)));
  } else {
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(result, gpuResult[final_ret], sizeof(int) * cols)
            .wait()));
  }

  /*
  DPCT1012:889: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:890: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  stop_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);
  checkCudaErrors(0);
  checkCudaErrors(DPCT_CHECK_ERROR(
      (elapsedTime =
           std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
               .count())));
  transferTime += elapsedTime * 1.e-3;  // convert to seconds

  /// <summary>	Output the results to a file. </summary>
  string outfile = op.getOptionString("outputFile");
  if (!outfile.empty()) {
    std::fstream fs;
    fs.open(outfile.c_str(), std::fstream::app);
    fs << "***DATA***" << std::endl;
    for (int i = 0; i < cols; i++) {
      fs << data[i] << " ";
    }
    fs << std::endl;
    fs << "***RESULT***" << std::endl;
    for (int i = 0; i < cols; i++) {
      fs << result[i] << " ";
    }
    fs << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// <summary>	cleanup. </summary>
  ///
  /// <remarks>	Ed, 5/20/2020. </remarks>
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // cudaFree(gpuWall);
  sycl::free(gpuResult[0], dpct::get_default_queue());
  sycl::free(gpuResult[1], dpct::get_default_queue());

if (!uvm && !uvm_advise && !uvm_prefetch && !uvm_prefetch_advise) {
  delete[] data;
  delete[] wall;
  delete[] result;
}

  string atts = toString(rows) + "x" + toString(cols);
#ifdef HYPERQ
  /// <summary>	The result db. add result. </summary>
  resultDB.AddResult("pathfinder_hyperq_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("pathfinder_hyperq_kernel_time", atts, "sec", hyperqKernelTime);
  resultDB.AddResult("pathfinder_hyperq_total_time", atts, "sec", hyperqKernelTime + transferTime);
  resultDB.AddResult("pathfinder_hyperq_parity", atts, "N",
                     transferTime / hyperqKernelTime);
  resultDB.AddResult("pathfinder_hyperq_speedup", atts, "sec",
                     kernelTime/hyperqKernelTime);
#else
  resultDB.AddResult("pathfinder_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("pathfinder_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("pathfinder_total_time", atts, "sec", kernelTime + transferTime);
  resultDB.AddResult("pathfinder_parity", atts, "N",
                     transferTime / kernelTime);

#endif
  resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
