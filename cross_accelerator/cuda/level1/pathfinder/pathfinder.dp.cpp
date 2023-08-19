////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	\altis\src\cuda\level1\pathfinder\pathfinder.cu
//
// summary:	Pathfinder class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
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

#define BLOCK_SIZE 256 // For P630 this needs to be <= 256
#define STR_SIZE   256
#define HALO \
    1 // halo width along one direction when advancing to the next iteration
#define SEED 7

#ifdef _FPGA
#define ATTRIBUTE                                          \
        [[sycl::reqd_work_group_size(1, 1, BLOCK_SIZE),    \
          intel::max_work_group_size(1, 1, BLOCK_SIZE)]] 
#else
#define ATTRIBUTE
#endif

void run(int             borderCols,
         int             smallBlockCol,
         int             blockCols,
         ResultDatabase &resultDB,
         OptionParser   &op,
         size_t          device_idx);

int   rows, cols;
int  *data;
int **wall;
int  *result;
int   pyramid_height;

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
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("rows", OPT_INT, "0", "number of rows");
    op.addOption("cols", OPT_INT, "0", "number of cols");
    op.addOption("pyramidHeight", OPT_INT, "0", "pyramid height");
    op.addOption(
        "instances", OPT_INT, "32", "number of pathfinder instances to run");
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
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running Pathfinder\n");

    bool quiet         = op.getOptionInt("quiet");
    int  rowLen        = op.getOptionInt("rows");
    int  colLen        = op.getOptionInt("cols");
    int  pyramidHeight = op.getOptionInt("pyramidHeight");

    if (rowLen == 0 || colLen == 0 || pyramidHeight == 0)
    {
        printf("Using preset problem size %d\n", (int)op.getOptionInt("size"));
        int rowSizes[5]     = { 8, 16, 32, 40, 48 };
        int colSizes[5]     = { 8, 16, 32, 40, 48 };
        int pyramidSizes[5] = { 4, 8, 16, 32, 36 };
        rows                = rowSizes[op.getOptionInt("size") - 1] * 1024;
        cols                = colSizes[op.getOptionInt("size") - 1] * 1024;
        pyramid_height      = pyramidSizes[op.getOptionInt("size") - 1];
    }
    else
    {
        rows           = rowLen;
        cols           = colLen;
        pyramid_height = pyramidHeight;
    }

    if (!quiet)
    {
        printf("Row length: %d\n", rows);
        printf("Column length: %d\n", cols);
        printf("Pyramid height: %d\n", pyramid_height);
    }

    /* --------------- pyramid parameters --------------- */
    int borderCols    = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols
        = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    if (!quiet)
    {
        printf(
            "gridSize: "
            "[%d],border:[%d],blockSize:[%d],blockGrid:[%d],targetBlock:[%d]\n",
            cols,
            borderCols,
            BLOCK_SIZE,
            blockCols,
            smallBlockCol);
    }

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        if (!quiet)
            printf("Pass %d: ", i);

        run(borderCols, smallBlockCol, blockCols, resultDB, op, device_idx);
        if (!quiet)
            printf("Done.\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes parameters for computing.  </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="op">	[in,out] The option specified by the user. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
init(OptionParser &op, sycl::queue& queue)
{
    const bool uvm                 = op.getOptionBool("uvm");
    const bool uvm_advise          = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch        = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        ::data = (int *)sycl::malloc_shared(sizeof(int) * rows * cols, queue);
        wall = sycl::malloc_shared<int *>(rows, queue);
        for (int n = 0; n < rows; n++)
            wall[n] = ::data + (int)cols * n;
    }
    else
    {
        ::data = new int[rows * cols];
        wall = new int *[rows];
        for (int n = 0; n < rows; n++)
            wall[n] = ::data + (int)cols * n;
        result = new int[cols];
    }

    srand(SEED);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[i][j] = rand() % 10;
    string outfile = op.getOptionString("outputFile");
    if (outfile != "")
    {
        std::fstream fs;
        fs.open(outfile.c_str(), std::fstream::in);
        fs.close();
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b)                ((a) <= (b) ? (a) : (b))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Dynproc kernel. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 1/7/2021. </remarks>
///
/// <param name="iteration"> 	The iteration. </param>
/// <param name="gpuWall">   	[in,out] If non-null, the GPU wall. </param>
/// <param name="gpuSrc">	 	[in,out] If non-null, the GPU source.
/// </param> <param name="gpuResults">	[in,out] If non-null, the GPU results.
/// </param> <param name="cols">		 	The cols. </param>
/// <param name="rows">		 	The rows. </param>
/// <param name="startStep"> 	The start step. </param>
/// <param name="border">	 	The border. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
dynproc_kernel(int              iteration,
               int             *gpuWall,
               int             *gpuSrc,
               int             *gpuResults,
               int              cols,
               int              rows,
               int              startStep,
               int              border,
               sycl::nd_item<3> item_ct1,
               int             *prev,
               int             *result)
{

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
    int blkX    = (int)small_block_cols * (int)bx - (int)border;
    int blkXmax = blkX + (int)BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int xidx = blkX + (int)tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > (int)cols - 1)
                        ? (int)BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1)
                        : (int)BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, (int)cols - 1))
        prev[tx] = gpuSrc[xidx];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    bool computed;
    for (int i = 0; i < iteration; i++)
    {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid)
        {
            computed     = true;
            int left     = prev[W];
            int up       = prev[tx];
            int right    = prev[E];
            int shortest = MIN(left, up);
            shortest     = MIN(shortest, right);
            int index    = cols * (startStep + i) + xidx;
            result[tx]   = shortest + gpuWall[index];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (i == iteration - 1)
            break;
        if (computed) // Assign the computation range
            prev[tx] = result[tx];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed)
        gpuResults[xidx] = result[tx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the path. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="gpuWall">		 	[in,out] If non-null, the GPU
/// wall.
/// </param> <param name="gpuResult">	 	[in,out] If non-null, the GPU
/// result. </param>
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

int
calc_path(int    *gpuWall,
          int    *gpuResult[2],
          int     rows,
          int     cols,
          int     pyramid_height,
          int     blockCols,
          int     borderCols,
          double &kernelTime,
          bool    hyperq,
          int     numStreams,
          sycl::queue& queue)
{
    sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, 1, blockCols);

    sycl::event                                        start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float                                              elapsedTime;

    sycl::queue *streams[numStreams];
    for (int s = 0; s < numStreams; s++)
        streams[s] = dpct::get_current_device().create_queue();
    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height)
    {
        for (int s = 0; s < numStreams; s++)
        {
            int temp = src;
            src      = dst;
            dst      = temp;

            if (hyperq)
            {
                if (t == 0 && s == 0)
                    start_ct1 = std::chrono::steady_clock::now();
                
                stop = streams[s]->submit([&](sycl::handler &cgh) {
                    sycl::accessor<int,
                                   1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        prev_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<int,
                                   1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        result_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                    auto gpuResult_src_ct2 = gpuResult[src];
                    auto gpuResult_dst_ct3 = gpuResult[dst];

                    cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                        [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
                            dynproc_kernel(MIN(pyramid_height, rows - t - 1),
                                           gpuWall,
                                           gpuResult_src_ct2,
                                           gpuResult_dst_ct3,
                                           cols,
                                           rows,
                                           t,
                                           borderCols,
                                           item_ct1,
                                           prev_acc_ct1.get_pointer(),
                                           result_acc_ct1.get_pointer());
                        });
                });
                if (t + pyramid_height >= rows - 1 && s == numStreams - 1)
                {
                    dpct::get_current_device().queues_wait_and_throw();
                    stop.wait();
                    stop_ct1 = std::chrono::steady_clock::now();
                    elapsedTime = std::chrono::duration<float, std::milli>(
                                      stop_ct1 - start_ct1)
                                      .count();
                    kernelTime += elapsedTime * 1.e-3;
                }
            }
            else
            {
                sycl::event kernel_event
                    = queue.submit([&](sycl::handler &cgh) {
                          sycl::accessor<int,
                                         1,
                                         sycl::access_mode::read_write,
                                         sycl::access::target::local>
                              prev_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                          sycl::accessor<int,
                                         1,
                                         sycl::access_mode::read_write,
                                         sycl::access::target::local>
                              result_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                          auto gpuResult_src_ct2 = gpuResult[src];
                          auto gpuResult_dst_ct3 = gpuResult[dst];

                          cgh.parallel_for(
                              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                              [=](sycl::nd_item<3> item_ct1) ATTRIBUTE{
                                  dynproc_kernel(
                                      MIN(pyramid_height, rows - t - 1),
                                      gpuWall,
                                      gpuResult_src_ct2,
                                      gpuResult_dst_ct3,
                                      cols,
                                      rows,
                                      t,
                                      borderCols,
                                      item_ct1,
                                      prev_acc_ct1.get_pointer(),
                                      result_acc_ct1.get_pointer());
                              });
                      });
                kernel_event.wait();
                elapsedTime = kernel_event.get_profiling_info<
                                  sycl::info::event_profiling::command_end>()
                              - kernel_event.get_profiling_info<
                                  sycl::info::event_profiling::command_start>();
                kernelTime += elapsedTime * 1.e-9;
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
/// <param name="resultDB">			[in,out] The result database.
/// </param> <param name="op">				[in,out] The options.
/// </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
run(int             borderCols,
    int             smallBlockCol,
    int             blockCols,
    ResultDatabase &resultDB,
    OptionParser   &op,
    size_t          device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    // initialize data
    init(op, queue);
    
    const bool uvm                 = op.getOptionBool("uvm");
    const bool uvm_advise          = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch        = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    int *gpuWall, *gpuResult[2];
    int  size = rows * cols;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        gpuResult[0] = ::data;
        gpuResult[1] = sycl::malloc_shared<int>(cols, queue);
        gpuWall      = sycl::malloc_shared<int>((size - cols), queue);
        // gpuWall = data + cols;
    }
    else
    {
        gpuResult[0] = sycl::malloc_device<int>(cols, queue);
        gpuResult[1] = sycl::malloc_device<int>(cols, queue);
        gpuWall      = sycl::malloc_device<int>((size - cols), queue);
    }

    // Cuda events and times
    sycl::event                                        start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float                                              elapsedTime;
    double                                             transferTime = 0.;
    double                                             kernelTime   = 0;

    start_ct1 = std::chrono::steady_clock::now();
    if (uvm)
    {
        // do nothing
    }
    else if (uvm_advise)
    {
        queue.mem_advise(gpuResult[0], sizeof(int) * cols, 0);
        queue.mem_advise(gpuWall, sizeof(int) * (size - cols), 0);
        queue.mem_advise(gpuWall, sizeof(int) * (size - cols), 0);
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(gpuResult[0], sizeof(int) * cols);
        queue.prefetch(gpuWall, sizeof(int) * (size - cols));
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(gpuResult[0], sizeof(int) * cols, 0);
        queue.mem_advise(gpuWall, sizeof(int) * (size - cols), 0);
        queue.mem_advise(gpuWall, sizeof(int) * (size - cols), 0);
        queue.prefetch(gpuResult[0], sizeof(int) * cols);
        queue.prefetch(gpuWall, sizeof(int) * (size - cols));
    }
    else
    {
        queue.memcpy(gpuResult[0], ::data, sizeof(int) * cols).wait();
        queue.memcpy(gpuWall, ::data + cols, sizeof(int) * (size - cols)).wait();
    }

    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    int instances = op.getOptionInt("instances");

#ifdef HYPERQ
    double hyperqKernelTime = 0;
    /// <summary>	Calc the path with hyperQ enabled. </summary>
    int final_ret = calc_path(gpuWall,
                              gpuResult,
                              rows,
                              cols,
                              pyramid_height,
                              blockCols,
                              borderCols,
                              hyperqKernelTime,
                              true,
                              instances,
                              queue);
#else
    int final_ret = calc_path(gpuWall,
                              gpuResult,
                              rows,
                              cols,
                              pyramid_height,
                              blockCols,
                              borderCols,
                              kernelTime,
                              false,
                              instances,
                              queue);
#endif

    start_ct1 = std::chrono::steady_clock::now();
    if (uvm)
    {
        result = gpuResult[final_ret];
    }
    else if (uvm_advise)
    {
        result = gpuResult[final_ret];
        queue.mem_advise(gpuResult[final_ret], sizeof(int) * cols, 0);
    }
    else if (uvm_prefetch)
    {
        result = gpuResult[final_ret];
        queue.prefetch(result, sizeof(int) * cols);
    }
    else if (uvm_prefetch_advise)
    {
        result = gpuResult[final_ret];
        queue.mem_advise(gpuResult[final_ret], sizeof(int) * cols, 0);
        queue.prefetch(result, sizeof(int) * cols);
    }
    else
    {
        queue.memcpy(result, gpuResult[final_ret], sizeof(int) * cols).wait();
    }
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    /// <summary>	Output the results to a file. </summary>
    string outfile = op.getOptionString("outputFile");
    if (!outfile.empty())
    {
        std::fstream fs;
        fs.open(outfile.c_str(), std::fstream::app);
        fs << "***DATA***" << std::endl;
        for (int i = 0; i < cols; i++)
            fs << ::data[i] << " ";
        fs << std::endl;
        fs << "***RESULT***" << std::endl;
        for (int i = 0; i < cols; i++)
            fs << result[i] << " ";
        fs << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	cleanup. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // cudaFree(gpuWall);
    sycl::free(gpuResult[0], queue);
    sycl::free(gpuResult[1], queue);

    if (!uvm && !uvm_advise && !uvm_prefetch && !uvm_prefetch_advise)
    {
        delete[] ::data;
        delete[] wall;
        delete[] result;
    }

    string atts = toString(rows) + "x" + toString(cols);
#ifdef HYPERQ
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult(
        "pathfinder_hyperq_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult(
        "pathfinder_hyperq_kernel_time", atts, "sec", hyperqKernelTime);
    resultDB.AddResult("pathfinder_hyperq_total_time",
                       atts,
                       "sec",
                       hyperqKernelTime + transferTime);
    resultDB.AddResult(
        "pathfinder_hyperq_parity", atts, "N", transferTime / hyperqKernelTime);
    resultDB.AddResult("pathfinder_hyperq_speedup",
                       atts,
                       "sec",
                       kernelTime / hyperqKernelTime);
#else
    resultDB.AddResult("pathfinder_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("pathfinder_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult(
        "pathfinder_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult(
        "pathfinder_parity", atts, "N", transferTime / kernelTime);

#endif
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}
