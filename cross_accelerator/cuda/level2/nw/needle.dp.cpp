////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\nw\needle.cu
//
// summary:	Needle class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <chrono>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "needle.h"
#include "needle_kernel.dp.cpp"

#define LIMIT -999
#define SEED  7

int max_rows, max_cols, penalty;

void runTest(ResultDatabase &resultDB, OptionParser &op, size_t device_idx);

/// <summary>	The blosum 62[24][24]. </summary>
int blosum62[24][24] = { { 4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                           -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4 },
                         { -1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                           -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4 },
                         { -2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                           -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4 },
                         { -2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                           -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4 },
                         { 0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                           -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4 },
                         { -1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                           0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4 },
                         { -1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                           -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4 },
                         { 0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                           -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4 },
                         { -2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                           -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4 },
                         { -1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                           1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4 },
                         { -1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                           2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4 },
                         { -1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                           -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4 },
                         { -1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                           5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4 },
                         { -2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                           0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4 },
                         { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                           -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4 },
                         { 1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                           -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4 },
                         { 0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                           -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4 },
                         { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                           -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4 },
                         { -2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                           -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4 },
                         { 0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                           1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4 },
                         { -2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                           -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4 },
                         { -1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                           -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4 },
                         { 0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4 },
                         { -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                           -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1 } };

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Add benchmark specific options parsing.  The user is allowed to
/// specify the size of the input data in kiB.. </summary>
///
/// <remarks>	Ed, 5/20/2020.
/// 			Anthony Danalis, 9/08, 2009
///
/// <param name="op">	[in,out] the options parser / parameter database.
/// </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("dimensions", OPT_INT, "0", "dimensions");
    op.addOption("penalty", OPT_INT, "10", "penalty");
    op.addOption("resultsfile", OPT_STRING, "", "file to write results to");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020.
/// 			Kyle Spafford, 8/13/2009 </remarks>
///
/// <param name="resultDB">	[in,out] results from the benchmark are stored
/// in
/// this db. </param> <param name="op">	   	[in,out] the options parser /
/// parameter database. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running Needleman-Wunsch\n");

    bool quiet = op.getOptionBool("quiet");
    int  dim   = op.getOptionInt("dimensions");
    penalty    = op.getOptionInt("penalty");

    if (penalty < 0)
    {
        printf("Error: Penalty must be a positive number.\n");
        return;
    }
    if (dim < 0)
    {
        printf("Error: Dimensions must be positive.\n");
        return;
    }

    if (dim == 0)
    {
        int problemSizes[4] = { 100, 1000, 6000, 40000 };
        dim                 = problemSizes[op.getOptionInt("size") - 1];
    }

    long long num_items = (long long)dim * (long long)dim;
    if (num_items >= INT_MAX)
    {
        printf("Error: Total size cannot exceed INT_MAX");
        return;
    }

    if (!quiet)
    {
        printf("WG size of kernel = %d \n", BLOCK_SIZE);
        printf("Max Rows x Cols: %dx%d\n", dim, dim);
        printf("Penalty: %d\n\n", penalty);
    }
    srand(SEED);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        if (!quiet)
            printf("Pass %d: ", i);

        max_rows = dim;
        max_cols = dim;
        runTest(resultDB, op, device_idx);
        if (!quiet)
            printf("Done.\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
runTest(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    bool quiet = op.getOptionBool("quiet");

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;

    int *referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(referrence);
    int *input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(input_itemsets);
    int *output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(output_itemsets);

    if (!input_itemsets)
    {
        fprintf(stderr, "Error: Can not allocate memory\n");
        exit(0);
    }

    for (int i = 0; i < max_cols; i++)
        for (int j = 0; j < max_rows; j++)
            input_itemsets[i * max_cols + j] = 0;

    for (int i = 1; i < max_rows; i++) // please define your own sequence.
        input_itemsets[i * max_cols] = i % 10 + 1;

    for (int j = 1; j < max_cols; j++) // please define your own sequence.
        input_itemsets[j] = j % 10 + 1;

    for (int i = 1; i < max_cols; i++)
        for (int j = 1; j < max_rows; j++)
            referrence[i * max_cols + j]
                = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];

    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j * penalty;

    int size = max_cols * max_rows;

    int *referrence_cuda = sycl::malloc_device<int>(size, queue);
    int *matrix_cuda     = sycl::malloc_device<int>(size, queue);

    std::chrono::time_point<std::chrono::high_resolution_clock> start_ct1;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_ct1;

    float  elapsedTime;
    double transferTime = 0;
    double kernelTime   = 0;

    start_ct1 = std::chrono::high_resolution_clock::now();
    queue.memcpy(referrence_cuda, referrence, sizeof(int) * size);
    queue.memcpy(matrix_cuda, input_itemsets, sizeof(int) * size);
    queue.wait_and_throw();
    stop_ct1    = std::chrono::high_resolution_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    sycl::range<3> dimGrid(1, 1, 1);
    sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
    int            block_width = (max_cols - 1) / BLOCK_SIZE;

    // process top-left matrix
    for (int i = 1; i <= block_width; i++)
    {
        dimGrid[2] = i;
        dimGrid[1] = 1;

        sycl::event k1_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<int,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(sycl::range<2>(BLOCK_SIZE + 1,
                                            BLOCK_SIZE + 1),
                             cgh);
            sycl::accessor<int,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                ref_acc_ct1(
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            auto max_cols_ct2 = max_cols;
            auto penalty_ct3  = penalty;

            cgh.parallel_for<class nd_step_1>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    needle_cuda_shared_1(referrence_cuda,
                                         matrix_cuda,
                                         max_cols_ct2,
                                         penalty_ct3,
                                         i,
                                         block_width,
                                         item_ct1,
                                         temp_acc_ct1.get_pointer(),
                                         ref_acc_ct1.get_pointer());
                });
        });
        k1_event.wait();
        elapsedTime = k1_event.get_profiling_info<
                          sycl::info::event_profiling::command_end>()
                      - k1_event.get_profiling_info<
                          sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;
    }
    // process bottom-right matrix
    for (int i = block_width - 1; i >= 1; i--)
    {
        dimGrid[2] = i;
        dimGrid[1] = 1;

        sycl::event k2_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<int,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(sycl::range<2>(BLOCK_SIZE + 1,
                                            BLOCK_SIZE + 1),
                             cgh);
            sycl::accessor<int,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                ref_acc_ct1(
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            auto max_cols_ct2 = max_cols;
            auto penalty_ct3  = penalty;

            cgh.parallel_for<class nd_step_2>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    needle_cuda_shared_2(referrence_cuda,
                                         matrix_cuda,
                                         max_cols_ct2,
                                         penalty_ct3,
                                         i,
                                         block_width,
                                         item_ct1,
                                         temp_acc_ct1.get_pointer(),
                                         ref_acc_ct1.get_pointer());
                });
        });
        k2_event.wait();
        elapsedTime = k2_event.get_profiling_info<
                          sycl::info::event_profiling::command_end>()
                      - k2_event.get_profiling_info<
                          sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;
    }

    start_ct1 = std::chrono::high_resolution_clock::now();
    queue.memcpy(output_itemsets, matrix_cuda, sizeof(int) * size).wait();
    stop_ct1    = std::chrono::high_resolution_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    string outfile = op.getOptionString("outputFile");
    if (outfile != "")
    {
        FILE *fpo = fopen(outfile.c_str(), "w");
        if (!quiet)
        {
            fprintf(fpo, "Print traceback value GPU to %s:\n", outfile.c_str());
        }

        for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
        {
            int nw, n, w, traceback;
            if (i == max_rows - 2 && j == max_rows - 2)
            {
                // print the first element
                fprintf(fpo, "%d ", output_itemsets[i * max_cols + j]);
            }
            if (i == 0 && j == 0)
                break;
            if (i > 0 && j > 0)
            {
                nw = output_itemsets[(i - 1) * max_cols + j - 1];
                w  = output_itemsets[i * max_cols + j - 1];
                n  = output_itemsets[(i - 1) * max_cols + j];
            }
            else if (i == 0)
            {
                nw = n = LIMIT;
                w      = output_itemsets[i * max_cols + j - 1];
            }
            else if (j == 0)
            {
                nw = w = LIMIT;
                n      = output_itemsets[(i - 1) * max_cols + j];
            }

            // traceback = maximum(nw, w, n);
            int new_nw, new_w, new_n;
            new_nw = nw + referrence[i * max_cols + j];
            new_w  = w - penalty;
            new_n  = n - penalty;

            traceback = maximum(new_nw, new_w, new_n);
            if (traceback == new_nw)
                traceback = nw;
            if (traceback == new_w)
                traceback = w;
            if (traceback == new_n)
                traceback = n;

            fprintf(fpo, "%d ", traceback);
            if (traceback == nw)
            {
                i--;
                j--;
                continue;
            }
            else if (traceback == w)
            {
                j--;
                continue;
            }
            else if (traceback == n)
            {
                i--;
                continue;
            }
            else
            {
            }
        }
        fclose(fpo);
    }

    // Cleanup memory
    sycl::free(referrence_cuda, queue);
    sycl::free(matrix_cuda, queue);
    free(referrence);
    free(input_itemsets);
    free(output_itemsets);

    char tmp[32];
    sprintf(tmp, "%ditems", size);
    string atts = string(tmp);
    resultDB.AddResult("NW-TransferTime", atts, "sec", transferTime);
    resultDB.AddResult("NW-KernelTime", atts, "sec", kernelTime);
    resultDB.AddResult("NW-TotalTime", atts, "sec", transferTime + kernelTime);
    resultDB.AddResult("NW-Rate_Parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}
