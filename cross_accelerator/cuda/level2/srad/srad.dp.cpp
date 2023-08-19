////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\srad\srad.cu
//
// summary:	Srad class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <chrono>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "srad.h"
#include "srad_kernel.dp.cpp"

float                                              kernelTime   = 0.0f;
float                                              transferTime = 0.0f;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float                                              elapsed;
float                                             *check;

#ifdef _FPGA
#define ATTRIBUTE                                            \
    [[sycl::reqd_work_group_size(1, BLOCK_SIZE, BLOCK_SIZE), \
      intel::max_work_group_size(1, BLOCK_SIZE, BLOCK_SIZE)]]
#else
#define ATTRIBUTE
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

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
/// <summary>	Srads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix.
/// </param> <param name="imageSize">  	Size of the image. </param> <param
/// name="speckleSize">	Size of the speckle. </param> <param name="iters">
/// The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad(ResultDatabase &resultDB,
           OptionParser   &op,
           float          *matrix,
           int             imageSize,
           int             speckleSize,
           int             iters,
           size_t          device_idx);

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
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("speckleSize", OPT_INT, "0", "speckle height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm");
    op.addOption("gen_input", OPT_BOOL, "0", "create input file for given size");
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
    printf("Running SRAD\n");

    srand(SEED);
    bool quiet = op.getOptionBool("quiet");

    // set parameters
    int imageSize   = op.getOptionInt("imageSize");
    int speckleSize = op.getOptionInt("speckleSize");
    int iters       = op.getOptionInt("iterations");
    if (imageSize == 0 || speckleSize == 0 || iters == 0)
    {
        int imageSizes[5] = { 128, 512, 4096, 8192, 16384 };
        int iterSizes[5]  = { 5, 1, 15, 20, 40 };
        imageSize         = imageSizes[op.getOptionInt("size") - 1];
        speckleSize       = imageSize / 2;
        iters             = iterSizes[op.getOptionInt("size") - 1];
    }

    if (!quiet)
    {
        printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
        printf("Image Size: %d x %d\n", imageSize, imageSize);
        printf("Speckle size: %d x %d\n", speckleSize, speckleSize);
        printf("Num Iterations: %d\n\n", iters);
    }

    bool gen_input = op.getOptionBool("gen_input");
    int rows = imageSize; // number of rows in the domain
    int cols = imageSize; // number of cols in the domain
    if (gen_input)
    {
        std::ofstream ostrm("input.txt");
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ostrm << rand() / (float)RAND_MAX << '\n';
    }

    // run workload
    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        float *matrix = (float *)malloc(imageSize * imageSize * sizeof(float));
        assert(matrix);

        string        inputFile = op.getOptionString("inputFile");
        std::ifstream file(inputFile.c_str());

        if (inputFile != "")
        {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    float val;
                    file >> val;
                    matrix[i * cols + j] = val;
                }
        }
        else
        {
            random_matrix(matrix, imageSize, imageSize);
        }

        if (!quiet)
            printf("Pass %d:\n", i);

        float time = srad(
            resultDB, op, matrix, imageSize, speckleSize, iters, device_idx);
        if (!quiet)
            printf("Running SRAD...Done.\n");

        free(matrix);
    }
}

float
srad(ResultDatabase &resultDB,
     OptionParser   &op,
     float          *matrix,
     int             imageSize,
     int             speckleSize,
     int             iters,
     size_t          device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    kernelTime   = 0.0f;
    transferTime = 0.0f;

    int rows = imageSize; // number of rows in the domain
    int cols = imageSize; // number of cols in the domain
    if ((rows % 16 != 0) || (cols % 16 != 0))
    {
        fprintf(stderr, "rows and cols must be multiples of 16\n");
        exit(1);
    }

    unsigned int r1     = 0;           // y1 position of the speckle
    unsigned int r2     = speckleSize; // y2 position of the speckle
    unsigned int c1     = 0;           // x1 position of the speckle
    unsigned int c2     = speckleSize; // x2 position of the speckle
    float        lambda = 0.5;         // Lambda value
    int          niter  = iters;       // number of iterations

    int size_I = cols * rows;
    int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    float *I = (float *)malloc(size_I * sizeof(float));
    assert(I);
    float *J = (float *)malloc(size_I * sizeof(float));
    assert(J);
    float *c = (float *)malloc(sizeof(float) * size_I);
    assert(c);

    // Allocate device memory
    float *J_cuda = sycl::malloc_device<float>(size_I, queue);
    float *C_cuda = sycl::malloc_device<float>(size_I, queue);
    float *E_C    = sycl::malloc_device<float>(size_I, queue);
    float *W_C    = sycl::malloc_device<float>(size_I, queue);
    float *S_C    = sycl::malloc_device<float>(size_I, queue);
    float *N_C    = sycl::malloc_device<float>(size_I, queue);

    // copy random matrix
    memcpy(I, matrix, rows * cols * sizeof(float));

    for (int k = 0; k < size_I; k++)
        J[k] = (float)exp(I[k]);

    float sum, sum2, tmp;
    for (int iter = 0; iter < niter; iter++)
    {
        sum  = 0;
        sum2 = 0;
        for (int i = r1; i <= r2; i++)
            for (int j = c1; j <= c2; j++)
            {
                tmp = J[i * cols + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }

        float meanROI = sum / size_R;
        float varROI  = (sum2 / size_R) - meanROI * meanROI;
        float q0sqr   = varROI / (meanROI * meanROI);

        // Currently the input size must be divided by 16 - the block size
        int block_x = cols / BLOCK_SIZE;
        int block_y = rows / BLOCK_SIZE;

        sycl::range<2> dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        sycl::range<2> dimGrid(block_y, block_x);

        // Copy data from main memory to device memory
        start_ct1 = std::chrono::steady_clock::now();
        queue.memcpy(J_cuda, J, sizeof(float) * size_I).wait();
        stop_ct1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
        transferTime += elapsed * 1.e-3;

        // Run kernels
        queue.wait_and_throw();
        sycl::event k1_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                temp_result_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE),
                                    cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                north_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                south_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                east_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                west_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for<class srad_fpga_1>(
                sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<2> item_ct1) ATTRIBUTE {
                    srad_cuda_1(E_C,
                                W_C,
                                N_C,
                                S_C,
                                J_cuda,
                                C_cuda,
                                cols,
                                rows,
                                q0sqr,
                                item_ct1,
                                temp_acc_ct1.get_pointer(),
                                temp_result_acc_ct1.get_pointer(),
                                north_acc_ct1.get_pointer(),
                                south_acc_ct1.get_pointer(),
                                east_acc_ct1.get_pointer(),
                                west_acc_ct1.get_pointer());
                });
        });
        k1_event.wait();
        elapsed = k1_event.get_profiling_info<
                      sycl::info::event_profiling::command_end>()
                  - k1_event.get_profiling_info<
                      sycl::info::event_profiling::command_start>();
        kernelTime += elapsed * 1.e-9;

        start_ct1            = std::chrono::steady_clock::now();
        sycl::event k2_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                south_c_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                east_c_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                c_cuda_temp_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE),
                                    cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                c_cuda_result_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE),
                                      cgh);
            sycl::accessor<float,
                           2,
                           sycl::access_mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for<class srad_fpga_2>(
                sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<2> item_ct1) ATTRIBUTE {
                    srad_cuda_2(E_C,
                                W_C,
                                N_C,
                                S_C,
                                J_cuda,
                                C_cuda,
                                cols,
                                rows,
                                lambda,
                                q0sqr,
                                item_ct1,
                                south_c_acc_ct1,
                                east_c_acc_ct1,
                                c_cuda_temp_acc_ct1,
                                c_cuda_result_acc_ct1,
                                temp_acc_ct1);
                });
        });
        k2_event.wait();
        elapsed = k2_event.get_profiling_info<
                      sycl::info::event_profiling::command_end>()
                  - k2_event.get_profiling_info<
                      sycl::info::event_profiling::command_start>();
        kernelTime += elapsed * 1.e-9;

        // Copy data from device memory to main memory
        start_ct1 = std::chrono::steady_clock::now();
        queue.memcpy(J, J_cuda, sizeof(float) * size_I).wait();
        stop_ct1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
        transferTime += elapsed * 1.e-3;
    }

    char atts[1024];
    sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
    resultDB.AddResult("srad_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("srad_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult(
        "srad_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult("srad_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

    string outfile = op.getOptionString("outputFile");
    if (!outfile.empty())
    {
        // Printing output
        if (!op.getOptionBool("quiet"))
            printf("Writing output to %s\n", outfile.c_str());

        FILE *fp = NULL;
        fp       = fopen(outfile.c_str(), "w");
        if (!fp)
        {
            printf("Error: Unable to write to file %s\n", outfile.c_str());
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                    fprintf(fp, "%.5f ", J[i * cols + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
    // write results to validate with srad_gridsync
    check = (float *)malloc(sizeof(float) * size_I);
    assert(check);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            check[i * cols + j] = J[i * cols + j];

    free(I);
    free(J);
    free(c);
    sycl::free(C_cuda, queue);
    sycl::free(J_cuda, queue);
    sycl::free(E_C, queue);
    sycl::free(W_C, queue);
    sycl::free(N_C, queue);
    sycl::free(S_C, queue);

    return kernelTime;
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

void
random_matrix(float *I, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            I[i * cols + j] = rand() / (float)RAND_MAX;
}
