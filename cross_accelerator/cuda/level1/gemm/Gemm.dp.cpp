////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	C:\Users\ed\source\repos\altis\src\cuda\level1\gemm\Gemm.cu
//
// summary:	Gemm class
//
// origin: SHOC (https://github.com/vetter/shocp)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"

#include "Utility.h"
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include "cudacommon.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#define SEED 7
/// <summary>	Length of the object field. </summary>
static const int FIELD_LENGTH = 128;

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void RunTest(string          testName,
             ResultDatabase &resultDB,
             OptionParser   &op,
             size_t          device_idx);

// origianlly don't need handle in v1 cublas

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gemm operation wrapper. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A T to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A T to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a T to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
inline void devGEMM(sycl::queue           *handle,
                    oneapi::mkl::transpose transa,
                    oneapi::mkl::transpose transb,
                    int                    m,
                    int                    n,
                    int                    k,
                    const T               *alpha,
                    const T               *A,
                    int                    lda,
                    const T               *B,
                    int                    ldb,
                    const T               *beta,
                    T                     *C,
                    int                    ldc);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Filling memory. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">   	[in,out] If non-null,  pointer to the array to
/// initialize. </param> <param name="n">   number of elements in the array.
/// </param> <param name="maxi">	The maxi. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void
fill(T *A, int n, int maxi)
{
    for (int j = 0; j < n; j++)
    {
        if constexpr (std::is_same<T, float>::value
                      || std::is_same<T, double>::value)
            A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / T(maxi + 1.);
        else if (std::is_same<T, sycl::half>::value)
            A[j]
                = sycl::vec<float, 1> { float((rand() % (maxi * 2 + 1)) - maxi)
                                        / (maxi + 1.) }
                      .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads a matrix. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">	   	[in,out] If non-null, pointer to matrix A.
/// </param>
/// <param name="B">	   	[in,out] If non-null, pointer to matrix B.
/// </param>
/// <param name="C">	   	[in,out] If non-null, pointer to matrix C.
/// </param> <param name="n">	   	An int to process. </param> <param
/// name="filename">	Filename of the file. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void
readMatrix(T *A, T *B, T *C, int n, string filename)
{
    std::ifstream mfs(filename.c_str());
    string        line;
    // Ignore header line because it was already checked
    getline(mfs, line);
    float a, b, c;
    for (int j = 0; j < n; j++)
    {
        sscanf(line.c_str(), "%f %f %f", &a, &b, &c);
        A[j] = T(a);
        B[j] = T(b);
        C[j] = T(c);
    }
}

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
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the single precision general
//   matrix multiplication (SGEMM) operation in GFLOPS.  Data transfer time
//   over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    cout << "Running GEMM" << endl;

    srand(SEED);

    bool quiet = op.getOptionBool("quiet");

    try
    {
        if (!quiet)
            cout << "Running single precision test" << endl;
        RunTest<float>("SGEMM", resultDB, op, device_idx);
    }
    catch (sycl::exception const &exc)
    {
        std::cout << "Error: Single kernels not supported by this device."
                  << endl;
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    try
    {
        if (!quiet)
            cout << "Running double precision test" << endl;
        RunTest<double>("DGEMM", resultDB, op, device_idx);
    }
    catch (sycl::exception const &exc)
    {
        std::cout << "Error: Double kernels not supported by this device."
                  << endl;
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    try
    {
        if (!quiet)
            cout << "Running half precision test" << endl;
        RunTest<sycl::half>("HGEMM", resultDB, op, device_idx);
    }
    catch (sycl::exception const &exc)
    {
        std::cout << "Error: Half kernels not supported by this device."
                  << endl;
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }
}

static const int tile_size = 32;
sycl::range      group_size { size_t(1), size_t(tile_size) };

// template <class T>
// class custom_gemm
// {
// public:
//   custom_gemm(size_t _dimm, T* __restrict _dA, T* __restrict _dB, T*
//   __restrict _dC) :
//     dimm(_dimm), dA(_dA), dB(_dB), dC(_dC) {}

//   void operator()(sycl::group<2> group) const
//   {
//     T tileA[tile_size]; // LocalMem

//     for (int kk = 0; kk < dimm; kk += tile_size)
//     {
//       group.parallel_for_work_item([&](sycl::h_item<2> item) {
//         int m = item.get_global_id()[0];
//         int i = item.get_local_id()[1];
//         tileA[i] = dA[m + (kk + i) * dimm];
//       });

//       // Implicit barrier

//       group.parallel_for_work_item([&](sycl::h_item<2> item) {
//         int m = item.get_global_id()[0];
//         int n = item.get_global_id()[1];
//         for (int k = 0; k < tile_size; k++)
//           dC[m + n * dimm] += tileA[k] * dB[kk + k + n * dimm];
//       });
//     }
//   }

// private:
//   const size_t dimm;
//   T* __restrict dA;
//   T* __restrict dB;
//   T* __restrict dC;
// };

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void
RunTest(string          testName,
        ResultDatabase &resultDB,
        OptionParser   &op,
        size_t          device_idx)
{
    int                           passes  = op.getOptionInt("passes");
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    // Use preset problem size or read data from input file
    int    kib;
    string filename = op.getOptionString("inputFile");
    if (filename == "")
    {
        int probSizes[5] = { 1, 3, 20, 60, 120 };
        kib              = probSizes[op.getOptionInt("size") - 1];
    }
    else
    {
        std::ifstream mfs(filename.c_str());
        std::string   line;
        char          object[FIELD_LENGTH];
        sscanf(line.c_str(), "%s %d", object, &kib);
    }

    int N = kib * 1024 / sizeof(T);
    T  *A = new T[N * N];
    T  *B = new T[N * N];
    T  *C = new T[N * N];

    T *dA = sycl::malloc_device<T>(N * N, queue);
    T *dB = sycl::malloc_device<T>(N * N, queue);
    T *dC = sycl::malloc_device<T>(N * N, queue);
    if (dA == nullptr || dB == nullptr || dC == nullptr)
    {
        std::cerr << "Could not allocate memory on device." << std::endl;
        exit(42);
    }

    // Fill matrix or read from input file
    if (filename == "")
    {
        fill<T>(A, N * N, 31);
        fill<T>(B, N * N, 31);
        fill<T>(C, N * N, 31);
    }
    else
    {
        readMatrix(A, B, C, N * N, filename);
    }

    double transferTime = 0;
    auto   start_ct1    = std::chrono::steady_clock::now();

    queue.memcpy(dA, A, N * N * sizeof(T));
    queue.memcpy(dB, B, N * N * sizeof(T));
    queue.wait();

    auto  stop_ct1 = std::chrono::steady_clock::now();
    float elapsedTime
        = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
              .count();
    transferTime += elapsedTime * 1.e-3;

    bool first = true;
    for (int j = 0; j < passes; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            constexpr int                BL    = 16;
            const T                      alpha = 1;
            const T                      beta  = 0; //-1;
            const oneapi::mkl::transpose transa
                = oneapi::mkl::transpose::nontrans;
            const oneapi::mkl::transpose transb
                = i ? oneapi::mkl::transpose::trans
                    : oneapi::mkl::transpose::nontrans;

            float kernelTime = 0.0f;
            for (int ii = 0; ii < 4; ++ii)
            {
                start_ct1 = std::chrono::steady_clock::now();

                queue
                    .submit([&](sycl::handler &h) {
                        sycl::device_ptr<T> dev_a(dA);
                        sycl::device_ptr<T> dev_b(dB);

                        sycl::accessor<T,
                                       2,
                                       sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            A_local(sycl::range<2> { BL, BL }, h);
                        sycl::accessor<T,
                                       2,
                                       sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            B_local(sycl::range<2> { BL, BL }, h);

                        const int      work_group_size = N / BL;
                        sycl::range<2> num_items(BL, BL);

                        // h.single_task([=]() [[intel::kernel_args_restrict]] {
                        //   sycl::device_ptr<T> dev_c(dC);

                        //   [[intelfpga::ivdep]]
                        //   for (int32_t item = 0; item < N * N; item++)
                        //   {
                        //     int32_t row = i / N;
                        //     int32_t col = i % N;

                        //     T sum = 0;

                        //     //#pragma unroll 16
                        //     [[intelfpga::ivdep]]
                        //     for (int32_t k = 0; k < N; k ++)
                        //       sum += dev_a[row + k * N] * dev_b[k + col * N]
                        //       * alpha;
                        //     dev_c[row + col * N] = sum; // + beta * dev_c[row
                        //     + col * N];
                        //   }
                        // });
                        h.single_task([=]() [[intel::kernel_args_restrict]] {
                            sycl::device_ptr<T> dev_c(dC);

#pragma unroll 1
                            for (int32_t block_x = 0; block_x < work_group_size;
                                 block_x++)
                            {
#pragma unroll 1
                                for (int32_t block_y = 0;
                                     block_y < work_group_size;
                                     block_y++)
                                {
#pragma unroll 1
                                    for (int32_t local_x = 0; local_x < BL;
                                         local_x++)
                                    {
#pragma unroll 1
                                        for (int32_t local_y = 0; local_y < BL;
                                             local_y++)
                                        {
                                            int a_start = N * BL * block_y;
                                            int a_end   = a_start + N - 1;
                                            int b_start = BL * block_x;

                                            T sum = 0.0f;

#pragma unroll 1
                                            [[intelfpga::ivdep(dev_c)]] //
                                            for (int a = a_start, b = b_start;
                                                 a <= a_end;
                                                 a += BL, b += (BL * N))
                                            {
                                                A_local[local_y][local_x]
                                                    = dev_a[a + N * local_y
                                                            + local_x];
                                                B_local[local_x][local_y]
                                                    = dev_b[b + N * local_y
                                                            + local_x];

#pragma unroll 4
                                                [[intelfpga::ivdep]] //
                                                for (int k = 0; k < BL; ++k) sum
                                                    += A_local[local_y][k]
                                                       * B_local[local_x][k];
                                            }

                                            dev_c[block_y * work_group_size
                                                  + local_y
                                                  + (block_x * work_group_size
                                                     + local_x)
                                                        * N]
                                                = sum;
                                        }
                                    }
                                }
                            }
                        });
                    })
                    .wait();

                stop_ct1          = std::chrono::steady_clock::now();
                float elapsedTime = std::chrono::duration<float, std::milli>(
                                        stop_ct1 - start_ct1)
                                        .count();
                kernelTime += elapsedTime;
            }
            double cublasTime = (kernelTime / 4.0) * 1.e-3;

            start_ct1 = std::chrono::steady_clock::now();

            queue.memcpy(C, dC, N * N * sizeof(T)).wait();

            stop_ct1 = std::chrono::steady_clock::now();
            float oTransferTime
                = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count()
                  * 1.e-3;

            // Add the PCIe transfer time to total transfer time only once
            if (first)
            {
                transferTime += oTransferTime;
                first = false;
            }

            double cublasGflops = 2. * N * N * N / cublasTime / 1.e9;
            double pcieGflops
                = 2. * N * N * N / (cublasTime + transferTime) / 1.e9;
            std::string transb_string
                = (transb == oneapi::mkl::transpose::trans) ? "T" : "N";
            string atts = "dim:" + toString(N);
            resultDB.AddResult(testName + "-" + transb_string + "-TransferTime",
                               atts,
                               "sec",
                               transferTime);
            resultDB.AddResult(testName + "-" + transb_string + "-KernelTime",
                               atts,
                               "sec",
                               cublasTime);
            resultDB.AddResult(testName + "-" + transb_string + "-TotalTime",
                               atts,
                               "sec",
                               transferTime + cublasTime);
            resultDB.AddResult(
                testName + "-" + transb_string, atts, "GFlops", cublasGflops);
            resultDB.AddResult(testName + "-" + transb_string + "_PCIe",
                               atts,
                               "GFlops",
                               pcieGflops);
            resultDB.AddResult(testName + "-" + transb_string + "_Parity",
                               atts,
                               "N",
                               transferTime / cublasTime);
            resultDB.AddOverall("GFlops", "", cublasGflops);
        }
    }

    // Clean Up
    //
    sycl::free(dA, queue);
    sycl::free(dB, queue);
    sycl::free(dC, queue);
}

template<>
inline void
devGEMM<double>(sycl::queue           *handle,
                oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb,
                int                    m,
                int                    n,
                int                    k,
                const double          *alpha,
                const double          *A,
                int                    lda,
                const double          *B,
                int                    ldb,
                const double          *beta,
                double                *C,
                int                    ldc)
{
    using oneapi::mkl::blas::column_major::gemm;
    gemm(*handle,
         transa,
         transb,
         int64_t(m),
         int64_t(n),
         int64_t(k),
         dpct::get_value(alpha, *handle),
         A,
         int64_t(lda),
         B,
         int64_t(ldb),
         dpct::get_value(beta, *handle),
         C,
         int64_t(ldc))
        .wait();
}

template<>
inline void
devGEMM<float>(sycl::queue           *handle,
               oneapi::mkl::transpose transa,
               oneapi::mkl::transpose transb,
               int                    m,
               int                    n,
               int                    k,
               const float           *alpha,
               const float           *A,
               int                    lda,
               const float           *B,
               int                    ldb,
               const float           *beta,
               float                 *C,
               int                    ldc)
{
    using oneapi::mkl::blas::column_major::gemm;
    gemm(*handle,
         transa,
         transb,
         m,
         n,
         k,
         dpct::get_value(alpha, *handle),
         A,
         lda,
         B,
         ldb,
         dpct::get_value(beta, *handle),
         C,
         ldc)
        .wait();
}

template<>
inline void
devGEMM<sycl::half>(sycl::queue           *handle,
                    oneapi::mkl::transpose transa,
                    oneapi::mkl::transpose transb,
                    int                    m,
                    int                    n,
                    int                    k,
                    const sycl::half      *alpha,
                    const sycl::half      *A,
                    int                    lda,
                    const sycl::half      *B,
                    int                    ldb,
                    const sycl::half      *beta,
                    sycl::half            *C,
                    int                    ldc)
{
    using oneapi::mkl::blas::column_major::gemm;
    gemm(*handle,
         transa,
         transb,
         m,
         n,
         k,
         dpct::get_value(alpha, *handle),
         A,
         lda,
         B,
         ldb,
         dpct::get_value(beta, *handle),
         C,
         ldc)
        .wait();
}
