////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	C:\Users\ed\source\repos\altis\src\cuda\level1\gemm\Gemm.cu
//
// summary:	Gemm class
// 
// origin: SHOC (https://github.com/vetter/shocp)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "OptionParser.h"
#include "ResultDatabase.h"
// #include "Timer.h"
#include "Utility.h"
#include <dpct/blas_utils.hpp>

//#include "cublas.h"
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

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

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

template <class T>
inline void devGEMM(dpct::queue_ptr handle, oneapi::mkl::transpose transa,
                    oneapi::mkl::transpose transb, int m, int n, int k,
                    const T *alpha, const T *A, int lda, const T *B, int ldb,
                    const T *beta, T *C, int ldc);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Filling memory. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">   	[in,out] If non-null,  pointer to the array to initialize. </param>
/// <param name="n">   number of elements in the array. </param>
/// <param name="maxi">	The maxi. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void fill(T *A, int n, int maxi) {
  for (int j = 0; j < n; j++) {
      if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
          A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / T(maxi + 1.);
      else if (std::is_same<T, sycl::half>::value)
          A[j] = sycl::vec<float, 1>{(float((rand() % (maxi * 2 + 1)) - maxi) /
                                      (maxi + 1.))}
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads a matrix. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">	   	[in,out] If non-null, pointer to matrix A. </param>
/// <param name="B">	   	[in,out] If non-null, pointer to matrix B. </param>
/// <param name="C">	   	[in,out] If non-null, pointer to matrix C. </param>
/// <param name="n">	   	An int to process. </param>
/// <param name="filename">	Filename of the file. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void readMatrix(T *A, T *B, T *C, int n, string filename) {
  std::ifstream mfs(filename.c_str());
  string line;
  // Ignore header line because it was already checked
  getline(mfs, line);
  float a, b, c;
  for (int j = 0; j < n; j++) {
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
void addBenchmarkSpecOptions(OptionParser &op) {}

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
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
   cout << "Running GEMM" << endl;
  int device;
  device = dpct::dev_mgr::instance().current_device_id();
  dpct::device_info deviceProp;
  dpct::dev_mgr::instance().get_device(device).get_device_info(deviceProp);

  srand(SEED);

  bool quiet = op.getOptionBool("quiet");

  if(!quiet) {
    cout << "Running single precision test" << endl;
  }
  RunTest<float>("SGEMM", resultDB, op);


  // Test to see if this device supports double precision
  /*
  DPCT1005:923: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if ((deviceProp.get_major_version() == 1 &&
       deviceProp.get_minor_version() >= 3) ||
      /*
      DPCT1005:924: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      (deviceProp.get_major_version() >= 2)) {
    if(!quiet) {
        cout << "Running double precision test" << endl;
    }
    RunTest<double>("DGEMM", resultDB, op);
  }

  /*
  DPCT1005:925: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if ((deviceProp.get_major_version() >= 6)) {
    if (!quiet) {
        cout << "Running half preicsion test" << endl;
    }
    RunTest<sycl::half>("HGEMM", resultDB, op);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) try {
  int passes = op.getOptionInt("passes");
  int device = op.getOptionInt("device");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  int kib;

  // Use preset problem size or read data from input file
  string filename = op.getOptionString("inputFile");
  if (filename == "") {
    int probSizes[5] = {1, 3, 20, 60, 120};
    kib = probSizes[op.getOptionInt("size") - 1];
  } else {
    std::ifstream mfs(filename.c_str());
    std::string line;
    char object[FIELD_LENGTH];
    sscanf(line.c_str(), "%s %d", object, &kib);
  }

  // Dimensions of matrix
  int N = kib * 1024 / sizeof(T);

  // Initialize the cublas library
  dpct::queue_ptr handle; // CUBLAS context
  int stat = DPCT_CHECK_ERROR(handle = &dpct::get_default_queue());
  if (stat != 0) {
        std::cerr << "CUBLAS initialization failed" << std::endl;
        safe_exit(-1);
  }

  // Allocate GPU memory
  T *dA, *dB, *dC;
  T *A;
  T *B;
  T *C;
  if (uvm || uvm_prefetch || uvm_advise || uvm_prefetch_advise) {
      /*
      DPCT1064:926: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dA = (T *)sycl::malloc_shared(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:927: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dB = (T *)sycl::malloc_shared(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:928: Migrated cudaMallocManaged call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dC = (T *)sycl::malloc_shared(
                               N * N * sizeof(T), dpct::get_default_queue())));

      if (filename == "") {
          fill<T>(dA, N * N, 31);
          fill<T>(dB, N * N, 31);
          fill<T>(dC, N * N, 31);
      } else {
          readMatrix(dA, dB, dC, N * N, filename);
      }
  }
  else {
      /*
      DPCT1064:929: Migrated cudaMalloc call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dA = (T *)sycl::malloc_device(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:930: Migrated cudaMalloc call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dB = (T *)sycl::malloc_device(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:931: Migrated cudaMalloc call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(dC = (T *)sycl::malloc_device(
                               N * N * sizeof(T), dpct::get_default_queue())));

      /*
      DPCT1064:932: Migrated cudaMallocHost call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(A = (T *)sycl::malloc_host(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:933: Migrated cudaMallocHost call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(B = (T *)sycl::malloc_host(
                               N * N * sizeof(T), dpct::get_default_queue())));
      /*
      DPCT1064:934: Migrated cudaMallocHost call is used in a macro/template
      definition and may not be valid for all macro/template uses. Adjust the
      code.
      */
      checkCudaErrors(
          DPCT_CHECK_ERROR(C = (T *)sycl::malloc_host(
                               N * N * sizeof(T), dpct::get_default_queue())));

      // Fill matrix or read from input file
      if (filename == "") {
          fill<T>(A, N * N, 31);
          fill<T>(B, N * N, 31);
          fill<T>(C, N * N, 31);
      } else {
        readMatrix(A, B, C, N * N, filename);
      }
  }

  // Copy input to GPU
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));
  float elapsedTime;

  // Copy inputs to GPU

  double transferTime = 0;
  /*
  DPCT1012:903: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:904: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  start_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);

  if (uvm) {
      // Do nothing
  } else if (uvm_prefetch) {
      // could ignore this to test demand paging performance affect
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dA, N * N * sizeof(T))));
      dpct::queue_ptr s1;
      checkCudaErrors(
          DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
      checkCudaErrors(DPCT_CHECK_ERROR(s1->prefetch(dB, N * N * sizeof(T))));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
      // checkCudaErrors(cudaStreamSynchronize(0));
      // checkCudaErrors(cudaStreamSynchronize((cudaStream_t)1));
  } else if (uvm_advise) {
      // Do nothing for demand paging
      /*
      DPCT1063:905: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), 0, device));
      /*
      DPCT1063:906: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), 0, device));
  } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:907: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), 0, device));
      /*
      DPCT1063:908: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), 0, device));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dA, N * N * sizeof(T))));
      dpct::queue_ptr s1;
      checkCudaErrors(
          DPCT_CHECK_ERROR(s1 = dpct::get_current_device().create_queue()));
      checkCudaErrors(DPCT_CHECK_ERROR(s1->prefetch(dB, N * N * sizeof(T))));
      checkCudaErrors(
          DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(s1)));
  } else {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_default_queue().memcpy(dA, A, N * N * sizeof(T)).wait()));
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_default_queue().memcpy(dB, B, N * N * sizeof(T)).wait()));
  }

  /*
  DPCT1012:909: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:910: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  stop_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);
  checkCudaErrors(0);
  elapsedTime =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsedTime * 1.e-3;

  bool first = true;
/// <summary>	. </summary>
  for (int j = 0; j < passes; j++) {
    for (int i = 0; i < 2; i++) {
      const oneapi::mkl::transpose transa = oneapi::mkl::transpose::nontrans;
      const oneapi::mkl::transpose transb =
          i ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
      const int nb = 128;
      const int idim = N / nb;

      int dim = idim * nb;

      const int m = dim;
      const int n = dim;
      const int k = dim;
      const int lda = dim;
      const int ldb = dim;
      const int ldc = dim;
      const T alpha = 1;
      const T beta = 0; //-1;

      // Warm Up
      devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                    ldc);
      dpct::get_current_device().queues_wait_and_throw();
      CHECK_CUDA_ERROR();

      double cublasTime;
      float kernelTime = 0.0f;
      for (int ii = 0; ii < 4; ++ii) {
          /*
          DPCT1012:911: Detected kernel execution time measurement pattern and
          generated an initial code for time measurements in SYCL. You can
          change the way time is measured depending on your goals.
          */
          /*
          DPCT1024:912: The original code returned the error code that was
          further consumed by the program logic. This original code was replaced
          with 0. You may need to rewrite the program logic consuming the error
          code.
          */
          start_ct1 = std::chrono::steady_clock::now();
          checkCudaErrors(0);
          devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                    ldc);
          /*
          DPCT1012:913: Detected kernel execution time measurement pattern and
          generated an initial code for time measurements in SYCL. You can
          change the way time is measured depending on your goals.
          */
          /*
          DPCT1024:914: The original code returned the error code that was
          further consumed by the program logic. This original code was replaced
          with 0. You may need to rewrite the program logic consuming the error
          code.
          */
          stop_ct1 = std::chrono::steady_clock::now();
          checkCudaErrors(0);
          checkCudaErrors(0);
          CHECK_CUDA_ERROR();
          float currTime = 0.0f;
          checkCudaErrors(DPCT_CHECK_ERROR((
              currTime =
                  std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count())));
          kernelTime += currTime;
      }
      cublasTime = (kernelTime / 4.0) * 1.e-3;

      /*
      DPCT1012:915: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:916: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      start_ct1 = std::chrono::steady_clock::now();
      checkCudaErrors(0); // timing may be affected by async

      if (uvm) {
        // Do nothing
      } else if (uvm_prefetch) {
          checkCudaErrors(
              DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
                  dC, N * N * sizeof(T))));
          // checkCudaErrors(cudaStreamSynchronize(0));
      } else if (uvm_advise) {
          /*
          DPCT1063:917: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          checkCudaErrors(
              cudaMemAdvise(dC, N * N * sizeof(T), 0, cudaCpuDeviceId));
          /*
          DPCT1063:918: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          checkCudaErrors(
              cudaMemAdvise(dC, N * N * sizeof(T), 0, cudaCpuDeviceId));
      } else if (uvm_prefetch_advise) {
          /*
          DPCT1063:919: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          checkCudaErrors(
              cudaMemAdvise(dC, N * N * sizeof(T), 0, cudaCpuDeviceId));
          /*
          DPCT1063:920: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          checkCudaErrors(
              cudaMemAdvise(dC, N * N * sizeof(T), 0, cudaCpuDeviceId));
          checkCudaErrors(
              DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(
                  dC, N * N * sizeof(T))));
      } else {
          checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_default_queue()
                                               .memcpy(C, dC, N * N * sizeof(T))
                                               .wait()));
      }

      /*
      DPCT1012:921: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:922: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      stop_ct1 = std::chrono::steady_clock::now();
      checkCudaErrors(0);
      checkCudaErrors(0);
      float oTransferTime = 0.0f;
      checkCudaErrors(DPCT_CHECK_ERROR(
          (oTransferTime =
               std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                   .count())));
      oTransferTime *= 1.e-3;

      // Add the PCIe transfer time to total transfer time only once
      if (first) {
        transferTime += oTransferTime;
        first = false;
      }

      double cublasGflops = 2. * m * n * k / cublasTime / 1e9;
      double pcieGflops = 2. * m * n * k / (cublasTime + transferTime) / 1e9;
      std::string transb_string =
          (transb == oneapi::mkl::transpose::trans) ? "T" : "N";
      string atts = "dim:" + toString(dim);
      resultDB.AddResult(testName + "-" + transb_string + "-TransferTime", atts, "sec", transferTime);
      resultDB.AddResult(testName + "-" + transb_string + "-KernelTime", atts, "sec", cublasTime);
      resultDB.AddResult(testName + "-" + transb_string + "-TotalTime", atts, "sec", transferTime + cublasTime);
      resultDB.AddResult(testName + "-" + transb_string, atts, "GFlops", cublasGflops);
      resultDB.AddResult(testName + "-" + transb_string + "_PCIe", atts, "GFlops", pcieGflops);
      resultDB.AddResult(testName + "-" + transb_string + "_Parity", atts, "N", transferTime / cublasTime);
      resultDB.AddOverall("GFlops", "", cublasGflops);
    }
  }

  // Clean Up

  checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(dA, dpct::get_default_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(dB, dpct::get_default_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(dC, dpct::get_default_queue())));
  if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
    checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(A, dpct::get_default_queue())));
    checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(B, dpct::get_default_queue())));
    checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(C, dpct::get_default_queue())));
  }

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
  handle = nullptr;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   gemm kernel (double). </summary>
///
/// <typeparam name="double">	Type of the double. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A double to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A double to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a double to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline void
devGEMM<double>(dpct::queue_ptr handle, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, int m, int n, int k,
                const double *alpha, const double *A, int lda, const double *B,
                int ldb, const double *beta, double *C, int ldc) {
  oneapi::mkl::blas::column_major::gemm(
      *handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), A, lda,
      B, ldb, dpct::get_value(beta, *handle), C, ldc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	gemm kernel (float). </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A float to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A float to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a float to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline void
devGEMM<float>(dpct::queue_ptr handle, oneapi::mkl::transpose transa,
               oneapi::mkl::transpose transb, int m, int n, int k,
               const float *alpha, const float *A, int lda, const float *B,
               int ldb, const float *beta, float *C, int ldc) {
  oneapi::mkl::blas::column_major::gemm(
      *handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), A, lda,
      B, ldb, dpct::get_value(beta, *handle), C, ldc);
}

template <>
inline void
devGEMM<sycl::half>(dpct::queue_ptr handle, oneapi::mkl::transpose transa,
                    oneapi::mkl::transpose transb, int m, int n, int k,
                    const sycl::half *alpha, const sycl::half *A, int lda,
                    const sycl::half *B, int ldb, const sycl::half *beta,
                    sycl::half *C, int ldc) {
  oneapi::mkl::blas::column_major::gemm(
      *handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), A, lda,
      B, ldb, dpct::get_value(beta, *handle), C, ldc);
}

