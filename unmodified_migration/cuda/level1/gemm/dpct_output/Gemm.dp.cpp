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
// #include "Timer.h"
#include "Utility.h"
#include <oneapi/mkl.hpp>
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
inline void devGEMM(sycl::queue *handle, oneapi::mkl::transpose transa,
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
      if (std::is_same<T, float>::value || std::is_same<T, double>::value)
          A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / T(maxi + 1.);
      else if (std::is_same<T, sycl::half>::value)
          A[j] = sycl::vec<float, 1>{float((rand() % (maxi * 2 + 1)) - maxi) /
                                     (maxi + 1.)}
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
      else
          safe_exit(-1);
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
  DPCT1005:0: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if ((deviceProp.get_major_version() == 1 &&
       deviceProp.get_minor_version() >= 3) ||
      /*
      DPCT1005:1: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      (deviceProp.get_major_version() >= 2)) {
    if(!quiet) {
        cout << "Running double precision test" << endl;
    }
    RunTest<double>("DGEMM", resultDB, op);
  }

  /*
  DPCT1005:2: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if ((deviceProp.get_major_version() >= 6)) {
    if (!quiet) {
        cout << "Running half preicsion test" << endl;
    }
    RunTest<half>("HGEMM", resultDB, op);
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
   dpct::device_ext &dev_ct1 = dpct::get_current_device();
   sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  sycl::queue *handle; // CUBLAS context
  /*
  DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  int stat = (handle = &q_ct1, 0);
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
      DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dA = (T *)sycl::malloc_shared(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dB = (T *)sycl::malloc_shared(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dC = (T *)sycl::malloc_shared(N * N * sizeof(T), q_ct1), 0));

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
      DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dA = (T *)sycl::malloc_device(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dB = (T *)sycl::malloc_device(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (dC = (T *)sycl::malloc_device(N * N * sizeof(T), q_ct1), 0));

      /*
      DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (A = (T *)sycl::malloc_host(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (B = (T *)sycl::malloc_host(N * N * sizeof(T), q_ct1), 0));
      /*
      DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors(
          (C = (T *)sycl::malloc_host(N * N * sizeof(T), q_ct1), 0));

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
  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  /*
  DPCT1027:13: The call to cudaEventCreate was replaced with 0 because this call
  is redundant in DPC++.
  */
  checkCudaErrors(0);
  /*
  DPCT1027:14: The call to cudaEventCreate was replaced with 0 because this call
  is redundant in DPC++.
  */
  checkCudaErrors(0);
  float elapsedTime;

  // Copy inputs to GPU

  double transferTime = 0;
  /*
  DPCT1012:15: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:16: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  start_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);

  if (uvm) {
      // Do nothing
  } else if (uvm_prefetch) {
      // could ignore this to test demand paging performance affect
      /*
      DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dA, N * N * sizeof(T)),
          0));
      sycl::queue *s1;
      /*
      DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((s1 = dev_ct1.create_queue(), 0));
      /*
      DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((s1->prefetch(dB, N * N * sizeof(T)), 0));
      /*
      DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dev_ct1.destroy_queue(s1), 0));
      // checkCudaErrors(cudaStreamSynchronize(0));
      // checkCudaErrors(cudaStreamSynchronize((cudaStream_t)1));
  } else if (uvm_advise) {
      // Do nothing for demand paging
      /*
      DPCT1063:21: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      /*
      DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_device(device).default_queue().mem_advise(
                           dA, N * N * sizeof(T), 0),
                       0));
      /*
      DPCT1063:23: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      /*
      DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_device(device).default_queue().mem_advise(
                           dB, N * N * sizeof(T), 0),
                       0));
  } else if (uvm_prefetch_advise) {
      /*
      DPCT1063:25: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      /*
      DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_device(device).default_queue().mem_advise(
                           dA, N * N * sizeof(T), 0),
                       0));
      /*
      DPCT1063:27: Advice parameter is device-defined and was set to 0. You may
      need to adjust it.
      */
      /*
      DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_device(device).default_queue().mem_advise(
                           dB, N * N * sizeof(T), 0),
                       0));
      /*
      DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((
          dpct::dev_mgr::instance().get_device(device).default_queue().prefetch(
              dA, N * N * sizeof(T)),
          0));
      sycl::queue *s1;
      /*
      DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((s1 = dev_ct1.create_queue(), 0));
      /*
      DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((s1->prefetch(dB, N * N * sizeof(T)), 0));
      /*
      DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dev_ct1.destroy_queue(s1), 0));
  } else {
      /*
      DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memcpy(dA, A, N * N * sizeof(T)).wait(), 0));
      /*
      DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memcpy(dB, B, N * N * sizeof(T)).wait(), 0));
  }

  /*
  DPCT1012:35: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:36: The original code returned the error code that was further
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
      dev_ct1.queues_wait_and_throw();
      CHECK_CUDA_ERROR();

      double cublasTime;
      float kernelTime = 0.0f;
      for (int ii = 0; ii < 4; ++ii) {
          /*
          DPCT1012:37: Detected kernel execution time measurement pattern and
          generated an initial code for time measurements in SYCL. You can
          change the way time is measured depending on your goals.
          */
          /*
          DPCT1024:38: The original code returned the error code that was
          further consumed by the program logic. This original code was replaced
          with 0. You may need to rewrite the program logic consuming the error
          code.
          */
          start_ct1 = std::chrono::steady_clock::now();
          checkCudaErrors(0);
          devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                    ldc);
          /*
          DPCT1012:39: Detected kernel execution time measurement pattern and
          generated an initial code for time measurements in SYCL. You can
          change the way time is measured depending on your goals.
          */
          /*
          DPCT1024:40: The original code returned the error code that was
          further consumed by the program logic. This original code was replaced
          with 0. You may need to rewrite the program logic consuming the error
          code.
          */
          stop_ct1 = std::chrono::steady_clock::now();
          checkCudaErrors(0);
          checkCudaErrors(0);
          CHECK_CUDA_ERROR();
          float currTime = 0.0f;
          /*
          DPCT1003:41: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((currTime = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                           0));
          kernelTime += currTime;
      }
      cublasTime = (kernelTime / 4.0) * 1.e-3;

      /*
      DPCT1012:42: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:43: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      start_ct1 = std::chrono::steady_clock::now();
      checkCudaErrors(0); // timing may be affected by async

      if (uvm) {
        // Do nothing
      } else if (uvm_prefetch) {
          /*
          DPCT1003:44: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::dev_mgr::instance()
                               .get_device(cudaCpuDeviceId)
                               .default_queue()
                               .prefetch(dC, N * N * sizeof(T)),
                           0));
          // checkCudaErrors(cudaStreamSynchronize(0));
      } else if (uvm_advise) {
          /*
          DPCT1063:45: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          /*
          DPCT1003:46: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::cpu_device().default_queue().mem_advise(
                               dC, N * N * sizeof(T), 0),
                           0));
          /*
          DPCT1063:47: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          /*
          DPCT1003:48: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::cpu_device().default_queue().mem_advise(
                               dC, N * N * sizeof(T), 0),
                           0));
      } else if (uvm_prefetch_advise) {
          /*
          DPCT1063:49: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          /*
          DPCT1003:50: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::cpu_device().default_queue().mem_advise(
                               dC, N * N * sizeof(T), 0),
                           0));
          /*
          DPCT1063:51: Advice parameter is device-defined and was set to 0. You
          may need to adjust it.
          */
          /*
          DPCT1003:52: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::cpu_device().default_queue().mem_advise(
                               dC, N * N * sizeof(T), 0),
                           0));
          /*
          DPCT1003:53: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((dpct::dev_mgr::instance()
                               .get_device(cudaCpuDeviceId)
                               .default_queue()
                               .prefetch(dC, N * N * sizeof(T)),
                           0));
      } else {
          /*
          DPCT1003:54: Migrated API does not return error code. (*, 0) is
          inserted. You may need to rewrite this code.
          */
          checkCudaErrors((q_ct1.memcpy(C, dC, N * N * sizeof(T)).wait(), 0));
      }

      /*
      DPCT1012:55: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      /*
      DPCT1024:56: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0. You
      may need to rewrite the program logic consuming the error code.
      */
      stop_ct1 = std::chrono::steady_clock::now();
      checkCudaErrors(0);
      checkCudaErrors(0);
      float oTransferTime = 0.0f;
      /*
      DPCT1003:57: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((oTransferTime = std::chrono::duration<float, std::milli>(
                                           stop_ct1 - start_ct1)
                                           .count(),
                       0));
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

  /*
  DPCT1003:58: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(dA, q_ct1), 0));
  /*
  DPCT1003:59: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(dB, q_ct1), 0));
  /*
  DPCT1003:60: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(dC, q_ct1), 0));
  if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
    /*
    DPCT1003:61: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(A, q_ct1), 0));
    /*
    DPCT1003:62: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(B, q_ct1), 0));
    /*
    DPCT1003:63: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(C, q_ct1), 0));
  }

  /*
  DPCT1027:64: The call to cudaEventDestroy was replaced with 0 because this
  call is redundant in DPC++.
  */
  checkCudaErrors(0);
  /*
  DPCT1027:65: The call to cudaEventDestroy was replaced with 0 because this
  call is redundant in DPC++.
  */
  checkCudaErrors(0);
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
inline void devGEMM<double>(sycl::queue *handle, oneapi::mkl::transpose transa,
                            oneapi::mkl::transpose transb, int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta,
                            double *C, int ldc) {
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
inline void devGEMM<float>(sycl::queue *handle, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
  oneapi::mkl::blas::column_major::gemm(
      *handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), A, lda,
      B, ldb, dpct::get_value(beta, *handle), C, ldc);
}

template <>
inline void devGEMM<half>(sycl::queue *handle, oneapi::mkl::transpose transa,
                          oneapi::mkl::transpose transb, int m, int n, int k,
                          const sycl::half *alpha, const sycl::half *A, int lda,
                          const sycl::half *B, int ldb, const sycl::half *beta,
                          sycl::half *C, int ldc) {
  oneapi::mkl::blas::column_major::gemm(
      *handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), A, lda,
      B, ldb, dpct::get_value(beta, *handle), C, ldc);
}

