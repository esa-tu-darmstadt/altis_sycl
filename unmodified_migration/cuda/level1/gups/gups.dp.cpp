/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */

/*
 * This code has been contributed by the DARPA HPCS program.  Contact
 * David Koester <dkoester@mitre.org> or Bob Lucas <rflucas@isi.edu>
 * if you have questions.
 *
 *
 * GUPS (Giga UPdates per Second) is a measurement that profiles the memory
 * architecture of a system and is a measure of performance similar to MFLOPS.
 * The HPCS HPCchallenge RandomAccess benchmark is intended to exercise the
 * GUPS capability of a system, much like the LINPACK benchmark is intended to
 * exercise the MFLOPS capability of a computer.  In each case, we would
 * expect these benchmarks to achieve close to the "peak" capability of the
 * memory system. The extent of the similarities between RandomAccess and
 * LINPACK are limited to both benchmarks attempting to calculate a peak system
 * capability.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term "randomly"
 * means that there is little relationship between one address to be updated and
 * the next, except that they occur in the space of one half the total system
 * memory.  An update is a read-modify-write operation on a table of 64-bit words.
 * An address is generated, the value at that address read from memory, modified
 * by an integer operation (add, and, or, xor) with a literal value, and that
 * new value is written back to memory.
 *
 * We are interested in knowing the GUPS performance of both entire systems and
 * system subcomponents --- e.g., the GUPS rating of a distributed memory
 * multiprocessor the GUPS rating of an SMP node, and the GUPS rating of a
 * single processor.  While there is typically a scaling of FLOPS with processor
 * count, a similar phenomenon may not always occur for GUPS.
 *
 * Select the memory size to be the power of two such that 2^n <= 1/2 of the
 * total memory.  Each CPU operates on its own address stream, and the single
 * table may be distributed among nodes. The distribution of memory to nodes
 * is left to the implementer.  A uniform data distribution may help balance
 * the workload, while non-uniform data distributions may simplify the
 * calculations that identify processor location by eliminating the requirement
 * for integer divides. A small (less than 1%) percentage of missed updates
 * are permitted.
 *
 * When implementing a benchmark that measures GUPS on a distributed memory
 * multiprocessor system, it may be required to define constraints as to how
 * far in the random address stream each node is permitted to "look ahead".
 * Likewise, it may be required to define a constraint as to the number of
 * update messages that can be stored before processing to permit multi-level
 * parallelism for those systems that support such a paradigm.  The limits on
 * "look ahead" and "stored updates" are being implemented to assure that the
 * benchmark meets the intent to profile memory architecture and not induce
 * significant artificial data locality. For the purpose of measuring GUPS,
 * we will stipulate that each thread is permitted to look ahead no more than
 * 1024 random address stream samples with the same number of update messages
 * stored before processing.
 *
 * The supplied MPI-1 code generates the input stream {A} on all processors
 * and the global table has been distributed as uniformly as possible to
 * balance the workload and minimize any Amdahl fraction.  This code does not
 * exploit "look-ahead".  Addresses are sent to the appropriate processor
 * where the table entry resides as soon as each address is calculated.
 * Updates are performed as addresses are received.  Each message is limited
 * to a single 64 bit long integer containing element ai from {A}.
 * Local offsets for T[ ] are extracted by the destination processor.
 *
 * If the number of processors is equal to a power of two, then the global
 * table can be distributed equally over the processors.  In addition, the
 * processor number can be determined from that portion of the input stream
 * that identifies the address into the global table by masking off log2(p)
 * bits in the address.
 *
 * If the number of processors is not equal to a power of two, then the global
 * table cannot be equally distributed between processors.  In the MPI-1
 * implementation provided, there has been an attempt to minimize the differences
 * in workloads and the largest difference in elements of T[ ] is one.  The
 * number of values in the input stream generated by each processor will be
 * related to the number of global table entries on each processor.
 *
 * The MPI-1 version of RandomAccess treats the potential instance where the
 * number of processors is a power of two as a special case, because of the
 * significant simplifications possible because processor location and local
 * offset can be determined by applying masks to the input stream values.
 * The non power of two case uses an integer division to determine the processor
 * location.  The integer division will be more costly in terms of machine
 * cycles to perform than the bit masking operations
 *
 * For additional information on the GUPS metric, the HPCchallenge RandomAccess
 * Benchmark,and the rules to run RandomAccess or modify it to optimize
 * performance -- see http://icl.cs.utk.edu/hpcc/
 *
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\dwt2d\gups\gups.cu
 //
 // summary:	Random access class
 // 
 // origin: HPCchallenge RandomAccess Benchmark(http://icl.cs.utk.edu/hpcc/)
 ////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#include <cstdlib>
#include <iostream>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines default logn for memory size. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEFAULT_LOGN 20

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Polygon. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define POLY 0x0000000000000007ULL

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines default GPU. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEFAULT_GPU 0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A benchtype. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

union benchtype {
  /// <summary>	The 64. </summary>
  uint64_t u64;
  /// <summary>	The 32. </summary>
  sycl::uint2 u32{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the 2[64]. </summary>
///
/// <value>	The c m 2[64]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<uint64_t, 1> c_m2(64);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the error[ 1]. </summary>
///
/// <value>	The d error[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::global_memory<uint32_t, 1> d_error(1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes this.  </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="n">	A size_t to process. </param>
/// <param name="t">	[in,out] If non-null, a benchtype to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

static void
d_init(size_t n, benchtype *t, const sycl::nd_item<3> &item_ct1)
{
  for (ptrdiff_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
       i < n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    t[i].u64 = i;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Starts the given n. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="n">	A size_t to process. </param>
///
/// <returns>	An uint64_t. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

static uint64_t
d_starts(size_t n, uint64_t *c_m2)
{
  if (n == 0) {
    return 1;
  }

  int i = 63 - sycl::clz((long long)n);

  uint64_t ran = 2;
  while (i > 0) {
    uint64_t temp = 0;
    for (int j = 0; j < 64; j++) {
      if ((ran >> j) & 1) {
        temp ^= c_m2[j];
      }
    }
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) {
      ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0);
    }
  }

  return ran;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Values that represent atomictype ts. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

enum atomictype_t {
  ATOMICTYPE_CAS,
  ATOMICTYPE_XOR,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Benches. </summary>
///
/// <typeparam name="ATOMICTYPE">	Type of the atomictype. </typeparam>
/// <param name="n">	A size_t to process. </param>
/// <param name="t">	[in,out] If non-null, a benchtype to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<atomictype_t ATOMICTYPE>
void
d_bench(size_t n, benchtype *t, const sycl::nd_item<3> &item_ct1,
        uint64_t *c_m2)
{
  size_t num_threads =
      item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
  size_t thread_num = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      item_ct1.get_local_id(2);
  size_t start = thread_num * 4 * n / num_threads;
  size_t end = (thread_num + 1) * 4 * n / num_threads;
  benchtype ran;
  ran.u64 = d_starts(start, c_m2);
  for (ptrdiff_t i = start; i < end; ++i) {
    ran.u64 = (ran.u64 << 1) ^ ((int64_t) ran.u64 < 0 ? POLY : 0);
    switch (ATOMICTYPE) {
    case ATOMICTYPE_CAS:
      unsigned long long int *address, old, assumed;
      address = (unsigned long long int *)&t[ran.u64 & (n - 1)].u64;
      old = *address;
      do {
        assumed = old;
        old = dpct::atomic_compare_exchange_strong<
            sycl::access::address_space::generic_space>(address, assumed,
                                                        assumed ^ ran.u64);
      } while  (assumed != old);
      break;
    case ATOMICTYPE_XOR:
      dpct::atomic_fetch_xor<sycl::access::address_space::generic_space>(
          &t[ran.u64 & (n - 1)].u32.x(), ran.u32.x());
      dpct::atomic_fetch_xor<sycl::access::address_space::generic_space>(
          &t[ran.u64 & (n - 1)].u32.y(), ran.u32.y());
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Checks. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="n">	A size_t to process. </param>
/// <param name="t">	[in,out] If non-null, a benchtype to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

static void
d_check(size_t n, benchtype *t, const sycl::nd_item<3> &item_ct1,
        uint32_t *d_error)
{
  for (ptrdiff_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
       i < n; i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    if (t[i].u64 != i) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          d_error, 1);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Starts this.  </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

static void
starts()
{
  uint64_t m2[64];
  uint64_t temp = 1;
  for (ptrdiff_t i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
  }
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_default_queue().memcpy(c_m2.get_ptr(), m2, sizeof(m2)).wait()));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
   // TODO, maybe add benchmark specs 
  op.addOption("shifts", OPT_INT, "20", "specify bit shift for the number of elements in update table", '\0');
}

//int main(int argc, char *argv[])
//{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="DB">	[in,out] The database. </param>
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {
  std::cout << "Running GUPS" << std::endl;
  size_t n = 0;
  const int passes = op.getOptionInt("passes");
  const bool uvm = op.getOptionBool("uvm");
  int device = 0;
  checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

  // Specify table size
  int problemSizes[5] = {20, 22, 24, 26, 32}; // size 5 might be extremely long!
  int toShifts = problemSizes[op.getOptionInt("size") - 1];

  int logn = op.getOptionInt("shifts");
  // TODO: watch out size
  if (logn > 0 && logn != 20) {
    n = (size_t) 1 << logn;
  } else {
    n = (size_t) 1 << toShifts;
  }

  std::cout << "Total table size = " << n << " (" << n*sizeof(uint64_t) << " bytes.)" << std::endl;

  starts();

  int ndev;
  checkCudaErrors(
      DPCT_CHECK_ERROR(ndev = dpct::dev_mgr::instance().device_count()));
  int dev = op.getOptionInt("device");

  dpct::device_info prop;
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dev_mgr::instance().get_device(device).get_device_info(prop)));
  /*
  DPCT1093:714: The "dev" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(dev)));
  printf("Using GPU %d of %d GPUs.\n", dev, ndev);
  printf("Warp size = %d.\n", prop.get_max_sub_group_size());
  printf("Multi-processor count = %d.\n", prop.get_max_compute_units());
  printf("Max threads per multi-processor = %d.\n",
         prop.get_max_work_items_per_compute_unit());

  benchtype *d_t = NULL;
  if (uvm) {
    /*
    DPCT1064:723: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        d_t = sycl::malloc_shared<benchtype>(n, dpct::get_default_queue())));
  } else {
    /*
    DPCT1064:724: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        d_t = sycl::malloc_device<benchtype>(n, dpct::get_default_queue())));
  }

  // max warp size
  sycl::range<3> grid(1, 1,
                      prop.get_max_compute_units() *
                          (prop.get_max_work_items_per_compute_unit() /
                           prop.get_max_sub_group_size()));
  // # as if scheduling warps instead of blocks
  sycl::range<3> thread(1, 1, prop.get_max_sub_group_size());
  dpct::event_ptr begin, end;
  std::chrono::time_point<std::chrono::steady_clock> begin_ct1;
  std::chrono::time_point<std::chrono::steady_clock> end_ct1;
  checkCudaErrors(DPCT_CHECK_ERROR(begin = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(end = new sycl::event()));
  void *p_error;
  checkCudaErrors(DPCT_CHECK_ERROR(*(&p_error) = d_error.get_ptr()));

  string atts = "ATOMICTYPE_CAS";
  for (int i = 0; i < passes; i++) {
    /*
    DPCT1049:149: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(grid * thread, thread),
                [=](sycl::nd_item<3> item_ct1) {
                      d_init(n, d_t, item_ct1);
                });
    /*
    DPCT1012:715: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:716: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    begin_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(
            *begin = dpct::get_default_queue().ext_oneapi_submit_barrier()));
    /*
    DPCT1049:150: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  c_m2.init();

                  auto c_m2_ptr_ct1 = c_m2.get_ptr();

                  cgh.parallel_for(sycl::nd_range<3>(grid * thread, thread),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         d_bench<ATOMICTYPE_CAS>(
                                             n, d_t, item_ct1, c_m2_ptr_ct1);
                                   });
            });
    /*
    DPCT1012:717: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:718: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    end_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(*end = dpct::get_default_queue()
                                                .ext_oneapi_submit_barrier()));
    checkCudaErrors(0);

    float ms;
    checkCudaErrors(DPCT_CHECK_ERROR(
        (ms = std::chrono::duration<float, std::milli>(end_ct1 - begin_ct1)
                  .count())));

    double time = ms * 1.0e-3;
    DB.AddResult("Elapsed time", atts, "seconds", time);
    double gups = 4 * n / (double) ms * 1.0e-6;
    DB.AddResult("Giga Updates per second", atts, "GUP/s", gups);

    dpct::get_default_queue()
        .memset(d_error.get_ptr(), 0, sizeof(uint32_t))
        .wait();
    /*
    DPCT1049:151: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  d_error.init();

                  auto d_error_ptr_ct1 = d_error.get_ptr();

                  cgh.parallel_for(sycl::nd_range<3>(grid * thread, thread),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         d_check(n, d_t, item_ct1,
                                                 d_error_ptr_ct1);
                                   });
            });
    uint32_t h_error;
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(&h_error, p_error, sizeof(uint32_t))
                             .wait()));
    if (op.getOptionBool("verbose")) {
      printf("Verification (ATOMICTYPE_CAS): Found %u errors.\n", h_error);
    }
  }

  atts = "ATOMICTYPE_XOR";
  for (int i = 0; i < passes; i++) {
    /*
    DPCT1049:152: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(grid * thread, thread),
                [=](sycl::nd_item<3> item_ct1) {
                      d_init(n, d_t, item_ct1);
                });
    /*
    DPCT1012:719: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:720: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    begin_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(
            *begin = dpct::get_default_queue().ext_oneapi_submit_barrier()));
    /*
    DPCT1049:153: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  c_m2.init();

                  auto c_m2_ptr_ct1 = c_m2.get_ptr();

                  cgh.parallel_for(sycl::nd_range<3>(grid * thread, thread),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         d_bench<ATOMICTYPE_XOR>(
                                             n, d_t, item_ct1, c_m2_ptr_ct1);
                                   });
            });
    /*
    DPCT1012:721: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:722: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    end_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(DPCT_CHECK_ERROR(*end = dpct::get_default_queue()
                                                .ext_oneapi_submit_barrier()));
    checkCudaErrors(0);

    float ms;
    checkCudaErrors(DPCT_CHECK_ERROR(
        (ms = std::chrono::duration<float, std::milli>(end_ct1 - begin_ct1)
                  .count())));

    double time = ms * 1.0e-3;
    DB.AddResult("Elapsed time", atts, "seconds", time);
    double gups = 4 * n / (double) ms * 1.0e-6;
    DB.AddResult("Giga Updates per second", atts, "GUP/s", gups);
    // d_bench<ATOMICTYPE_XOR><<<grid, thread>>>(n, d_t);
    // void *p_error;
    // checkCudaErrors(cudaGetSymbolAddress(&p_error, d_error));
    dpct::get_default_queue()
        .memset(d_error.get_ptr(), 0, sizeof(uint32_t))
        .wait();
    /*
    DPCT1049:154: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  d_error.init();

                  auto d_error_ptr_ct1 = d_error.get_ptr();

                  cgh.parallel_for(sycl::nd_range<3>(grid * thread, thread),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         d_check(n, d_t, item_ct1,
                                                 d_error_ptr_ct1);
                                   });
            });
    uint32_t h_error;
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(&h_error, p_error, sizeof(uint32_t))
                             .wait()));
    if (op.getOptionBool("verbose")) {
      printf("Verification (ATOMICTYPE_XOR): Found %u errors.\n", h_error);
    }
  }

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(end)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(begin)));
  checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(d_t, dpct::get_default_queue())));
}