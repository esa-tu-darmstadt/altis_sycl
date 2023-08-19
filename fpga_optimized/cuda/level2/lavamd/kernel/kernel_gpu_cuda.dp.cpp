#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#ifdef _STRATIX10
#define UNROLL_FAC 25
#endif
#ifdef _AGILEX
#define UNROLL_FAC 16
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//
//	origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

static constexpr int int_ceil(float f) {
  const int i = static_cast<int>(f);
  return f > i ? i + 1 : i;
}

constexpr unsigned bits_needed_for(unsigned x) {
  return x < 2 ? x : 1 + bits_needed_for(x >> 1);
}

constexpr int32_t elems_to_fetch =
    int_ceil(NUMBER_PAR_PER_BOX / float(NUMBER_THREADS));
using etf_t = ac_int<bits_needed_for(elems_to_fetch + 1), false>;

void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                     sycl::device_ptr<box_str> d_box_gpu,
                     sycl::device_ptr<FOUR_VECTOR> d_rv_gpu,
                     sycl::device_ptr<fp> d_qv_gpu,
                     sycl::device_ptr<FOUR_VECTOR> d_fv_gpu,
                     sycl::nd_item<1> item_ct1) {
  auto rA_shared_ptr =
      group_local_memory_for_overwrite<FOUR_VECTOR[NUMBER_PAR_PER_BOX]>(
          item_ct1.get_group());
  auto &rA_shared = *rA_shared_ptr;
  auto rB_shared_ptr =
      group_local_memory_for_overwrite<FOUR_VECTOR[NUMBER_PAR_PER_BOX]>(
          item_ct1.get_group());
  auto &rB_shared = *rB_shared_ptr;
  auto qB_shared_ptr =
      group_local_memory_for_overwrite<double[NUMBER_PAR_PER_BOX]>(
          item_ct1.get_group());
  auto &qB_shared = *qB_shared_ptr;

  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  //	THREAD PARAMETERS
  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

  const int bx =
      item_ct1.get_group(0); // get current horizontal block index (0-n)
  const int tx =
      item_ct1.get_local_id(0); // get current horizontal thread index (0-n)

  // parameters
  fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;

  //------------------------------------------------------------------------------------------------------------------------------------------------------160
  //	Home box
  //------------------------------------------------------------------------------------------------------------------------------------------------------160

  //----------------------------------------------------------------------------------------------------------------------------------140
  //	Setup parameters
  //----------------------------------------------------------------------------------------------------------------------------------140

  // home box - box parameters
  int first_i = d_box_gpu[bx].offset;

  // home box - distance, force, charge and type parameters
  FOUR_VECTOR *rA = &d_rv_gpu[first_i];
  FOUR_VECTOR *fA = &d_fv_gpu[first_i];

  //----------------------------------------------------------------------------------------------------------------------------------140
  //	Copy to shared memory
  //----------------------------------------------------------------------------------------------------------------------------------140

  // home box - shared memory
  for (etf_t e = 0; e < elems_to_fetch; e++)
    if (e * NUMBER_THREADS + tx < NUMBER_PAR_PER_BOX)
      rA_shared[e * NUMBER_THREADS + tx] = rA[e * NUMBER_THREADS + tx];

  // loop over neiing boxes of home box
  for (int16_t k = 0; k < (1 + d_box_gpu[bx].nn); k++) {
    int pointer;
    if (k == 0)
      pointer = bx;
    else
      pointer = d_box_gpu[bx].nei[k - 1].number;

    // nei box - box parameters
    int16_t first_j = d_box_gpu[pointer].offset;

    // nei box - distance, (force), charge and (type) parameters
    FOUR_VECTOR *rB = &d_rv_gpu[first_j];
    fp *qB = &d_qv_gpu[first_j];

    // nei box - shared memory
    for (etf_t e = 0; e < elems_to_fetch; e++)
      if (e * NUMBER_THREADS + tx < NUMBER_PAR_PER_BOX) {
        rB_shared[e * NUMBER_THREADS + tx] = rB[e * NUMBER_THREADS + tx];
        qB_shared[e * NUMBER_THREADS + tx] = qB[e * NUMBER_THREADS + tx];
      }

    // synchronize threads because in next section each thread accesses
    // data brought in by different threads here
    item_ct1.barrier(sycl::access::fence_space::local_space);

// loop for the number of particles in the home box
#pragma unroll
    for (etf_t e = 0; e < elems_to_fetch; e++)
      if (const uint8_t wtx = e * NUMBER_THREADS + tx;
          wtx < NUMBER_PAR_PER_BOX) {
        fp v = 0;
        fp x = 0;
        fp y = 0;
        fp z = 0;

// loop for the number of particles in the current nei box
#pragma unroll UNROLL_FAC
        for (uint8_t j = 0; j < NUMBER_PAR_PER_BOX; j++) {
          fp r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v -
                  DOT((fp)rA_shared[wtx], (fp)rB_shared[j]);
          fp u2 = a2 * r2;
          fp vij = sycl::exp(-u2);
          fp fs = 2 * vij;

          THREE_VECTOR d;
          d.x = (fp)rA_shared[wtx].x - (fp)rB_shared[j].x;
          fp fxij = fs * d.x;
          d.y = (fp)rA_shared[wtx].y - (fp)rB_shared[j].y;
          fp fyij = fs * d.y;
          d.z = (fp)rA_shared[wtx].z - (fp)rB_shared[j].z;
          fp fzij = fs * d.z;

          v += (double)((fp)qB_shared[j] * vij);
          x += (double)((fp)qB_shared[j] * fxij);
          y += (double)((fp)qB_shared[j] * fyij);
          z += (double)((fp)qB_shared[j] * fzij);
        }

        fA[wtx].v += v;
        fA[wtx].x += x;
        fA[wtx].y += y;
        fA[wtx].z += z;
      }

    // synchronize after finishing force contributions from current nei
    // box not to cause conflicts when starting next box
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}
