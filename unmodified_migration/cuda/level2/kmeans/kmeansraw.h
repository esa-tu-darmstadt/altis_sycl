///-------------------------------------------------------------------------------------------------
// file:	kmeansaw.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef __KMEANS_RAW_H__
#define __KMEANS_RAW_H__

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <vector>
#include <set>
#include <type_traits>
#include "genericvector.h"
#include "kmeans-common.h"
#include "cudacommon.h"
#include "ResultDatabase.h"
#include <assert.h>
#include <cfloat>
#include <iostream>
#include <chrono>

#include <cmath>

typedef double (*LPFNKMEANS)(ResultDatabase &DB,
    const int nSteps,
    void* h_Points,
    void* h_Centers,
    const int nPoints,
    const int nCenters,
    bool bVerify,
    bool bVerbose);

typedef void (*LPFNBNC)(ResultDatabase &DB,
    char* szFile,
    LPFNKMEANS lpfn,
    int nSteps,
    int nSeed,
    bool bVerify,
    bool bVerbose);

static dpct::constant_memory<float, 1> d_cnst_centers(CONSTMEMSIZE /
                                                      sizeof(float));

template <int R, int C> 
class centersmanagerRO {
protected: 
    float * m_pG;
    float * m_pRO;
public:
    centersmanagerRO(float * pG) : m_pG(pG), m_pRO(NULL) {} 
    bool useROMem() { return R*C<CONSTMEMSIZE/sizeof(float); }
    float * dataRW() { return m_pG; } 
    float * dataRO() { return d_cnst_centers; }
    bool update(float * p, bool bHtoD=false) { return (bHtoD ? updateHtoD(p) : updateDtoD(p)); }
    bool updateHtoD(float *p) try {
        return DPCT_CHECK_ERROR(dpct::get_default_queue()
                                    .memcpy(d_cnst_centers.get_ptr(), p,
                                            sizeof(float) * R * C)
                                    .wait()) == 0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    bool updateDtoD(float *p) try {
        return DPCT_CHECK_ERROR(dpct::get_default_queue()
                                    .memcpy(d_cnst_centers.get_ptr(), p,
                                            sizeof(float) * R * C)
                                    .wait()) == 0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
};

template <int R, int C> 
class centersmanagerGM {
protected: 
    float * m_pG;
    float * m_pRO;
public:
    centersmanagerGM(float * pG) : m_pG(pG), m_pRO(NULL) {} 
    bool useROMem() { return false; }
    float * dataRW() { return m_pG; }
    float * dataRO() { return m_pG; }
    bool update(float * p, bool bHtoD=false) { return true; }
    bool updateHtoD(float * p) { return true; }
    bool updateDToD(float * p) { return true; }
};


template <int R, int C>
void resetExplicit(float * pC, int * pCC, const sycl::nd_item<3> &item_ct1)  {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= C) return;
    // printf("resetExplicit: pCC[%d]=>%.3f/%d\n", idx, pC[idx*R], pCC[idx]);
	for(int i=0;i<R;i++) 
		pC[idx*R+i] = 0.0f;
    pCC[idx] = 0;
}

template <int R, int C>
void resetExplicitColumnMajor(float * pC, int * pCC,
                              const sycl::nd_item<3> &item_ct1)  {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= C) return;
    // printf("resetExplicitColumnMajor: pCC[%d]=>%.3f/%d\n", idx, pC[idx*R], pCC[idx]);
	for(int i=0;i<R;i++) 
		pC[(idx*C)+i] = 0.0f;
    pCC[idx] = 0;
}

template <int R, int C>
void finalizeCentersBasic(float * pC, int * pCC,
                          const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= C) return;
    int nNumerator = pCC[idx];
    if(nNumerator == 0) {       
	    for(int i=0;i<R;i++) 
            pC[(idx*R)+i] = 0.0f;
    } else {
	    for(int i=0;i<R;i++) 
		    pC[(idx*R)+i] /= pCC[idx];
    }
}

template <int R, int C>
void finalizeCentersShmap(float * pC, int * pCC,
                          const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(idx >= R*C) return;
    int cidx = idx/R;
    // int nNumerator = pCC[idx];   TODO check
    int nNumerator = pCC[cidx];
    if(nNumerator == 0) {       
        pC[idx] = 0.0f;
    } else {
        pC[idx] /= pCC[cidx];
    }
}

template <int R, int C>
void accumulateCenters(float * pP, float * pC, int * pCC, int * pCI, int nP,
                       const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= nP) return;
	int clusterid = pCI[idx];
	for(int i=0;i<R;i++)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &pC[(clusterid * R) + i], pP[(idx * R) + i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[clusterid], 1);
}

template<int R, int C> 
void accumulateCentersColumnMajor(float * pP, float * pC, int * pCC, int * pCI, int nP,
                                  const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= nP) return;
	int clusterid = pCI[idx];
	for(int i=0;i<R;i++) {
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &pC[clusterid + (i * C)], pP[idx + (i * nP)]);
        }
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[clusterid], 1);
}

template<int R, int C> 
void finalizeCentersColumnMajor(float * pC, int * pCC,
                                const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= C) return;
    
    int nNumerator = pCC[idx];
    if(nNumerator) {
	    for(int i=0;i<R;i++) {
		    pC[idx+(i*C)] /= pCC[idx];
        }
    } else {
	    for(int i=0;i<R;i++) {
		    pC[idx+(i*C)] = 0.0f;
        }
    }
}

template<int R, int C> 
void finalizeCentersColumnMajorShmap(float * pC, int * pCC,
                                     const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    int cidx = idx % C;
    if(cidx<C&&idx<R*C) {
        int nNumerator = pCC[cidx];
        if(nNumerator) {
            pC[idx] /= pCC[cidx];
        } else {
            pC[idx] = 0.0f;
        }
    }
}


template <int R, int C, bool ROWMAJ=true> 
class accumulatorGM {
public:

    static void reset(float * pC, int * pCC) { 
        if (ROWMAJ) {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, iDivUp(C, THREADBLOCK_SIZE)) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            resetExplicit<R, C>(pC, pCC, item_ct1);
                      });
        } else {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, iDivUp(C, THREADBLOCK_SIZE)) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            resetExplicitColumnMajor<R, C>(pC, pCC, item_ct1);
                      });
        }
    }

    static void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        const uint nPointsBlocks = iDivUp(nP, THREADBLOCK_SIZE);
        if (ROWMAJ) {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, nPointsBlocks) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            accumulateCenters<R, C>(pP, pC, pCC, pCI, nP,
                                                    item_ct1);
                      });
        } else {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, nPointsBlocks) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            accumulateCentersColumnMajor<R, C>(pP, pC, pCC, pCI,
                                                               nP, item_ct1);
                      });
        }
    }

    static void finalize(float * pC, int * pCC) {
        const int nAccums = iDivUp(C, THREADBLOCK_SIZE);
        if (ROWMAJ) {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, nAccums) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            finalizeCentersBasic<R, C>(pC, pCC, item_ct1);
                      });
        } else {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, nAccums) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            finalizeCentersColumnMajor<R, C>(pC, pCC, item_ct1);
                      });
        }
    }
};

template <int R, int C, bool ROWMAJ=true> 
class accumulatorGMMS {
public:

    static void reset(float * pC, int * pCC) {
        checkCudaErrors(
            DPCT_CHECK_ERROR(dpct::get_default_queue()
                                 .memset(pC, 0, R * C * sizeof(float))
                                 .wait()));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_default_queue().memset(pCC, 0, C * sizeof(int)).wait()));
    }

    static void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        accumulatorGM<R,C,ROWMAJ>::accumulate(pP, pC, pCC, pCI, nP);
    }

    static void finalize(float * pC, int * pCC) {
        accumulatorGM<R,C,ROWMAJ>::finalize(pC, pCC);
    }
};

template<int R, int C> 
void accumulateSM_RCeqBlockSize(float * pP, float * pC, int * pCC, int * pCI, int nP,
                                const sycl::nd_item<3> &item_ct1, float *accums,
                                int *cnts) {

    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    dassert(R*C <= 1024);
    dassert(threadIdx.x < R*C);
    if (item_ct1.get_local_id(2) < R * C) accums[item_ct1.get_local_id(2)] =
        0.0f;
    if (item_ct1.get_local_id(2) < C) cnts[item_ct1.get_local_id(2)] = 0;
    /*
    DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &accums[(clusterid * R) + i], pP[(idx * R) + i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnts[clusterid], 1);
    }
    /*
    DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < R * C)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pC[item_ct1.get_local_id(2)], accums[item_ct1.get_local_id(2)]);
    if (item_ct1.get_local_id(2) < C)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[item_ct1.get_local_id(2)], cnts[item_ct1.get_local_id(2)]);
}

template<int R, int C, int nLDElemsPerThread> 
void accumulateSM(float * pP, float * pC, int * pCC, int * pCI, int nP,
                  const sycl::nd_item<3> &item_ct1, float *accums, int *cnts) {

    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if (item_ct1.get_local_id(2) < C) cnts[item_ct1.get_local_id(2)] = 0;
        for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nCenterIdx = ridx*C;
                int nLDIdx = item_ct1.get_local_id(2) + nCenterIdx;
        if(nLDIdx < R*C) accums[nLDIdx] = 0.0f;
    }
    /*
    DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &accums[(clusterid * R) + i], pP[(idx * R) + i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnts[clusterid], 1);
    }
    /*
    DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < C)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[item_ct1.get_local_id(2)], cnts[item_ct1.get_local_id(2)]);
        for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nCenterIdx = ridx*C;
                int nLDIdx = item_ct1.get_local_id(2) + nCenterIdx;
        if(nLDIdx < R*C)
                        dpct::atomic_fetch_add<
                            sycl::access::address_space::generic_space>(
                            &pC[nLDIdx], accums[nLDIdx]);
    }
}

template<int R, int C> 
void accumulateSMColumnMajor_RCeqBS(float * pP, float * pC, int * pCC, int * pCI, int nP,
                                    const sycl::nd_item<3> &item_ct1,
                                    float *accums, int *cnts) {

    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if (item_ct1.get_local_id(2) < R * C) accums[item_ct1.get_local_id(2)] =
        0.0f;
    if (item_ct1.get_local_id(2) < C) cnts[item_ct1.get_local_id(2)] = 0;
    /*
    DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &accums[clusterid + (C * i)], pP[idx + (nP * i)]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnts[clusterid], 1);
    }
    /*
    DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < R * C)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pC[item_ct1.get_local_id(2)], accums[item_ct1.get_local_id(2)]);
    if (item_ct1.get_local_id(2) < C)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[item_ct1.get_local_id(2)], cnts[item_ct1.get_local_id(2)]);
}

template<int R, int C> 
void accumulateSMColumnMajor(float * pP, float * pC, int * pCC, int * pCI, int nP,
                             const sycl::nd_item<3> &item_ct1, float *accums,
                             int *cnts) {

    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if (item_ct1.get_local_id(2) < C) {
        cnts[item_ct1.get_local_id(2)] = 0;
        for(int i=0;i<R;i++)
            accums[item_ct1.get_local_id(2) * R + i] = 0.0f;
    }
    /*
    DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &accums[clusterid + (C * i)], pP[idx + (nP * i)]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnts[clusterid], 1);
    }
    /*
    DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < C) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &pCC[item_ct1.get_local_id(2)], cnts[item_ct1.get_local_id(2)]);
        for(int i=0;i<R;i++)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &pC[item_ct1.get_local_id(2) * R + i],
                accums[item_ct1.get_local_id(2) * R + i]);
    }
}

template <int R, int C, bool ROWMAJ=true> 
class accumulatorSM {
public:
    static void reset(float * pC, int * pCC) { accumulatorGM<R,C,ROWMAJ>::reset(pC, pCC); }
    static void finalize(float * pC, int * pCC) { accumulatorGMMS<R,C,ROWMAJ>::finalize(pC, pCC); }

    static void
    accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) { 
        if (R*C>SHMEMACCUM_FLOATS) {
            accumulatorGM<R,C,ROWMAJ>::accumulate(pP, pC, pCC, pCI, nP);
        } else {
            if(R*C<MAXBLOCKTHREADS) {
                accumulateRCeqBS(pP, pC, pCC, pCI, nP);
            } else {
                accumulateGeneral(pP, pC, pCC, pCI, nP);
            }
        }
    }

protected:

    static void
    accumulateRCeqBS(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        // if R*C < max thread group, we can reset and merge
        // accumulators in a more efficient way than if R*C is larger
        const int nMinRCSize = std::max(R * C, THREADBLOCK_SIZE);
        const int nBlockSize = MAXBLOCKTHREADS; // min(nMinRCSize, MAXBLOCKTHREADS);
        const int nMinAccumBlocks = iDivUp(R*C, nBlockSize);
        const int nMinPointsBlocks = iDivUp(nP, nBlockSize);
        const int nBlocks = std::max(nMinAccumBlocks, nMinPointsBlocks);
        if(ROWMAJ) {
            // printf("accumulateSM_RCeqBlockSize<%d,%d><<<%d,%d>>>(...)\n", R, C, nBlocks, nBlockSize);
            /*
            DPCT1049:51: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<float, 1> accums_acc_ct1(
                            sycl::range<1>(R * C), cgh);
                        sycl::local_accessor<int, 1> cnts_acc_ct1(
                            sycl::range<1>(C), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, nBlocks) *
                                    sycl::range<3>(1, 1, nBlockSize),
                                sycl::range<3>(1, 1, nBlockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  accumulateSM_RCeqBlockSize<R, C>(
                                      pP, pC, pCC, pCI, nP, item_ct1,
                                      accums_acc_ct1.get_pointer(),
                                      cnts_acc_ct1.get_pointer());
                            });
                  });
        } else {
            // printf("accumulateSMColumnMajor_RCeqBS<%d,%d><<<%d,%d>>>(...)\n", R, C, nBlocks, nBlockSize);
            /*
            DPCT1049:52: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<float, 1> accums_acc_ct1(
                            sycl::range<1>(R * C), cgh);
                        sycl::local_accessor<int, 1> cnts_acc_ct1(
                            sycl::range<1>(C), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, nBlocks) *
                                    sycl::range<3>(1, 1, nBlockSize),
                                sycl::range<3>(1, 1, nBlockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  accumulateSMColumnMajor_RCeqBS<R, C>(
                                      pP, pC, pCC, pCI, nP, item_ct1,
                                      accums_acc_ct1.get_pointer(),
                                      cnts_acc_ct1.get_pointer());
                            });
                  });
        }
    }

    static void
    accumulateGeneral(float * pP, float * pC, int * pCC, int * pCI, int nP) {
		const int nBlockSize = THREADBLOCK_SIZE;
		const int nElemsPerThread = (R*C)/nBlockSize;
        const int nMinAccumBlocks = iDivUp(C, nBlockSize);
        const int nMinPointsBlocks = iDivUp(nP, nBlockSize);
        const int nBlocks = std::max(nMinAccumBlocks, nMinPointsBlocks);
        if(ROWMAJ) {
			// printf("accumulateSM<%d, %d, %d><<<%d,%d>>>()\n", R,C,nElemsPerThread,nBlocks,nBlockSize);
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<float, 1> accums_acc_ct1(
                            sycl::range<1>(R * C), cgh);
                        sycl::local_accessor<int, 1> cnts_acc_ct1(
                            sycl::range<1>(C), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, nBlocks) *
                                    sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                                sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  accumulateSM<R, C, nElemsPerThread>(
                                      pP, pC, pCC, pCI, nP, item_ct1,
                                      accums_acc_ct1.get_pointer(),
                                      cnts_acc_ct1.get_pointer());
                            });
                  });
        } else {
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<float, 1> accums_acc_ct1(
                            sycl::range<1>(R * C), cgh);
                        sycl::local_accessor<int, 1> cnts_acc_ct1(
                            sycl::range<1>(C), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, nBlocks) *
                                    sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                                sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  accumulateSMColumnMajor<R, C>(
                                      pP, pC, pCC, pCI, nP, item_ct1,
                                      accums_acc_ct1.get_pointer(),
                                      cnts_acc_ct1.get_pointer());
                            });
                  });
        }
    }
};

template <int R, int C, bool ROWMAJ=true> 
class accumulatorSMMAP {
public:
    static void reset(float * pC, int * pCC) { accumulatorSM<R,C,ROWMAJ>::reset(pC, pCC); }
    static void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) { accumulatorSM<R,C,ROWMAJ>::accumulate(pP, pC, pCC, pCI, nP); }
    static void finalize(float * pC, int * pCC) { 
        if (ROWMAJ) {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1,
                                         iDivUp(R * C, THREADBLOCK_SIZE)) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            finalizeCentersShmap<R, C>(pC, pCC, item_ct1);
                      });
        } else {
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1,
                                         iDivUp(R * C, THREADBLOCK_SIZE)) *
                              sycl::range<3>(1, 1, THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) {
                            finalizeCentersColumnMajorShmap<R, C>(pC, pCC,
                                                                  item_ct1);
                      });
        }
    }
};


template <int R, int C> 
float 
_vdistancef(float * a, float * b) {
    float accum = 0.0f;
    for(int i=0; i<R; i++) {
        float delta = a[i]-b[i];
        accum += delta*delta;
    }
    return sycl::sqrt(accum);
}

template<int R, int C>
float 
_vdistancefcm(
    int nAIndex,
    float * pAVectors,    
    int nAVectorCount,
    int nBIndex,
    float * pBVectors,
    int nBVectorCount
    ) 
{
    // assumes perfect packing 
    // (no trailing per-row pitch) 
    float accum = 0.0f;
    float * pAStart = &pAVectors[nAIndex];
    float * pBStart = &pBVectors[nBIndex];
    for(int i=0; i<R; i++) {
        float a = (*(pAStart + i*nAVectorCount));
        float b = (*(pBStart + i*nBVectorCount));
        float delta = a-b;
        accum += delta*delta;
    }
    return sycl::sqrt(accum);
}

template <int R, int C>
int 
nearestCenter(float * pP, float * pC) {
    float mindist = FLT_MAX;
    int minidx = 0;
	int clistidx = 0;
    for(int i=0; i<C;i++) {
		clistidx = i*R;
        float dist = _vdistancef<R,C>(pP, &pC[clistidx]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template<int R, int C>
int 
nearestCenterColumnMajor(float * pP, float * pC, int nPointIndex, int nP){
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<C;i++) {
        float dist = _vdistancefcm<R,C>(nPointIndex, pP, nP, i, pC, C);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template <int R, int C, bool bRO> 
void 
mapPointsToCenters(float * pP, float * pC, int * pCI, int nP,
                   const sycl::nd_item<3> &item_ct1, float *d_cnst_centers) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= nP) return;
	pCI[idx] = nearestCenter<R,C>(&pP[idx*R], bRO ? d_cnst_centers : pC);
}

template<int R, int C, bool bRO> 
void 
mapPointsToCentersColumnMajor(float * pP, float * pC, int * pCI, int nP,
                              const sycl::nd_item<3> &item_ct1,
                              float *d_cnst_centers) {
        int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if(idx >= nP) return;
	pCI[idx] = nearestCenterColumnMajor<R,C>(pP, bRO ? d_cnst_centers : pC, idx, nP);
}


template<int R, 
         int C, 
         typename CM,
         typename SM,
         bool ROWMAJ=true> 
class kmeansraw {
public:

    kmeansraw(
	    int     nSteps, 
	    float * d_Points,  
	    float * d_Centers,
	    int *   d_ClusterCounts,
	    int *   d_ClusterIds,
	    int     nPoints,
	    int     nCenters) : m_nSteps(nSteps),
                                 m_dPoints(d_Points),
                                 m_dCenters(d_Centers),
                                 m_dClusterCounts(d_ClusterCounts),
                                 m_dClusterIds(d_ClusterIds),
                                 m_nPoints(nPoints),
                                 m_nCenters(nCenters),
                                 m_centers(d_Centers) {}

    bool execute() {
        assert(m_nCenters == C);
        const uint nCentersBlocks = iDivUp(m_nCenters, THREADBLOCK_SIZE);
        const uint nPointsBlocks = iDivUp(m_nPoints, THREADBLOCK_SIZE);
        for (int i=0; i<m_nSteps; i++) {
            _V(updatecentersIn(m_dCenters));    // deal with centers data
		    _V(mapCenters(m_dPoints, m_dClusterIds, m_nPoints));
		    _V(resetAccumulators(m_dCenters, m_dClusterCounts));
		    _V(accumulate(m_dPoints, m_dCenters, m_dClusterCounts, m_dClusterIds, m_nPoints));
		    _V(finalizeAccumulators(m_dCenters, m_dClusterCounts));
	    }
        return true;
    }


protected:

    int     m_nSteps;
    float * m_dPoints;
    float * m_dCenters;
    int *   m_dClusterCounts;
    int *   m_dClusterIds;
    int     m_nPoints;
    int     m_nCenters;

    CM m_centers; 
    SM m_accumulator;
    bool updatecentersIn(float * p_Centers) { return m_centers.update(p_Centers); }
    bool initcentersInput(float * p_Centers) { return m_centers.updateHtoD(p_Centers); }
    void resetAccumulators(float * pC, int * pCC) { m_accumulator.reset(pC, pCC); }
    void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) { m_accumulator.accumulate(pP, pC, pCC, pCI, nP); }
    void finalizeAccumulators(float * pC, int * pCI) { m_accumulator.finalize(pC, pCI); }
    float * centers() { return m_centers.data(); }

    void mapCenters(float * pP, int * pCI, int nP) {
        const uint nCentersBlocks = iDivUp(C, THREADBLOCK_SIZE);
        const uint nPointsBlocks = iDivUp(nP, THREADBLOCK_SIZE);	    
        float * pC = m_centers.dataRO();
        if (ROWMAJ) {
            if (m_centers.useROMem()) {
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  d_cnst_centers.init();

                                  auto d_cnst_centers_ptr_ct1 =
                                      d_cnst_centers.get_ptr();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(
                                          sycl::range<3>(1, 1, nPointsBlocks) *
                                              sycl::range<3>(1, 1,
                                                             THREADBLOCK_SIZE),
                                          sycl::range<3>(1, 1,
                                                         THREADBLOCK_SIZE)),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            mapPointsToCenters<R, C, true>(
                                                pP, pC, pCI, nP, item_ct1,
                                                d_cnst_centers_ptr_ct1);
                                      });
                            });
            } else {
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  d_cnst_centers.init();

                                  auto d_cnst_centers_ptr_ct1 =
                                      d_cnst_centers.get_ptr();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(
                                          sycl::range<3>(1, 1, nPointsBlocks) *
                                              sycl::range<3>(1, 1,
                                                             THREADBLOCK_SIZE),
                                          sycl::range<3>(1, 1,
                                                         THREADBLOCK_SIZE)),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            mapPointsToCenters<R, C, false>(
                                                pP, pC, pCI, nP, item_ct1,
                                                d_cnst_centers_ptr_ct1);
                                      });
                            });
            }
        } else {
            if (m_centers.useROMem()) {
                        dpct::get_default_queue().submit(
                            [&](sycl::handler &cgh) {
                                  d_cnst_centers.init();

                                  auto d_cnst_centers_ptr_ct1 =
                                      d_cnst_centers.get_ptr();

                                  cgh.parallel_for(
                                      sycl::nd_range<3>(
                                          sycl::range<3>(1, 1, nPointsBlocks) *
                                              sycl::range<3>(1, 1,
                                                             THREADBLOCK_SIZE),
                                          sycl::range<3>(1, 1,
                                                         THREADBLOCK_SIZE)),
                                      [=](sycl::nd_item<3> item_ct1) {
                                            mapPointsToCentersColumnMajor<R, C,
                                                                          true>(
                                                pP, pC, pCI, nP, item_ct1,
                                                d_cnst_centers_ptr_ct1);
                                      });
                            });
            } else {
                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                              d_cnst_centers.init();

                              auto d_cnst_centers_ptr_ct1 =
                                  d_cnst_centers.get_ptr();

                              cgh.parallel_for(
                                  sycl::nd_range<3>(
                                      sycl::range<3>(1, 1, nPointsBlocks) *
                                          sycl::range<3>(1, 1,
                                                         THREADBLOCK_SIZE),
                                      sycl::range<3>(1, 1, THREADBLOCK_SIZE)),
                                  [=](sycl::nd_item<3> item_ct1) {
                                        mapPointsToCentersColumnMajor<R, C,
                                                                      false>(
                                            pP, pC, pCI, nP, item_ct1,
                                            d_cnst_centers_ptr_ct1);
                                  });
                        });
            }
        }
    }

    static const int MAX_CHAR_PER_LINE = 512;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads an input file. </summary>
    ///
    /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. Adapted from code in
    ///             serban/kmeans which appears adapted from code in STAMP (variable names only
    ///             appear to be changed), or perhaps each was adapted from code in other places.
    ///             
    ///             ****************************** Note that AMP requires template int args to be
    ///             statically known at compile time. Hence, for now, you have to have built this
    ///             program with DEFAULTRANK defined to match the DEFAULTRANK of your input file! This restriction
    ///             is easy to fix, but not the best use of my time at the moment...
    ///             ******************************.
    ///             </remarks>
    ///
    /// <param name="filename">     If non-null, filename of the file. </param>
    /// <param name="points">       The points. </param>
    /// <param name="numObjs">      If non-null, number of objects. </param>
    /// <param name="numCoords">    If non-null, number of coords. </param>
    /// <param name="_debug">       The debug flag. </param>
    ///
    /// <returns>   The input. </returns>
    ///-------------------------------------------------------------------------------------------------

    static int
    ReadInput(
        char * filename,
        std::vector<pt<R>>& points,
        int * numObjs,
        int * numCoords,
        int _debug
        ) 
    {
        #pragma warning(disable:4996)
        float **objects;
        int     i, j, len;

        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            safe_exit(-1);
        }

        /* first find the number of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = (int)strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);
        if (_debug) printf("lineLen = %d\n",lineLen);

        /* find the no. objects of each object */
        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first coordiinate): numCoords = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }

        /* allocate space for objects[][] and read all objects */
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        /* read all objects */
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = (float)atof(strtok(NULL, " ,\t\n"));
            i++;
        }

		points.reserve(static_cast<int>(*numObjs));
        for(int idx=0; idx<(*numObjs); idx++) {
            pt<R> point(objects[idx]);
            points.push_back(point);
        }

        fclose(infile);
        free(line);
        free(objects[0]);
        free(objects);
        return 0;
        #pragma warning(default:4996)
    }

    static void
    ChooseInitialCenters(
        std::vector<pt<R>> &points,
        std::vector<pt<R>> &centers,
        std::vector<pt<R>> &refcenters,
        int nRandomSeed
        )
    {
        srand(nRandomSeed);
        std::set<int> chosenidx;
        while(chosenidx.size() < (size_t)C) {
            // sets don't allow dups...
            int idx = rand() % points.size();
            chosenidx.insert(idx);
        }
        std::set<int>::iterator si;
        for(si=chosenidx.begin(); si!=chosenidx.end(); si++) {
            centers.push_back(points[*si]);
            refcenters.push_back(points[*si]);
        }    
    }

    static void
    PrintCenters(
        FILE * fp,
        std::vector<pt<R>> &centers,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        fprintf(fp, "\n");
        int nRows = 0;
        for(auto x = centers.begin(); x != centers.end(); x++) {
            if(++nRows > nRowLimit) {
                fprintf(fp, "...");
                break;
            }
            x->dump(fp,nColLimit);
        }
        fprintf(fp, "\n");
    }

    static void PrintCenters(
        FILE * fp,
        pt<R> * pCenters,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        nRowLimit = (nRowLimit == 0) ? C : std::min(nRowLimit, C);
            for(int i=0; i<nRowLimit; i++)
            pCenters[i].dump(fp,nColLimit);
        if(nRowLimit < C) 
            fprintf(fp, "...");
        fprintf(fp, "\n");
    }

    static void MyPrintCenters(
        float * pCenters,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        nRowLimit = (nRowLimit == 0) ? C : std::min(nRowLimit, C);

        for (int r = 0; r < nRowLimit; r++) {
            for (int c = 0; c < nColLimit; c++) {
                int index = r * C + c;
                std::cout << pCenters[index] << "  ";
            }
            std::cout << std::endl;
        }
    }

    static void PrintCentersTransposed(
        FILE * fp,
        float * pCenters,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        nRowLimit = (nRowLimit == 0) ? C : std::min(nRowLimit, C);
        nColLimit = (nColLimit == 0) ? R : std::min(nColLimit, R);
        for(int c=0; c<nRowLimit; c++) {
	        for(int i=0; i<nColLimit; i++) {
                int fidx = i*C+c;
                if(i>0) fprintf(fp, ", ");
                fprintf(fp, "%.3f", pCenters[fidx]);
            }
            if(nColLimit < R)
                fprintf(fp, "....");
            fprintf(fp, "\n");
        }
        if(nRowLimit < C) 
            fprintf(fp, "....\n");
    }

    static void
    PrintResults(
        FILE * fp,
        std::vector<pt<R>> &centers,
        std::vector<pt<R>> &refcenters,
        bool bSuccess,
        double dKmeansExecTime,
        double dKmeansCPUTime,
        bool bVerbose=false
        )
    {
        fprintf(fp, "%s: GPU: %.5f sec, CPU: %.5f sec\n", 
                (bSuccess?"SUCCESS":"FAILURE"),
                dKmeansExecTime, dKmeansCPUTime);
        if(!bSuccess || bVerbose) {
            fprintf(fp, "final centers:\n");
		    PrintCenters(fp, centers);
            fprintf(fp, "reference centers:\n");
		    PrintCenters(fp, refcenters);
        }
    }

    static bool
    CompareResults_obs(
        std::vector<pt<R>> &centers,    
        std::vector<pt<R>> &refcenters,
        float EPSILON=0.0001f,
        bool bVerbose=true,
        int nRowLimit=16
        )
    {
        std::set<pt<R>*> unmatched;
        for(auto vi=centers.begin(); vi!=centers.end(); vi++) {
            bool bFound = false;
            for(auto xi=refcenters.begin(); xi!=refcenters.end(); xi++) {
                if(EPSILON > hdistance(*vi, *xi)) {
                    bFound = true;
                    break;
                }
            }
            if(!bFound)
                unmatched.insert(&(*vi));
        }
        bool bSuccess = unmatched.size() == 0;
        if(bVerbose && !bSuccess) {
            int nDumped = 0;
            fprintf(stderr, "Could not match %d centers:\n", unmatched.size());
            for(auto si=unmatched.begin(); si!=unmatched.end(); si++) {
                if(++nDumped > nRowLimit) {
                    printf("...\n");
                    break;
                }
                (*si)->dump(stderr);        
            }
        }
        return bSuccess;
    }

    static bool
    CompareResults(
        std::vector<pt<R>> &centers,    
        std::vector<pt<R>> &refcenters,
        float EPSILON=0.0001f,
        bool bVerbose=true,
        int nRowLimit=16
        )
    {
        int nRows=0;
        std::map<int, pt<R>*> unmatched;
        std::map<int, float> unmatched_deltas;
        std::map<int, int> matched;
        std::map<int, int> revmatched;
        int nCenterIdx=0;    
        for(auto vi=centers.begin(); vi!=centers.end(); vi++, nCenterIdx++) {
            bool bFound = false;        
            if(EPSILON*R > hdistance(*vi, refcenters[nCenterIdx])) {
                bFound = true;
                matched[nCenterIdx] = nCenterIdx;
                revmatched[nCenterIdx] = nCenterIdx;
            } else {
                int nRefIdx=0;
                for(auto xi=refcenters.begin(); xi!=refcenters.end(); xi++, nRefIdx++) {
                    if(EPSILON*R > hdistance(*vi, *xi)) {
                        bFound = true;
                        matched[nCenterIdx] = nRefIdx;
                        revmatched[nRefIdx] = nCenterIdx;
                        break;
                    }
                }
            }
            if(!bFound) {
                unmatched[nCenterIdx] = (&(*vi));
                unmatched_deltas[nCenterIdx] = hdistance(*vi, refcenters[nCenterIdx]);
            }
        }
        bool bSuccess = unmatched.size() == 0;
        if (bVerbose && !bSuccess) {
            std::cerr << "Could not match " << unmatched.size() << " centers: " << std::endl;
            for(auto si=unmatched.begin(); si!=unmatched.end(); si++) {
                if(++nRows > nRowLimit) {
                    fprintf(stderr, "...\n");
                    break;
                }
                fprintf(stdout, "IDX(%d): ", si->first);
                (si->second)->dump(stderr);        
            }
        }
        return bSuccess;
    }

public:

    static float * transpose(float * pP, int nP, float * pTxP=NULL) {
        size_t  uiPointsBytes = nP * R * sizeof(float);
        float * pInput = reinterpret_cast<float*>(pP);
        if(pTxP == NULL) {
            pTxP = (float*)malloc(uiPointsBytes);
            /* ALTIS_MALLOC((void *)pTxP, uiPointsBytes); */
        }
        for(int i=0; i<nP; i++) {
            for(int j=0; j<R; j++) { 
                int nInputIdx = (i*R)+j;
                int nTxIdx = j*nP+i;
                pTxP[nTxIdx] = pInput[nInputIdx];
            }
        }
        return pTxP;
    }

    static float * transpose(pt<R>* h_Points, int nP, float * pTxP=NULL) {
        float * pInput = reinterpret_cast<float*>(h_Points);
        return transpose(pInput, nP, pTxP);
    }

    static float * rtranspose(float * pP, int nP, float * pTxP=NULL) {
        size_t  uiPointsBytes = nP * R * sizeof(float);
        float * pInput = reinterpret_cast<float*>(pP);
        if(pTxP == NULL) {
            pTxP = (float*)malloc(uiPointsBytes);
            /* ALTIS_MALLOC((void *)pTxP, uiPointsBytes); */
        }
        for(int i=0; i<nP; i++) {
            for(int j=0; j<R; j++) { 
                int nTxIdx = (i*R)+j;
                int nInputIdx = j*nP+i;
                pTxP[nTxIdx] = pInput[nInputIdx];
            }
        }
        return pTxP;
    }

    static float * rtranspose(pt<R>* h_Points, int nP, float * pTxP=NULL) {
        float * pInput = reinterpret_cast<float*>(h_Points);
        return rtranspose(pInput, nP, pTxP);
    }

    static double
    benchmark(
        ResultDatabase &DB,
	    const int nSteps,
	    void * lpvPoints,
	    void * lpvCenters,
	    const int nPoints,
	    const int nCenters,
	    bool bVerify,
	    bool bVerbose
	    )
    {
        assert(nCenters == C);

        float * h_TxPoints = NULL;
        float * h_TxCenters = NULL;
	    pt<R> * h_InPoints = reinterpret_cast<pt<R>*>(lpvPoints);
	    pt<R> * h_InCenters = reinterpret_cast<pt<R>*>(lpvCenters);        
	    float * h_Points = reinterpret_cast<float*>(h_InPoints);
	    float * h_Centers = reinterpret_cast<float*>(h_InCenters);
        if (!ROWMAJ) {
            h_TxPoints = transpose(h_InPoints, nPoints);
            h_TxCenters = transpose(h_InCenters, nCenters);
            h_Points = h_TxPoints;
            h_Centers = h_TxCenters;
        }

        float * d_Points = NULL;
	    float * d_Centers = NULL;
	    int *   d_ClusterIds = NULL;
	    int *   d_ClusterCounts = NULL;
	    size_t  uiPointsBytes = nPoints * R * sizeof(float);
	    size_t  uiCentersBytes = nCenters * R * sizeof(float);
	    size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	    size_t  uiClusterCountsBytes = nCenters * sizeof(int);

        //INFORM(bVerbose, "Initializing data...\n");
        #ifndef UNIFIED_MEMORY
        /*
        DPCT1064:384: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_Points = (float *)sycl::malloc_device(
                                 uiPointsBytes, dpct::get_default_queue())));
        /*
        DPCT1064:385: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_Centers = (float *)sycl::malloc_device(
                                 uiCentersBytes, dpct::get_default_queue())));
        /*
        DPCT1064:386: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_ClusterIds = (int *)sycl::malloc_device(
                uiClusterIdsBytes, dpct::get_default_queue())));
        /*
        DPCT1064:387: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_ClusterCounts = (int *)sycl::malloc_device(
                uiClusterCountsBytes, dpct::get_default_queue())));
            checkCudaErrors(
                DPCT_CHECK_ERROR(dpct::get_default_queue()
                                     .memcpy(d_Points, h_Points, uiPointsBytes)
                                     .wait()));
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_default_queue()
                    .memcpy(d_Centers, h_Centers, uiCentersBytes)
                    .wait()));
#else
        d_Points = h_Points;
        d_Centers = h_Centers;
        ALTIS_CUDA_MALLOC(d_ClusterIds, uiClusterIdsBytes);
        ALTIS_CUDA_MALLOC(d_ClusterCounts, uiClusterCountsBytes);
        #endif
	    //INFORM(bVerbose, "Starting up kmeans-raw...\n\n");

        // fprintf(stdout, "initial centers:\n");
	    // PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);

        kmeansraw<R,C,CM,SM,ROWMAJ>* pKMeans = 
            new kmeansraw<R,C,CM,SM,ROWMAJ>(nSteps,
                                            d_Points,
                                            d_Centers,
                                            d_ClusterCounts,
                                            d_ClusterIds,
                                            nPoints,
                                            nCenters);

        float elapsed;
        dpct::event_ptr start, stop;
        std::chrono::time_point<std::chrono::steady_clock> start_ct1;
        std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
        checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
        checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

        /*
        DPCT1012:380: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:381: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);

        (*pKMeans).execute();
            checkCudaErrors(DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()));

        /*
        DPCT1012:382: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:383: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            (elapsed =
                 std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));

        checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));

        char atts[1024];
        sprintf(atts, "iterations:%d, centers:%d, rank:%d", nSteps, C, R);
        DB.AddResult("kmeans total execution time", atts, "sec", elapsed * 1.0e-3);
        DB.AddResult("kmeans execution time per iteration", atts, "sec", elapsed * 1.0e-3 / nSteps);

	    if (bVerbose) {
		    uint byteCount = (uint)(uiPointsBytes + uiCentersBytes);
            DB.AddResult("kmeans thoughput", atts, "MB/sec", ((double)byteCount * 1.0e-6) / (elapsed * 1.0e-3));
		    //shrLogEx(LOGBOTH | MASTER, 0, "kmeans, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
					    //(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, THREADBLOCK_SIZE); 
	    }

	    if (bVerify) {
		    std::cout << " ...reading back GPU results" << std::endl;
            #ifndef UNIFIED_MEMORY
                    checkCudaErrors(DPCT_CHECK_ERROR(
                        dpct::get_default_queue()
                            .memcpy(h_Centers, d_Centers, uiCentersBytes)
                            .wait()));
#else
            /* h_Centers = d_Centers, the ptr will be freed in bncmain. */
            h_Centers = d_Centers;  // TODO check
            #endif
            if (!ROWMAJ) {
                rtranspose(h_TxCenters, nCenters, (float*)h_InCenters);
            }
	    }

	    //shrLog("cleaning up device resources...\n");
        #ifndef UNIFIED_MEMORY
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free((void *)d_Points, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free((void *)d_Centers, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(
                sycl::free((void *)d_ClusterIds, dpct::get_default_queue())));
            checkCudaErrors(DPCT_CHECK_ERROR(sycl::free(
                (void *)d_ClusterCounts, dpct::get_default_queue())));
#else
        ALTIS_FREE(d_ClusterIds);
        ALTIS_FREE(d_ClusterCounts);
        #endif
        if (!ROWMAJ) {
            // free(h_TxCenters);
            ALTIS_FREE(h_TxCenters);
            // free(h_TxPoints);
            ALTIS_FREE(h_TxPoints);
        }
        return (double)elapsed * 1.0e-3;
    }

    static float
    hdistance(
        pt<R> &a,
        pt<R> &b
        ) 
    {
        float accum = 0.0f;
        for(int i=0; i<R; i++) {
            float delta = a.m_v[i]-b.m_v[i];
            accum += delta*delta;
        }
        return sqrt(accum);
    }

    static int 
    NearestCenter(
        pt<R> &point,
        std::vector<pt<R>> &centers
        ) 
    {
        float mindist = FLT_MAX;
        int minidx = 0;
        for(size_t i=0; i<centers.size();i++) {
            float dist = hdistance(point, centers[i]);
            if(dist < mindist) {
                minidx = static_cast<int>(i);
                mindist = dist;
            }
        }
        return minidx;
    }

    static void
    MapPointsToCentersSequential(
        std::vector<pt<R>> &vcenters,
        std::vector<pt<R>> &vpoints,
        std::vector<int>& vclusterids
        )
    {
        int index = 0;
        for(auto vi=vpoints.begin(); vi!=vpoints.end(); vi++) 
            vclusterids[index++] = NearestCenter(*vi, vcenters);
    }

    static void
    UpdateCentersSequential(
        std::vector<pt<R>> &vcenters,
        std::vector<pt<R>> &vpoints,
        std::vector<int>& vclusterids
        )
    {
        std::vector<int> counts;
        for(size_t i=0; i<vcenters.size(); i++) {
            vcenters[i].set(0.0f);
            counts.push_back(0);
        }
        for(size_t i=0; i<vpoints.size(); i++) {
            int clusterid = vclusterids[i];
            vcenters[clusterid] += vpoints[i];
            counts[clusterid] += 1;
        }
        for(size_t i=0; i<vcenters.size(); i++) {
            vcenters[i] /= counts[i];
        }
    }

    static void 
    bncmain(
        ResultDatabase &DB,
        char * lpszInputFile, 
        LPFNKMEANS lpfnKMeans,
        int nSteps,
        int nSeed,
        bool bVerify,
        bool bVerbose
        )
    {
        int nP = 0;
        int nD = 0;
        std::vector<pt<R>> points;
        std::vector<pt<R>> centers;
        std::vector<pt<R>> refcenters;

        ReadInput(lpszInputFile, points, &nP, &nD, 0);
	    if (points.size()==0) {
		    fprintf(stderr, "Error loading points from %s!\n", lpszInputFile);
	    }
        std::vector<int> clusterids(points.size());
        std::vector<int> refclusterids(points.size());

        // be careful to make sure you're choosing the same
        // random seed every run. If you upgrade this code to actually
        // do divergence tests to decide when to terminate, it will be
        // important to use the same random seed every time, since the 
        // choice of initial centers can have a profound impact on the
        // number of iterations required to converge. Failure to be 
        // consistent will introduce a ton of noice in your data. 
    
        ChooseInitialCenters(points,                   // points to choose from              
                             centers,                  // destination array of initial centers
                             refcenters,               // save a copy for the reference impl to check
                             nSeed);                   // random seed. ACHTUNG! Beware benchmarkers

	    int nPoints = (int)points.size();
	    int nCenters = C;
	    size_t uiPointsBytes = nPoints * R * sizeof(float);
	    size_t uiCentersBytes = C * R * sizeof(float);
	    size_t uiClusterIdsBytes = nPoints * sizeof(int);
		
		bool bSuccess = false;
		double dAvgSecs = 0.0;
	    pt<R> *h_Points = NULL;
	    pt<R> *h_Centers = NULL;
	    // int * h_ClusterIds = NULL;
		bool bTooBig = (uiPointsBytes > UINT_MAX);

		// if the points won't fit in GPU memory, there is
		// no point in going through the exercise of watching 
		// the GPU exec fail (particularly if we still want to
		// collect the CPU comparison number). If it's obviously
		// too big, skip the CUDA rigmaroll.

		if (!bTooBig) {

			pt<R> *h_Points = (pt<R>*)malloc(uiPointsBytes);
            /* ALTIS_MALLOC(h_Points, uiPointsBytes); */
			pt<R> *h_Centers = (pt<R>*)malloc(uiCentersBytes);
            /* ALTIS_MALLOC(h_Centers, uiCentersBytes); */
			int * h_ClusterIds = (int*)malloc(uiClusterIdsBytes);
            // ALTIS_MALLOC(h_ClusterIds, uiClusterIdsBytes);
			// memset(h_ClusterIds, 0, uiClusterIdsBytes);

			pt<R>* pPoints = h_Points;
			for (auto vi=points.begin(); vi!=points.end(); vi++) {
				*pPoints++ = *vi;
            }
			pt<R>* pCenters = h_Centers;
			for (auto vi=centers.begin(); vi!=centers.end(); vi++) {
				*pCenters++ = *vi;
            }


			//fprintf(stdout, "initial centers:\n");
			//PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);
			dAvgSecs = (*lpfnKMeans)(DB,
                                     nSteps, 
									 h_Points, 
									 h_Centers, 
									 nPoints, 
									 nCenters, 
									 bVerify,
									 bVerbose);
		}

	    if (bVerify) {
		    //shrLog("\nValidating GPU results...\n");
		    for(int nStep=0; nStep<nSteps; nStep++) {
			    MapPointsToCentersSequential(refcenters, points, refclusterids);
			    UpdateCentersSequential(refcenters, points, refclusterids);
		    }

		    // compare the results, complaining loudly on a mismatch,
		    // and print the final output along with timing data for each impl.
		    
			if (!bTooBig) {
				pt<R>* pCenters = h_Centers;
				for(auto vi=centers.begin(); vi!=centers.end(); vi++) 
					*vi = *pCenters++;
				bSuccess = CompareResults(centers, refcenters,  0.1f, bVerbose);
			}
            // TODO placeholder
            //double dAvgSecs = 0;
		    //PrintResults(stdout, centers, refcenters, bSuccess, dAvgSecs, dRefAvgSecs, bVerbose);
	    }

	    //shrLog("Cleaning up...\n");
        // free(h_Centers);
        ALTIS_FREE(h_Centers);
        // free(h_Points);
        ALTIS_FREE(h_Points);
        // free(h_ClusterIds);
    }
};


#endif
