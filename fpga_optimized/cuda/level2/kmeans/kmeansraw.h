///-------------------------------------------------------------------------------------------------
// file:	kmeansaw.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef __KMEANS_RAW_H__
#define __KMEANS_RAW_H__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <type_traits>
#include <vector>

#include "ResultDatabase.h"
#include "cudacommon.h"
#include "fpga_pwr.hpp"
#include "genericvector.h"
#include "kmeans-common.h"
#include "tuple.hpp"

inline size_t g_device_idx = 0;

typedef double (*LPFNKMEANS)(ResultDatabase &DB, const int nSteps,
                             void *h_Points, void *h_Centers, const int nPoints,
                             const int nCenters, bool bVerify, bool bVerbose);

typedef void (*LPFNBNC)(ResultDatabase &DB, char *szFile, LPFNKMEANS lpfn,
                        int nSteps, int nSeed, bool bVerify, bool bVerbose);

template <int R, int C> class centersmanagerGM {
protected:
  float *m_pG;
  float *m_pRO;

public:
  centersmanagerGM(float *pG, sycl::queue &q) : m_pG(pG), m_pRO(NULL) {}
  float *dataRW() { return m_pG; }
  float *dataRO() { return m_pG; }
  bool update(float *p, bool bHtoD = false) { return true; }
  bool updateHtoD(float *p) { return true; }
  bool updateDToD(float *p) { return true; }
};

constexpr unsigned bits_needed_for(unsigned x) {
  return x < 2 ? x : 1 + bits_needed_for(x >> 1);
}

// Maximum we will see is 2^16
constexpr int32_t max_points = 65536;
using idx_t = ac_int<bits_needed_for(max_points + 1), false>;
// Maximum we will see is 2^16
constexpr int32_t point_block_size = 16;
using pb_t = ac_int<bits_needed_for(max_points / point_block_size + 1), false>;
// Maximum we will see is R
constexpr int32_t max_rank = 16;
using rank_t = ac_int<bits_needed_for(max_rank + 1), false>;
// Maximum we will see is C
constexpr int32_t max_centers = 64;
using center_t = ac_int<bits_needed_for(max_centers + 1), false>;
// Maximum we allow is 1000
constexpr int32_t max_steps = 1000;
using step_t = ac_int<bits_needed_for(max_steps + 1), false>;

using cntr_pt_t = fpga_tools::Tuple<center_t, sycl::float16>;

using points_streamer_pipe =
    sycl::ext::intel::pipe<class points_streamer_pipe_id, sycl::float16, 16>;
using points_streamer_pipe2 =
    sycl::ext::intel::pipe<class points_streamer_pipe2_id, sycl::float16, 16>;
using center_idx_pipe =
    sycl::ext::intel::pipe<class center_idx_pipe_id, cntr_pt_t, 32>;
using acc_out_pipe =
    sycl::ext::intel::pipe<class acc_out_pipe_id, sycl::float16, 16>;

template <int R, int C, bool ROWMAJ = true> class accumulatorGM {
public:
  static void reset(float *pC, int *pCC, sycl::queue &queue) {
    // Sets all elements of pC and pCC to zero. This is done now in accumulate
    // directly!
  }

  static void accumulate(float *pP, float *pC, int _nP, sycl::queue &queue,
                         int16_t steps) {
    queue
        .submit([&](sycl::handler &cgh) {
          const idx_t nP = _nP;
          cgh.single_task<class accumulator_kernel_id>(
              [=]()
                  [[intel::kernel_args_restrict, intel::max_global_work_dim(0),
                    intel::no_global_work_offset(1)]] {
                    [[intel::max_concurrency(1)]] //
                    for (step_t s = 0; s < steps && s < 1000; s++) {
                      sycl::float16 centers_local[C];
                      [[intel::initiation_interval(1),
                        intel::loop_coalesce(2)]] //
                      for (center_t center = 0; center < C; center++)
                        for (rank_t rank = 0; rank < R; rank++)
                          centers_local[center][rank] = 0.0f;

                      int cc_local[C];
                      [[intel::initiation_interval(1),
                        intel::speculated_iterations(0)]] //
                      for (center_t i = 0; i < C; i++)
                        cc_local[i] = 0;

                      [[intel::speculated_iterations(0)]] //
                      for (idx_t point = 0; point < nP; point++) {
                        auto tuple = center_idx_pipe::read();
                        center_t clusterid = tuple.get<0>();
                        cc_local[clusterid] += 1;

                        sycl::float16 val = tuple.get<1>();
                        centers_local[clusterid] += val;
                      } // LOOP(point)

                      [[intel::ivdep(pC), intel::initiation_interval(1),
                        intel::speculated_iterations(0)]] //
                      for (center_t c = 0; c < C; c++) {
                        const int32_t cc = cc_local[c];
                        const sycl::float16 new_center =
                            ((cc == 0) ? 0.0f : 1.0f) * centers_local[c] /
                            ((cc == 0) ? 1.0f : cc);
                        if (s == steps - 1)
                          ((sycl::float16 *)pC)[c] = new_center;
                        else
                          acc_out_pipe::write(new_center);
                      } // LOOP(c)
                    }   // LOOP(s)
                  });
        })
        .wait();
  }

  static void finalize(float *pC, int *pCC, sycl::queue &queue) {
    // Just done in accumulate.
  }
};

class map_centers_kernel_rowmaj_id;

template <int R, int C, typename CM, typename SM, bool ROWMAJ = true>
class kmeansraw {
public:
  kmeansraw(int nSteps, float *d_Points, float *d_Centers, int *d_ClusterCounts,
            int *d_ClusterIds, int nPoints, int nCenters, sycl::queue &_queue)
      : m_nSteps(nSteps), m_dPoints(d_Points), m_dCenters(d_Centers),
        m_dClusterCounts(d_ClusterCounts), m_dClusterIds(d_ClusterIds),
        m_nPoints(nPoints), m_nCenters(nCenters), m_centers(d_Centers, _queue),
        queue(_queue) {}

  bool execute(int16_t steps) {
    assert(m_nCenters == C);

    // NOTE: reset and finalize merged into accumulate!
    //
    FPGA_PWR_MEAS_START
    mapCenters(m_dPoints, m_nPoints, steps);
    // resetAccumulators(m_dCenters, m_dClusterCounts);
    accumulate(m_dPoints, m_dCenters, m_nPoints, steps);
    // finalizeAccumulators(m_dCenters, m_dClusterCounts);
    FPGA_PWR_MEAS_END

    return true;
  }

protected:
  int m_nSteps;
  float *m_dPoints;
  float *m_dCenters;
  int *m_dClusterCounts;
  int *m_dClusterIds;
  int m_nPoints;
  int m_nCenters;
  sycl::queue &queue;

  CM m_centers;
  SM m_accumulator;
  bool updatecentersIn(float *p_Centers) { return m_centers.update(p_Centers); }
  bool initcentersInput(float *p_Centers) {
    return m_centers.updateHtoD(p_Centers);
  }
  void resetAccumulators(float *pC, int *pCC) {
    m_accumulator.reset(pC, pCC, queue);
  }
  void accumulate(float *pP, float *pC, int nP, int16_t steps) {
    m_accumulator.accumulate(pP, pC, nP, queue, steps);
  }
  void finalizeAccumulators(float *pC, int *pCI) {
    m_accumulator.finalize(pC, pCI, queue);
  }
  float *centers() { return m_centers.data(); }

  void mapCenters(float *pP, int _nP, int16_t steps) {
    float *pC = m_centers.dataRO();
    queue.submit([&](sycl::handler &cgh) {
      const idx_t nP = _nP;
      cgh.single_task<class map_centers_streamer_id>(
          [=]() [[intel::kernel_args_restrict, intel::max_global_work_dim(0),
                  intel::no_global_work_offset(1)]] {
            [[intel::initiation_interval(1)]] //
            for (step_t s = 0; s < steps && s < 1000; s++) {
              [[intel::initiation_interval(1),
                intel::speculated_iterations(0)]] //
              for (idx_t p = 0; p < nP; p++) {
                const sycl::float16 x = ((sycl::float16 *)pP)[p];
                points_streamer_pipe::write(x);
              }
            }
          });
    });
    queue.submit([&](sycl::handler &cgh) {
      const idx_t nP = _nP;
      cgh.single_task<class map_centers_kernel_id>(
          [=]() [[intel::kernel_args_restrict, intel::max_global_work_dim(0),
                  intel::no_global_work_offset(1)]] {
            sycl::float16 first_centers_local[C];
            [[intel::initiation_interval(1)]] //
            for (center_t i = 0; i < C; i++)
              first_centers_local[i] = ((sycl::float16 *)pC)[i];

            [[intel::max_concurrency(1)]] //
            for (step_t s = 0; s < steps && s < 1000; s++) {
              // Load in centers, in the first iteration they come out of DDR.
              //
              sycl::float16 centers_local[C];
              [[intel::initiation_interval(1),
                intel::speculated_iterations(0)]] for (center_t i = 0; i < C;
                                                       i++)
                centers_local[i] =
                    (s == 0) ? first_centers_local[i] : acc_out_pipe::read();

              const pb_t point_blocks = nP / point_block_size;
              [[intel::initiation_interval(1)]] //
              for (pb_t pb = 0; pb < point_blocks; pb++) {
                center_t point_block_result[point_block_size];
                const idx_t pb_begin = pb * point_block_size;

                [[intel::private_copies(32)]] //
                sycl::float16 points_local[point_block_size];
                [[intel::initiation_interval(1),
                  intel::speculated_iterations(0)]] //
                for (int8_t i = 0; i < point_block_size; i++)
                  points_local[i] = points_streamer_pipe::read();

#pragma unroll
                for (idx_t p_idx = 0; p_idx < point_block_size; p_idx++) {
                  idx_t point = pb_begin + p_idx;

                  // Use shift register for minimum-distances to get II=1.
                  //(Without it's II=8 -> Use this as shift-reg-size!)
                  //
                  using tuple_t = fpga_tools::Tuple<float, center_t>;
                  constexpr int32_t II_CYLCE = 8;
                  tuple_t shift_reg[II_CYLCE];
#pragma unroll
                  for (int8_t i = 0; i < II_CYLCE; i++)
                    shift_reg[i] = {FLT_MAX, 0};

                  [[intel::initiation_interval(1),
                    intel::speculated_iterations(0)]] //
                  for (center_t c = 0; c < C; c++) {
                    float accum = 0.0f;

                    const sycl::float16 dist_vec =
                        points_local[p_idx] - centers_local[c];
#pragma unroll
                    for (rank_t j = 0; j < R; j++)
                      accum += dist_vec[j] * dist_vec[j];

                    const float dist = sycl::sqrt(accum);
                    const float last_dist_in_sr =
                        shift_reg[II_CYLCE - 1].template get<0>();
                    shift_reg[0] = (last_dist_in_sr < dist)
                                       ? shift_reg[II_CYLCE - 1]
                                       : tuple_t(dist, c);

#pragma unroll
                    for (int8_t j = II_CYLCE - 1; j > 0; j--)
                      shift_reg[j] = shift_reg[j - 1];
                  } // FOR(c)

                  float mindist = FLT_MAX;
                  int minidx = 0;
#pragma unroll
                  for (int8_t i = 0; i < II_CYLCE; ++i)
                    if (const auto d = shift_reg[i].template get<0>();
                        d < mindist) {
                      minidx = shift_reg[i].template get<1>();
                      mindist = d;
                    }

                  point_block_result[p_idx] = minidx;
                } // FOR(p_idx)

                [[intel::initiation_interval(1),
                  intel::speculated_iterations(0)]] //
                for (int8_t res_i = 0; res_i < point_block_size; res_i++)
                  center_idx_pipe::write(cntr_pt_t(point_block_result[res_i],
                                                   points_local[res_i]));
              } // FOR(pb)
            }   // FOR(s)
          });
    });
  }

  static const int MAX_CHAR_PER_LINE = 512;

  ///-------------------------------------------------------------------------------------------------
  /// <summary>   Reads an input file. </summary>
  ///
  /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. Adapted
  /// from code in
  ///             serban/kmeans which appears adapted from code in STAMP
  ///             (variable names only appear to be changed), or perhaps each
  ///             was adapted from code in other places.
  ///
  ///             ****************************** Note that AMP requires
  ///             template int args to be statically known at compile time.
  ///             Hence, for now, you have to have built this program with
  ///             DEFAULTRANK defined to match the DEFAULTRANK of your input
  ///             file! This restriction is easy to fix, but not the best use
  ///             of my time at the moment...
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

  static int ReadInput(char *filename, std::vector<pt<R>> &points, int *numObjs,
                       int *numCoords, int _debug) {
#pragma warning(disable : 4996)
    float **objects;
    int i, j, len;

    FILE *infile;
    char *line, *ret;
    int lineLen;

    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      safe_exit(-1);
    }

    /* first find the number of objects */
    lineLen = MAX_CHAR_PER_LINE;
    line = (char *)malloc(lineLen);
    assert(line != NULL);

    (*numObjs) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
      /* check each line to find the max line length */
      while (strlen(line) == lineLen - 1) {
        /* this line read is not complete */
        len = (int)strlen(line);
        fseek(infile, -len, SEEK_CUR);

        /* increase lineLen */
        lineLen += MAX_CHAR_PER_LINE;
        line = (char *)realloc(line, lineLen);
        assert(line != NULL);

        ret = fgets(line, lineLen, infile);
        assert(ret != NULL);
      }

      if (strtok(line, " \t\n") != 0)
        (*numObjs)++;
    }
    rewind(infile);
    if (_debug)
      printf("lineLen = %d\n", lineLen);

    /* find the no. objects of each object */
    (*numCoords) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first coordiinate): numCoords = 1; */
        while (strtok(NULL, " ,\t\n") != NULL)
          (*numCoords)++;
        break; /* this makes read from 1st object */
      }
    }
    rewind(infile);
    if (_debug) {
      printf("File %s numObjs   = %d\n", filename, *numObjs);
      printf("File %s numCoords = %d\n", filename, *numCoords);
    }

    /* allocate space for objects[][] and read all objects */
    len = (*numObjs) * (*numCoords);
    objects = (float **)malloc((*numObjs) * sizeof(float *));
    assert(objects != NULL);
    objects[0] = (float *)malloc(len * sizeof(float));
    assert(objects[0] != NULL);
    for (i = 1; i < (*numObjs); i++)
      objects[i] = objects[i - 1] + (*numCoords);

    i = 0;
    /* read all objects */
    while (fgets(line, lineLen, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL)
        continue;
      for (j = 0; j < (*numCoords); j++)
        objects[i][j] = (float)atof(strtok(NULL, " ,\t\n"));
      i++;
    }

    points.reserve(static_cast<int>(*numObjs));
    for (int idx = 0; idx < (*numObjs); idx++) {
      pt<R> point(objects[idx]);
      points.push_back(point);
    }

    fclose(infile);
    free(line);
    free(objects[0]);
    free(objects);
    return 0;
#pragma warning(default : 4996)
  }

  static void ChooseInitialCenters(std::vector<pt<R>> &points,
                                   std::vector<pt<R>> &centers,
                                   std::vector<pt<R>> &refcenters,
                                   int nRandomSeed) {
    srand(nRandomSeed);
    std::set<int> chosenidx;
    while (chosenidx.size() < (size_t)C) {
      // sets don't allow dups...
      int idx = rand() % points.size();
      chosenidx.insert(idx);
    }
    std::set<int>::iterator si;
    for (si = chosenidx.begin(); si != chosenidx.end(); si++) {
      centers.push_back(points[*si]);
      refcenters.push_back(points[*si]);
    }
  }

  static void PrintCenters(FILE *fp, std::vector<pt<R>> &centers,
                           int nColLimit = 8, int nRowLimit = 16) {
    fprintf(fp, "\n");
    int nRows = 0;
    for (auto x = centers.begin(); x != centers.end(); x++) {
      if (++nRows > nRowLimit) {
        fprintf(fp, "...");
        break;
      }
      x->dump(fp, nColLimit);
    }
    fprintf(fp, "\n");
  }

  static void PrintCenters(FILE *fp, pt<R> *pCenters, int nColLimit = 8,
                           int nRowLimit = 16) {
    nRowLimit = (nRowLimit == 0) ? C : std::min(nRowLimit, C);
    for (int i = 0; i < nRowLimit; i++)
      pCenters[i].dump(fp, nColLimit);
    if (nRowLimit < C)
      fprintf(fp, "...");
    fprintf(fp, "\n");
  }

  static void PrintResults(FILE *fp, std::vector<pt<R>> &centers,
                           std::vector<pt<R>> &refcenters, bool bSuccess,
                           double dKmeansExecTime, double dKmeansCPUTime,
                           bool bVerbose = false) {
    fprintf(fp, "%s: GPU: %.5f sec, CPU: %.5f sec\n",
            (bSuccess ? "SUCCESS" : "FAILURE"), dKmeansExecTime,
            dKmeansCPUTime);
    if (!bSuccess || bVerbose) {
      fprintf(fp, "final centers:\n");
      PrintCenters(fp, centers);
      fprintf(fp, "reference centers:\n");
      PrintCenters(fp, refcenters);
    }
  }

  static bool CompareResults_obs(std::vector<pt<R>> &centers,
                                 std::vector<pt<R>> &refcenters,
                                 float EPSILON = 0.0001f, bool bVerbose = true,
                                 int nRowLimit = 16) {
    std::set<pt<R> *> unmatched;
    for (auto vi = centers.begin(); vi != centers.end(); vi++) {
      bool bFound = false;
      for (auto xi = refcenters.begin(); xi != refcenters.end(); xi++) {
        if (EPSILON > hdistance(*vi, *xi)) {
          bFound = true;
          break;
        }
      }
      if (!bFound)
        unmatched.insert(&(*vi));
    }
    bool bSuccess = unmatched.size() == 0;
    if (bVerbose && !bSuccess) {
      int nDumped = 0;
      fprintf(stderr, "Could not match %d centers:\n", unmatched.size());
      for (auto si = unmatched.begin(); si != unmatched.end(); si++) {
        if (++nDumped > nRowLimit) {
          printf("...\n");
          break;
        }
        (*si)->dump(stderr);
      }
    }
    return bSuccess;
  }

  static bool CompareResults(std::vector<pt<R>> &centers,
                             std::vector<pt<R>> &refcenters,
                             float EPSILON = 0.0001f, bool bVerbose = true,
                             int nRowLimit = 16) {
    int nRows = 0;
    std::map<int, pt<R> *> unmatched;
    std::map<int, float> unmatched_deltas;
    std::map<int, int> matched;
    std::map<int, int> revmatched;
    int nCenterIdx = 0;
    for (auto vi = centers.begin(); vi != centers.end(); vi++, nCenterIdx++) {
      bool bFound = false;
      if (EPSILON * R > hdistance(*vi, refcenters[nCenterIdx])) {
        bFound = true;
        matched[nCenterIdx] = nCenterIdx;
        revmatched[nCenterIdx] = nCenterIdx;
      } else {
        int nRefIdx = 0;
        for (auto xi = refcenters.begin(); xi != refcenters.end();
             xi++, nRefIdx++) {
          if (EPSILON * R > hdistance(*vi, *xi)) {
            bFound = true;
            matched[nCenterIdx] = nRefIdx;
            revmatched[nRefIdx] = nCenterIdx;
            break;
          }
        }
      }
      if (!bFound) {
        unmatched[nCenterIdx] = (&(*vi));
        unmatched_deltas[nCenterIdx] = hdistance(*vi, refcenters[nCenterIdx]);
      }
    }
    bool bSuccess = unmatched.size() == 0;
    if (bVerbose && !bSuccess) {
      std::cerr << "Could not match " << unmatched.size()
                << " centers: " << std::endl;
      for (auto si = unmatched.begin(); si != unmatched.end(); si++) {
        if (++nRows > nRowLimit) {
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
  static double benchmark(ResultDatabase &DB, const int nSteps, void *lpvPoints,
                          void *lpvCenters, const int nPoints,
                          const int nCenters, bool bVerify, bool bVerbose) {
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue queue(devices[g_device_idx]);

    assert(nCenters == C);

    float *h_TxPoints = NULL;
    float *h_TxCenters = NULL;
    pt<R> *h_InPoints = reinterpret_cast<pt<R> *>(lpvPoints);
    pt<R> *h_InCenters = reinterpret_cast<pt<R> *>(lpvCenters);
    float *h_Points = reinterpret_cast<float *>(h_InPoints);
    float *h_Centers = reinterpret_cast<float *>(h_InCenters);

    std::cout << "nPoints: " << nPoints << std::endl;

    size_t uiPointsBytes = nPoints * R * sizeof(float);
    size_t uiCentersBytes = nCenters * R * sizeof(float);
    size_t uiClusterIdsBytes =
        (nPoints + (THREADBLOCK_SIZE - nPoints % THREADBLOCK_SIZE)) *
        sizeof(int);
    size_t uiClusterCountsBytes = nCenters * sizeof(int);
    float *d_Points = sycl::malloc_device<float>(uiPointsBytes, queue);
    float *d_Centers = sycl::malloc_device<float>(uiCentersBytes, queue);
    int *d_ClusterIds = sycl::malloc_device<int>(uiClusterIdsBytes, queue);
    int *d_ClusterCounts =
        sycl::malloc_device<int>(uiClusterCountsBytes, queue);
    if (nullptr == d_Points || nullptr == d_Centers ||
        nullptr == d_ClusterIds || nullptr == d_ClusterCounts) {
      std::cerr << "Error allocating memory on device." << std::endl;
      std::terminate();
    }

    kmeansraw<R, C, CM, SM, ROWMAJ> *pKMeans =
        new kmeansraw<R, C, CM, SM, ROWMAJ>(nSteps, d_Points, d_Centers,
                                            d_ClusterCounts, d_ClusterIds,
                                            nPoints, nCenters, queue);

    // Warmup.
    (*pKMeans).execute(0);

    // Copy real input data after warmup.
    //
    queue.memcpy(d_Points, h_Points, uiPointsBytes);
    queue.memcpy(d_Centers, h_Centers, uiCentersBytes);
    queue.wait_and_throw();

    float elapsed;
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    start_ct1 = std::chrono::steady_clock::now();
    (*pKMeans).execute(nSteps);
    queue.wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

    char atts[1024];
    sprintf(atts, "iterations:%d, centers:%d, rank:%d", nSteps, C, R);
    DB.AddResult("kmeans total execution time", atts, "sec", elapsed * 1.0e-3);
    DB.AddResult("kmeans execution time per iteration", atts, "sec",
                 elapsed * 1.0e-3 / nSteps);

    if (bVerbose) {
      uint byteCount = (uint)(uiPointsBytes + uiCentersBytes);
      DB.AddResult("kmeans thoughput", atts, "MB/sec",
                   ((double)byteCount * 1.0e-6) / (elapsed * 1.0e-3));
      // shrLogEx(LOGBOTH | MASTER, 0, "kmeans, Throughput = %.4f
      // MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u,
      // Workgroup = %u\n", (1.0e-6 * (double)byteCount / dAvgSecs),
      // dAvgSecs, byteCount, 1, THREADBLOCK_SIZE);
    }

    if (bVerify) {
      std::cout << " Reading back GPU results..." << std::endl;
      queue.memcpy(h_Centers, d_Centers, uiCentersBytes).wait();
      queue.wait_and_throw();
    }

    sycl::free(d_Points, queue);
    sycl::free(d_Centers, queue);
    sycl::free(d_ClusterIds, queue);
    sycl::free(d_ClusterCounts, queue);

    return (double)elapsed * 1.0e-3;
  }

  static float hdistance(pt<R> &a, pt<R> &b) {
    float accum = 0.0f;
    for (int i = 0; i < R; i++) {
      float delta = a.m_v[i] - b.m_v[i];
      accum += delta * delta;
    }
    return sqrt(accum);
  }

  static int NearestCenter(pt<R> &point, std::vector<pt<R>> &centers) {
    float mindist = FLT_MAX;
    int minidx = 0;
    for (size_t i = 0; i < centers.size(); i++) {
      float dist = hdistance(point, centers[i]);
      if (dist < mindist) {
        minidx = static_cast<int>(i);
        mindist = dist;
      }
    }
    return minidx;
  }

  static void MapPointsToCentersSequential(std::vector<pt<R>> &vcenters,
                                           std::vector<pt<R>> &vpoints,
                                           std::vector<int> &vclusterids) {
    int index = 0;
    for (auto vi = vpoints.begin(); vi != vpoints.end(); vi++)
      vclusterids[index++] = NearestCenter(*vi, vcenters);
  }

  static void UpdateCentersSequential(std::vector<pt<R>> &vcenters,
                                      std::vector<pt<R>> &vpoints,
                                      std::vector<int> &vclusterids) {
    std::vector<int> counts;
    for (size_t i = 0; i < vcenters.size(); i++) {
      vcenters[i].set(0.0f);
      counts.push_back(0);
    }
    for (size_t i = 0; i < vpoints.size(); i++) {
      int clusterid = vclusterids[i];
      vcenters[clusterid] += vpoints[i];
      counts[clusterid] += 1;
    }
    for (size_t i = 0; i < vcenters.size(); i++) {
      vcenters[i] /= counts[i];
    }
  }

  static void bncmain(ResultDatabase &DB, char *lpszInputFile,
                      LPFNKMEANS lpfnKMeans, int nSteps, int nSeed,
                      bool bVerify, bool bVerbose) {
    int nP = 0;
    int nD = 0;
    std::vector<pt<R>> points;
    std::vector<pt<R>> centers;
    std::vector<pt<R>> refcenters;

    ReadInput(lpszInputFile, points, &nP, &nD, 0);
    if (points.size() == 0) {
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

    ChooseInitialCenters(
        points,     // points to choose from
        centers,    // destination array of initial centers
        refcenters, // save a copy for the reference impl to check
        nSeed);     // random seed. ACHTUNG! Beware benchmarkers

    int nPoints = (int)points.size();
    int nCenters = C;
    size_t uiPointsBytes = nPoints * R * sizeof(float);
    size_t uiCentersBytes = C * R * sizeof(float);
    size_t uiClusterIdsBytes = nPoints * sizeof(int);

    bool bSuccess = false;
    double dAvgSecs = 0.0;
    pt<R> *h_Centers = NULL;
    bool bTooBig = (uiPointsBytes > UINT_MAX);

    // if the points won't fit in GPU memory, there is
    // no point in going through the exercise of watching
    // the GPU exec fail (particularly if we still want to
    // collect the CPU comparison number). If it's obviously
    // too big, skip the CUDA rigmaroll.

    if (!bTooBig) {

      pt<R> *h_Points = (pt<R> *)malloc(uiPointsBytes);
      h_Centers = (pt<R> *)malloc(uiCentersBytes);

      pt<R> *pPoints = h_Points;
      for (auto vi = points.begin(); vi != points.end(); vi++)
        *pPoints++ = *vi;

      pt<R> *pCenters = h_Centers;
      for (auto vi = centers.begin(); vi != centers.end(); vi++)
        *pCenters++ = *vi;

      // fprintf(stdout, "initial centers:\n");
      // PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);
      dAvgSecs = (*lpfnKMeans)(DB, nSteps, h_Points, h_Centers, nPoints,
                               nCenters, bVerify, bVerbose);
    }

    if (bVerify) {
      std::cout << "Calculating reference results..." << std::endl;
      for (int nStep = 0; nStep < nSteps; nStep++) {
        MapPointsToCentersSequential(refcenters, points, refclusterids);
        UpdateCentersSequential(refcenters, points, refclusterids);
      }

      // compare the results, complaining loudly on a mismatch,
      // and print the final output along with timing data for each
      // impl.

      if (!bTooBig) {
        for (uint64_t i = 0ull; i < centers.size(); i++)
          centers[i] = h_Centers[i];
        bSuccess = CompareResults(centers, refcenters, 0.1f, bVerbose);
      }

      double dAvgSecs = 0;
      PrintResults(stdout, centers, refcenters, bSuccess, dAvgSecs, dAvgSecs,
                   bVerbose);
    }

    std::cout << "Cleaning up..." << std::endl;
    if (h_Centers)
      free(h_Centers);
  }
};

#endif
