///
/// @file    common.h
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @brief   Common stuff for all CUDA dwt functions.
/// @date    2011-01-20 14:19
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///

#ifndef DWT_COMMON_H
#define DWT_COMMON_H

#include <cstdio>
#include <algorithm>
#include <vector>

inline constexpr int g_sizeX = 960;

// compile time minimum macro
#define CTMIN(a, b) (((a) < (b)) ? (a) : (b))

namespace dwt_cuda {

/// Divide and round up.
template<typename T>
inline T
divRndUp(const T &n, const T &d)
{
    return (n / d) + ((n % d) ? 1 : 0);
}

// 9/7 forward DWT lifting schema coefficients
const float f97Predict1 = -1.586134342;   ///< forward 9/7 predict 1
const float f97Update1  = -0.05298011854; ///< forward 9/7 update 1
const float f97Predict2 = 0.8829110762;   ///< forward 9/7 predict 2
const float f97Update2  = 0.4435068522;   ///< forward 9/7 update 2

// 9/7 reverse DWT lifting schema coefficients
const float r97update2  = -f97Update2;  ///< undo 9/7 update 2
const float r97predict2 = -f97Predict2; ///< undo 9/7 predict 2
const float r97update1  = -f97Update1;  ///< undo 9/7 update 1
const float r97Predict1 = -f97Predict1; ///< undo 9/7 predict 1

// FDWT 9/7 scaling coefficients
const float scale97Mul = 1.23017410491400f;
const float scale97Div = 1.0 / scale97Mul;

// 5/3 forward DWT lifting schema coefficients
const float forward53Predict = -0.5f; /// forward 5/3 predict
const float forward53Update  = 0.25f; /// forward 5/3 update

// 5/3 forward DWT lifting schema coefficients
const float reverse53Update  = -forward53Update;  /// undo 5/3 update
const float reverse53Predict = -forward53Predict; /// undo 5/3 predict

/// Functor which adds scaled sum of neighbors to given central pixel.
struct AddScaledSum
{
    const float scale; // scale of neighbors
    AddScaledSum(const float scale)
        : scale(scale)
    {
    }
    void operator()(const float p, float &c, const float n) const
    {
        c += scale * (p + n);
    }
};

/// Returns index ranging from 0 to num threads, such that first half
/// of threads get even indices and others get odd indices. Each thread
/// gets different index.
/// Example: (for 8 threads)   threadIdx.x:   0  1  2  3  4  5  6  7
///                              parityIdx:   0  2  4  6  1  3  5  7
/// @tparam THREADS  total count of participating threads
/// @return parity-separated index of thread
template<int THREADS>
inline int
parityIdx(sycl::nd_item<3> item_ct1)
{
    return (item_ct1.get_local_id(2) * 2)
           - (THREADS - 1) * (item_ct1.get_local_id(2) / (THREADS / 2));
}

/// size of shared memory
// 16384 is Arria10
const int SHM_SIZE = 16384; //48 * 1024;

} // end of namespace dwt_cuda

#endif // DWT_COMMON_CUDA_H
