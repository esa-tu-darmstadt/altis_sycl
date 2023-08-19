// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level1\sort\sort_kernel.cu
//
// summary:	Declares the sort class
// 
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sort_kernel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans a LSB. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="val">   	The value. </param>
/// <param name="s_data">	[in,out] If non-null, the data. </param>
///
/// <returns>	An uint. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

uint scanLSB(const uint val, uint* s_data, const sycl::nd_item<3> &item_ct1)
{
    // Shared mem is 256 uints long, set first half to 0's
    int idx = item_ct1.get_local_id(2);
    s_data[idx] = 0;
    /*
    DPCT1065:177: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += item_ct1.get_local_range(2); // += 128 in this case

    // Unrolled scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    /*
    DPCT1065:178: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] = val; item_ct1.barrier();
    /*
    DPCT1065:179: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 1]; item_ct1.barrier();
    /*
    DPCT1065:180: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:181: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 2]; item_ct1.barrier();
    /*
    DPCT1065:182: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:183: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 4]; item_ct1.barrier();
    /*
    DPCT1065:184: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:185: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 8]; item_ct1.barrier();
    /*
    DPCT1065:186: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:187: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 16]; item_ct1.barrier();
    /*
    DPCT1065:188: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:189: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 32]; item_ct1.barrier();
    /*
    DPCT1065:190: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:191: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 64]; item_ct1.barrier();
    /*
    DPCT1065:192: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans a 4. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="idata">	The idata. </param>
/// <param name="ptr">  	[in,out] If non-null, the pointer. </param>
///
/// <returns>	An uint4. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

sycl::uint4 scan4(sycl::uint4 idata, uint *ptr,
                  const sycl::nd_item<3> &item_ct1)
{
    sycl::uint4 val4 = idata;
    sycl::uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x() = val4.x();
    sum.y() = val4.y() + sum.x();
    sum.z() = val4.z() + sum.y();
    uint val = val4.w() + sum.z();

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr, item_ct1);

    val4.x() = val;
    val4.y() = val + sum.x();
    val4.z() = val + sum.y();
    val4.w() = val + sum.z();

    return val4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	adixSortBlocks sorts all blocks of data independently in shared
///  memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements.
///  The radix sort is done in two stages.  This stage calls radixSortBlock
///  on each block independently, sorting on the basis of bits
///  (startbit) -> (startbit + nbits) </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="nbits">		The nbits. </param>
/// <param name="startbit"> 	The startbit. </param>
/// <param name="keysOut">  	[in,out] If non-null, the keys out. </param>
/// <param name="valuesOut">	[in,out] If non-null, the values out. </param>
/// <param name="keysIn">   	[in,out] If non-null, the keys in. </param>
/// <param name="valuesIn"> 	[in,out] If non-null, the values in. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL void radixSortBlocks(const uint nbits, const uint startbit,
                                   sycl::uint4 *keysOut, sycl::uint4 *valuesOut,
                                   sycl::uint4 *keysIn, sycl::uint4 *valuesIn,
                                   const sycl::nd_item<3> &item_ct1, uint *sMem,
                                   uint &numtrue)
{

    // Get Indexing information
    const uint i = item_ct1.get_local_id(2) +
                   (item_ct1.get_group(2) * item_ct1.get_local_range(2));
    const uint tid = item_ct1.get_local_id(2);
    const uint localSize = item_ct1.get_local_range(2);

    // Load keys and vals from global memory
    sycl::uint4 key, value;
    key = keysIn[i];
    value = valuesIn[i];

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        sycl::uint4 lsb;
        lsb.x() = !((key.x() >> shift) & 0x1);
        lsb.y() = !((key.y() >> shift) & 0x1);
        lsb.z() = !((key.z() >> shift) & 0x1);
        lsb.w() = !((key.w() >> shift) & 0x1);

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        sycl::uint4 address = scan4(lsb, sMem, item_ct1);

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
            numtrue = address.w() + lsb.w();
        }
        /*
        DPCT1065:193: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        sycl::uint4 rank;
        const int idx = tid*4;
        rank.x() = lsb.x() ? address.x() : numtrue + idx - address.x();
        rank.y() = lsb.y() ? address.y() : numtrue + idx + 1 - address.y();
        rank.z() = lsb.z() ? address.z() : numtrue + idx + 2 - address.z();
        rank.w() = lsb.w() ? address.w() : numtrue + idx + 3 - address.w();

        // Scatter keys into local mem
        sMem[(rank.x() & 3) * localSize + (rank.x() >> 2)] = key.x();
        sMem[(rank.y() & 3) * localSize + (rank.y() >> 2)] = key.y();
        sMem[(rank.z() & 3) * localSize + (rank.z() >> 2)] = key.z();
        sMem[(rank.w() & 3) * localSize + (rank.w() >> 2)] = key.w();
        /*
        DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        key.x() = sMem[tid];
        key.y() = sMem[tid + localSize];
        key.z() = sMem[tid + 2 * localSize];
        key.w() = sMem[tid + 3 * localSize];
        /*
        DPCT1065:195: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Scatter values into local mem
        sMem[(rank.x() & 3) * localSize + (rank.x() >> 2)] = value.x();
        sMem[(rank.y() & 3) * localSize + (rank.y() >> 2)] = value.y();
        sMem[(rank.z() & 3) * localSize + (rank.z() >> 2)] = value.z();
        sMem[(rank.w() & 3) * localSize + (rank.w() >> 2)] = value.w();
        /*
        DPCT1065:196: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        value.x() = sMem[tid];
        value.y() = sMem[tid + localSize];
        value.z() = sMem[tid + 2 * localSize];
        value.w() = sMem[tid + 3 * localSize];
        /*
        DPCT1065:197: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Given an array with blocks sorted according to a 4-bit radix group, each
///  block counts the number of keys that fall into each radix in the group, and
///  finds the starting offset of each radix in the block.  It then writes the
///  radix counts to the counters array, and the starting offsets to the
///  blockOffsets array.. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="keys">		   	[in,out] If non-null, the keys. </param>
/// <param name="counters">	   	[in,out] If non-null, the counters. </param>
/// <param name="blockOffsets">	[in,out] If non-null, the block offsets. </param>
/// <param name="startbit">	   	The startbit. </param>
/// <param name="numElements"> 	Number of elements. </param>
/// <param name="totalBlocks"> 	The total blocks. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL void findRadixOffsets(sycl::uint2 *keys, uint *counters,
                                    uint *blockOffsets, uint startbit,
                                    uint numElements, uint totalBlocks,
                                    const sycl::nd_item<3> &item_ct1,
                                    uint8_t *dpct_local, uint *sStartPointers)
{

    auto sRadix1 = (uint *)dpct_local;

    uint groupId = item_ct1.get_group(2);
    uint localId = item_ct1.get_local_id(2);
    uint groupSize = item_ct1.get_local_range(2);

    sycl::uint2 radix2;
    radix2 = keys[item_ct1.get_local_id(2) +
                  (item_ct1.get_group(2) * item_ct1.get_local_range(2))];

    sRadix1[2 * localId] = (radix2.x() >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y() >> startbit) & 0xF;

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    /*
    DPCT1065:198: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    /*
    DPCT1065:199: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    /*
    DPCT1065:200: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Compute the sizes of each block.
    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] =
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] =
            localId + groupSize - sStartPointers[sRadix1[localId +
                                                         groupSize - 1]];
    }

    if(localId == groupSize - 1)
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] =
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    /*
    DPCT1065:201: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	reorderData shuffles data in the array globally after the radix offsets
/// have been found. On compute version 1.1 and earlier GPUs, this code depends
/// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="startbit">	   	The startbit. </param>
/// <param name="outKeys">	   	[in,out] If non-null, the out keys. </param>
/// <param name="outValues">   	[in,out] If non-null, the out values. </param>
/// <param name="keys">		   	[in,out] If non-null, the keys. </param>
/// <param name="values">	   	[in,out] If non-null, the values. </param>
/// <param name="blockOffsets">	[in,out] If non-null, the block offsets. </param>
/// <param name="offsets">	   	[in,out] If non-null, the offsets. </param>
/// <param name="sizes">	   	[in,out] If non-null, the sizes. </param>
/// <param name="totalBlocks"> 	The total blocks. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL void reorderData(uint startbit, uint *outKeys, uint *outValues,
                               sycl::uint2 *keys, sycl::uint2 *values,
                               uint *blockOffsets, uint *offsets, uint *sizes,
                               uint totalBlocks,
                               const sycl::nd_item<3> &item_ct1,
                               sycl::uint2 *sKeys2, sycl::uint2 *sValues2,
                               uint *sOffsets, uint *sBlockOffsets)
{
    uint GROUP_SIZE = item_ct1.get_local_range(2);

    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint blockId = item_ct1.get_group(2);

    uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    sKeys2[item_ct1.get_local_id(2)] = keys[i];
    sValues2[item_ct1.get_local_id(2)] = values[i];

    if (item_ct1.get_local_id(2) < 16)
    {
        sOffsets[item_ct1.get_local_id(2)] =
            offsets[item_ct1.get_local_id(2) * totalBlocks + blockId];
        sBlockOffsets[item_ct1.get_local_id(2)] =
            blockOffsets[blockId * 16 + item_ct1.get_local_id(2)];
    }
    /*
    DPCT1065:202: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    uint radix = (sKeys1[item_ct1.get_local_id(2)] >> startbit) & 0xF;
    uint globalOffset =
        sOffsets[radix] + item_ct1.get_local_id(2) - sBlockOffsets[radix];

    outKeys[globalOffset] = sKeys1[item_ct1.get_local_id(2)];
    outValues[globalOffset] = sValues1[item_ct1.get_local_id(2)];

    radix = (sKeys1[item_ct1.get_local_id(2) + GROUP_SIZE] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + item_ct1.get_local_id(2) + GROUP_SIZE -
                   sBlockOffsets[radix];

    outKeys[globalOffset] = sKeys1[item_ct1.get_local_id(2) + GROUP_SIZE];
    outValues[globalOffset] = sValues1[item_ct1.get_local_id(2) + GROUP_SIZE];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans a local memory. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="val">   	The value. </param>
/// <param name="s_data">	[in,out] If non-null, the data. </param>
///
/// <returns>	An uint. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

uint scanLocalMem(const uint val, uint* s_data,
                  const sycl::nd_item<3> &item_ct1)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = item_ct1.get_local_id(2);
    s_data[idx] = 0.0f;
    /*
    DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += item_ct1.get_local_range(2); // += 256

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    /*
    DPCT1065:204: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] = val; item_ct1.barrier();
    /*
    DPCT1065:205: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 1]; item_ct1.barrier();
    /*
    DPCT1065:206: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:207: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 2]; item_ct1.barrier();
    /*
    DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:209: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 4]; item_ct1.barrier();
    /*
    DPCT1065:210: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:211: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 8]; item_ct1.barrier();
    /*
    DPCT1065:212: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:213: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 16]; item_ct1.barrier();
    /*
    DPCT1065:214: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:215: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 32]; item_ct1.barrier();
    /*
    DPCT1065:216: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:217: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 64]; item_ct1.barrier();
    /*
    DPCT1065:218: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();
    /*
    DPCT1065:219: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    t = s_data[idx - 128]; item_ct1.barrier();
    /*
    DPCT1065:220: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    s_data[idx] += t; item_ct1.barrier();

    return s_data[idx-1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="g_odata">	  	[in,out] If non-null, the odata. </param>
/// <param name="g_idata">	  	[in,out] If non-null, the idata. </param>
/// <param name="g_blockSums">	[in,out] If non-null, the block sums. </param>
/// <param name="n">		  	An int to process. </param>
/// <param name="fullBlock">  	True to full block. </param>
/// <param name="storeSum">   	True to store sum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL void scan(uint *g_odata, uint *g_idata, uint *g_blockSums,
                        const int n, const bool fullBlock, const bool storeSum,
                        const sycl::nd_item<3> &item_ct1, uint *s_data)
{

    // Load data into shared mem
    sycl::uint4 tempData;
    sycl::uint4 threadScanT;
    uint res;
    sycl::uint4 *inData = (sycl::uint4 *)g_idata;

    const int gid = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                    item_ct1.get_local_id(2);
    const int tid = item_ct1.get_local_id(2);
    const int i = gid * 4;

    // If possible, read from global mem in a uint4 chunk
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x() = tempData.x();
        threadScanT.y() = tempData.y() + threadScanT.x();
        threadScanT.z() = tempData.z() + threadScanT.y();
        threadScanT.w() = tempData.w() + threadScanT.z();
        res = threadScanT.w();
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x() = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y() =
            ((i + 1 < n) ? g_idata[i + 1] : 0.0f) + threadScanT.x();
        threadScanT.z() =
            ((i + 2 < n) ? g_idata[i + 2] : 0.0f) + threadScanT.y();
        threadScanT.w() =
            ((i + 3 < n) ? g_idata[i + 3] : 0.0f) + threadScanT.z();
        res = threadScanT.w();
    }

    res = scanLocalMem(res, s_data, item_ct1);
    /*
    DPCT1065:221: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == item_ct1.get_local_range(2) - 1) {
        g_blockSums[item_ct1.get_group(2)] = res + threadScanT.w();
    }

    // write results to global memory
    sycl::uint4 *outData = (sycl::uint4 *)g_odata;

    tempData.x() = res;
    tempData.y() = res + threadScanT.x();
    tempData.z() = res + threadScanT.y();
    tempData.w() = res + threadScanT.z();

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if (i < n) { g_odata[i] = tempData.x();
        if ((i + 1) < n) { g_odata[i + 1] = tempData.y();
        if ((i+2) < n) { g_odata[i+2] = tempData.z(); } } }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Vector add, 4 elem per thread. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_vector">  	[in,out] If non-null, the vector. </param>
/// <param name="d_uniforms">	The uniforms. </param>
/// <param name="n">		 	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL void vectorAddUniform4(uint *d_vector, const uint *d_uniforms,
                                     const int n,
                                     const sycl::nd_item<3> &item_ct1,
                                     uint *uni)
{

    if (item_ct1.get_local_id(2) == 0)
    {
        uni[0] = d_uniforms[item_ct1.get_group(2)];
    }

    unsigned int address =
        item_ct1.get_local_id(2) +
        (item_ct1.get_group(2) * item_ct1.get_local_range(2) * 4);

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += item_ct1.get_local_range(2);
    }
}
