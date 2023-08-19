////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\nw\needle.h
//
// summary:	Declares the needle class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _NEEDLE_H_
#define _NEEDLE_H_

constexpr int32_t g_block_size = 64;
constexpr int32_t g_ref_w = g_block_size;
constexpr int32_t g_temp_w = g_block_size + 1;

using no_cache_no_burst_lsu =
    sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<false>,
                          sycl::ext::intel::statically_coalesce<false>,
                          sycl::ext::intel::cache<0>>;

#endif
