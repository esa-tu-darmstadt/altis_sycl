////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\nw\needle_kernel.cu
//
// summary:	Needle kernel class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "needle.h"

inline int maximum(int a, int b, int c) {
  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}

template <int cu_id>
void needle_cuda_shared(int *referrence, int *matrix_cuda, int b_index_x_offset,
                        int b_index_y_offset, int cols, int penalty, int i,
                        int bx_off, int block_width,
                        sycl::nd_item<1> item_ct1) {
  int bx = item_ct1.get_group(0) + bx_off;
  int tx = item_ct1.get_local_id(0);

  int b_index_x = bx + b_index_x_offset;
  int b_index_y = -bx - 1 + b_index_y_offset;

  int index = cols * g_block_size * b_index_y + g_block_size * b_index_x + tx +
              (cols + 1);
  int index_n =
      cols * g_block_size * b_index_y + g_block_size * b_index_x + tx + (1);
  int index_w =
      cols * g_block_size * b_index_y + g_block_size * b_index_x + (cols);
  int index_nw = cols * g_block_size * b_index_y + g_block_size * b_index_x;

  // SMem allocation 1.
  //
  auto ref_ptr = group_local_memory_for_overwrite<int[g_ref_w][g_ref_w]>(
      item_ct1.get_group());
  auto temp_ptr = group_local_memory_for_overwrite<int[g_temp_w][g_temp_w]>(
      item_ct1.get_group());
  auto &ref = *ref_ptr;
  auto &temp = *temp_ptr;

  // SMem initialization 1.
  //
  for (int8_t ty = 0; ty < g_block_size; ty++)
    ref[ty][tx] = referrence[index + cols * ty];

  if (tx == 0)
    temp[0][0] = no_cache_no_burst_lsu::load(
        sycl::device_ptr<int>(matrix_cuda) + index_nw);
  temp[tx + 1][0] = no_cache_no_burst_lsu::load(
      sycl::device_ptr<int>(matrix_cuda) + index_w + cols * tx);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  temp[0][tx + 1] =
      no_cache_no_burst_lsu::load(sycl::device_ptr<int>(matrix_cuda) + index_n);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Calculation 1.
  //
  for (int8_t m = 0; m < g_block_size; m++) {
    if (tx <= m) {
      int t_index_x = tx + 1;
      int t_index_y = m - tx + 1;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_x - 1][t_index_y - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_x][t_index_y - 1] - penalty,
                  temp[t_index_x - 1][t_index_y] - penalty);
    }

    item_ct1.barrier(cl::sycl::access::fence_space::local_space);
  }

  // Calculation 2.
  //
  for (int8_t m = g_block_size - 2; m >= 0; m--) {
    if (tx <= m) {
      int t_index_x = tx + g_block_size - m;
      int t_index_y = g_block_size - tx;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_y - 1][t_index_x - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_y][t_index_x - 1] - penalty,
                  temp[t_index_y - 1][t_index_x] - penalty);
    }

    item_ct1.barrier(cl::sycl::access::fence_space::local_space);
  }

  // Writeback.
  //
  for (int8_t ty = 0; ty < g_block_size; ty++)
    matrix_cuda[index + ty * cols] = temp[tx + 1][ty + 1];
}
