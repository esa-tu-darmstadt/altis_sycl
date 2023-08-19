#pragma once

constexpr int32_t g_threads_x = 8;
constexpr int32_t g_threads_y = 8;
constexpr int32_t g_max_materials = 512;
constexpr int32_t g_max_objects = 512;

#ifdef _FPGA
#define ATTRIBUTE                                                              \
  [[intel::kernel_args_restrict, intel::no_global_work_offset(1),              \
    intel::num_simd_work_items(1),                                             \
    sycl::reqd_work_group_size(1, g_threads_y, g_threads_x),                   \
    intel::max_work_group_size(1, g_threads_y, g_threads_x)]]
#else
#define ATTRIBUTE
#endif
