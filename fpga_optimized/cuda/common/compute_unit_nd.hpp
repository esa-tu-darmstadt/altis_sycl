#include <CL/sycl.hpp>
#include <utility>
#include <iostream>

//#define LOG_LEVEL_EXTREME 1

// NOTE: Really bad code. Remember if you change something for
//       ND-Range 1 (only x-axis), do it also for ND-Range 2 functions !!

namespace {
template<typename Func,
         int32_t N,
         int32_t block_size_x,
         template<std::size_t>
         typename Name,
         std::size_t Index>
class SubmitOneComputeUnitND1
{
public:
    SubmitOneComputeUnitND1(Func                      &&f,
                           sycl::queue                &q,
                           std::array<sycl::event, N> &events,
                           const sycl::range<1>       &grid_dim,
                           int32_t*                    grids_per_cu,
                           const sycl::range<1>       &block_dim)
    {
        constexpr int32_t cu = std::integral_constant<std::size_t, Index>();

        int32_t grid_offset = 0;
        for (int32_t i = 0; i < cu; i++)
            grid_offset += grids_per_cu[i];
            
        if (grids_per_cu[cu] > 0)
        {
#ifdef LOG_LEVEL_EXTREME
            std::cout << "Compute Unit " << cu << " is running." << std::endl;
#endif

            const int bx_offset = grid_offset / block_dim[0];
            events[std::integral_constant<std::size_t, Index>()] = q.parallel_for<Name<Index>>(
                sycl::nd_range<1>(grids_per_cu[cu], block_dim),
                [=](sycl::nd_item<1> item_ct1)
                    [[intel::kernel_args_restrict,
                      intel::no_global_work_offset(1),
                      sycl::reqd_work_group_size(1, 1, block_size_x),
                      intel::max_work_group_size(1, 1, block_size_x)]] {
                        f(item_ct1,
                          bx_offset,
                          std::integral_constant<std::size_t, Index>());
                    });
        }
    }
};
template<typename Func,
         int32_t N,
         int32_t block_size_x,
         int32_t block_size_y,
         template<std::size_t>
         typename Name,
         std::size_t Index>
class SubmitOneComputeUnitND2
{
public:
    SubmitOneComputeUnitND2(Func                      &&f,
                           sycl::queue                &q,
                           std::array<sycl::event, N> &events,
                           const sycl::range<2>       &grid_dim,
                           int32_t*                    grids_per_cu,
                           const sycl::range<2>       &block_dim)
    {
        constexpr int32_t cu = std::integral_constant<std::size_t, Index>();

        int32_t grid_offset = 0;
        for (int32_t i = 0; i < cu; i++)
            grid_offset += grids_per_cu[i];
            
        if (grids_per_cu[cu] > 0)
        {
#ifdef LOG_LEVEL_EXTREME
            std::cout << "Compute Unit " << cu << " is running." << std::endl;
#endif

            const int bx_offset = grid_offset / block_dim[0];
            events[std::integral_constant<std::size_t, Index>()] = q.parallel_for<Name<Index>>(
                sycl::nd_range<2>(sycl::range<2>(grids_per_cu[cu], grid_dim[1]), block_dim),
                [=](sycl::nd_item<2> item_ct1)
                    [[intel::kernel_args_restrict,
                      intel::no_global_work_offset(1),
                      sycl::reqd_work_group_size(1, block_size_y, block_size_x),
                      intel::max_work_group_size(1, block_size_y, block_size_x)]] {
                        f(item_ct1,
                          bx_offset,
                          std::integral_constant<std::size_t, Index>());
                    });
        }
    }
};

template<template<std::size_t> typename Name,
         int32_t N,
         int32_t block_size,
         typename Func,
         std::size_t... Indices>
inline constexpr void
ComputeUnitUnrollerND1(sycl::queue                &q,
                      std::array<sycl::event, N> &events,
                      const sycl::range<1>       &grid_dim,
                      int32_t*                    grids_per_cu,
                      const sycl::range<1>       &block_dim,
                      Func                      &&f,
                      std::index_sequence<Indices...>)
{
    (SubmitOneComputeUnitND1<Func, N, block_size, Name, Indices>(
         f, q, events, grid_dim, grids_per_cu, block_dim),
     ...); // fold expression
}
template<template<std::size_t> typename Name,
         int32_t N,
         int32_t block_size_x,
         int32_t block_size_y,
         typename Func,
         std::size_t... Indices>
inline constexpr void
ComputeUnitUnrollerND2(sycl::queue                &q,
                      std::array<sycl::event, N> &events,
                      const sycl::range<2>       &grid_dim,
                      int32_t*                    grids_per_cu,
                      const sycl::range<2>       &block_dim,
                      Func                      &&f,
                      std::index_sequence<Indices...>)
{
    (SubmitOneComputeUnitND2<Func, N, block_size_x, block_size_y, Name, Indices>(
         f, q, events, grid_dim, grids_per_cu, block_dim),
     ...); // fold expression
}

} // namespace

template<std::size_t N, // Number of compute units
         template<std::size_t ID>
         typename Name, // Name for the compute units
         int32_t block_size, // Max Blocksize for ND-Range kernel to be called
         typename Func> // Callable defining compute
                        // units' functionality
// Func must take a single argument. This argument is the compute unit's ID.
// The compute unit ID is a constexpr, and it can be used to specialize
// the kernel's functionality.
// Note: the type of Func's single argument must be 'auto', because Func
// will be called with various indices (i.e., the ID for each compute unit)
constexpr double
SubmitComputeUnitsND1(sycl::queue          &q,
                     const sycl::range<1> &grid_dim,
                     const sycl::range<1> &block_dim,
                     Func                &&f)
{
    int32_t grid_per_cu[N];
    for (int32_t i = 0; i < N; i++)
        grid_per_cu[i] = 0;

    int32_t grid_per_cu_idx = 0;
    int32_t combined_grid_size = 0;
    for (int32_t i = 0; i < std::numeric_limits<int32_t>::max(); i++)
    {
        grid_per_cu[grid_per_cu_idx] += block_dim[0];

        combined_grid_size += block_dim[0];
        if (combined_grid_size >= grid_dim[0])
            break;

        grid_per_cu_idx = (grid_per_cu_idx + 1) % N;
    }

#ifdef LOG_LEVEL_EXTREME
    std::cout << "Starting ND-Range Compute Units with Grid=" << grid_dim[0]
              << ",Block=" << block_dim[0] << ",PerCu=[";
    for (int32_t i = 0; i < N; i++)
        std::cout << grid_per_cu[i] << ";";
    std::cout << "]" << std::endl;
#endif

    std::make_index_sequence<N> indices;
    std::array<sycl::event, N> events;
    ComputeUnitUnrollerND1<Name, N, block_size>(
        q, events, grid_dim, grid_per_cu, block_dim, f, indices);
    q.wait();

    double max_kernel_time = std::numeric_limits<double>::min();
    for (size_t i = 0; i < N; i++)
    {
        if (grid_per_cu[i] > 0)
        {
            double kernel_time
                = events[i].template get_profiling_info<
                      sycl::info::event_profiling::command_end>()
                  - events[i].template get_profiling_info<
                      sycl::info::event_profiling::command_start>();
            if (kernel_time > max_kernel_time)
                max_kernel_time = kernel_time;
        }
    }
    max_kernel_time *= 1.e-9;

#ifdef LOG_LEVEL_EXTREME
    std::cout << "Compute Units done!" << std::endl;
#endif

    return max_kernel_time;
}
template<std::size_t N, // Number of compute units
         template<std::size_t ID>
         typename Name, // Name for the compute units
         int32_t block_size_x, // Max Blocksize for ND-Range kernel to be called
         int32_t block_size_y, // Max Blocksize for ND-Range kernel to be called
         typename Func> // Callable defining compute
                        // units' functionality
// Func must take a single argument. This argument is the compute unit's ID.
// The compute unit ID is a constexpr, and it can be used to specialize
// the kernel's functionality.
// Note: the type of Func's single argument must be 'auto', because Func
// will be called with various indices (i.e., the ID for each compute unit)
constexpr double
SubmitComputeUnitsND2(sycl::queue          &q,
                     const sycl::range<2> &grid_dim,
                     const sycl::range<2> &block_dim,
                     Func                &&f)
{
    int32_t grid_per_cu[N];
    for (int32_t i = 0; i < N; i++)
        grid_per_cu[i] = 0;

    int32_t grid_per_cu_idx = 0;
    int32_t combined_grid_size = 0;
    for (int32_t i = 0; i < std::numeric_limits<int32_t>::max(); i++)
    {
        grid_per_cu[grid_per_cu_idx] += block_dim[0];

        combined_grid_size += block_dim[0];
        if (combined_grid_size >= grid_dim[0])
            break;

        grid_per_cu_idx = (grid_per_cu_idx + 1) % N;
    }

#ifdef LOG_LEVEL_EXTREME
    std::cout << "Starting ND-Range Compute Units with Grid=" << grid_dim[0]
              << ",Block=" << block_dim[0] << ",PerCu=[";
    for (int32_t i = 0; i < N; i++)
        std::cout << grid_per_cu[i] << ";";
    std::cout << "]" << std::endl;
#endif

    std::make_index_sequence<N> indices;
    std::array<sycl::event, N> events;
    ComputeUnitUnrollerND2<Name, N, block_size_x, block_size_y>(
        q, events, grid_dim, grid_per_cu, block_dim, f, indices);
    q.wait();

    double max_kernel_time = std::numeric_limits<double>::min();
    for (size_t i = 0; i < N; i++)
    {
        if (grid_per_cu[i] > 0)
        {
            double kernel_time
                = events[i].template get_profiling_info<
                      sycl::info::event_profiling::command_end>()
                  - events[i].template get_profiling_info<
                      sycl::info::event_profiling::command_start>();
            if (kernel_time > max_kernel_time)
                max_kernel_time = kernel_time;
        }
    }
    max_kernel_time *= 1.e-9;

#ifdef LOG_LEVEL_EXTREME
    std::cout << "Compute Units done!" << std::endl;
#endif

    return max_kernel_time;
}
