#ifndef __MEMORY_UTILS_HPP__
#define __MEMORY_UTILS_HPP__

#include <type_traits>

#include "metaprogramming_utils.hpp"

//
// The utilities in this file are used for converting streaming data to/from
// memory from/to a pipe.
//

namespace fpga_tools {

namespace detail {

//
// Helper to check if a SYCL pipe and pointer have the same base type
//
template <typename PipeT, typename PtrT>
struct pipe_and_pointer_have_same_base {
  using PipeBaseT =
      std::conditional_t<fpga_tools::has_subscript_v<PipeT>,
                         std::decay_t<decltype(std::declval<PipeT>()[0])>,
                         PipeT>;
  using PtrBaseT = std::decay_t<decltype(std::declval<PtrT>()[0])>;
  static constexpr bool value = std::is_same_v<PipeBaseT, PtrBaseT>;
};

template <typename PipeT, typename PtrT>
inline constexpr bool pipe_and_pointer_have_same_base_v =
    pipe_and_pointer_have_same_base<PipeT, PtrT>::value;

//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeRemainder(PtrT in_ptr, size_t full_count,
                           size_t remainder_count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < full_count; i++) {
    PipeT pipe_data;
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }

  PipeT pipe_data;
  for (size_t i = 0; i < remainder_count; i++) {
    pipe_data[i] = in_ptr[full_count * elements_per_cycle + i];
  }
  Pipe::write(pipe_data);
}

//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeNoRemainder(PtrT in_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    PipeT pipe_data;
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryRemainder(PtrT out_ptr, size_t full_count,
                           size_t remainder_count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < full_count; i++) {
    auto pipe_data = Pipe::read();
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }

  auto pipe_data = Pipe::read();
  for (size_t i = 0; i < remainder_count; i++) {
    out_ptr[full_count * elements_per_cycle + i] = pipe_data[i];
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryNoRemainder(PtrT out_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    auto pipe_data = Pipe::read();
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }
}

}  // namespace detail

//
// Streams data from memory to a SYCL pipe 1 element a time
//
template <typename Pipe, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(detail::pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    Pipe::write(in_ptr[i]);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  if constexpr (!remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr, count);
  } else {
    // might have a remainder and it was not specified, so calculate it
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (!remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr,
                                                              full_count);
  } else {
    // might have a remainder that was specified by the user
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 1 element a time
//
template <typename Pipe, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(detail::pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    out_ptr[i] = Pipe::read();
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  if constexpr (!remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr, count);
  } else {
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (!remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr,
                                                              full_count);
  } else {
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count,
                                                            remainder_count);
  }
}

//
// The Producer kernel reads 'literals_per_cycle' elements at a time from
// memory (in_ptr) and writes them into InPipe. We use the utilities from
// DirectProgramming/DPC++FPGA/include/memory_utils.hpp to do this.
//
//  Template parameters:
//    Id: the type to use for the kernel ID
//    InPipe: a SYCL pipe that streams bytes into the decompression engine,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the number of bytes to read from the pointer and
//      write to the pipe at once.
//
//  Arguments:
//    q: the SYCL queue
//    in_count_padded: the total number of bytes to read from in_ptr and write
//      to the input pipe. In this design, we pad the size to be a multiple of
//      literals_per_cycle.
//    in_ptr: a pointer to the input data
//
template <typename Id, typename InPipe, unsigned literals_per_cycle>
sycl::event SubmitProducer(sycl::queue& q, unsigned in_count_padded,
                           unsigned char* in_ptr) {
  assert(in_count_padded % literals_per_cycle == 0);
  auto iteration_count = in_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // Use the MemoryToPipe utility to read from in_ptr 'literals_per_cycle'
    // elements at once and write them to 'InPipe'.
    // The 'false' template argument is our way of guaranteeing to the library
    // that 'literals_per_cycle is a multiple 'iteration_count'. In both the
    // GZIP and SNAPPY designs, we guarantee this in the DecompressBytes
    // functions in ../gzip/gzip_decompressor.hpp and
    // ../snappy/snappy_decompressor.hpp respectively.
    sycl::device_ptr<unsigned char> in(in_ptr);
    fpga_tools::MemoryToPipe<InPipe, literals_per_cycle, false>(
        in, iteration_count);
  });
}

//
// Same idea as SubmitProducer but in the opposite direction. Data is streamed
// from the SYCL pipe (OutPipe) and written to memory (out_ptr).
//
//  Template parameters:
//    Id: the type to use for the kernel ID
//    InPipe: a SYCL pipe that streams bytes from the decompression engine,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the number of bytes to read from pipe and write to the
//      pointer at once.
//
//  Arguments:
//    q: the SYCL queue
//    out_count_padded: the total number of bytes to read from input pipe and
//      write to out_ptr. In this design, we pad the size to be a multiple of
//      literals_per_cycle.
//    out_ptr: a pointer to the output data
//
template <typename Id, typename OutPipe, unsigned literals_per_cycle>
sycl::event SubmitConsumer(sycl::queue& q, unsigned out_count_padded,
                           unsigned char* out_ptr) {
  assert(out_count_padded % literals_per_cycle == 0);
  auto iteration_count = out_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // Use the PipeToMemory utility to read 'literals_per_cycle'
    // elements at once from 'OutPipe' and write them to 'out_ptr'.
    // For details about the 'false' template parameter, see the SubmitProducer
    // function above.
    sycl::device_ptr<unsigned char> out(out_ptr);
    fpga_tools::PipeToMemory<OutPipe, literals_per_cycle, false>(
        out, iteration_count);
  });
}

}  // namespace fpga_tools

#endif /* __MEMORY_UTILS_HPP__ */