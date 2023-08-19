#pragma once

// #include <dpct/dpct.hpp>
// #include <dpct/rng_utils.hpp>
#include "oneapi/dpl/random"
// #include <oneapi/mkl.hpp>
// #include <oneapi/mkl/rng/device.hpp>

constexpr float rnd_fac = (0.999999f / std::numeric_limits<uint32_t>::max());

struct lfsr_prng {
  // uint32_t lfsr;
  // lfsr_prng(sycl::nd_item<2> item) : lfsr(item.get_global_linear_id()) {}

  // float rand() {
  //     unsigned char lsb = lfsr & 0x01u;

  //     lfsr >>= 1;
  //     lfsr ^= (-lsb) & 0xA3000000u;
  //     return rnd_fac*lfsr;
  // }

  using engine_t = oneapi::dpl::minstd_rand;
  engine_t engine;
  oneapi::dpl::uniform_real_distribution<float> distr =
      oneapi::dpl::uniform_real_distribution<float>(0.0f, 1.0f);

  lfsr_prng(sycl::nd_item<2> item) : engine(item.get_global_linear_id(), 0) {}

  float rand() { return distr(engine); }
};
