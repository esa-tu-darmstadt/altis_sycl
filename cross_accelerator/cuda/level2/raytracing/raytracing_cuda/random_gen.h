#pragma once

// #include <dpct/dpct.hpp>
// #include <dpct/rng_utils.hpp>
#include "oneapi/dpl/random"
// #include <oneapi/mkl.hpp>
// #include <oneapi/mkl/rng/device.hpp>

struct 
lfsr_prng
{
    // uint32_t lfsr = 1984;
    // float rand() {
    //     unsigned char lsb = lfsr & 0x01u;

    //     lfsr >>= 1;
    //     lfsr ^= (-lsb) & 0xA3000000u;
    //     return (0.999999f/std::numeric_limits<uint32_t>::max())*lfsr;
    // }

    using engine_t = oneapi::dpl::minstd_rand;
    engine_t engine;
    oneapi::dpl::uniform_real_distribution<float> distr =
        oneapi::dpl::uniform_real_distribution<float>(0.0f, 1.0f);

    lfsr_prng(sycl::nd_item<3> item) : engine(item.get_global_linear_id(), 0) {}

    float rand() 
    {
        return distr(engine);
    }
};
