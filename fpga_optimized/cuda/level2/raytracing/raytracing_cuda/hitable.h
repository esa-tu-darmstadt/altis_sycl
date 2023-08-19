////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\hitable.h
//
// summary:	Declares the hitable class
//
//  origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include <CL/sycl.hpp>

class material;

struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
  uint16_t mat_idx;
};

#endif
