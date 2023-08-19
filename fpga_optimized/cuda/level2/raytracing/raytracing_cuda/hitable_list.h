////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\hitable_list.h
//
// summary:	Declares the hitable list class
//
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HITABLELISTH
#define HITABLELISTH

#include "constants.hpp"
#include "sphere.h"
#include <CL/sycl.hpp>

#ifdef _STRATIX10
#define UNROLL_FAC 30
#endif
#ifdef _AGILEX
#define UNROLL_FAC 16
#endif

class hitable_list {
public:
  hitable_list() {}
  hitable_list(sphere *l, int n) {
    list = l;
    list_size = n;
  }

  SYCL_EXTERNAL static bool hit(const ray &r, float tmin, float tmax,
                                hit_record &rec, const sphere *spheres,
                                uint16_t obj_count) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = tmax;
#pragma unroll UNROLL_FAC
    for (uint16_t i = 0; i < obj_count; i++) {
      if (spheres[i].hit(r, tmin, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
    return hit_anything;
  }

  sphere *list;
  int list_size;
};

#endif
