////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\sphere.h
//
// summary:	Declares the sphere class
//
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SPHEREH
#define SPHEREH

#include <CL/sycl.hpp>

#include "hitable.h"

class sphere {
public:
  sphere() {}
  sphere(vec3 cen, float r, uint16_t m) {
    data[0] = cen.x();
    data[1] = cen.y();
    data[2] = cen.z();
    data[3] = r;
    data[4] = m;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// <summary>	Hits. </summary>
  ///
  /// <remarks>	Ed, 5/20/2020. </remarks>
  ///
  /// <param name="r">   	A ray to process. </param>
  /// <param name="tmin">	The tmin. </param>
  /// <param name="tmax">	The tmax. </param>
  /// <param name="rec"> 	[in,out] The record. </param>
  ///
  /// <returns>	True if it succeeds, false if it fails. </returns>
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  SYCL_EXTERNAL bool hit(const ray &r, float tmin, float tmax,
                         hit_record &rec) const {
    vec3 oc = r.origin() - vec3(data[0], data[1], data[2]);
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - data[3] * data[3];
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sycl::sqrt(discriminant)) / a;
      if (temp < tmax && temp > tmin) {
        rec.t = temp;
        rec.p = r.point_at_parameter(rec.t);
        rec.normal = (rec.p - vec3(data[0], data[1], data[2])) / data[3];
        rec.mat_idx = static_cast<uint16_t>(data[4]);
        return true;
      }
      temp = (-b + sycl::sqrt(discriminant)) / a;
      if (temp < tmax && temp > tmin) {
        rec.t = temp;
        rec.p = r.point_at_parameter(rec.t);
        rec.normal = (rec.p - vec3(data[0], data[1], data[2])) / data[3];
        rec.mat_idx = static_cast<uint16_t>(data[4]);
        return true;
      }
    }
    return false;
  }

  // NOTE: Why this weird array? Because the HLS will synthesize this with
  // single members into ridicioulusly complex memory systems :/ 0-2: center 3:
  // radius 4: mat_idx
  sycl::float8 data;
};

#endif
