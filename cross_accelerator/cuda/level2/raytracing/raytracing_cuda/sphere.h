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
        sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};

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

        SYCL_EXTERNAL bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
            vec3 oc = r.origin() - center;
            float a = dot(r.direction(), r.direction());
            float b = dot(oc, r.direction());
            float c = dot(oc, oc) - radius*radius;
            float discriminant = b*b - a*c;
            if (discriminant > 0) {
                float temp = (-b - sycl::sqrt(discriminant)) / a;
                if (temp < tmax && temp > tmin) {
                    rec.t = temp;
                    rec.p = r.point_at_parameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    rec.mat_ptr = mat_ptr;
                    return true;
                }
                temp = (-b + sycl::sqrt(discriminant)) / a;
                if (temp < tmax && temp > tmin) {
                    rec.t = temp;
                    rec.p = r.point_at_parameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    rec.mat_ptr = mat_ptr;
                    return true;
                }
            }
            return false;
        }
        
        vec3 center;
        float radius;
        material *mat_ptr;
};

#endif
