////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\hitable_list.h
//
// summary:	Declares the hitable list class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HITABLELISTH
#define HITABLELISTH

#include <CL/sycl.hpp>
#include "sphere.h"

class hitable_list {
    public:
        hitable_list() {}
        hitable_list(sphere *l, int n) {list = l; list_size = n; }

        SYCL_EXTERNAL bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

        sphere *list;
        int list_size;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Hits. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="r">		A ray to process. </param>
/// <param name="t_min">	The minimum. </param>
/// <param name="t_max">	The maximum. </param>
/// <param name="rec">  	[in,out] The record. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i].hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
