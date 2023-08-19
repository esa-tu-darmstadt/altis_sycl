////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\hitable.h
//
// summary:	Declares the hitable class
// 
//  origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HITABLEH
#define HITABLEH

#include <CL/sycl.hpp>
#include "ray.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

#endif
