////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\camera.h
//
// summary:	Declares the camera class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CAMERAH
#define CAMERAH

#include <CL/sycl.hpp>

#include "random_gen.h"
#include "ray.h"

vec3 random_in_unit_disk(lfsr_prng &rndstate) {
    vec3 p;
    do {
        p = 2.0f * vec3(rndstate.rand(),
                        rndstate.rand(),
                        0.0f) -
            vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera {
public:
    camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = sycl::tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        
        lower_left_corner = origin - half_width * focus_dist * u -
                            half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Gets a ray. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="s">			   	A float to process. </param>
    /// <param name="t">			   	A float to process. </param>
    ///
    /// <returns>	The ray. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ray
    get_ray(float s, float t, lfsr_prng &rndstate) {
        vec3 rd = lens_radius * random_in_unit_disk(rndstate);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    vec3 u, v, w;
    float lens_radius;
};

#endif
