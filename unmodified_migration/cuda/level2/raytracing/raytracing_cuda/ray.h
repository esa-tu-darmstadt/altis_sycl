////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\ray.h
//
// summary:	Declares the ray class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef RAYH
#define RAYH
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "vec3.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A ray. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class ray
{
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Default constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ray() {}

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="a">	A vec3 to process. </param>
        /// <param name="b">	A vec3 to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ray(const vec3& a, const vec3& b) { A = a; B = b; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Gets the origin. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <returns>	A vec3. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        vec3 origin() const       { return A; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Gets the direction. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <returns>	A vec3. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        vec3 direction() const    { return B; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Point at parameter. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="t">	A float to process. </param>
        ///
        /// <returns>	A vec3. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        vec3 point_at_parameter(float t) const { return A + t*B; }

        /// <summary>	A vec3 to process. </summary>
        vec3 A;
        /// <summary>	A vec3 to process. </summary>
        vec3 B;
};

#endif
