////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\material.h
//
// summary:	Declares the material class
//
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MATERIALH
#define MATERIALH

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Information about the hit. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct hit_record;

#include <CL/sycl.hpp>

#include "random_gen.h"
#include "ray.h"
#include "hitable.h"

float
schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0       = r0 * r0;
    return r0 + (1.0f - r0) * sycl::pow<float>((1.0f - cosine), 5.0f);
}

bool
refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted)
{
    vec3  uv           = unit_vector(v);
    float dt           = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sycl::sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3                 \
        vec3(rndstate.rand(),    \
             rndstate.rand(),    \
             rndstate.rand())

vec3 random_in_unit_sphere(
    lfsr_prng &rndstate) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

vec3
reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

class material
{
public:
    enum type : uint8_t
    {
        lambertian,
        metal,
        dielectric
    };

    type m_type;
    vec3  albedo;  // For lambertian and metal
    float fuzz;    // For metal
    float ref_idx; // for dielectric

    material(const vec3 &a)
        : albedo(a)
    {
        m_type = type::lambertian;
    }

    material(const vec3 &a, float f)
        : albedo(a)
    {
        m_type = type::metal;
        if (f < 1)
            fuzz = f;
        else
            fuzz = 1;
    }

    material(float ri)
        : ref_idx(ri)
    {
        m_type = type::dielectric;
    }

    SYCL_EXTERNAL bool scatter(
        const ray        &r_in,
        const hit_record &rec,
        vec3             &attenuation,
        ray              &scattered,
        lfsr_prng        &rndstate) const
    {
        switch (m_type)
        {
            case type::lambertian:
                return scatter_lambertian(
                    r_in, rec, attenuation, scattered, rndstate);
            case type::metal:
                return scatter_metal(
                    r_in, rec, attenuation, scattered, rndstate);
            case type::dielectric:
                return scatter_dielectric(
                    r_in, rec, attenuation, scattered, rndstate);
        }
    };

    SYCL_EXTERNAL bool scatter_dielectric(
        const ray        &r_in,
        const hit_record &rec,
        vec3             &attenuation,
        ray              &scattered,
        lfsr_prng        &rndstate) const
    {
        attenuation = vec3(1.0f, 1.0f, 1.0f);

        vec3  outward_normal;
        float ni_over_nt;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f)
        {
            outward_normal = -rec.normal;
            ni_over_nt     = ref_idx;
            cosine
                = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine
                = sycl::sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt     = 1.0f / ref_idx;
            cosine         = -dot(r_in.direction(), rec.normal)
                     / r_in.direction().length();
        }

        vec3  refracted;
        float reflect_prob;
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;

        vec3  reflected = reflect(r_in.direction(), rec.normal);
        if (rndstate.rand() < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    SYCL_EXTERNAL bool scatter_metal(
        const ray        &r_in,
        const hit_record &rec,
        vec3             &attenuation,
        ray              &scattered,
        lfsr_prng        &rndstate) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered      = ray(
            rec.p, reflected + fuzz * random_in_unit_sphere(rndstate));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    SYCL_EXTERNAL bool scatter_lambertian(
        const ray        &r_in,
        const hit_record &rec,
        vec3             &attenuation,
        ray              &scattered,
        lfsr_prng        &rndstate) const
    {
        vec3 target
            = rec.p + rec.normal + random_in_unit_sphere(rndstate);
        scattered   = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }
};

#endif
