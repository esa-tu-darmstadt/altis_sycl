// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\cfd\euler3d.cu
//
// summary:	Sort class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#define NUM_STREAMS         2
#define SEED                7
#define GAMMA               1.4f
#define iterations          10
#define NDIM                3
#define NNB                 4
#define RK                  3 // 3rd order RK
#define ff_mach             1.2f
#define deg_angle_of_attack 0.0f

#define BLOCK_SIZE 192

#define VAR_DENSITY        0
#define VAR_MOMENTUM       1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR               (VAR_DENSITY_ENERGY + 1)

#ifdef _FPGA
#define ATTRIBUTE                                   \
    [[sycl::reqd_work_group_size(1, 1, BLOCK_SIZE), \
      intel::max_work_group_size(1, 1, BLOCK_SIZE)]]
#else
#define ATTRIBUTE
#endif

float kernelTime   = 0.0f;
float transferTime = 0.0f;

sycl::event                                        start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float                                              elapsed;

template<typename T>
T *
alloc(int N, sycl::queue &queue)
{
    return (T *)sycl::malloc_device(sizeof(T) * N, queue);
}

template<typename T>
void
dealloc(T *array, sycl::queue &queue)
{
    sycl::free((void *)array, queue);
}

template<typename T>
sycl::event
copy(T *dst, T *src, int N, sycl::queue &queue)
{
    return queue.memcpy((void *)dst, (void *)src, N * sizeof(T));
}

template<typename T>
void
upload(T *dst, T *src, int N, sycl::queue &queue)
{
    start_ct1 = std::chrono::steady_clock::now();

    queue.memcpy((void *)dst, (void *)src, N * sizeof(T)).wait();

    stop_ct1 = std::chrono::steady_clock::now();
    elapsed  = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
    transferTime += elapsed * 1.e-3;
}

template<typename T>
void
download(T *dst, T *src, int N, sycl::queue &queue)
{
    start_ct1 = std::chrono::steady_clock::now();

    queue.memcpy((void *)dst, (void *)src, N * sizeof(T)).wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed  = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                  .count();
    transferTime += elapsed * 1.e-3;
}

void
dump(float *variables, int nel, int nelr, sycl::queue &queue)
{
    float *h_variables = new float[nelr * NVAR];
    download(h_variables, variables, nelr * NVAR, queue);

    {
        std::ofstream file("density");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
    }

    {
        std::ofstream file("momentum");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++)
        {
            for (int j = 0; j != NDIM; j++)
                file << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            file << std::endl;
        }
    }

    {
        std::ofstream file("density_energy");
        file << nel << " " << nelr << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
    }
    delete[] h_variables;
}

std::array<float, NVAR> h_ff_variable;
sycl::float3 h_ff_flux_contribution_momentum_x;
sycl::float3 h_ff_flux_contribution_momentum_y;
sycl::float3 h_ff_flux_contribution_momentum_z;
sycl::float3 h_ff_flux_contribution_density_energy;

void
initialize_variables(int nelr, float *variables, sycl::queue &queue)
{
    sycl::buffer<float> buff{h_ff_variable.begin(), h_ff_variable.end()};

    sycl::range<3> Dg(1, 1, nelr / BLOCK_SIZE), Db(1, 1, BLOCK_SIZE);
    sycl::event    k_event = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buff, cgh, sycl::read_only};
        
        cgh.parallel_for<class initialize_variables>(
            sycl::nd_range<3>(Dg * Db, Db),
            [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
                const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2)
                            + item_ct1.get_local_id(2));
                for (int j = 0; j < NVAR; j++)
                    variables[i + j * nelr] = acc[j];
            });
    });
    k_event.wait();
    elapsed
        = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
          - k_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsed * 1.e-9;
}

SYCL_EXTERNAL inline void
compute_flux_contribution(const float        &density,
                          const sycl::float3 &momentum,
                          const float        &density_energy,
                          const float        &pressure,
                          const sycl::float3 &velocity,
                          sycl::float3       &fc_momentum_x,
                          sycl::float3       &fc_momentum_y,
                          sycl::float3       &fc_momentum_z,
                          sycl::float3       &fc_density_energy)
{
    fc_momentum_x.x() = velocity.x() * momentum.x() + pressure;
    fc_momentum_x.y() = velocity.x() * momentum.y();
    fc_momentum_x.z() = velocity.x() * momentum.z();

    fc_momentum_y.x() = fc_momentum_x.y();
    fc_momentum_y.y() = velocity.y() * momentum.y() + pressure;
    fc_momentum_y.z() = velocity.y() * momentum.z();

    fc_momentum_z.x() = fc_momentum_x.z();
    fc_momentum_z.y() = fc_momentum_y.z();
    fc_momentum_z.z() = velocity.z() * momentum.z() + pressure;

    float de_p            = density_energy + pressure;
    fc_density_energy.x() = velocity.x() * de_p;
    fc_density_energy.y() = velocity.y() * de_p;
    fc_density_energy.z() = velocity.z() * de_p;
}

SYCL_EXTERNAL inline void
compute_velocity(float &density, sycl::float3 &momentum, sycl::float3 &velocity)
{
    velocity.x() = momentum.x() / density;
    velocity.y() = momentum.y() / density;
    velocity.z() = momentum.z() / density;
}

SYCL_EXTERNAL inline float
compute_speed_sqd(sycl::float3 &velocity)
{
    return velocity.x() * velocity.x() + velocity.y() * velocity.y()
           + velocity.z() * velocity.z();
}

SYCL_EXTERNAL inline float
compute_pressure(float &density, float &density_energy, float &speed_sqd)
{
    return (float(GAMMA) - float(1.0f))
           * (density_energy - float(0.5f) * density * speed_sqd);
}

SYCL_EXTERNAL inline float
compute_speed_of_sound(float &density, float &pressure)
{
    return sycl::sqrt(float(GAMMA) * pressure / density);
}

SYCL_EXTERNAL void
cuda_compute_step_factor(int              nelr,
                         float           *variables,
                         float           *areas,
                         float           *step_factors,
                         sycl::nd_item<3> item_ct1)
{
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2)
                   + item_ct1.get_local_id(2));

    float        density = variables[i + VAR_DENSITY * nelr];
    sycl::float3 momentum;
    momentum.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

    sycl::float3 velocity;
    compute_velocity(density, momentum, velocity);
    float speed_sqd      = compute_speed_sqd(velocity);
    float pressure       = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);

    // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time
    // stepping, this later would need to be divided by the area, so we just do
    // it all at once
    step_factors[i]
        = float(0.5f)
          / (sycl::sqrt(areas[i]) * (sycl::sqrt(speed_sqd) + speed_of_sound));
}

void
compute_step_factor(int          nelr,
                    float       *variables,
                    float       *areas,
                    float       *step_factors,
                    sycl::queue &queue)
{
    sycl::range<3> Dg(1, 1, nelr / BLOCK_SIZE), Db(1, 1, BLOCK_SIZE);
    sycl::event    k_event = queue.parallel_for<class compute_step_factor>(
        sycl::nd_range<3>(Dg * Db, Db),
        [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
            cuda_compute_step_factor(
                nelr, variables, areas, step_factors, item_ct1);
        });
    k_event.wait();
    elapsed
        = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
          - k_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsed * 1.e-9;
}

void
cuda_compute_flux(int              nelr,
                  int             *elements_surrounding_elements,
                  float           *normals,
                  float           *variables,
                  float           *fluxes,
                  sycl::nd_item<3> item_ct1,
                  float           ff_variable_idx1,
                  float           ff_variable_idx2,
                  float           ff_variable_idx3,
                  sycl::float3    ff_flux_contribution_momentum_x,
                  sycl::float3    ff_flux_contribution_momentum_y,
                  sycl::float3    ff_flux_contribution_momentum_z,
                  sycl::float3    ff_flux_contribution_density_energy)
{
    const float smoothing_coefficient = float(0.2f);
    const int   i = (item_ct1.get_local_range(2) * item_ct1.get_group(2)
                   + item_ct1.get_local_id(2));

    int          j, nb;
    sycl::float3 normal;
    float        normal_len;
    float        factor;

    float        density_i = variables[i + VAR_DENSITY * nelr];
    sycl::float3 momentum_i;
    momentum_i.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum_i.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum_i.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

    sycl::float3 velocity_i;
    compute_velocity(density_i, momentum_i, velocity_i);
    float speed_sqd_i = compute_speed_sqd(velocity_i);
    float speed_i     = sycl::sqrt(speed_sqd_i);
    float pressure_i
        = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
    sycl::float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
        flux_contribution_i_momentum_z;
    sycl::float3 flux_contribution_i_density_energy;
    compute_flux_contribution(density_i,
                              momentum_i,
                              density_energy_i,
                              pressure_i,
                              velocity_i,
                              flux_contribution_i_momentum_x,
                              flux_contribution_i_momentum_y,
                              flux_contribution_i_momentum_z,
                              flux_contribution_i_density_energy);

    float        flux_i_density = float(0.0f);
    sycl::float3 flux_i_momentum;
    flux_i_momentum.x()         = float(0.0f);
    flux_i_momentum.y()         = float(0.0f);
    flux_i_momentum.z()         = float(0.0f);
    float flux_i_density_energy = float(0.0f);

    sycl::float3 velocity_nb;
    float        density_nb, density_energy_nb;
    sycl::float3 momentum_nb;
    sycl::float3 flux_contribution_nb_momentum_x,
        flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
    sycl::float3 flux_contribution_nb_density_energy;
    float        speed_sqd_nb, speed_of_sound_nb, pressure_nb;

// Unrolling the following loop would result in a way too large hardware on an
// FPGA.
#ifndef _FPGA
//#pragma unroll 
#endif
    for (j = 0; j < NNB; j++)
    {
        nb         = elements_surrounding_elements[i + j * nelr];
        normal.x() = normals[i + (j + 0 * NNB) * nelr];
        normal.y() = normals[i + (j + 1 * NNB) * nelr];
        normal.z() = normals[i + (j + 2 * NNB) * nelr];
        normal_len
            = sycl::sqrt(normal.x() * normal.x() + normal.y() * normal.y()
                         + normal.z() * normal.z());

        if (nb >= 0) // a legitimate neighbor
        {
            density_nb        = variables[nb + VAR_DENSITY * nelr];
            momentum_nb.x()   = variables[nb + (VAR_MOMENTUM + 0) * nelr];
            momentum_nb.y()   = variables[nb + (VAR_MOMENTUM + 1) * nelr];
            momentum_nb.z()   = variables[nb + (VAR_MOMENTUM + 2) * nelr];
            density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
            compute_velocity(density_nb, momentum_nb, velocity_nb);
            speed_sqd_nb = compute_speed_sqd(velocity_nb);
            pressure_nb
                = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
            speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
            compute_flux_contribution(density_nb,
                                      momentum_nb,
                                      density_energy_nb,
                                      pressure_nb,
                                      velocity_nb,
                                      flux_contribution_nb_momentum_x,
                                      flux_contribution_nb_momentum_y,
                                      flux_contribution_nb_momentum_z,
                                      flux_contribution_nb_density_energy);

            // artificial viscosity
            factor = -normal_len * smoothing_coefficient * float(0.5f)
                     * (speed_i + sycl::sqrt(speed_sqd_nb) + speed_of_sound_i
                        + speed_of_sound_nb);
            flux_i_density += factor * (density_i - density_nb);
            flux_i_density_energy
                += factor * (density_energy_i - density_energy_nb);
            flux_i_momentum.x() += factor * (momentum_i.x() - momentum_nb.x());
            flux_i_momentum.y() += factor * (momentum_i.y() - momentum_nb.y());
            flux_i_momentum.z() += factor * (momentum_i.z() - momentum_nb.z());

            // accumulate cell-centered fluxes
            factor = float(0.5f) * normal.x();
            flux_i_density += factor * (momentum_nb.x() + momentum_i.x());
            flux_i_density_energy
                += factor
                   * (flux_contribution_nb_density_energy.x()
                      + flux_contribution_i_density_energy.x());
            flux_i_momentum.x() += factor
                                   * (flux_contribution_nb_momentum_x.x()
                                      + flux_contribution_i_momentum_x.x());
            flux_i_momentum.y() += factor
                                   * (flux_contribution_nb_momentum_y.x()
                                      + flux_contribution_i_momentum_y.x());
            flux_i_momentum.z() += factor
                                   * (flux_contribution_nb_momentum_z.x()
                                      + flux_contribution_i_momentum_z.x());

            factor = float(0.5f) * normal.y();
            flux_i_density += factor * (momentum_nb.y() + momentum_i.y());
            flux_i_density_energy
                += factor
                   * (flux_contribution_nb_density_energy.y()
                      + flux_contribution_i_density_energy.y());
            flux_i_momentum.x() += factor
                                   * (flux_contribution_nb_momentum_x.y()
                                      + flux_contribution_i_momentum_x.y());
            flux_i_momentum.y() += factor
                                   * (flux_contribution_nb_momentum_y.y()
                                      + flux_contribution_i_momentum_y.y());
            flux_i_momentum.z() += factor
                                   * (flux_contribution_nb_momentum_z.y()
                                      + flux_contribution_i_momentum_z.y());

            factor = float(0.5f) * normal.z();
            flux_i_density += factor * (momentum_nb.z() + momentum_i.z());
            flux_i_density_energy
                += factor
                   * (flux_contribution_nb_density_energy.z()
                      + flux_contribution_i_density_energy.z());
            flux_i_momentum.x() += factor
                                   * (flux_contribution_nb_momentum_x.z()
                                      + flux_contribution_i_momentum_x.z());
            flux_i_momentum.y() += factor
                                   * (flux_contribution_nb_momentum_y.z()
                                      + flux_contribution_i_momentum_y.z());
            flux_i_momentum.z() += factor
                                   * (flux_contribution_nb_momentum_z.z()
                                      + flux_contribution_i_momentum_z.z());
        }
        else if (nb == -1) // a wing boundary
        {
            flux_i_momentum.x() += normal.x() * pressure_i;
            flux_i_momentum.y() += normal.y() * pressure_i;
            flux_i_momentum.z() += normal.z() * pressure_i;
        }
        else if (nb == -2) // a far field boundary
        {
            factor = float(0.5f) * normal.x();
            flux_i_density
                += factor * (ff_variable_idx1 + momentum_i.x());
            flux_i_density_energy
                += factor
                   * (ff_flux_contribution_density_energy.x()
                      + flux_contribution_i_density_energy.x());
            flux_i_momentum.x() += factor
                                   * (ff_flux_contribution_momentum_x.x()
                                      + flux_contribution_i_momentum_x.x());
            flux_i_momentum.y() += factor
                                   * (ff_flux_contribution_momentum_y.y()
                                      + flux_contribution_i_momentum_y.x());
            flux_i_momentum.z() += factor
                                   * (ff_flux_contribution_momentum_z.z()
                                      + flux_contribution_i_momentum_z.x());

            factor = float(0.5f) * normal.y();
            flux_i_density
                += factor * (ff_variable_idx2 + momentum_i.y());
            flux_i_density_energy
                += factor
                   * (ff_flux_contribution_density_energy.y()
                      + flux_contribution_i_density_energy.y());
            flux_i_momentum.x() += factor
                                   * (ff_flux_contribution_momentum_x.x()
                                      + flux_contribution_i_momentum_x.y());
            flux_i_momentum.y() += factor
                                   * (ff_flux_contribution_momentum_y.y()
                                      + flux_contribution_i_momentum_y.y());
            flux_i_momentum.z() += factor
                                   * (ff_flux_contribution_momentum_z.z()
                                      + flux_contribution_i_momentum_z.y());

            factor = float(0.5f) * normal.z();
            flux_i_density
                += factor * (ff_variable_idx3 + momentum_i.z());
            flux_i_density_energy
                += factor
                   * (ff_flux_contribution_density_energy.z()
                      + flux_contribution_i_density_energy.z());
            flux_i_momentum.x() += factor
                                   * (ff_flux_contribution_momentum_x.x()
                                      + flux_contribution_i_momentum_x.z());
            flux_i_momentum.y() += factor
                                   * (ff_flux_contribution_momentum_y.y()
                                      + flux_contribution_i_momentum_y.z());
            flux_i_momentum.z() += factor
                                   * (ff_flux_contribution_momentum_z.z()
                                      + flux_contribution_i_momentum_z.z());
        }
    }

    fluxes[i + VAR_DENSITY * nelr]        = flux_i_density;
    fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x();
    fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y();
    fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z();
    fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}

void
compute_flux(int          nelr,
             int         *elements_surrounding_elements,
             float       *normals,
             float       *variables,
             float       *fluxes,
             sycl::queue &queue)
{
    sycl::range<3> Dg(1, 1, nelr / BLOCK_SIZE), Db(1, 1, BLOCK_SIZE);
    sycl::event    k_event = queue.submit([&](sycl::handler &cgh) {
        float ff_variable_idx1 = h_ff_variable[1];
        float ff_variable_idx2 = h_ff_variable[2];
        float ff_variable_idx3 = h_ff_variable[3];
        sycl::float3 ff_flux_contribution_momentum_x = h_ff_flux_contribution_momentum_x;
        sycl::float3 ff_flux_contribution_momentum_y = h_ff_flux_contribution_momentum_y;
        sycl::float3 ff_flux_contribution_momentum_z = h_ff_flux_contribution_momentum_z;
        sycl::float3 ff_flux_contribution_density_energy = h_ff_flux_contribution_density_energy;

        cgh.parallel_for<class compute_flux>(
            sycl::nd_range<3>(Dg * Db, Db),
            [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
                cuda_compute_flux(nelr,
                                  elements_surrounding_elements,
                                  normals,
                                  variables,
                                  fluxes,
                                  item_ct1,
                                  ff_variable_idx1,
                                  ff_variable_idx2,
                                  ff_variable_idx3,
                                  ff_flux_contribution_momentum_x,
                                  ff_flux_contribution_momentum_y,
                                  ff_flux_contribution_momentum_z,
                                  ff_flux_contribution_density_energy);
            });
    });
    k_event.wait();
    elapsed
        = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
          - k_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsed * 1.e-9;
}

SYCL_EXTERNAL void
cuda_time_step(int              j,
               int              nelr,
               float           *old_variables,
               float           *variables,
               float           *step_factors,
               float           *fluxes,
               sycl::nd_item<3> item_ct1)
{
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2)
                   + item_ct1.get_local_id(2));

    float factor = step_factors[i] / float(RK + 1 - j);

    variables[i + VAR_DENSITY * nelr]
        = old_variables[i + VAR_DENSITY * nelr]
          + factor * fluxes[i + VAR_DENSITY * nelr];
    variables[i + VAR_DENSITY_ENERGY * nelr]
        = old_variables[i + VAR_DENSITY_ENERGY * nelr]
          + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
    variables[i + (VAR_MOMENTUM + 0) * nelr]
        = old_variables[i + (VAR_MOMENTUM + 0) * nelr]
          + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
    variables[i + (VAR_MOMENTUM + 1) * nelr]
        = old_variables[i + (VAR_MOMENTUM + 1) * nelr]
          + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
    variables[i + (VAR_MOMENTUM + 2) * nelr]
        = old_variables[i + (VAR_MOMENTUM + 2) * nelr]
          + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
}

void
time_step(int          j,
          int          nelr,
          float       *old_variables,
          float       *variables,
          float       *step_factors,
          float       *fluxes,
          sycl::queue &queue)
{
    sycl::range<3> Dg(1, 1, nelr / BLOCK_SIZE), Db(1, 1, BLOCK_SIZE);
    sycl::event    k_event = queue.parallel_for<class time_step>(
        sycl::nd_range<3>(Dg * Db, Db),
        [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
            cuda_time_step(j,
                           nelr,
                           old_variables,
                           variables,
                           step_factors,
                           fluxes,
                           item_ct1);
        });
    k_event.wait();
    elapsed
        = k_event.get_profiling_info<sycl::info::event_profiling::command_end>()
          - k_event.get_profiling_info<
              sycl::info::event_profiling::command_start>();
    kernelTime += elapsed * 1.e-9;
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
}

void cfd(ResultDatabase &resultDB, OptionParser &op, size_t device_idx);

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running CFDSolver\n");
    bool quiet = op.getOptionBool("quiet");
    if (!quiet)
    {
        printf(
            "WG size of kernel:initialize = %d, WG size of "
            "kernel:compute_step_factor = %d, WG size of kernel:compute_flux = "
            "%d, WG size of kernel:time_step = %d\n",
            BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE);
    }

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        kernelTime   = 0.0f;
        transferTime = 0.0f;
        if (!quiet)
            printf("Pass %d:\n", i);

        cfd(resultDB, op, device_idx);
        if (!quiet)
            printf("Done.\n");
    }
}

void
cfd(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    // set far field conditions and load them into constant memory on the gpu
    {
        const float angle_of_attack
            = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

        h_ff_variable[VAR_DENSITY] = float(1.4);

        float ff_pressure = float(1.0f);
        float ff_speed_of_sound
            = sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
        float ff_speed = float(ff_mach) * ff_speed_of_sound;

        sycl::float3 ff_velocity;
        ff_velocity.x() = ff_speed * float(cos((float)angle_of_attack));
        ff_velocity.y() = ff_speed * float(sin((float)angle_of_attack));
        ff_velocity.z() = 0.0f;

        h_ff_variable[VAR_MOMENTUM + 0]
            = h_ff_variable[VAR_DENSITY] * ff_velocity.x();
        h_ff_variable[VAR_MOMENTUM + 1]
            = h_ff_variable[VAR_DENSITY] * ff_velocity.y();
        h_ff_variable[VAR_MOMENTUM + 2]
            = h_ff_variable[VAR_DENSITY] * ff_velocity.z();

        h_ff_variable[VAR_DENSITY_ENERGY]
            = h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed))
              + (ff_pressure / float(GAMMA - 1.0f));

        sycl::float3 h_ff_momentum;
        h_ff_momentum.x() = *(h_ff_variable.data() + VAR_MOMENTUM + 0);
        h_ff_momentum.y() = *(h_ff_variable.data() + VAR_MOMENTUM + 1);
        h_ff_momentum.z() = *(h_ff_variable.data() + VAR_MOMENTUM + 2);
        compute_flux_contribution(h_ff_variable[VAR_DENSITY],
                                  h_ff_momentum,
                                  h_ff_variable[VAR_DENSITY_ENERGY],
                                  ff_pressure,
                                  ff_velocity,
                                  h_ff_flux_contribution_momentum_x,
                                  h_ff_flux_contribution_momentum_y,
                                  h_ff_flux_contribution_momentum_z,
                                  h_ff_flux_contribution_density_energy);
    }

    int nel;
    int nelr;

    // read in domain geometry
    float *areas;
    int   *elements_surrounding_elements;
    float *normals;
    {
        string        inputFile = op.getOptionString("inputFile");
        std::ifstream file(inputFile.c_str());

        if (inputFile != "")
        {
            file >> nel;
        }
        else
        {
            int problemSizes[4] = { 97000, 200000, 40000000, 60000000 };
            nel                 = problemSizes[op.getOptionInt("size") - 1];
        }
        nelr
            = BLOCK_SIZE * ((nel / BLOCK_SIZE) + std::min(1, nel % BLOCK_SIZE));

        float *h_areas                         = new float[nelr];
        int   *h_elements_surrounding_elements = new int[nelr * NNB];
        float *h_normals                       = new float[nelr * NDIM * NNB];

        srand(SEED);

        // read in data
        for (int i = 0; i < nel; i++)
        {
            if (inputFile != "")
                file >> h_areas[i];
            else
                h_areas[i] = 1.0 * rand() / RAND_MAX;

            for (int j = 0; j < NNB; j++) // NNB is always 4
            {
                if (inputFile != "")
                {
                    file >> h_elements_surrounding_elements[i + j * nelr];
                }
                else
                {
                    int val = i + (rand() % 20) - 10;
                    h_elements_surrounding_elements[i + j * nelr] = val;
                }
                if (h_elements_surrounding_elements[i + j * nelr] < 0)
                    h_elements_surrounding_elements[i + j * nelr] = -1;
                h_elements_surrounding_elements
                    [i + j * nelr]--; // it's coming in with Fortran numbering

                for (int k = 0; k < NDIM; k++) // NDIM is always 3
                {
                    if (inputFile != "")
                        file >> h_normals[i + (j + k * NNB) * nelr];
                    else
                        h_normals[i + (j + k * NNB) * nelr]
                            = 1.0 * rand() / RAND_MAX - 0.5;

                    h_normals[i + (j + k * NNB) * nelr]
                        = -h_normals[i + (j + k * NNB) * nelr];
                }
            }
        }

        // fill in remaining data
        int last = nel - 1;
        for (int i = nel; i < nelr; i++)
        {
            h_areas[i] = h_areas[last];
            for (int j = 0; j < NNB; j++)
            {
                // duplicate the last element
                h_elements_surrounding_elements[i + j * nelr]
                    = h_elements_surrounding_elements[last + j * nelr];
                for (int k = 0; k < NDIM; k++)
                    h_normals[last + (j + k * NNB) * nelr]
                        = h_normals[last + (j + k * NNB) * nelr];
            }
        }

        areas = alloc<float>(nelr, queue);
        upload<float>(areas, h_areas, nelr, queue);

        elements_surrounding_elements = alloc<int>(nelr * NNB, queue);
        upload<int>(elements_surrounding_elements,
                    h_elements_surrounding_elements,
                    nelr * NNB,
                    queue);

        normals = alloc<float>(nelr * NDIM * NNB, queue);
        upload<float>(normals, h_normals, nelr * NDIM * NNB, queue);

        delete[] h_areas;
        delete[] h_elements_surrounding_elements;
        delete[] h_normals;
    }

    // Create arrays and set initial conditions
    float *variables = alloc<float>(nelr * NVAR, queue);
    initialize_variables(nelr, variables, queue);

    float *old_variables = alloc<float>(nelr * NVAR, queue);
    initialize_variables(nelr, old_variables, queue);

    float *fluxes = alloc<float>(nelr * NVAR, queue);
    initialize_variables(nelr, fluxes, queue);

    float *step_factors = alloc<float>(nelr, queue);
    queue.memset((void *)step_factors, 0, sizeof(float) * nelr).wait();

    // make sure CUDA isn't still doing something before we start timing
    queue.wait_and_throw();

    // Begin iterations
    for (int i = 0; i < iterations; i++)
    {
        // Time will need to be recomputed, more aggressive optimization TODO
        auto cpy = copy<float>(old_variables, variables, nelr * NVAR, queue);

        // for the first iteration we compute the time step
        compute_step_factor(nelr, variables, areas, step_factors, queue);
        cpy.wait();
        auto end = cpy.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        auto start = cpy.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        transferTime += (end - start) / 1.0e9;

        for (int j = 0; j < RK; j++)
        {
            compute_flux(nelr,
                         elements_surrounding_elements,
                         normals,
                         variables,
                         fluxes,
                         queue);
            time_step(
                j, nelr, old_variables, variables, step_factors, fluxes, queue);
        }
    }

    queue.wait_and_throw();
    if (op.getOptionBool("verbose"))
    {
        if (iterations > 3)
            std::cout << "Dumping after more than 3 iterations. Results will "
                         "probably nan. This is however, the same behaviour "
                         "this benchmarks shows with its original CUDA code."
                      << std::endl;
        dump(variables, nel, nelr, queue);
    }

    dealloc<float>(areas, queue);
    dealloc<int>(elements_surrounding_elements, queue);
    dealloc<float>(normals, queue);

    dealloc<float>(variables, queue);
    dealloc<float>(old_variables, queue);
    dealloc<float>(fluxes, queue);
    dealloc<float>(step_factors, queue);

    char atts[1024];
    sprintf(atts, "numelements:%d", nel);
    resultDB.AddResult("cfd_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("cfd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("cfd_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}
