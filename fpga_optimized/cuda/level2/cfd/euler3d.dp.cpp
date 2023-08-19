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

#include "compute_unit.hpp"
#include "fpga_pwr.hpp"
#include "pipe_utils.hpp"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

// Compute untis used for the most computationally intensive algorithm.
//
#ifdef _STRATIX10
constexpr int32_t num_cu = 4;
#endif
#ifdef _AGILEX
constexpr int32_t num_cu = 8;
#endif

struct var_t {
  // Why this weird array instead of just members?
  // -> Somehow, when using seperate members, the
  // compiler generates a seperate LSU for each one...
  float data[6];
  float &density() { return data[0]; }
  float &momentum_x() { return data[1]; }
  float &momentum_y() { return data[2]; }
  float &momentum_z() { return data[3]; }
  float &density_energy() { return data[4]; }
  int nb() { return (int)data[5]; }
};

#define SEED 7
#define GAMMA 1.4f
#define iterations 10
#define NDIM 3
#define NNB 4
#define RK 3 // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define BLOCK_SIZE 256

#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

float kernelTime = 0.0f;
float transferTime = 0.0f;

sycl::event start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float elapsed;

template <typename T> T *alloc(int N, sycl::queue &queue) {
  return (T *)sycl::malloc_device(sizeof(T) * N, queue);
}

template <typename T> void dealloc(T *array, sycl::queue &queue) {
  sycl::free((void *)array, queue);
}

template <typename T>
sycl::event copy(T *dst, T *src, int N, sycl::queue &queue) {
  return queue.memcpy((void *)dst, (void *)src, N * sizeof(T));
}

template <typename T> void upload(T *dst, T *src, int N, sycl::queue &queue) {
  start_ct1 = std::chrono::steady_clock::now();

  queue.memcpy((void *)dst, (void *)src, N * sizeof(T)).wait();

  stop_ct1 = std::chrono::steady_clock::now();
  elapsed =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsed * 1.e-3;
}

template <typename T> void download(T *dst, T *src, int N, sycl::queue &queue) {
  start_ct1 = std::chrono::steady_clock::now();

  queue.memcpy((void *)dst, (void *)src, N * sizeof(T)).wait();
  stop_ct1 = std::chrono::steady_clock::now();
  elapsed =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  transferTime += elapsed * 1.e-3;
}

void dump(var_t *variables, int nel, int nelr, sycl::queue &queue) {
  var_t *h_variables = new var_t[nelr];
  download(h_variables, variables, nelr, queue);

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++)
      file << h_variables[i].density() << std::endl;
  }

  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++) {
      file << h_variables[i].momentum_x() << " ";
      file << h_variables[i].momentum_y() << " ";
      file << h_variables[i].momentum_z() << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++)
      file << h_variables[i].density_energy() << std::endl;
  }
  delete[] h_variables;
}

std::array<float, NVAR> h_ff_variable;
sycl::float3 h_ff_flux_contribution_momentum_x;
sycl::float3 h_ff_flux_contribution_momentum_y;
sycl::float3 h_ff_flux_contribution_momentum_z;
sycl::float3 h_ff_flux_contribution_density_energy;

void initialize_variables(int nelr, var_t *variables, sycl::queue &queue) {
  sycl::range<1> Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
  sycl::event k_event = queue.submit([&](sycl::handler &cgh) {
    const float a0 = h_ff_variable[0];
    const float a1 = h_ff_variable[1];
    const float a2 = h_ff_variable[2];
    const float a3 = h_ff_variable[3];
    const float a4 = h_ff_variable[4];

    cgh.parallel_for<class initialize_variables>(
        sycl::nd_range<1>(Dg * Db, Db),
        [=](sycl::nd_item<1> item_ct1)
            [[intel::kernel_args_restrict, intel::no_global_work_offset(1),
              intel::num_simd_work_items(4),
              sycl::reqd_work_group_size(1, 1, BLOCK_SIZE),
              intel::max_work_group_size(1, 1, BLOCK_SIZE)]] {
              const int i = item_ct1.get_global_id(0);
              if (i >= nelr)
                return;

              var_t var;
              var.density() = a0;
              var.momentum_x() = a1;
              var.momentum_y() = a2;
              var.momentum_z() = a3;
              var.density_energy() = a4;

              variables[i] = var;
            });
  });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

SYCL_EXTERNAL inline void compute_flux_contribution(
    const float &density, const sycl::float3 &momentum,
    const float &density_energy, const float &pressure,
    const sycl::float3 &velocity, sycl::float3 &fc_momentum_x,
    sycl::float3 &fc_momentum_y, sycl::float3 &fc_momentum_z,
    sycl::float3 &fc_density_energy) {
  fc_momentum_x.x() = velocity.x() * momentum.x() + pressure;
  fc_momentum_x.y() = velocity.x() * momentum.y();
  fc_momentum_x.z() = velocity.x() * momentum.z();

  fc_momentum_y.x() = fc_momentum_x.y();
  fc_momentum_y.y() = velocity.y() * momentum.y() + pressure;
  fc_momentum_y.z() = velocity.y() * momentum.z();

  fc_momentum_z.x() = fc_momentum_x.z();
  fc_momentum_z.y() = fc_momentum_y.z();
  fc_momentum_z.z() = velocity.z() * momentum.z() + pressure;

  float de_p = density_energy + pressure;
  fc_density_energy.x() = velocity.x() * de_p;
  fc_density_energy.y() = velocity.y() * de_p;
  fc_density_energy.z() = velocity.z() * de_p;
}

SYCL_EXTERNAL inline void compute_velocity(const float &density,
                                           const sycl::float3 &momentum,
                                           sycl::float3 &velocity) {
  velocity.x() = momentum.x() / density;
  velocity.y() = momentum.y() / density;
  velocity.z() = momentum.z() / density;
}

SYCL_EXTERNAL inline float compute_speed_sqd(const sycl::float3 &velocity) {
  return velocity.x() * velocity.x() + velocity.y() * velocity.y() +
         velocity.z() * velocity.z();
}

SYCL_EXTERNAL inline float compute_pressure(const float &density,
                                            const float &density_energy,
                                            const float &speed_sqd) {
  return (float(GAMMA) - float(1.0f)) *
         (density_energy - float(0.5f) * density * speed_sqd);
}

SYCL_EXTERNAL inline float compute_speed_of_sound(const float &density,
                                                  const float &pressure) {
  return sycl::sqrt(float(GAMMA) * pressure / density);
}

SYCL_EXTERNAL void cuda_compute_step_factor(int nelr, var_t *variables,
                                            float *areas, float *step_factors,
                                            sycl::nd_item<1> item_ct1) {
  const int i = item_ct1.get_global_id(0);
  if (i >= nelr)
    return;

  var_t var = variables[i];
  float density = var.density();
  float one_div_density = 1.0f / density;
  sycl::float3 momentum = {var.momentum_x(), var.momentum_y(),
                           var.momentum_z()};
  float density_energy = var.density_energy();

  sycl::float3 velocity{momentum.x() * one_div_density,
                        momentum.y() * one_div_density,
                        momentum.z() * one_div_density};
  float speed_sqd = compute_speed_sqd(velocity);
  float pressure = compute_pressure(density, density_energy, speed_sqd);
  float speed_of_sound = sycl::sqrt(float(GAMMA) * pressure * one_div_density);

  // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time
  // stepping, this later would need to be divided by the area, so we just do
  // it all at once
  step_factors[i] = float(0.5f) / (sycl::sqrt(areas[i]) *
                                   (sycl::sqrt(speed_sqd) + speed_of_sound));
}

void compute_step_factor(int nelr, var_t *variables, float *areas,
                         float *step_factors, sycl::queue &queue) {
  sycl::range<1> Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
  sycl::event k_event = queue.parallel_for<class compute_step_factor>(
      sycl::nd_range<1>(Dg * Db, Db),
      [=](sycl::nd_item<1> item_ct1)
          [[intel::kernel_args_restrict, intel::no_global_work_offset(1),
            intel::num_simd_work_items(4),
            sycl::reqd_work_group_size(1, 1, BLOCK_SIZE),
            intel::max_work_group_size(1, 1, BLOCK_SIZE)]] {
            cuda_compute_step_factor(nelr, variables, areas, step_factors,
                                     item_ct1);
          });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

template <size_t cu> class var_nb_streamer_cu;
template <size_t cu> class normal_streamer_cu;
template <size_t cu> class flux_writeback_cu;
template <size_t cu> class compute_cu;

using var_i_streamer_pipe =
    fpga_tools::PipeArray<class var_i_streamer_pipe_id, var_t, 16, num_cu>;
using var_nb_streamer_pipe =
    fpga_tools::PipeArray<class var_nb_streamer_pipe_id, var_t, 32, num_cu>;
using normal_streamer_pipe =
    fpga_tools::PipeArray<class normal_streamer_pipe_id, sycl::float4, 32,
                          num_cu>;
using flux_writeback_pipe =
    fpga_tools::PipeArray<class flux_writeback_pipe_id, var_t, 16, num_cu>;

void compute_flux(int nelr, int *elements_surrounding_elements,
                  sycl::float3 *normals, var_t *variables, var_t *fluxes,
                  sycl::queue &queue) {
  const int32_t per_cu = nelr / num_cu;

  std::array<sycl::event, num_cu> var_nb_streamer_events;
  SubmitComputeUnits<num_cu, var_nb_streamer_cu>(
      queue, var_nb_streamer_events, [=](auto ID) {
        const int32_t s = ID * per_cu;
        const int32_t e = ID * per_cu + per_cu;

        [[intel::initiation_interval(1)]] for (int32_t i = s; i < e; i++) {
          int32_t nbs[NNB];
#pragma unroll
          for (int8_t j = 0; j < NNB; j++)
            nbs[j] = elements_surrounding_elements[i * NNB + j];

          [[intel::speculated_iterations(0),
            intel::initiation_interval(1)]] for (int8_t j = 0; j < NNB; j++) {
            // We are calculating all three cases at once: nb >= 0, nb == -1 and
            // else. Therefore be sure to bound the access to variables!!!
            int nb = nbs[j];
            var_t var = nb < 0 ? var_t() : variables[nb];
            var.data[7] = nb;
            var_nb_streamer_pipe::PipeAt<ID>::write(var);
          }
        }
      });

  std::array<sycl::event, num_cu> normal_streamer_events;
  SubmitComputeUnits<num_cu, normal_streamer_cu>(
      queue, normal_streamer_events, [=](auto ID) {
        const int32_t s = ID * per_cu;
        const int32_t e = ID * per_cu + per_cu;

        [[intel::initiation_interval(1)]] for (int32_t i = s; i < e; i++) {
          [[intel::speculated_iterations(0),
            intel::initiation_interval(1)]] for (int8_t j = 0; j < NNB; j++) {
            const sycl::float3 normal = normals[i * NNB + j];
            const float normal_len =
                sycl::sqrt(normal.x() * normal.x() + normal.y() * normal.y() +
                           normal.z() * normal.z());
            normal_streamer_pipe::PipeAt<ID>::write(
                {normal[ID], normal[1], normal[2], normal_len});
          }
        }
      });

  std::array<sycl::event, num_cu> flux_writeback_events;
  SubmitComputeUnits<num_cu, flux_writeback_cu>(
      queue, flux_writeback_events, [=](auto ID) {
        const int32_t s = ID * per_cu;
        const int32_t e = ID * per_cu + per_cu;

        [[intel::initiation_interval(1), ivdep(fluxes)]] for (int32_t i = s;
                                                              i < e; i++)
          fluxes[i] = flux_writeback_pipe::PipeAt<ID>::read();
      });

  std::array<sycl::event, num_cu> compute_events;
  const float ff_variable_idx1 = h_ff_variable[1];
  const float ff_variable_idx2 = h_ff_variable[2];
  const float ff_variable_idx3 = h_ff_variable[3];
  const sycl::float3 ff_flux_contribution_momentum_x =
      h_ff_flux_contribution_momentum_x;
  const sycl::float3 ff_flux_contribution_momentum_y =
      h_ff_flux_contribution_momentum_y;
  const sycl::float3 ff_flux_contribution_momentum_z =
      h_ff_flux_contribution_momentum_z;
  const sycl::float3 ff_flux_contribution_density_energy =
      h_ff_flux_contribution_density_energy;
  SubmitComputeUnits<num_cu, compute_cu>(queue, compute_events, [=](auto ID) {
    constexpr float smoothing_coefficient = 0.2f;

    const int32_t s = ID * per_cu;
    const int32_t e = ID * per_cu + per_cu;

    [[intel::initiation_interval(1)]] for (int32_t i = s; i < e; i++) {
      var_t var = variables[i];
      float density_i = var.density();
      sycl::float3 momentum_i = {var.momentum_x(), var.momentum_y(),
                                 var.momentum_z()};
      const float density_energy_i = var.density_energy();

      const float one_div_density_i = 1.0f / density_i;
      const sycl::float3 velocity_i{momentum_i[0] * one_div_density_i,
                                    momentum_i[1] * one_div_density_i,
                                    momentum_i[2] * one_div_density_i};
      const float speed_sqd_i = compute_speed_sqd(velocity_i);
      const float speed_i = sycl::sqrt(speed_sqd_i);
      const float pressure_i =
          compute_pressure(density_i, density_energy_i, speed_sqd_i);
      const float speed_of_sound_i =
          sycl::sqrt(float(GAMMA) * pressure_i * one_div_density_i);

      sycl::float3 flux_contribution_i_momentum_x,
          flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
      sycl::float3 flux_contribution_i_density_energy;
      compute_flux_contribution(
          density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
          flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
          flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

      float flux_i_density_arr[NNB];
      float flux_i_momentum_x_arr[NNB];
      float flux_i_momentum_y_arr[NNB];
      float flux_i_momentum_z_arr[NNB];
      float flux_i_density_energy_arr[NNB];

      var_t vars[NNB];
      [[intel::speculated_iterations(0),
        intel::initiation_interval(1)]] for (int8_t i = 0; i < NNB; i++)
        vars[i] = var_nb_streamer_pipe::PipeAt<ID>::read();
      sycl::float4 normals[NNB];
      [[intel::speculated_iterations(0),
        intel::initiation_interval(1)]] for (int8_t i = 0; i < NNB; i++)
        normals[i] = normal_streamer_pipe::PipeAt<ID>::read();

#pragma unroll
      [[intel::speculated_iterations(0),
        intel::initiation_interval(1)]] for (int8_t j = 0; j < NNB; j++) {
        auto normal_payload = normals[j];
        const sycl::float3 normal = {normal_payload[0], normal_payload[1],
                                     normal_payload[2]};
        const float normal_len = normal_payload[3];

        const float factor2 = 0.5f * normal.x();
        const float factor3 = 0.5f * normal.y();
        const float factor4 = 0.5f * normal.z();

        float fid_if_nb_gt0;
        float fid_if_nb_en2;
        sycl::float3 fim_if_nb_gt0;
        sycl::float3 fim_if_nb_en1;
        sycl::float3 fim_if_nb_en2;
        float fide_if_nb_gt0;
        float fide_if_nb_en2;

        auto var_nb = vars[j];
        const int32_t nb = var_nb.nb();

        float density_nb = var_nb.density();
        sycl::float3 momentum_nb = {var_nb.momentum_x(), var_nb.momentum_y(),
                                    var_nb.momentum_z()};
        float density_energy_nb = var_nb.density_energy();

        const float one_div_density_nb = 1.0f / density_nb;
        const sycl::float3 velocity_nb{momentum_nb[0] * density_nb,
                                       momentum_nb[1] * density_nb,
                                       momentum_nb[2] * density_nb};
        const float speed_sqd_nb = compute_speed_sqd(velocity_nb);
        const float speed_nb = sycl::sqrt(speed_sqd_nb);
        const float pressure_nb =
            compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
        const float speed_of_sound_nb =
            sycl::sqrt(float(GAMMA) * pressure_nb * one_div_density_nb);

        sycl::float3 flux_contribution_nb_momentum_x,
            flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
        sycl::float3 flux_contribution_nb_density_energy;
        compute_flux_contribution(
            density_nb, momentum_nb, density_energy_nb, pressure_nb,
            velocity_nb, flux_contribution_nb_momentum_x,
            flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z,
            flux_contribution_nb_density_energy);

        // artificial viscosity
        const float factor = -normal_len * smoothing_coefficient * 0.5f *
                             (speed_i + sycl::sqrt(speed_sqd_nb) +
                              speed_of_sound_i + speed_of_sound_nb);

        // accumulate cell-centered fluxes

        fid_if_nb_gt0 = factor * (density_i - density_nb) +
                        factor2 * (momentum_nb.x() + momentum_i.x()) +
                        factor3 * (momentum_nb.y() + momentum_i.y()) +
                        factor4 * (momentum_nb.z() + momentum_i.z());
        fide_if_nb_gt0 = factor * (density_energy_i - density_energy_nb) +
                         factor2 * (flux_contribution_nb_density_energy.x() +
                                    flux_contribution_i_density_energy.x()) +
                         factor3 * (flux_contribution_nb_density_energy.y() +
                                    flux_contribution_i_density_energy.y()) +
                         factor4 * (flux_contribution_nb_density_energy.z() +
                                    flux_contribution_i_density_energy.z());
        fim_if_nb_gt0.x() = factor * (momentum_i.x() - momentum_nb.x()) +
                            factor2 * (flux_contribution_nb_momentum_x.x() +
                                       flux_contribution_i_momentum_x.x()) +
                            factor3 * (flux_contribution_nb_momentum_x.y() +
                                       flux_contribution_i_momentum_x.y()) +
                            factor4 * (flux_contribution_nb_momentum_x.z() +
                                       flux_contribution_i_momentum_x.z());
        fim_if_nb_gt0.y() = factor * (momentum_i.y() - momentum_nb.y()) +
                            factor2 * (flux_contribution_nb_momentum_y.x() +
                                       flux_contribution_i_momentum_y.x()) +
                            factor3 * (flux_contribution_nb_momentum_y.y() +
                                       flux_contribution_i_momentum_y.y()) +
                            factor4 * (flux_contribution_nb_momentum_y.z() +
                                       flux_contribution_i_momentum_y.z());
        fim_if_nb_gt0.z() = factor * (momentum_i.z() - momentum_nb.z()) +
                            factor2 * (flux_contribution_nb_momentum_z.x() +
                                       flux_contribution_i_momentum_z.x()) +
                            factor3 * (flux_contribution_nb_momentum_z.y() +
                                       flux_contribution_i_momentum_z.y()) +
                            factor4 * (flux_contribution_nb_momentum_z.z() +
                                       flux_contribution_i_momentum_z.z());

        fim_if_nb_en1.x() = normal.x() * pressure_i;
        fim_if_nb_en1.y() = normal.y() * pressure_i;
        fim_if_nb_en1.z() = normal.z() * pressure_i;

        fid_if_nb_en2 = factor2 * (ff_variable_idx1 + momentum_i.x()) +
                        factor3 * (ff_variable_idx2 + momentum_i.y()) +
                        factor4 * (ff_variable_idx3 + momentum_i.z());
        fide_if_nb_en2 = factor2 * (ff_flux_contribution_density_energy.x() +
                                    flux_contribution_i_density_energy.x()) +
                         factor3 * (ff_flux_contribution_density_energy.y() +
                                    flux_contribution_i_density_energy.y()) +
                         factor4 * (ff_flux_contribution_density_energy.z() +
                                    flux_contribution_i_density_energy.z());
        fim_if_nb_en2.x() = factor2 * (ff_flux_contribution_momentum_x.x() +
                                       flux_contribution_i_momentum_x.x()) +
                            factor3 * (ff_flux_contribution_momentum_x.y() +
                                       flux_contribution_i_momentum_x.y()) +
                            factor4 * (ff_flux_contribution_momentum_x.z() +
                                       flux_contribution_i_momentum_x.z());
        fim_if_nb_en2.y() = factor2 * (ff_flux_contribution_momentum_y.x() +
                                       flux_contribution_i_momentum_y.x()) +
                            factor3 * (ff_flux_contribution_momentum_y.y() +
                                       flux_contribution_i_momentum_y.y()) +
                            factor4 * (ff_flux_contribution_momentum_y.z() +
                                       flux_contribution_i_momentum_y.z());
        fim_if_nb_en2.z() = factor2 * (ff_flux_contribution_momentum_z.x() +
                                       flux_contribution_i_momentum_z.x()) +
                            factor3 * (ff_flux_contribution_momentum_z.y() +
                                       flux_contribution_i_momentum_z.y()) +
                            factor4 * (ff_flux_contribution_momentum_z.z() +
                                       flux_contribution_i_momentum_z.z());

        flux_i_density_arr[j] =
            (nb >= 0) ? fid_if_nb_gt0 : ((nb == -2) ? fid_if_nb_en2 : 0.0f);
        flux_i_momentum_x_arr[j] = (nb >= 0) ? fim_if_nb_gt0.x()
                                             : ((nb == -1)   ? fim_if_nb_en1.x()
                                                : (nb == -2) ? fim_if_nb_en2.x()
                                                             : 0.0f);
        flux_i_momentum_y_arr[j] = (nb >= 0) ? fim_if_nb_gt0.y()
                                             : ((nb == -1)   ? fim_if_nb_en1.y()
                                                : (nb == -2) ? fim_if_nb_en2.y()
                                                             : 0.0f);
        flux_i_momentum_z_arr[j] = (nb >= 0) ? fim_if_nb_gt0.z()
                                             : ((nb == -1)   ? fim_if_nb_en1.z()
                                                : (nb == -2) ? fim_if_nb_en2.z()
                                                             : 0.0f);
        flux_i_density_energy_arr[j] = (nb >= 0)    ? fide_if_nb_gt0
                                       : (nb == -2) ? fide_if_nb_en2
                                                    : 0.0f;
      } // FOR(j)

      var_t flux;
      flux.density() = 0.0f;
      flux.momentum_x() = 0.0f;
      flux.momentum_y() = 0.0f;
      flux.momentum_z() = 0.0f;
      flux.density_energy() = 0.0f;

#pragma unroll
      for (int8_t j = 0; j < NNB; j++) {
        flux.density() += flux_i_density_arr[j];
        flux.momentum_x() += flux_i_momentum_x_arr[j];
        flux.momentum_y() += flux_i_momentum_y_arr[j];
        flux.momentum_z() += flux_i_momentum_z_arr[j];
        flux.density_energy() += flux_i_density_energy_arr[j];
      }

      flux_writeback_pipe::PipeAt<ID>::write(flux);
    } // FOR(i)
  });
  queue.wait();

  float elapsed = .0f;
  for (auto &e : compute_events) {
    float t =
        e.get_profiling_info<sycl::info::event_profiling::command_end>() -
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    if (t > elapsed)
      elapsed = t;
  }
  kernelTime += elapsed * 1.e-9;
}

void time_step(int j, int nelr, var_t *old_variables, var_t *variables,
               float *step_factors, var_t *fluxes, sycl::queue &queue) {
  sycl::range<1> Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
  sycl::event k_event = queue.submit([&](sycl::handler &cgh) {
    const float one_div_rk = 1.0f / float(RK + 1 - j);

    cgh.parallel_for<class time_step>(
        sycl::nd_range<1>(Dg * Db, Db),
        [=](sycl::nd_item<1> item_ct1)
            [[intel::kernel_args_restrict, intel::no_global_work_offset(1),
              intel::num_simd_work_items(4),
              sycl::reqd_work_group_size(1, 1, BLOCK_SIZE),
              intel::max_work_group_size(1, 1, BLOCK_SIZE)]] {
              const int i = (item_ct1.get_global_id(0));
              if (i >= nelr)
                return;

              const float factor = step_factors[i] * one_div_rk;

              var_t flux = fluxes[i];
              var_t old_var = old_variables[i];

              var_t var;
              var.density() = old_var.density() + factor * flux.density();
              var.momentum_x() =
                  old_var.momentum_x() + factor * flux.momentum_x();
              var.momentum_y() =
                  old_var.momentum_y() + factor * flux.momentum_y();
              var.momentum_z() =
                  old_var.momentum_z() + factor * flux.momentum_z();
              var.density_energy() =
                  old_var.density_energy() + factor * flux.density_energy();
              variables[i] = var;
            });
  });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

void addBenchmarkSpecOptions(OptionParser &op) {}

void cfd(ResultDatabase &resultDB, OptionParser &op, size_t device_idx);

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op,
                  size_t device_idx) {
  printf("Running CFDSolver\n");
  bool quiet = op.getOptionBool("quiet");
  if (!quiet) {
    printf("WG size of kernel:initialize = %d, WG size of "
           "kernel:compute_step_factor = %d, WG size of kernel:compute_flux = "
           "%d, WG size of kernel:time_step = %d\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  }

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    kernelTime = 0.0f;
    transferTime = 0.0f;
    if (!quiet)
      printf("Pass %d:\n", i);

    cfd(resultDB, op, device_idx);
    if (!quiet)
      printf("Done.\n");
  }
}

void cfd(ResultDatabase &resultDB, OptionParser &op, size_t device_idx) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  sycl::queue queue(devices[device_idx],
                    sycl::property::queue::enable_profiling{});

  // set far field conditions and load them into constant memory on the gpu
  {
    const float angle_of_attack =
        float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

    h_ff_variable[VAR_DENSITY] = float(1.4);

    float ff_pressure = float(1.0f);
    float ff_speed_of_sound =
        sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
    float ff_speed = float(ff_mach) * ff_speed_of_sound;

    sycl::float3 ff_velocity;
    ff_velocity.x() = ff_speed * float(cos((float)angle_of_attack));
    ff_velocity.y() = ff_speed * float(sin((float)angle_of_attack));
    ff_velocity.z() = 0.0f;

    h_ff_variable[VAR_MOMENTUM + 0] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.x();
    h_ff_variable[VAR_MOMENTUM + 1] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.y();
    h_ff_variable[VAR_MOMENTUM + 2] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.z();

    h_ff_variable[VAR_DENSITY_ENERGY] =
        h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) +
        (ff_pressure / float(GAMMA - 1.0f));

    sycl::float3 h_ff_momentum;
    h_ff_momentum.x() = *(h_ff_variable.data() + VAR_MOMENTUM + 0);
    h_ff_momentum.y() = *(h_ff_variable.data() + VAR_MOMENTUM + 1);
    h_ff_momentum.z() = *(h_ff_variable.data() + VAR_MOMENTUM + 2);
    compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum,
                              h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
                              ff_velocity, h_ff_flux_contribution_momentum_x,
                              h_ff_flux_contribution_momentum_y,
                              h_ff_flux_contribution_momentum_z,
                              h_ff_flux_contribution_density_energy);
  }

  int nel;
  int nelr;

  // read in domain geometry
  float *areas;
  int *elements_surrounding_elements;
  sycl::float3 *normals;
  {
    string inputFile = op.getOptionString("inputFile");
    std::ifstream file(inputFile.c_str());

    if (inputFile != "") {
      file >> nel;
    } else {
      int problemSizes[4] = {97000, 200000, 40000000, 60000000};
      nel = problemSizes[op.getOptionInt("size") - 1];
    }
    nelr = BLOCK_SIZE * ((nel / BLOCK_SIZE) + std::min(1, nel % BLOCK_SIZE));

    float *h_areas = new float[nelr];
    int *h_elements_surrounding_elements = new int[nelr * NNB];
    sycl::float3 *h_normals = new sycl::float3[nelr * NNB];

    srand(SEED);

    // read in data
    for (int i = 0; i < nel; i++) {
      if (inputFile != "")
        file >> h_areas[i];
      else
        h_areas[i] = 1.0 * rand() / RAND_MAX;

      for (int j = 0; j < NNB; j++) // NNB is always 4
      {
        if (inputFile != "") {
          file >> h_elements_surrounding_elements[i * NNB + j];
        } else {
          int val = i + (rand() % 20) - 10;
          h_elements_surrounding_elements[i * NNB + j] = val;
        }
        if (h_elements_surrounding_elements[i * NNB + j] < 0)
          h_elements_surrounding_elements[i * NNB + j] = -1;
        h_elements_surrounding_elements[i * NNB + j]--; // it's coming in with
                                                        // Fortran numbering

        for (int k = 0; k < NDIM; k++) // NDIM is always 3
        {
          if (inputFile != "") {
            file >> h_normals[i * NNB + j][k];
          } else
            h_normals[i * NNB + j][k] = 1.0 * rand() / RAND_MAX - 0.5;

          h_normals[i * NNB + j][k] = -h_normals[i * NNB + j][k];
        }
      }
    }

    // fill in remaining data
    int last = nel - 1;
    for (int i = nel; i < nelr; i++) {
      h_areas[i] = h_areas[last];
      for (int j = 0; j < NNB; j++) {
        // duplicate the last element
        h_elements_surrounding_elements[i * NNB + j] =
            h_elements_surrounding_elements[last * NNB + j];
      }
    }

    areas = alloc<float>(nelr, queue);
    upload<float>(areas, h_areas, nelr, queue);

    elements_surrounding_elements = alloc<int>(nelr * NNB, queue);
    upload<int>(elements_surrounding_elements, h_elements_surrounding_elements,
                nelr * NNB, queue);

    normals = alloc<sycl::float3>(nelr * NNB, queue);
    upload<sycl::float3>(normals, h_normals, nelr * NNB, queue);

    delete[] h_areas;
    delete[] h_elements_surrounding_elements;
    delete[] h_normals;
  }

  // Create arrays and set initial conditions
  var_t *variables = alloc<var_t>(nelr * NVAR, queue);
  initialize_variables(nelr, variables, queue);

  var_t *old_variables = alloc<var_t>(nelr * NVAR, queue);
  initialize_variables(nelr, old_variables, queue);

  var_t *fluxes = alloc<var_t>(nelr * NVAR, queue);
  initialize_variables(nelr, fluxes, queue);

  float *step_factors = alloc<float>(nelr, queue);
  queue.memset((void *)step_factors, 0, sizeof(float) * nelr).wait();

  // make sure CUDA isn't still doing something before we start timing
  queue.wait_and_throw();

  FPGA_PWR_MEAS_START

  // Begin iterations
  for (int i = 0; i < iterations; i++) {
    // Time will need to be recomputed, more aggressive optimization TODO
    auto cpy = copy<var_t>(old_variables, variables, nelr * NVAR, queue);

    // for the first iteration we compute the time step
    compute_step_factor(nelr, variables, areas, step_factors, queue);
    cpy.wait();
    auto end =
        cpy.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start =
        cpy.get_profiling_info<sycl::info::event_profiling::command_start>();
    transferTime += (end - start) / 1.0e9;

    for (int j = 0; j < RK; j++) {
      compute_flux(nelr, elements_surrounding_elements, normals, variables,
                   fluxes, queue);
      time_step(j, nelr, old_variables, variables, step_factors, fluxes, queue);
    }
  }

  queue.wait_and_throw();
  FPGA_PWR_MEAS_END
  if (op.getOptionBool("verbose")) {
    if (iterations > 3)
      std::cout << "Dumping after more than 3 iterations. Results will "
                   "probably nan. This is however, the same behaviour "
                   "this benchmarks shows with its original CUDA code."
                << std::endl;
    dump(variables, nel, nelr, queue);
  }

  dealloc<float>(areas, queue);
  dealloc<int>(elements_surrounding_elements, queue);
  dealloc<sycl::float3>(normals, queue);

  dealloc<var_t>(variables, queue);
  dealloc<var_t>(old_variables, queue);
  dealloc<var_t>(fluxes, queue);
  dealloc<float>(step_factors, queue);

  char atts[1024];
  sprintf(atts, "numelements:%d", nel);
  resultDB.AddResult("cfd_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("cfd_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("cfd_parity", atts, "N", transferTime / kernelTime);
  resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}
