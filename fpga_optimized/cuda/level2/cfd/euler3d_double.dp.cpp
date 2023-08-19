// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\cfd\euler3d_double.cu
//
// summary:	Sort class
//
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include "fpga_pwr.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#define SEED 7
#define GAMMA 1.4
#define iterations 2000
#define block_length 128
#define NDIM 3
#define NNB 4
#define RK 3 // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0

#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#ifdef _STRATIX10
#define SIMD 2
#endif
#ifdef _AGILEX
#define SIMD 1
#endif

#ifdef _FPGA
#define ATTRIBUTE                                                              \
  [[intel::kernel_args_restrict,                    \          
      intel::no_global_work_offset(1),                \           
      intel::num_simd_work_items(SIMD),                  \
      sycl::reqd_work_group_size(1, 1, block_length), \
      intel::max_work_group_size(1, 1, block_length)]]
#else
#define ATTRIBUTE
#endif

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

void dump(double *variables, int nel, int nelr, sycl::queue &queue) {
  double *h_variables = new double[nelr * NVAR];
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
    for (int i = 0; i < nel; i++) {
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

std::array<double, NVAR> h_ff_variable;
sycl::double3 h_ff_flux_contribution_momentum_x;
sycl::double3 h_ff_flux_contribution_momentum_y;
sycl::double3 h_ff_flux_contribution_momentum_z;
sycl::double3 h_ff_flux_contribution_density_energy;

void initialize_variables(int nelr, double *variables, sycl::queue &queue) {
  sycl::buffer<double> buff{h_ff_variable.begin(), h_ff_variable.end()};

  sycl::range<1> Dg(nelr / block_length), Db(block_length);
  sycl::event k_event = queue.submit([&](sycl::handler &cgh) {
    const double a0 = h_ff_variable[0];
    const double a1 = h_ff_variable[1];
    const double a2 = h_ff_variable[2];
    const double a3 = h_ff_variable[3];
    const double a4 = h_ff_variable[4];

    cgh.parallel_for<class initialize_variables>(
        sycl::nd_range<1>(Dg * Db, Db),
        [=](sycl::nd_item<1> item_ct1) ATTRIBUTE {
          const int i = item_ct1.get_global_id(0);
          variables[i] = a0;
          variables[i + nelr] = a1;
          variables[i + 2 * nelr] = a2;
          variables[i + 3 * nelr] = a3;
          variables[i + 4 * nelr] = a4;
        });
  });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

SYCL_EXTERNAL inline void compute_flux_contribution(
    double &density, sycl::double3 &momentum, double &density_energy,
    double &pressure, sycl::double3 &velocity, sycl::double3 &fc_momentum_x,
    sycl::double3 &fc_momentum_y, sycl::double3 &fc_momentum_z,
    sycl::double3 &fc_density_energy) {
  fc_momentum_x.x() = velocity.x() * momentum.x() + pressure;
  fc_momentum_x.y() = velocity.x() * momentum.y();
  fc_momentum_x.z() = velocity.x() * momentum.z();

  fc_momentum_y.x() = fc_momentum_x.y();
  fc_momentum_y.y() = velocity.y() * momentum.y() + pressure;
  fc_momentum_y.z() = velocity.y() * momentum.z();

  fc_momentum_z.x() = fc_momentum_x.z();
  fc_momentum_z.y() = fc_momentum_y.z();
  fc_momentum_z.z() = velocity.z() * momentum.z() + pressure;

  double de_p = density_energy + pressure;
  fc_density_energy.x() = velocity.x() * de_p;
  fc_density_energy.y() = velocity.y() * de_p;
  fc_density_energy.z() = velocity.z() * de_p;
}

SYCL_EXTERNAL inline void compute_velocity(double &density,
                                           sycl::double3 &momentum,
                                           sycl::double3 &velocity) {
  velocity.x() = momentum.x() / density;
  velocity.y() = momentum.y() / density;
  velocity.z() = momentum.z() / density;
}

SYCL_EXTERNAL inline double compute_speed_sqd(sycl::double3 &velocity) {
  return velocity.x() * velocity.x() + velocity.y() * velocity.y() +
         velocity.z() * velocity.z();
}

SYCL_EXTERNAL inline double
compute_pressure(double &density, double &density_energy, double &speed_sqd) {
  return (double(GAMMA) - double(1.0)) *
         (density_energy - double(0.5) * density * speed_sqd);
}

SYCL_EXTERNAL inline double compute_speed_of_sound(double &density,
                                                   double &pressure) {
  return sycl::sqrt(double(GAMMA) * pressure / density);
}

SYCL_EXTERNAL void cuda_compute_step_factor(int nelr, double *variables,
                                            double *areas, double *step_factors,
                                            sycl::nd_item<1> item_ct1) {
  const int i = item_ct1.get_global_id(0);

  double density = variables[i + VAR_DENSITY * nelr];
  sycl::double3 momentum;
  momentum.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
  momentum.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
  momentum.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

  double density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

  sycl::double3 velocity;
  compute_velocity(density, momentum, velocity);
  double speed_sqd = compute_speed_sqd(velocity);
  double pressure = compute_pressure(density, density_energy, speed_sqd);
  double speed_of_sound = compute_speed_of_sound(density, pressure);

  // dt = double(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time
  // stepping, this later would need to be divided by the area, so we just do
  // it all at once
  step_factors[i] = double(0.5) / (sycl::sqrt(areas[i]) *
                                   (sycl::sqrt(speed_sqd) + speed_of_sound));
}

void compute_step_factor(int nelr, double *variables, double *areas,
                         double *step_factors, sycl::queue &queue) {
  sycl::range<1> Dg(nelr / block_length), Db(block_length);
  sycl::event k_event = queue.parallel_for<class compute_step_factor>(
      sycl::nd_range<1>(Dg * Db, Db), [=](sycl::nd_item<1> item_ct1) ATTRIBUTE {
        cuda_compute_step_factor(nelr, variables, areas, step_factors,
                                 item_ct1);
      });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

void cuda_compute_flux(int nelr, int *elements_surrounding_elements,
                       double *normals, double *variables, double *fluxes,
                       sycl::nd_item<1> item_ct1, double ff_variable_idx1,
                       double ff_variable_idx2, double ff_variable_idx3,
                       sycl::double3 ff_flux_contribution_momentum_x,
                       sycl::double3 ff_flux_contribution_momentum_y,
                       sycl::double3 ff_flux_contribution_momentum_z,
                       sycl::double3 ff_flux_contribution_density_energy) {
  constexpr double smoothing_coefficient = 0.2;
  const int i = item_ct1.get_global_id(0);

  double density_i = variables[i + VAR_DENSITY * nelr];
  sycl::double3 momentum_i;
  momentum_i.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
  momentum_i.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
  momentum_i.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

  double density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

  sycl::double3 velocity_i;
  compute_velocity(density_i, momentum_i, velocity_i);
  double speed_sqd_i = compute_speed_sqd(velocity_i);
  double speed_i = sycl::sqrt(speed_sqd_i);
  double pressure_i =
      compute_pressure(density_i, density_energy_i, speed_sqd_i);
  double speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
  sycl::double3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
      flux_contribution_i_momentum_z;
  sycl::double3 flux_contribution_i_density_energy;
  compute_flux_contribution(
      density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
      flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
      flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

  double flux_i_density = 0.0;
  sycl::double3 flux_i_momentum;
  flux_i_momentum.x() = 0.0;
  flux_i_momentum.y() = 0.0;
  flux_i_momentum.z() = 0.0;
  double flux_i_density_energy = 0.0;

  sycl::double3 velocity_nb;
  double density_nb, density_energy_nb;
  sycl::double3 momentum_nb;
  sycl::double3 flux_contribution_nb_momentum_x,
      flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
  sycl::double3 flux_contribution_nb_density_energy;
  double speed_sqd_nb, speed_of_sound_nb, pressure_nb;

  for (uint8_t j = 0; j < NNB; j++) {
    double factor;
    int nb = elements_surrounding_elements[i + j * nelr];
    sycl::double3 normal(normals[i + (j + 0 * NNB) * nelr],
                         normal.y() = normals[i + (j + 1 * NNB) * nelr],
                         normal.z() = normals[i + (j + 2 * NNB) * nelr]);
    const double normal_len =
        sycl::sqrt(normal.x() * normal.x() + normal.y() * normal.y() +
                   normal.z() * normal.z());

    if (nb >= 0) // a legitimate neighbor
    {
      density_nb = variables[nb + VAR_DENSITY * nelr];
      momentum_nb.x() = variables[nb + (VAR_MOMENTUM + 0) * nelr];
      momentum_nb.y() = variables[nb + (VAR_MOMENTUM + 1) * nelr];
      momentum_nb.z() = variables[nb + (VAR_MOMENTUM + 2) * nelr];
      density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
      compute_velocity(density_nb, momentum_nb, velocity_nb);
      speed_sqd_nb = compute_speed_sqd(velocity_nb);
      pressure_nb =
          compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
      speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
      compute_flux_contribution(
          density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb,
          flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y,
          flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

      // artificial viscosity
      factor = -normal_len * smoothing_coefficient * double(0.5) *
               (speed_i + sycl::sqrt(speed_sqd_nb) + speed_of_sound_i +
                speed_of_sound_nb);
      flux_i_density += factor * (density_i - density_nb);
      flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
      flux_i_momentum.x() += factor * (momentum_i.x() - momentum_nb.x());
      flux_i_momentum.y() += factor * (momentum_i.y() - momentum_nb.y());
      flux_i_momentum.z() += factor * (momentum_i.z() - momentum_nb.z());

      // accumulate cell-centered fluxes
      factor = double(0.5) * normal.x();
      flux_i_density += factor * (momentum_nb.x() + momentum_i.x());
      flux_i_density_energy +=
          factor * (flux_contribution_nb_density_energy.x() +
                    flux_contribution_i_density_energy.x());
      flux_i_momentum.x() += factor * (flux_contribution_nb_momentum_x.x() +
                                       flux_contribution_i_momentum_x.x());
      flux_i_momentum.y() += factor * (flux_contribution_nb_momentum_y.x() +
                                       flux_contribution_i_momentum_y.x());
      flux_i_momentum.z() += factor * (flux_contribution_nb_momentum_z.x() +
                                       flux_contribution_i_momentum_z.x());

      factor = double(0.5) * normal.y();
      flux_i_density += factor * (momentum_nb.y() + momentum_i.y());
      flux_i_density_energy +=
          factor * (flux_contribution_nb_density_energy.y() +
                    flux_contribution_i_density_energy.y());
      flux_i_momentum.x() += factor * (flux_contribution_nb_momentum_x.y() +
                                       flux_contribution_i_momentum_x.y());
      flux_i_momentum.y() += factor * (flux_contribution_nb_momentum_y.y() +
                                       flux_contribution_i_momentum_y.y());
      flux_i_momentum.z() += factor * (flux_contribution_nb_momentum_z.y() +
                                       flux_contribution_i_momentum_z.y());

      factor = double(0.5) * normal.z();
      flux_i_density += factor * (momentum_nb.z() + momentum_i.z());
      flux_i_density_energy +=
          factor * (flux_contribution_nb_density_energy.z() +
                    flux_contribution_i_density_energy.z());
      flux_i_momentum.x() += factor * (flux_contribution_nb_momentum_x.z() +
                                       flux_contribution_i_momentum_x.z());
      flux_i_momentum.y() += factor * (flux_contribution_nb_momentum_y.z() +
                                       flux_contribution_i_momentum_y.z());
      flux_i_momentum.z() += factor * (flux_contribution_nb_momentum_z.z() +
                                       flux_contribution_i_momentum_z.z());
    } else if (nb == -1) // a wing boundary
    {
      flux_i_momentum.x() += normal.x() * pressure_i;
      flux_i_momentum.y() += normal.y() * pressure_i;
      flux_i_momentum.z() += normal.z() * pressure_i;
    } else if (nb == -2) // a far field boundary
    {
      factor = double(0.5) * normal.x();
      flux_i_density += factor * (ff_variable_idx1 + momentum_i.x());
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy.x() +
                    flux_contribution_i_density_energy.x());
      flux_i_momentum.x() += factor * (ff_flux_contribution_momentum_x.x() +
                                       flux_contribution_i_momentum_x.x());
      flux_i_momentum.y() += factor * (ff_flux_contribution_momentum_y.x() +
                                       flux_contribution_i_momentum_y.x());
      flux_i_momentum.z() += factor * (ff_flux_contribution_momentum_z.x() +
                                       flux_contribution_i_momentum_z.x());

      factor = double(0.5) * normal.y();
      flux_i_density += factor * (ff_variable_idx2 + momentum_i.y());
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy.y() +
                    flux_contribution_i_density_energy.y());
      flux_i_momentum.x() += factor * (ff_flux_contribution_momentum_x.y() +
                                       flux_contribution_i_momentum_x.y());
      flux_i_momentum.y() += factor * (ff_flux_contribution_momentum_y.y() +
                                       flux_contribution_i_momentum_y.y());
      flux_i_momentum.z() += factor * (ff_flux_contribution_momentum_z.y() +
                                       flux_contribution_i_momentum_z.y());

      factor = double(0.5) * normal.z();
      flux_i_density += factor * (ff_variable_idx3 + momentum_i.z());
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy.z() +
                    flux_contribution_i_density_energy.z());
      flux_i_momentum.x() += factor * (ff_flux_contribution_momentum_x.z() +
                                       flux_contribution_i_momentum_x.z());
      flux_i_momentum.y() += factor * (ff_flux_contribution_momentum_y.z() +
                                       flux_contribution_i_momentum_y.z());
      flux_i_momentum.z() += factor * (ff_flux_contribution_momentum_z.z() +
                                       flux_contribution_i_momentum_z.z());
    }
  }

  fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
  fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x();
  fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y();
  fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z();
  fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}

void compute_flux(int nelr, int *elements_surrounding_elements, double *normals,
                  double *variables, double *fluxes, sycl::queue &queue) {
  sycl::range<1> Dg(nelr / block_length), Db(block_length);
  sycl::event k_event = queue.submit([&](sycl::handler &cgh) {
    double ff_variable_idx1 = h_ff_variable[1];
    double ff_variable_idx2 = h_ff_variable[2];
    double ff_variable_idx3 = h_ff_variable[3];
    sycl::double3 ff_flux_contribution_momentum_x =
        h_ff_flux_contribution_momentum_x;
    sycl::double3 ff_flux_contribution_momentum_y =
        h_ff_flux_contribution_momentum_y;
    sycl::double3 ff_flux_contribution_momentum_z =
        h_ff_flux_contribution_momentum_z;
    sycl::double3 ff_flux_contribution_density_energy =
        h_ff_flux_contribution_density_energy;

    cgh.parallel_for<class compute_flux>(
        sycl::nd_range<1>(Dg * Db, Db),
        [=](sycl::nd_item<1> item_ct1)
            [[intel::kernel_args_restrict, intel::no_global_work_offset(1),
              intel::num_simd_work_items(2),
              sycl::reqd_work_group_size(1, 1, block_length),
              intel::max_work_group_size(1, 1, block_length)]] {
              cuda_compute_flux(nelr, elements_surrounding_elements, normals,
                                variables, fluxes, item_ct1, ff_variable_idx1,
                                ff_variable_idx2, ff_variable_idx3,
                                ff_flux_contribution_momentum_x,
                                ff_flux_contribution_momentum_y,
                                ff_flux_contribution_momentum_z,
                                ff_flux_contribution_density_energy);
            });
  });
  k_event.wait();
  elapsed =
      k_event.get_profiling_info<sycl::info::event_profiling::command_end>() -
      k_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  kernelTime += elapsed * 1.e-9;
}

SYCL_EXTERNAL void cuda_time_step(int j, int nelr, double *old_variables,
                                  double *variables, double *step_factors,
                                  double *fluxes, sycl::nd_item<3> item_ct1) {
  const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                 item_ct1.get_local_id(2));

  double factor = step_factors[i] / double(RK + 1 - j);

  variables[i + VAR_DENSITY * nelr] = old_variables[i + VAR_DENSITY * nelr] +
                                      factor * fluxes[i + VAR_DENSITY * nelr];
  variables[i + VAR_DENSITY_ENERGY * nelr] =
      old_variables[i + VAR_DENSITY_ENERGY * nelr] +
      factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
  variables[i + (VAR_MOMENTUM + 0) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 0) * nelr] +
      factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
  variables[i + (VAR_MOMENTUM + 1) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 1) * nelr] +
      factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
  variables[i + (VAR_MOMENTUM + 2) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 2) * nelr] +
      factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
}

void time_step(int j, int nelr, double *old_variables, double *variables,
               double *step_factors, double *fluxes, sycl::queue &queue) {
  sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
  sycl::event k_event = queue.parallel_for<class time_step>(
      sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
        cuda_time_step(j, nelr, old_variables, variables, step_factors, fluxes,
                       item_ct1);
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
  printf("Running CFDSolver (double)\n");
  bool quiet = op.getOptionBool("quiet");
  if (!quiet)
    printf("WG size of %d\n", block_length);

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
    const double angle_of_attack =
        double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);

    h_ff_variable[VAR_DENSITY] = double(1.4);

    double ff_pressure = double(1.0);
    double ff_speed_of_sound =
        sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
    double ff_speed = double(ff_mach) * ff_speed_of_sound;

    sycl::double3 ff_velocity;
    ff_velocity.x() = ff_speed * double(cos((double)angle_of_attack));
    ff_velocity.y() = ff_speed * double(sin((double)angle_of_attack));
    ff_velocity.z() = 0.0;

    h_ff_variable[VAR_MOMENTUM + 0] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.x();
    h_ff_variable[VAR_MOMENTUM + 1] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.y();
    h_ff_variable[VAR_MOMENTUM + 2] =
        h_ff_variable[VAR_DENSITY] * ff_velocity.z();

    h_ff_variable[VAR_DENSITY_ENERGY] =
        h_ff_variable[VAR_DENSITY] * (double(0.5) * (ff_speed * ff_speed)) +
        (ff_pressure / double(GAMMA - 1.0));

    sycl::double3 h_ff_momentum;
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
  double *areas;
  int *elements_surrounding_elements;
  double *normals;
  {
    string inputFile = op.getOptionString("inputFile");
    std::ifstream file(inputFile.c_str());

    if (inputFile != "") {
      file >> nel;
    } else {
      int problemSizes[4] = {97000, 200000, 1000000, 4000000};
      nel = problemSizes[op.getOptionInt("size") - 1];
    }
    nelr =
        block_length * ((nel / block_length) + std::min(1, nel % block_length));

    double *h_areas = new double[nelr];
    int *h_elements_surrounding_elements = new int[nelr * NNB];
    double *h_normals = new double[nelr * NDIM * NNB];

    srand(SEED);

    // read in data
    for (int i = 0; i < nel; i++) {
      if (inputFile != "")
        file >> h_areas[i];
      else
        h_areas[i] = 1.0 * rand() / RAND_MAX;

      for (int j = 0; j < NNB; j++) {
        if (inputFile != "") {
          file >> h_elements_surrounding_elements[i + j * nelr];
        } else {
          int val = i + (rand() % 20) - 10;
          h_elements_surrounding_elements[i + j * nelr] = val;
        }
        if (h_elements_surrounding_elements[i + j * nelr] < 0)
          h_elements_surrounding_elements[i + j * nelr] = -1;
        h_elements_surrounding_elements[i + j * nelr]--; // it's coming in with
                                                         // Fortran numbering

        for (int k = 0; k < NDIM; k++) {
          if (inputFile != "")
            file >> h_normals[i + (j + k * NNB) * nelr];
          else
            h_normals[i + (j + k * NNB) * nelr] = 1.0 * rand() / RAND_MAX - 0.5;

          h_normals[i + (j + k * NNB) * nelr] =
              -h_normals[i + (j + k * NNB) * nelr];
        }
      }
    }

    // fill in remaining data
    int last = nel - 1;
    for (int i = nel; i < nelr; i++) {
      h_areas[i] = h_areas[last];
      for (int j = 0; j < NNB; j++) {
        // duplicate the last element
        h_elements_surrounding_elements[i + j * nelr] =
            h_elements_surrounding_elements[last + j * nelr];
        for (int k = 0; k < NDIM; k++)
          h_normals[last + (j + k * NNB) * nelr] =
              h_normals[last + (j + k * NNB) * nelr];
      }
    }

    areas = alloc<double>(nelr, queue);
    upload<double>(areas, h_areas, nelr, queue);

    elements_surrounding_elements = alloc<int>(nelr * NNB, queue);
    upload<int>(elements_surrounding_elements, h_elements_surrounding_elements,
                nelr * NNB, queue);

    normals = alloc<double>(nelr * NDIM * NNB, queue);
    upload<double>(normals, h_normals, nelr * NDIM * NNB, queue);

    delete[] h_areas;
    delete[] h_elements_surrounding_elements;
    delete[] h_normals;
  }

  // Create arrays and set initial conditions
  double *variables = alloc<double>(nelr * NVAR, queue);
  initialize_variables(nelr, variables, queue);

  double *old_variables = alloc<double>(nelr * NVAR, queue);
  initialize_variables(nelr, old_variables, queue);

  double *fluxes = alloc<double>(nelr * NVAR, queue);
  initialize_variables(nelr, fluxes, queue);

  double *step_factors = alloc<double>(nelr, queue);
  queue.memset((void *)step_factors, 0, sizeof(double) * nelr).wait();

  // make sure CUDA isn't still doing something before we start timing
  queue.wait_and_throw();

  FPGA_PWR_MEAS_START

  // Begin iterations
  for (int i = 0; i < iterations; i++) {
    // Time will need to be recomputed, more aggressive optimization TODO
    auto cpy = copy<double>(old_variables, variables, nelr * NVAR, queue);

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
                   "probably nan. This is, however, the same behaviour "
                   "this benchmarks shows with its original CUDA code."
                << std::endl;
    dump(variables, nel, nelr, queue);
  }

  dealloc<double>(areas, queue);
  dealloc<int>(elements_surrounding_elements, queue);
  dealloc<double>(normals, queue);

  dealloc<double>(variables, queue);
  dealloc<double>(old_variables, queue);
  dealloc<double>(fluxes, queue);
  dealloc<double>(step_factors, queue);

  char atts[1024];
  sprintf(atts, "numelements:%d", nel);
  resultDB.AddResult("cfd_double_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("cfd_double_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("cfd_double_parity", atts, "N", transferTime / kernelTime);
  resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
}
