// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\cfd\euler3d_double.cu
//
// summary:	Sort class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////


// #include <cutil.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <fstream>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include <chrono>

#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

#if (SYCL_LANGUAGE_VERSION < 202000)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A double 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct double3
{
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Gets the z coordinate. </summary>
	///
	/// <value>	The z coordinate. </value>
	////////////////////////////////////////////////////////////////////////////////////////////////////

	double x, y, z;
};
#endif

/*
/// <summary>	. </summary>
 * Options 
 * 
 */ 
#define GAMMA 1.4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines iterations. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define iterations 2000
#ifndef block_length

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block length. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define block_length 128
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines ndim. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NDIM 3

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines nnb. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NNB 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines rk. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RK 3	// 3rd order RK

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines ff mach. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define ff_mach 1.2

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Degrees angle of attack. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define deg_angle_of_attack 0.0

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable density. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_DENSITY 0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable momentum. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_MOMENTUM  1

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable density energy. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines nvar. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NVAR (VAR_DENSITY_ENERGY+1)

/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
/// <summary>	The elapsed. </summary>
float elapsed;

/*

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 /// <summary>	Allocs. </summary>
 ///
 /// <remarks>	Ed, 5/20/2020. </remarks>
 ///
 /// <typeparam name="T">	Generic type parameter. </typeparam>
 /// <param name="N">	An int to process. </param>
 ///
 /// <returns>	Null if it fails, else a pointer to a T. </returns>
 ////////////////////////////////////////////////////////////////////////////////////////////////////

 * Generic functions
 */
template <typename T> T *alloc(int N) try {
        T* t;
        /*
        DPCT1064:313: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        CUDA_SAFE_CALL(
            DPCT_CHECK_ERROR(t = (T *)sycl::malloc_device(
                                 sizeof(T) * N, dpct::get_default_queue())));
        return t;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Deallocs the given array. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="array">	[in,out] If non-null, the array. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void dealloc(T *array) try {
        CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
            sycl::free((void *)array, dpct::get_default_queue())));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies this.  </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void copy(T *dst, T *src, int N) try {
    /*
    DPCT1012:296: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
            dpct::get_default_queue()
                .memcpy((void *)dst, (void *)src, N * sizeof(T))
                .wait()));
    /*
    DPCT1012:297: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Uploads. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void upload(T *dst, T *src, int N) try {
    /*
    DPCT1012:298: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
            dpct::get_default_queue()
                .memcpy((void *)dst, (void *)src, N * sizeof(T))
                .wait()));
    /*
    DPCT1012:299: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Downloads this.  </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void download(T *dst, T *src, int N) try {
    /*
    DPCT1012:300: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
            dpct::get_default_queue()
                .memcpy((void *)dst, (void *)src, N * sizeof(T))
                .wait()));
    /*
    DPCT1012:301: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsed * 1.e-3;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Dumps. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="variables">	[in,out] If non-null, the variables. </param>
/// <param name="nel">			The nel. </param>
/// <param name="nelr">			The nelr. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void dump(double* variables, int nel, int nelr)
{
	double* h_variables = new double[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

/*

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 /// <summary>	Gets the ff variable[ nvar]. </summary>
 ///
 /// <value>	The ff variable[ nvar]. </value>
 ////////////////////////////////////////////////////////////////////////////////////////////////////

 * Element-based Cell-centered FVM solver functions
 */
static dpct::constant_memory<double, 1> ff_variable(NVAR);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum x[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum x[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<sycl::double3, 1>
    ff_flux_contribution_momentum_x(1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum y[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum y[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<sycl::double3, 1>
    ff_flux_contribution_momentum_y(1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum z[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum z[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<sycl::double3, 1>
    ff_flux_contribution_momentum_z(1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution density energy[ 1]. </summary>
///
/// <value>	The ff flux contribution density energy[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

static dpct::constant_memory<sycl::double3, 1>
    ff_flux_contribution_density_energy(1);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda initialize variables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">			The nelr. </param>
/// <param name="variables">	[in,out] If non-null, the variables. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cuda_initialize_variables(int nelr, double* variables,
                               const sycl::nd_item<3> &item_ct1,
                               double *ff_variable)
{
        const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));
        for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes the variables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">			The nelr. </param>
/// <param name="variables">	[in,out] If non-null, the variables. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_variables(int nelr, double* variables)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
    /*
    DPCT1012:302: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      {
            dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                         {sycl::aspect::fp64});
            *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  ff_variable.init();

                  auto ff_variable_ptr_ct1 = ff_variable.get_ptr();

                  cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         cuda_initialize_variables(
                                             nelr, variables, item_ct1,
                                             ff_variable_ptr_ct1);
                                   });
            });
      }
    /*
    DPCT1012:303: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the flux contribution. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density">				[in,out] The density. </param>
/// <param name="momentum">				[in,out] The momentum. </param>
/// <param name="density_energy">   	[in,out] The density energy. </param>
/// <param name="pressure">				[in,out] The pressure. </param>
/// <param name="velocity">				[in,out] The velocity. </param>
/// <param name="fc_momentum_x">		[in,out] The fc momentum x coordinate. </param>
/// <param name="fc_momentum_y">		[in,out] The fc momentum y coordinate. </param>
/// <param name="fc_momentum_z">		[in,out] The fc momentum z coordinate. </param>
/// <param name="fc_density_energy">	[in,out] The fc density energy. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void compute_flux_contribution(double &density, sycl::double3 &momentum,
                                      double &density_energy, double &pressure,
                                      sycl::double3 &velocity,
                                      sycl::double3 &fc_momentum_x,
                                      sycl::double3 &fc_momentum_y,
                                      sycl::double3 &fc_momentum_z,
                                      sycl::double3 &fc_density_energy)
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

        double de_p = density_energy+pressure;
        fc_density_energy.x() = velocity.x() * de_p;
        fc_density_energy.y() = velocity.y() * de_p;
        fc_density_energy.z() = velocity.z() * de_p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the velocity. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density"> 	[in,out] The density. </param>
/// <param name="momentum">	[in,out] The momentum. </param>
/// <param name="velocity">	[in,out] The velocity. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void compute_velocity(double &density, sycl::double3 &momentum,
                             sycl::double3 &velocity)
{
        velocity.x() = momentum.x() / density;
        velocity.y() = momentum.y() / density;
        velocity.z() = momentum.z() / density;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the speed sqd. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="velocity">	[in,out] The velocity. </param>
///
/// <returns>	The calculated speed sqd. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline double compute_speed_sqd(sycl::double3 &velocity)
{
        return velocity.x() * velocity.x() + velocity.y() * velocity.y() +
               velocity.z() * velocity.z();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the pressure. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density">		 	[in,out] The density. </param>
/// <param name="density_energy">	[in,out] The density energy. </param>
/// <param name="speed_sqd">	 	[in,out] The speed sqd. </param>
///
/// <returns>	The calculated pressure. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline double compute_pressure(double& density, double& density_energy, double& speed_sqd)
{
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the speed of sound. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density"> 	[in,out] The density. </param>
/// <param name="pressure">	[in,out] The pressure. </param>
///
/// <returns>	The calculated speed of sound. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

inline double compute_speed_of_sound(double& density, double& pressure)
{
        return sycl::sqrt(double(GAMMA) * pressure / density);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda compute step factor. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">		   	The nelr. </param>
/// <param name="variables">   	[in,out] If non-null, the variables. </param>
/// <param name="areas">	   	[in,out] If non-null, the areas. </param>
/// <param name="step_factors">	[in,out] If non-null, the step factors. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cuda_compute_step_factor(int nelr, double* variables, double* areas, double* step_factors,
                              const sycl::nd_item<3> &item_ct1)
{
        const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

        double density = variables[i + VAR_DENSITY*nelr];
        sycl::double3 momentum;
        momentum.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
        momentum.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
        momentum.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

        double density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];

        sycl::double3 velocity; compute_velocity(density, momentum, velocity);
        double speed_sqd      = compute_speed_sqd(velocity);
	double pressure       = compute_pressure(density, density_energy, speed_sqd);
	double speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = double(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
        step_factors[i] =
            double(0.5) /
            (sycl::sqrt(areas[i]) * (sycl::sqrt(speed_sqd) + speed_of_sound));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the step factor. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">		   	The nelr. </param>
/// <param name="variables">   	[in,out] If non-null, the variables. </param>
/// <param name="areas">	   	[in,out] If non-null, the areas. </param>
/// <param name="step_factors">	[in,out] If non-null, the step factors. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
    /*
    DPCT1012:304: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      {
            dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                         {sycl::aspect::fp64});
            *stop = dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item_ct1) {
                      cuda_compute_step_factor(nelr, variables, areas,
                                               step_factors, item_ct1);
                });
      }
    /*
    DPCT1012:305: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

/*

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda compute flux. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">								The nelr. </param>
/// <param name="elements_surrounding_elements">	[in,out] If non-null, the elements
/// 												surrounding elements. </param>
/// <param name="normals">							[in,out] If non-null, the normals. </param>
/// <param name="variables">						[in,out] If non-null, the variables. </param>
/// <param name="fluxes">							[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

 *
 *
*/
/*
DPCT1110:19: The total declared local variable size in device function
cuda_compute_flux exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void cuda_compute_flux(int nelr, int *elements_surrounding_elements,
                       double *normals, double *variables, double *fluxes,
                       const sycl::nd_item<3> &item_ct1, double *ff_variable,
                       sycl::double3 *ff_flux_contribution_momentum_x,
                       sycl::double3 *ff_flux_contribution_momentum_y,
                       sycl::double3 *ff_flux_contribution_momentum_z,
                       sycl::double3 *ff_flux_contribution_density_energy)
{
	const double smoothing_coefficient = double(0.2f);
        const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

        int j, nb;
        sycl::double3 normal; double normal_len;
        double factor;
	
	double density_i = variables[i + VAR_DENSITY*nelr];
        sycl::double3 momentum_i;
        momentum_i.x() = variables[i + (VAR_MOMENTUM + 0) * nelr];
        momentum_i.y() = variables[i + (VAR_MOMENTUM + 1) * nelr];
        momentum_i.z() = variables[i + (VAR_MOMENTUM + 2) * nelr];

        double density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

        sycl::double3 velocity_i;
            compute_velocity(density_i, momentum_i, velocity_i);
        double speed_sqd_i                          = compute_speed_sqd(velocity_i);
        double speed_i = sycl::sqrt(speed_sqd_i);
        double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
        sycl::double3 flux_contribution_i_momentum_x,
            flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
        sycl::double3 flux_contribution_i_density_energy;
        compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	double flux_i_density = double(0.0);
        sycl::double3 flux_i_momentum;
        flux_i_momentum.x() = double(0.0);
        flux_i_momentum.y() = double(0.0);
        flux_i_momentum.z() = double(0.0);
        double flux_i_density_energy = double(0.0);

        sycl::double3 velocity_nb;
        double density_nb, density_energy_nb;
        sycl::double3 momentum_nb;
        sycl::double3 flux_contribution_nb_momentum_x,
            flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
        sycl::double3 flux_contribution_nb_density_energy;
        double speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
                normal.x() = normals[i + (j + 0 * NNB) * nelr];
                normal.y() = normals[i + (j + 1 * NNB) * nelr];
                normal.z() = normals[i + (j + 2 * NNB) * nelr];
                normal_len = sycl::sqrt(normal.x() * normal.x() +
                                        normal.y() * normal.y() +
                                        normal.z() * normal.z());

                if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
                        momentum_nb.x() = variables[nb + (VAR_MOMENTUM + 0) * nelr];
                        momentum_nb.y() = variables[nb + (VAR_MOMENTUM + 1) * nelr];
                        momentum_nb.z() = variables[nb + (VAR_MOMENTUM + 2) * nelr];
                        density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
												compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
			                                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
			
			// artificial viscosity
                        factor = -normal_len * smoothing_coefficient *
                                 double(0.5) *
                                 (speed_i + sycl::sqrt(speed_sqd_nb) +
                                  speed_of_sound_i + speed_of_sound_nb);
                        flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
                        flux_i_momentum.x() += factor * (momentum_i.x() - momentum_nb.x());
                        flux_i_momentum.y() += factor * (momentum_i.y() - momentum_nb.y());
                        flux_i_momentum.z() += factor * (momentum_i.z() - momentum_nb.z());

                        // accumulate cell-centered fluxes
                        factor = double(0.5) * normal.x();
                        flux_i_density += factor * (momentum_nb.x() + momentum_i.x());
                        flux_i_density_energy +=
                            factor * (flux_contribution_nb_density_energy.x() +
                                      flux_contribution_i_density_energy.x());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.x() +
                                      flux_contribution_i_momentum_x.x());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.x() +
                                      flux_contribution_i_momentum_y.x());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.x() +
                                      flux_contribution_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density += factor * (momentum_nb.y() + momentum_i.y());
                        flux_i_density_energy +=
                            factor * (flux_contribution_nb_density_energy.y() +
                                      flux_contribution_i_density_energy.y());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.y() +
                                      flux_contribution_i_momentum_x.y());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.y() +
                                      flux_contribution_i_momentum_y.y());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.y() +
                                      flux_contribution_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density += factor * (momentum_nb.z() + momentum_i.z());
                        flux_i_density_energy +=
                            factor * (flux_contribution_nb_density_energy.z() +
                                      flux_contribution_i_density_energy.z());
                        flux_i_momentum.x() +=
                            factor * (flux_contribution_nb_momentum_x.z() +
                                      flux_contribution_i_momentum_x.z());
                        flux_i_momentum.y() +=
                            factor * (flux_contribution_nb_momentum_y.z() +
                                      flux_contribution_i_momentum_y.z());
                        flux_i_momentum.z() +=
                            factor * (flux_contribution_nb_momentum_z.z() +
                                      flux_contribution_i_momentum_z.z());
                }
		else if(nb == -1)	// a wing boundary
		{
                        flux_i_momentum.x() += normal.x() * pressure_i;
                        flux_i_momentum.y() += normal.y() * pressure_i;
                        flux_i_momentum.z() += normal.z() * pressure_i;
                }
		else if(nb == -2) // a far field boundary
		{
                        factor = double(0.5) * normal.x();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x());
                        flux_i_density_energy +=
                            factor *
                            (ff_flux_contribution_density_energy[0].x() +
                             flux_contribution_i_density_energy.x());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].x() +
                                      flux_contribution_i_momentum_x.x());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].x() +
                                      flux_contribution_i_momentum_y.x());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].x() +
                                      flux_contribution_i_momentum_z.x());

                        factor = double(0.5) * normal.y();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y());
                        flux_i_density_energy +=
                            factor *
                            (ff_flux_contribution_density_energy[0].y() +
                             flux_contribution_i_density_energy.y());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].y() +
                                      flux_contribution_i_momentum_x.y());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].y() +
                                      flux_contribution_i_momentum_y.y());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].y() +
                                      flux_contribution_i_momentum_z.y());

                        factor = double(0.5) * normal.z();
                        flux_i_density +=
                            factor *
                            (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z());
                        flux_i_density_energy +=
                            factor *
                            (ff_flux_contribution_density_energy[0].z() +
                             flux_contribution_i_density_energy.z());
                        flux_i_momentum.x() +=
                            factor * (ff_flux_contribution_momentum_x[0].z() +
                                      flux_contribution_i_momentum_x.z());
                        flux_i_momentum.y() +=
                            factor * (ff_flux_contribution_momentum_y[0].z() +
                                      flux_contribution_i_momentum_y.z());
                        flux_i_momentum.z() +=
                            factor * (ff_flux_contribution_momentum_z[0].z() +
                                      flux_contribution_i_momentum_z.z());
                }
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
        fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x();
        fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y();
        fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z();
        fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the flux. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">								The nelr. </param>
/// <param name="elements_surrounding_elements">	[in,out] If non-null, the elements
/// 												surrounding elements. </param>
/// <param name="normals">							[in,out] If non-null, the normals. </param>
/// <param name="variables">						[in,out] If non-null, the variables. </param>
/// <param name="fluxes">							[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
    /*
    DPCT1012:306: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      {
            dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                         {sycl::aspect::fp64});
            *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  ff_variable.init();
                  ff_flux_contribution_momentum_x.init();
                  ff_flux_contribution_momentum_y.init();
                  ff_flux_contribution_momentum_z.init();
                  ff_flux_contribution_density_energy.init();

                  auto ff_variable_ptr_ct1 = ff_variable.get_ptr();
                  auto ff_flux_contribution_momentum_x_ptr_ct1 =
                      ff_flux_contribution_momentum_x.get_ptr();
                  auto ff_flux_contribution_momentum_y_ptr_ct1 =
                      ff_flux_contribution_momentum_y.get_ptr();
                  auto ff_flux_contribution_momentum_z_ptr_ct1 =
                      ff_flux_contribution_momentum_z.get_ptr();
                  auto ff_flux_contribution_density_energy_ptr_ct1 =
                      ff_flux_contribution_density_energy.get_ptr();

                  cgh.parallel_for(
                      sycl::nd_range<3>(Dg * Db, Db),
                      [=](sycl::nd_item<3> item_ct1) {
                            cuda_compute_flux(
                                nelr, elements_surrounding_elements, normals,
                                variables, fluxes, item_ct1,
                                ff_variable_ptr_ct1,
                                ff_flux_contribution_momentum_x_ptr_ct1,
                                ff_flux_contribution_momentum_y_ptr_ct1,
                                ff_flux_contribution_momentum_z_ptr_ct1,
                                ff_flux_contribution_density_energy_ptr_ct1);
                      });
            });
      }
    /*
    DPCT1012:307: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda time step. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="j">				An int to process. </param>
/// <param name="nelr">				The nelr. </param>
/// <param name="old_variables">	[in,out] If non-null, the old variables. </param>
/// <param name="variables">		[in,out] If non-null, the variables. </param>
/// <param name="step_factors"> 	[in,out] If non-null, the step factors. </param>
/// <param name="fluxes">			[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cuda_time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes,
                    const sycl::nd_item<3> &item_ct1)
{
        const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

        double factor = step_factors[i]/double(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Time step. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="j">				An int to process. </param>
/// <param name="nelr">				The nelr. </param>
/// <param name="old_variables">	[in,out] If non-null, the old variables. </param>
/// <param name="variables">		[in,out] If non-null, the variables. </param>
/// <param name="step_factors"> 	[in,out] If non-null, the step factors. </param>
/// <param name="fluxes">			[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
        sycl::range<3> Dg(1, 1, nelr / block_length), Db(1, 1, block_length);
    /*
    DPCT1012:308: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      {
            dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                         {sycl::aspect::fp64});
            *stop = dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item_ct1) {
                      cuda_time_step(j, nelr, old_variables, variables,
                                     step_factors, fluxes, item_ct1);
                });
      }
    /*
    DPCT1012:309: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop->wait();
    stop_ct1 = std::chrono::steady_clock::now();
    elapsed =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cfds. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cfd(ResultDatabase &resultDB, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running CFDSolver (double)\n");
    bool quiet = op.getOptionBool("quiet");
    if(!quiet) {
        printf("WG size of %d\n", block_length);
    }

    start = new sycl::event();
    stop = new sycl::event();

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        if(!quiet) {
            printf("Pass %d:\n", i);
        }
        cfd(resultDB, op);
        if(!quiet) {
            printf("Done.\n");
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cfds. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cfd(ResultDatabase &resultDB, OptionParser &op) try {
        // set far field conditions and load them into constant memory on the gpu
	{
		double h_ff_variable[NVAR];
		const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = double(1.4);
		
		double ff_pressure = double(1.0);
		double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		double ff_speed = double(ff_mach)*ff_speed_of_sound;

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

                h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(double(0.5)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0));

                sycl::double3 h_ff_momentum;
                h_ff_momentum.x() = *(h_ff_variable + VAR_MOMENTUM + 0);
                h_ff_momentum.y() = *(h_ff_variable + VAR_MOMENTUM + 1);
                h_ff_momentum.z() = *(h_ff_variable + VAR_MOMENTUM + 2);
                sycl::double3 h_ff_flux_contribution_momentum_x;
                sycl::double3 h_ff_flux_contribution_momentum_y;
                sycl::double3 h_ff_flux_contribution_momentum_z;
                sycl::double3 h_ff_flux_contribution_density_energy;
                compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
        /*
        DPCT1012:310: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        start_ct1 = std::chrono::steady_clock::now();

                CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(ff_variable.get_ptr(), h_ff_variable,
                                NVAR * sizeof(double))
                        .wait()));
                CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(ff_flux_contribution_momentum_x.get_ptr(),
                                &h_ff_flux_contribution_momentum_x,
                                sizeof(sycl::double3))
                        .wait()));
                CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(ff_flux_contribution_momentum_y.get_ptr(),
                                &h_ff_flux_contribution_momentum_y,
                                sizeof(sycl::double3))
                        .wait()));
                CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(ff_flux_contribution_momentum_z.get_ptr(),
                                &h_ff_flux_contribution_momentum_z,
                                sizeof(sycl::double3))
                        .wait()));
                CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                    dpct::get_default_queue()
                        .memcpy(ff_flux_contribution_density_energy.get_ptr(),
                                &h_ff_flux_contribution_density_energy,
                                sizeof(sycl::double3))
                        .wait()));

        /*
        DPCT1012:311: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
        transferTime += elapsed * 1.e-3;
	}
	int nel;
	int nelr;
	
	// read in domain geometry
	double* areas;
	int* elements_surrounding_elements;
	double* normals;
	{
        string inputFile = op.getOptionString("inputFile");
		std::ifstream file(inputFile.c_str());
        
        if(inputFile != "") {
		    file >> nel;
        } else {
            int problemSizes[4] = {97000, 200000, 1000000, 4000000};
            nel = problemSizes[op.getOptionInt("size") - 1];
        }
		nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

		double* h_areas = new double[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		double* h_normals = new double[nelr*NDIM*NNB];

        srand(SEED);
				
		// read in data
		for(int i = 0; i < nel; i++)
		{
            if(inputFile != "") {
		    	file >> h_areas[i];
            } else {
                h_areas[i] = 1.0 * rand() / RAND_MAX;
            }
			for(int j = 0; j < NNB; j++)
			{
                if(inputFile != "") {
				    file >> h_elements_surrounding_elements[i + j*nelr];
                } else {
                    int val = i + (rand() % 20) - 10;
                    h_elements_surrounding_elements[i + j * nelr] = val;
                }
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
				{
                    if(inputFile != "") {
					    file >> h_normals[i + (j + k*NNB)*nelr];
                    } else {
                        h_normals[i + (j + k*NNB)*nelr] = 1.0 * rand() / RAND_MAX - 0.5;
                    }
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		areas = alloc<double>(nelr);
		upload<double>(areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<double>(nelr*NDIM*NNB);
		upload<double>(normals, h_normals, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

	// Create arrays and set initial conditions
	double* variables = alloc<double>(nelr*NVAR);
	initialize_variables(nelr, variables);

	double* old_variables = alloc<double>(nelr*NVAR);   	
	double* fluxes = alloc<double>(nelr*NVAR);
	double* step_factors = alloc<double>(nelr); 

	// make sure all memory is doublely allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
        dpct::get_default_queue()
            .memset((void *)step_factors, 0, sizeof(double) * nelr)
            .wait();
        // make sure CUDA isn't still doing something before we start timing
        dpct::get_current_device().queues_wait_and_throw();

        // these need to be computed the first time in order to compute time step

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy<double>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
        CHECK_CUDA_ERROR();
		
		for(int j = 0; j < RK; j++)
		  {
		    compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
            CHECK_CUDA_ERROR();
		    time_step(j, nelr, old_variables, variables, step_factors, fluxes);
            CHECK_CUDA_ERROR();
		  }
	}

        dpct::get_current_device().queues_wait_and_throw();

    if(op.getOptionBool("verbose")) {
	    dump(variables, nel, nelr);
    }

	
	dealloc<double>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<double>(normals);
	
	dealloc<double>(variables);
	dealloc<double>(old_variables);
	dealloc<double>(fluxes);
	dealloc<double>(step_factors);

    char atts[1024];
    sprintf(atts, "numelements:%d", nel);
    resultDB.AddResult("cfd_double_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("cfd_double_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("cfd_double_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
