////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\raytracing.cu
//
// summary:	Raytracing class
// 
// origin: Raytracing(https://github.com/rogerallen/raytracinginoneweekendincuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <float.h>
#include <dpct/rng_utils.hpp>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Matching the C++ code would recurse enough into color() calls that
/// it was blowing up the stack, so we have to turn this into a
/// limited-depth loop instead.  Later code in the book limits to a max
/// depth of 50, so we adapt this a few chapters early on the GPU.. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="r">			   	A ray to process. </param>
/// <param name="world">		   	[in,out] If non-null, the world. </param>
/// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
///
/// <returns>	A vec3. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1110:65: The total declared local variable size in device function color
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
/*
DPCT1032:496: A different random number generator is used. You may need to
adjust the code.
*/
vec3 color(const ray &r, hitable **world,
           dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
               *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        /*
        DPCT1109:66: Virtual functions cannot be called in SYCL device code. You
        need to adjust the code.
        */
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            /*
            DPCT1109:67: Virtual functions cannot be called in SYCL device code.
            You need to adjust the code.
            */
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                                     local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random initialize. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1032:497: A different random number generator is used. You may need to
adjust the code.
*/
void rand_init(
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *rand_state,
    const sycl::nd_item<3> &item_ct1) {
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
        /*
        DPCT1105:498: The mcg59 random number generator is used. The subsequence
        argument "0" is ignored. You need to verify the migration.
        */
        *rand_state = dpct::rng::device::rng_generator<
            oneapi::mkl::rng::device::mcg59<1>>(1984, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	initialize rendering. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="max_x">	 	The maximum x coordinate. </param>
/// <param name="max_y">	 	The maximum y coordinate. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1032:499: A different random number generator is used. You may need to
adjust the code.
*/
void render_init(
    int max_x, int max_y,
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *rand_state,
    const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    /*
    DPCT1105:500: The mcg59 random number generator is used. The subsequence
    argument "pixel_index" is ignored. You need to verify the migration.
    */
    rand_state[pixel_index] =
        dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(
            1984, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Renders this.  </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="fb">		 	[in,out] If non-null, the fb. </param>
/// <param name="max_x">	 	The maximum x coordinate. </param>
/// <param name="max_y">	 	The maximum y coordinate. </param>
/// <param name="ns">		 	The ns. </param>
/// <param name="cam">		 	[in,out] If non-null, the camera. </param>
/// <param name="world">	 	[in,out] If non-null, the world. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1032:501: A different random number generator is used. You may need to
adjust the code.
*/
void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
            hitable **world,
            dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
                *rand_state,
            const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    /*
    DPCT1032:502: A different random number generator is used. You may need to
    adjust the code.
    */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u =
            float(
                i +
                local_rand_state
                    .generate<oneapi::mkl::rng::device::uniform<float>, 1>()) /
            float(max_x);
        float v =
            float(
                j +
                local_rand_state
                    .generate<oneapi::mkl::rng::device::uniform<float>, 1>()) /
            float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sycl::sqrt(col[0]);
    col[1] = sycl::sqrt(col[1]);
    col[2] = sycl::sqrt(col[2]);
    fb[pixel_index] = col;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Random. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RND                                                                    \
    (local_rand_state.generate<oneapi::mkl::rng::device::uniform<float>, 1>())

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a world. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_list">	 	[in,out] If non-null, the list. </param>
/// <param name="d_world">   	[in,out] If non-null, the world. </param>
/// <param name="d_camera">  	[in,out] If non-null, the camera. </param>
/// <param name="nx">		 	The nx. </param>
/// <param name="ny">		 	The ny. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1032:503: A different random number generator is used. You may need to
adjust the code.
*/
void create_world(
    hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny,
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *rand_state,
    const sycl::nd_item<3> &item_ct1) {
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
        /*
        DPCT1032:504: A different random number generator is used. You may need
        to adjust the code.
        */
        dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
            local_rand_state = *rand_state;
        /*
        DPCT1109:68: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        d_list[0] =
            new sphere(vec3(0, -1000.0, -1), 1000,
                       /*
                       DPCT1109:69: Memory storage allocation cannot be called
                       in SYCL device code. You need to adjust the code.
                       */
                       new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    /*
                    DPCT1109:70: Memory storage allocation cannot be called in
                    SYCL device code. You need to adjust the code.
                    */
                    d_list[i++] = new sphere(
                        center, 0.2,
                        /*
                        DPCT1109:71: Memory storage allocation cannot be called
                        in SYCL device code. You need to adjust the code.
                        */
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if(choose_mat < 0.95f) {
                    /*
                    DPCT1109:72: Memory storage allocation cannot be called in
                    SYCL device code. You need to adjust the code.
                    */
                    d_list[i++] = new sphere(
                        center, 0.2,
                        /*
                        DPCT1109:73: Memory storage allocation cannot be called
                        in SYCL device code. You need to adjust the code.
                        */
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                                       0.5f * (1.0f + RND)),
                                  0.5f * RND));
                }
                else {
                    /*
                    DPCT1109:74: Memory storage allocation cannot be called in
                    SYCL device code. You need to adjust the code.
                    */
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        /*
        DPCT1109:75: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        /*
        DPCT1109:76: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0,
                                 new lambertian(vec3(0.4, 0.2, 0.1)));
        /*
        DPCT1109:77: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        d_list[i++] =
            new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        /*
        DPCT1109:78: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        /*
        DPCT1109:79: Memory storage allocation cannot be called in SYCL device
        code. You need to adjust the code.
        */
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0,
                               float(nx) / float(ny), aperture, dist_to_focus);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Free world resources. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_list">  	[in,out] If non-null, the list. </param>
/// <param name="d_world"> 	[in,out] If non-null, the world. </param>
/// <param name="d_camera">	[in,out] If non-null, the camera. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("X", OPT_INT, "1200", "specify image x dimension", '\0');
    op.addOption("Y", OPT_INT, "800", "specify image y dimension", '\0');
    op.addOption("samples", OPT_INT, "10", "specify number of iamge samples", '\0');
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="DB">	[in,out] The database. </param>
/// <param name="op">	[in,out] The options specified. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    dpct::event_ptr total_start, total_stop;
    std::chrono::time_point<std::chrono::steady_clock> total_start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> total_stop_ct1;
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

    checkCudaErrors(DPCT_CHECK_ERROR(total_start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(total_stop = new sycl::event()));
    /*
    DPCT1012:505: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:506: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    total_start_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(
        DPCT_CHECK_ERROR(*total_start = dpct::get_default_queue()
                                            .ext_oneapi_submit_barrier()));

    // Predefined image resolutions
    int xDim[5] = {400, 1200, 4096, 15360, 20480};
    int yDim[5] = {300, 800, 2160, 8640, 17280};
    int size = op.getOptionInt("size") - 1;
    int nx = xDim[size];
    int ny = yDim[size];
    if (op.getOptionInt("X") != 1200 || op.getOptionInt("Y") != 800) {
        nx = op.getOptionInt("X");
        ny = op.getOptionInt("Y");
    }
    int ns = op.getOptionInt("samples");
    assert(ns > 0);
    int tx = 8;
    int ty = 8;
    int num_passes = op.getOptionInt("passes");

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    /*
    DPCT1064:524: Migrated cudaMallocManaged call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        fb = (vec3 *)sycl::malloc_shared(fb_size, dpct::get_default_queue())));

    // allocate random state
    /*
    DPCT1032:507: A different random number generator is used. You may need to
    adjust the code.
    */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *d_rand_state = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1032:508: A different random number generator is used. You may need
        to adjust the code.
        */
        /*
        DPCT1064:525: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_rand_state = sycl::malloc_shared<dpct::rng::device::rng_generator<
                oneapi::mkl::rng::device::mcg59<1>>>(
                num_pixels, dpct::get_default_queue())));
    } else {
        /*
        DPCT1032:509: A different random number generator is used. You may need
        to adjust the code.
        */
        /*
        DPCT1064:526: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_rand_state = sycl::malloc_device<dpct::rng::device::rng_generator<
                oneapi::mkl::rng::device::mcg59<1>>>(
                num_pixels, dpct::get_default_queue())));
    }

    /*
    DPCT1032:510: A different random number generator is used. You may need to
    adjust the code.
    */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *d_rand_state2 = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1032:511: A different random number generator is used. You may need
        to adjust the code.
        */
        /*
        DPCT1064:527: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_rand_state2 =
                sycl::malloc_shared<dpct::rng::device::rng_generator<
                    oneapi::mkl::rng::device::mcg59<1>>>(
                    num_pixels, dpct::get_default_queue())));
    } else {
        /*
        DPCT1032:512: A different random number generator is used. You may need
        to adjust the code.
        */
        /*
        DPCT1064:528: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_rand_state2 =
                sycl::malloc_device<dpct::rng::device::rng_generator<
                    oneapi::mkl::rng::device::mcg59<1>>>(
                    num_pixels, dpct::get_default_queue())));
    }

    // we need that 2nd random state to be initialized for the world creation
      *stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
                rand_init(d_rand_state2, item_ct1);
          });
    /*
    DPCT1010:513: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:529: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_list = sycl::malloc_shared<hitable *>(
                                 num_hitables, dpct::get_default_queue())));
    } else {
        /*
        DPCT1064:530: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_list = sycl::malloc_device<hitable *>(
                                 num_hitables, dpct::get_default_queue())));
    }
    
    hitable **d_world;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:531: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_world = sycl::malloc_shared<hitable *>(
                                 num_hitables, dpct::get_default_queue())));
    } else {
        /*
        DPCT1064:532: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(
            DPCT_CHECK_ERROR(d_world = sycl::malloc_device<hitable *>(
                                 num_hitables, dpct::get_default_queue())));
    }

    camera **d_camera;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        /*
        DPCT1064:533: Migrated cudaMallocManaged call is used in a
        macro/template definition and may not be valid for all macro/template
        uses. Adjust the code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_camera = (camera **)sycl::malloc_shared(
                num_hitables * sizeof(hitable *), dpct::get_default_queue())));
    } else {
        /*
        DPCT1064:534: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_camera = (camera **)sycl::malloc_device(
                num_hitables * sizeof(hitable *), dpct::get_default_queue())));
    }

      *stop = dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
                create_world(d_list, d_world, d_camera, nx, ny, d_rand_state2,
                             item_ct1);
          });
    /*
    DPCT1010:514: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    // use cudaevent
        char atts[1024];
    sprintf(atts, "img: %d by %d, samples: %d, iter:%d", nx, ny, ns, num_passes);
    int i = 0;
    for (; i < num_passes; i++) {
        /*
        DPCT1012:515: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:516: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(
            DPCT_CHECK_ERROR(*start = dpct::get_default_queue()
                                          .ext_oneapi_submit_barrier()));
        // Render our buffer
        sycl::range<3> blocks(1, ny / ty + 1, nx / tx + 1);
        sycl::range<3> threads(1, ty, tx);
        /*
        DPCT1049:80: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(blocks * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                      render_init(nx, ny, d_rand_state, item_ct1);
                });
        /*
        DPCT1010:517: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        /*
        DPCT1049:81: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            dpct::get_default_queue().parallel_for(
                sycl::nd_range<3>(blocks * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                      render(fb, nx, ny, ns, d_camera, d_world, d_rand_state,
                             item_ct1);
                });
        /*
        DPCT1010:518: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        /*
        DPCT1012:519: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:520: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(DPCT_CHECK_ERROR(
                *stop = dpct::get_default_queue().ext_oneapi_submit_barrier()));
        checkCudaErrors(0);
        float t = 0;
        checkCudaErrors(DPCT_CHECK_ERROR(
            (t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));
        DB.AddResult("raytracing rendering time", atts, "sec", t * 1.0e-3);
    }
    // std::cerr << "took " << timer_seconds << " seconds.\n";

#if 0
    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
#endif

    // clean up
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
      dpct::get_default_queue().parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
                free_world(d_list, d_world, d_camera);
          });
    /*
    DPCT1010:521: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(d_camera, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(d_world, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(d_list, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(d_rand_state, dpct::get_default_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(fb, dpct::get_default_queue())));

    /*
    DPCT1012:522: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:523: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    stop->wait();
    dpct::get_current_device().queues_wait_and_throw();
    total_stop_ct1 = std::chrono::steady_clock::now();
    checkCudaErrors(
        DPCT_CHECK_ERROR(*total_stop = dpct::get_default_queue()
                                           .ext_oneapi_submit_barrier()));
    checkCudaErrors(0);
    float total_time = 0.0f;
    checkCudaErrors(
        DPCT_CHECK_ERROR((total_time = std::chrono::duration<float, std::milli>(
                                           total_stop_ct1 - total_start_ct1)
                                           .count())));
    DB.AddResult("raytracing total execution time", atts, "sec", total_time * 1.0e-3);

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_current_device().reset()));
}
