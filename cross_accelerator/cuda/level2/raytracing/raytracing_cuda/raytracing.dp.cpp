////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\raytracing.cu
//
// summary:	Raytracing class
//
// origin:
// Raytracing(https://github.com/rogerallen/raytracinginoneweekendincuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>

#include <chrono>
#include <iostream>
#include <float.h>
#include <random>

#include "camera.h"
#include "hitable_list.h"
#include "material.h"
#include "random_gen.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include "oneapi/dpl/random"

constexpr int32_t g_threads_x = 8;
constexpr int32_t g_threads_y = 8;

#ifdef _FPGA
#define ATTRIBUTE // Nothing needed for compile
#else
#define ATTRIBUTE
#endif

vec3
color(const ray &r, hitable_list *world, lfsr_prng &rndstate)
{
    ray  cur_ray         = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if (world->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray  scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(
                    cur_ray, rec, attenuation, scattered, rndstate))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            vec3  unit_direction = unit_vector(cur_ray.direction());
            float t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3  c              = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f)
                     + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0f, 0.0f, 0.0f); // exceeded recursion
}

class RaytracingKernelID;
void
render(vec3            *fb,
       int              max_x,
       int              max_y,
       int              ns,
       camera          *cam,
       hitable_list    *world,
       sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2)
            + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int j = item_ct1.get_local_id(1)
            + item_ct1.get_group(1) * item_ct1.get_local_range(1);
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;

    vec3      col(0, 0, 0);
    lfsr_prng lfsr_state(item_ct1);

    for (int s = 0; s < ns; s++)
    {
        float u = float(i + lfsr_state.rand()) / float(max_x);
        float v = float(j + lfsr_state.rand()) / float(max_y);
        ray   r = cam->get_ray(u, v, lfsr_state);
        col += color(r, world, lfsr_state);
    }

    col /= float(ns);
    col[0]          = sycl::sqrt(col[0]);
    col[1]          = sycl::sqrt(col[1]);
    col[2]          = sycl::sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND distr(engine)

void
create_world(sphere       *h_spheres,
             material     *h_materials,
             hitable_list *h_world,
             camera       *h_camera,
             sphere       *d_spheres,
             material     *d_materials,
             hitable_list *d_world,
             camera       *d_camera,
             int           nx,
             int           ny,
             sycl::queue  &queue)
{
    std::cout << "Creating the world..." << std::endl;

    // Init random number generator.
    //
    std::random_device rd;
    oneapi::dpl::minstd_rand                      engine(rd(), 0);
    oneapi::dpl::uniform_real_distribution<float> distr(0.0f, 1.0f);

    // Initialize Camera.
    //
    vec3  lookfrom(13.0f, 2.0f, 3.0f);
    vec3  lookat(0.0f, 0.0f, 0.0f);
    float dist_to_focus = 10.0f;
    (lookfrom - lookat).length();
    float aperture = 0.1f;
    *h_camera      = camera(lookfrom,
                       lookat,
                       vec3(0.0f, 1.0f, 0.0f),
                       30.0f,
                       float(nx) / float(ny),
                       aperture,
                       dist_to_focus);

    // Initialize Materials and Spheres. Make sure to let material-pointers of
    // the spheres point to the corresponding material on the device.
    //
    h_spheres[0]   = sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, d_materials);
    h_materials[0] = material(vec3(0.5f, 0.5f, 0.5f));
    int16_t i      = 1;
    for (int8_t a = -11; a < 11; a++)
    {
        for (int8_t b = -11; b < 11; b++)
        {
            float choose_mat = RND;
            vec3  center(a + RND, 0.2f, b + RND);
            h_spheres[i] = sphere(center, 0.2f, &d_materials[i]);

            if (choose_mat < 0.8f)
                h_materials[i]
                    = material(vec3(RND * RND, RND * RND, RND * RND));
            else if (choose_mat < 0.95f)
                h_materials[i] = material(vec3(0.5f * (1.0f + RND),
                                               0.5f * (1.0f + RND),
                                               0.5f * (1.0f + RND)),
                                          0.5f * RND);
            else
                h_materials[i] = material(1.5f);

            i++;
        }
    }
    h_spheres[i]   = sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, &d_materials[i]);
    h_materials[i] = material(1.5f);
    i++;
    h_spheres[i]   = sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, &d_materials[i]);
    h_materials[i] = material(vec3(0.4f, 0.2f, 0.1f));
    i++;
    h_spheres[i]   = sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, &d_materials[i]);
    h_materials[i] = material(vec3(0.7f, 0.6f, 0.5f), 0.0f);
    i++;

    // Initialize World.
    //
    h_world->list      = d_spheres;
    h_world->list_size = i;

    queue.memcpy(d_spheres, h_spheres, i * sizeof(sphere));
    queue.memcpy(d_materials, h_materials, i * sizeof(material));
    queue.memcpy(d_world, h_world, sizeof(hitable_list));
    queue.memcpy(d_camera, h_camera, sizeof(camera));
    queue.wait_and_throw();

    std::cout << "Done." << std::endl;
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("X", OPT_INT, "1200", "specify image x dimension", '\0');
    op.addOption("Y", OPT_INT, "800", "specify image y dimension", '\0');
    op.addOption(
        "samples", OPT_INT, "10", "specify number of iamge samples", '\0');
}

void
RunBenchmark(ResultDatabase &DB, OptionParser &op, size_t device_idx)
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    std::chrono::time_point<std::chrono::steady_clock> total_start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> total_stop_ct1;

    total_start_ct1 = std::chrono::steady_clock::now();

    // Predefined image resolutions
    int xDim[5] = { 400, 1200, 4096, 15360, 20480 };
    int yDim[5] = { 300, 800, 2160, 8640, 17280 };
    int size    = op.getOptionInt("size") - 1;
    int nx      = xDim[size];
    int ny      = yDim[size];
    if (op.getOptionInt("X") != 1200 || op.getOptionInt("Y") != 800)
    {
        nx = op.getOptionInt("X");
        ny = op.getOptionInt("Y");
    }
    int ns = op.getOptionInt("samples");
    assert(ns > 0);
    int num_passes = op.getOptionInt("passes");

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns
              << " samples per pixel ";
    std::cerr << "in " << g_threads_x << "x" << g_threads_y << " blocks.\n";

    int num_pixels = nx * ny;

    // allocate FB
    const size_t       fb_size = num_pixels * sizeof(vec3);
    sycl::buffer<vec3> h_fb { sycl::range(fb_size) };

    // Make our world of hitables & the camera. The host data we only need for
    // the world-creation.
    //
    constexpr int num_hitables = 22 * 22 + 1 + 3;
    sphere       *h_spheres //
        = (sphere *)malloc(num_hitables * sizeof(sphere));
    material *h_materials //
        = (material *)malloc(num_hitables * sizeof(material));
    hitable_list *h_world //
        = (hitable_list *)malloc(1 * sizeof(hitable_list));
    camera *h_camera //
        = (camera *)malloc(1 * sizeof(camera));
    sphere *d_spheres //
        = sycl::malloc_device<sphere>(num_hitables, queue);
    material *d_materials //
        = sycl::malloc_device<material>(num_hitables, queue);
    hitable_list *d_world //
        = sycl::malloc_device<hitable_list>(1, queue);
    camera *d_camera //
        = sycl::malloc_device<camera>(1, queue);
    if (h_spheres == nullptr || h_materials == nullptr || h_world == nullptr
        || h_camera == nullptr || d_spheres == nullptr || d_materials == nullptr
        || d_camera == nullptr)
    {
        std::cerr << "Error allocating memory." << std::endl;
        std::terminate();
    }
    create_world(h_spheres,
                 h_materials,
                 h_world,
                 h_camera,
                 d_spheres,
                 d_materials,
                 d_world,
                 d_camera,
                 nx,
                 ny,
                 queue);
    free(h_spheres);
    free(h_materials);
    free(h_world);
    free(h_camera);

    char atts[1024];
    sprintf(
        atts, "img: %d by %d, samples: %d, iter:%d", nx, ny, ns, num_passes);
    for (int i = 0; i < num_passes; i++)
    {
        std::cout << "Pass " << i << std::endl;

        const sycl::range<3> blocks(
            1, ny / g_threads_y + 1, nx / g_threads_x + 1);
        const sycl::range<3> threads(1, g_threads_y, g_threads_x);

        sycl::event render_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor a_fb { h_fb, cgh, sycl::write_only, sycl::noinit };

            cgh.parallel_for<RaytracingKernelID>(
                sycl::nd_range<3>(blocks * threads, threads),
                [=](sycl::nd_item<3> item_ct1) ATTRIBUTE {
                    render(a_fb.get_pointer(),
                           nx,
                           ny,
                           ns,
                           d_camera,
                           d_world,
                           item_ct1);
                });
        });
        render_event.wait();
        const float elapsed = render_event.get_profiling_info<
                                  sycl::info::event_profiling::command_end>()
                              - render_event.get_profiling_info<
                                  sycl::info::event_profiling::command_start>();

        DB.AddResult("raytracing rendering time", atts, "sec", elapsed * 1.e-9);
    }

#if 1
    // Output FB as Image.
    //
    const int      filesize = 54 + 3 * nx * ny;
    unsigned char *img      = (unsigned char *)malloc(3 * nx * ny);
    memset(img, 0, 3 * nx * ny);

    sycl::host_accessor fb { h_fb, sycl::read_only };
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            int    r           = int(255.99 * fb[pixel_index].r());
            int    g           = int(255.99 * fb[pixel_index].g());
            int    b           = int(255.99 * fb[pixel_index].b());
            if (r > 255)
                r = 255;
            if (g > 255)
                g = 255;
            if (b > 255)
                b = 255;
            int x                     = i;
            int y                     = (ny - 1) - j;
            img[(x + y * nx) * 3 + 2] = (unsigned char)(r);
            img[(x + y * nx) * 3 + 1] = (unsigned char)(g);
            img[(x + y * nx) * 3 + 0] = (unsigned char)(b);
        }
    }

    unsigned char bmpfileheader[14]
        = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
    unsigned char bmpinfoheader[40]
        = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
    unsigned char bmppad[3] = { 0, 0, 0 };

    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);

    bmpinfoheader[4]  = (unsigned char)(nx);
    bmpinfoheader[5]  = (unsigned char)(nx >> 8);
    bmpinfoheader[6]  = (unsigned char)(nx >> 16);
    bmpinfoheader[7]  = (unsigned char)(nx >> 24);
    bmpinfoheader[8]  = (unsigned char)(ny);
    bmpinfoheader[9]  = (unsigned char)(ny >> 8);
    bmpinfoheader[10] = (unsigned char)(ny >> 16);
    bmpinfoheader[11] = (unsigned char)(ny >> 24);

    FILE *f = fopen("img.bmp", "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    for (int i = 0; i < ny; i++)
    {
        fwrite(img + (nx * (ny - i - 1) * 3), 3, nx, f);
        fwrite(bmppad, 1, (4 - (nx * 3) % 4) % 4, f);
    }

    free(img);
    fclose(f);
    std::cout << "Done!" << std::endl;
#endif

    // Clean up.
    //
    sycl::free(d_camera, queue);
    sycl::free(d_world, queue);
    sycl::free(d_spheres, queue);
    sycl::free(d_materials, queue);

    total_stop_ct1   = std::chrono::steady_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(
                           total_stop_ct1 - total_start_ct1)
                           .count();
    DB.AddResult(
        "raytracing total execution time", atts, "sec", total_time * 1.0e-3);
}
