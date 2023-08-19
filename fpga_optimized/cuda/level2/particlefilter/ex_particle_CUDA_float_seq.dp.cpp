////////////////////////////////////////////////////////////////////////////////////////////////////
// file:
// altis\src\cuda\level2\particlefilter\ex_particle_CUDA_float_seq.cu
//
// summary:	Exception particle cuda float sequence class
//
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <chrono>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "compute_unit.hpp"
#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "fpga_pwr.hpp"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#define PI 3.1415926535897932

constexpr int threads_per_block = 512; // 256 on P630, else 512

template <size_t cu>
class kernel_cu;
template <size_t cu>
class writeback_cu;

using cached_lsu =
    sycl::ext::intel::lsu</*
        sycl::ext::intel::statically_coalesce<false>,
        sycl::ext::intel::burst_coalesce<true>,
        sycl::ext::intel::cache<524288>*/>;

#ifdef _STRATIX10
constexpr int32_t num_cus = 10;
constexpr uint8_t u_vals_size = 50; // #ParticlesToProcessAtOnce
#endif
#ifdef _AGILEX
constexpr int32_t num_cus = 4;
constexpr uint8_t u_vals_size = 24; // #ParticlesToProcessAtOnce
#endif

using idx_pipe = 
    fpga_tools::PipeArray<class idx_pipe_id, int32_t, u_vals_size * 2, num_cus>;
using likelihood_to_sum_pipe = 
    fpga_tools::PipeArray<class likelihood_to_sum_id, double, 64, 1>;
using sum_to_find_pipe = 
    fpga_tools::PipeArray<class likelihood_to_sum_id, double, 64, 1>;

bool norand  = false;
bool verbose = false;
bool quiet   = false;

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
 */
long M = INT_MAX;
/**
@var A value for LCG
 */
int A = 1103515245;
/**
@var C value for LCG
 */
int C = 12345;

/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 -
 *(IK[IND] - 228)^2)/ 100 param 1 I 3D matrix param 2 current ind array param 3
 *length of ind array returns a double representing the sum
 ********************************/
double
calcLikelihoodSum(unsigned char *I, int *ind, int numOnes, int index)
{
    double likelihoodSum = 0.0;
    int    x;
    for (x = 0; x < numOnes; x++)
        likelihoodSum += ((double)(I[ind[index * numOnes + x]] - 100)
                              * (double)(I[ind[index * numOnes + x]] - 100)
                          - (double)(I[ind[index * numOnes + x]] - 228)
                                * (double)(I[ind[index * numOnes + x]] - 228))
                         / 50.0;
    return likelihoodSum;
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
 *****************************/
void
cdfCalc(double *CDF, double *weights, int Nparticles)
{
    double temp = weights[0];
    #pragma unroll 8
    for (int x = 1; x < Nparticles; x++)
    {
        double new_temp = weights[x] + temp;
        CDF[x] = new_temp;
        new_temp = temp;
    }
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a double representing a randomily generated number from a uniform
 *distribution with range [0, 1)
 ******************************/
double
d_randu(int *seed, int index)
{
    constexpr int    M     = INT_MAX;
    constexpr double M_INV = 1.0 / double(M);
    constexpr int    A     = 1103515245;
    constexpr int    C     = 12345;

    int num      = A * seed[index] + C;
    int new_seed = num % M;

    seed[index] = new_seed;

    return sycl::fabs(new_seed * M_INV);
}
sycl::double4
d_randu4(int *seed, int index)
{
    constexpr int    M     = INT_MAX;
    constexpr double M_INV = 1.0 / double(M);
    constexpr int    A     = 1103515245;
    constexpr int    C     = 12345;

    int num1      = A * seed[index] + C;
    int new_seed1 = num1 % M;
    double rand1  = sycl::fabs(new_seed1 * M_INV);

    int num2      = A * new_seed1 + C;
    int new_seed2 = num2 % M;
    double rand2  = sycl::fabs(new_seed2 * M_INV);

    int num3      = A * new_seed2 + C;
    int new_seed3 = num2 % M;
    double rand3  = sycl::fabs(new_seed2 * M_INV);

    int num4      = A * new_seed3 + C;
    int new_seed4 = num3 % M;
    double rand4  = sycl::fabs(new_seed3 * M_INV);

    seed[index] = new_seed4;

    return {rand1, rand2, rand3, rand4};
}

/**
   * Generates a uniformly distributed random number using the provided seed and
   * GCC's settings for the Linear Congruential Generator (LCG)
   * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
   * @note This function is thread-safe
   * @param seed The seed array
   * @param index The specific index of the seed to be advanced
   * @return a uniformly distributed number [0, 1)
   */

double
randu(int *seed, int index)
{
    int num     = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double)M));
}

/**
 * Generates a normally distributed random number using the Box-Muller
 * transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller
 * algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing
 * value for normal random distribution
 */
double
randn(int *seed, int index)
{
    /*Box-Muller algorithm*/
    double u      = randu(seed, index);
    double v      = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt     = -2 * log(u);
    return sqrt(rt) * cosine;
}

double
test_randn(int *seed, int index)
{
    // Box-Muller algortihm
    double pi     = 3.14159265358979323846;
    double u      = randu(seed, index);
    double v      = randu(seed, index);
    double cosine = cos(2 * pi * v);
    double rt     = -2 * log(u);
    return sqrt(rt) * cosine;
}

double
d_randn(int *seed, int index)
{
    // Box-Muller algortihm
    double pi     = 3.14159265358979323846;
    double u      = d_randu(seed, index);
    double v      = d_randu(seed, index);
    double cosine = sycl::cos(2 * pi * v);
    double rt     = -2 * sycl::log(u);
    return sycl::sqrt(rt) * cosine;
}

sycl::double2
d_randn2(int *seed, int index)
{
    // Box-Muller algortihm
    double pi     = 3.14159265358979323846;
    sycl::double4 randX2 = d_randu4(seed, index);
    double u      = randX2[0];
    double v      = randX2[1];
    double u2      = randX2[2];
    double v2      = randX2[3];
    double cosine = sycl::cos(2 * pi * v);
    double cosine2 = sycl::cos(2 * pi * v2);
    double rt     = -2 * sycl::log(u);
    double rt2     = -2 * sycl::log(u2);
    return {sycl::sqrt(rt) * cosine, sycl::sqrt(rt2) * cosine2};
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
 ****************************/
double
updateWeights(double *weights, double *likelihood, int Nparticles)
{
    int    x;
    double sum = 0;
    for (x = 0; x < Nparticles; x++)
    {
        weights[x] = weights[x] * sycl::exp(likelihood[x]);
        sum += weights[x];
    }
    return sum;
}

int
findIndexBin(double *CDF, int beginIndex, int endIndex, double value)
{
    if (endIndex < beginIndex)
        return -1;
    int middleIndex;
    while (endIndex > beginIndex)
    {
        middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
        if (CDF[middleIndex] >= value)
        {
            if (middleIndex == 0)
                return middleIndex;
            else if (CDF[middleIndex - 1] < value)
                return middleIndex;
            else if (CDF[middleIndex - 1] == value)
            {
                while (CDF[middleIndex] == value && middleIndex >= 0)
                    middleIndex--;
                middleIndex++;
                return middleIndex;
            }
        }
        if (CDF[middleIndex] > value)
            endIndex = middleIndex - 1;
        else
            beginIndex = middleIndex + 1;
    }
    return -1;
}

double
dev_round_double(double value)
{
    int newValue = (int)(value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}

/*****************************
 * CUDA Likelihood Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param2.5: CDF
 * param3: ind
 * param4: objxy
 * param5: likelihood
 * param6: I
 * param6.5: u
 * param6.75: weights
 * param7: Nparticles
 * param8: countOnes
 * param9: max_size
 * param10: k
 * param11: IszY
 * param12: Nfr
 *****************************/
void
likelihood_kernel(double          *arrayX,
                  double          *arrayY,
                  double          *xj,
                  double          *yj,
                  int             *ind,
                  int             *objxy,
                  double          *likelihood,
                  unsigned char   *I,
                  double          *weights,
                  int              Nparticles,
                  double           one_div_Nparticles,
                  int              countOnes,
                  double           one_div_countOnes,
                  int              max_size,
                  int              k,
                  int              IszY,
                  int              Nfr,
                  int             *seed,
                  double          *partial_sums,
                  sycl::nd_item<1> item_ct1)
{
    int i = item_ct1.get_global_id(0);
    int y;

    int indX, indY;

    auto buffer_ptr = 
        group_local_memory_for_overwrite<double[threads_per_block]>(item_ct1.get_group());
    auto& buffer = *buffer_ptr;

    buffer[item_ct1.get_local_id(0)] = (i < Nparticles) ? weights[i] : 0.0;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // this doesn't account for the last block that isn't full
    #pragma unroll
    for (unsigned int s = threads_per_block / 2; s > 0; s >>= 1)
    {
        if (item_ct1.get_local_id(0) < s)
            buffer[item_ct1.get_local_id(0)]
                += buffer[item_ct1.get_local_id(0) + s];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (item_ct1.get_local_id(0) == 0)
        partial_sums[item_ct1.get_group(0)] = buffer[0];
}

/**
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value
 * > input value
 */
double
roundDouble(double value)
{
    int newValue = (int)(value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the
 * testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void
setIf(int            testValue,
      int            newValue,
      unsigned char *array3D,
      int           *dimX,
      int           *dimY,
      int           *dimZ)
{
    int x, y, z;
    for (x = 0; x < *dimX; x++)
    {
        for (y = 0; y < *dimY; y++)
        {
            for (z = 0; z < *dimZ; z++)
            {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal
 * distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void
addNoise(unsigned char *array3D, int *dimX, int *dimY, int *dimZ, int *seed)
{
    int x, y, z;
    for (x = 0; x < *dimX; x++)
    {
        for (y = 0; y < *dimY; y++)
        {
            for (z = 0; z < *dimZ; z++)
            {
                array3D[x * *dimY * *dimZ + y * *dimZ + z]
                    = array3D[x * *dimY * *dimZ + y * *dimZ + z]
                      + (unsigned char)(5 * randn(seed, 0));
            }
        }
    }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void
strelDisk(int *disk, int radius)
{
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++)
    {
        for (y = 0; y < diameter; y++)
        {
            double distance
                = sqrt((double)(x - radius + 1) * (double)(x - radius + 1)
                       + (double)(y - radius + 1) * (double)(y - radius + 1));
            if (distance < radius)
            {
                disk[x * diameter + y] = 1;
            }
            else
            {
                disk[x * diameter + y] = 0;
            }
        }
    }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void
dilate_matrix(unsigned char *matrix,
              int            posX,
              int            posY,
              int            posZ,
              int            dimX,
              int            dimY,
              int            dimZ,
              int            error)
{
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++)
    {
        for (y = startY; y < endY; y++)
        {
            double distance = sqrt((double)(x - posX) * (double)(x - posX)
                                   + (double)(y - posY) * (double)(y - posY));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void
imdilate_disk(unsigned char *matrix,
              int            dimX,
              int            dimY,
              int            dimZ,
              int            error,
              unsigned char *newMatrix)
{
    int x, y, z;
    for (z = 0; z < dimZ; z++)
    {
        for (x = 0; x < dimX; x++)
        {
            for (y = 0; y < dimY; y++)
            {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1)
                {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void
getneighbors(int *se, int numOnes, int *neighbors, int radius)
{
    int x, y;
    int neighY   = 0;
    int center   = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++)
    {
        for (y = 0; y < diameter; y++)
        {
            if (se[x * diameter + y])
            {
                neighbors[neighY * 2]     = (int)(y - center);
                neighbors[neighY * 2 + 1] = (int)(x - center);
                neighY++;
            }
        }
    }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the background intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void
videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed)
{
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0                            = (int)roundDouble(IszY / 2.0);
    int y0                            = (int)roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++)
    {
        xk  = abs(x0 + (k - 1));
        yk  = abs(y0 - 2 * (k - 1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    unsigned char *newMatrix
        = (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++)
    {
        for (y = 0; y < IszY; y++)
        {
            for (k = 0; k < Nfr; k++)
            {
                I[x * IszY * Nfr + y * Nfr + k]
                    = newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 */
int
findIndex(double *CDF, int lengthCDF, double value)
{
    int index = -1;
    for (int x = 0; x < lengthCDF; x++)
        if (CDF[x] >= value)
        {
            index = x;
            break;
        }

    if (index == -1)
        return lengthCDF - 1;
    return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In
 * addition, it references a provided MATLAB function which takes the video, the
 * objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void
particleFilter(unsigned char  *I,
               int             IszX,
               int             IszY,
               int             Nfr,
               int            *seed,
               int             Nparticles,
               ResultDatabase &resultDB,
               size_t          device_idx)
try
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    float                                              kernelTime   = 0.0f;
    float                                              transferTime = 0.0f;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float                                              elapsedTime;

    int max_size = IszX * IszY * Nfr;
    // original particle centroid
    double xe = roundDouble(IszY / 2.0);
    double ye = roundDouble(IszX / 2.0);

    // expected object locations, compared to center
    constexpr int radius   = 5;
    constexpr int diameter = radius * 2 - 1;
    constexpr int countOnesMax = diameter * diameter;

    int *disk = (int *)malloc(diameter * diameter * sizeof(int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++)
        for (y = 0; y < diameter; y++)
            if (disk[x * diameter + y] == 1)
                countOnes++;

    int *objxy = (int *)malloc(countOnes * 2 * sizeof(int));
    getneighbors(disk, countOnes, objxy, radius);
    // initial weights are all equal (1/Nparticles)
    double *weights = (double *)malloc(sizeof(double) * Nparticles);
    for (x = 0; x < Nparticles; x++)
        weights[x] = 1 / ((double)(Nparticles));

    // initial likelihood to 0.0
    double *likelihood = (double *)malloc(sizeof(double) * Nparticles);
    double *arrayX     = (double *)malloc(sizeof(double) * Nparticles);
    double *arrayY     = (double *)malloc(sizeof(double) * Nparticles);
    double *xj         = (double *)malloc(sizeof(double) * Nparticles);
    double *yj         = (double *)malloc(sizeof(double) * Nparticles);
    double *CDF        = (double *)malloc(sizeof(double) * Nparticles);

    // GPU copies of arrays
    double *arrayX_GPU     = sycl::malloc_device<double>(Nparticles, queue);
    double *arrayY_GPU     = sycl::malloc_device<double>(Nparticles, queue);
    double *xj_GPU         = sycl::malloc_device<double>(Nparticles, queue);
    double *yj_GPU         = sycl::malloc_device<double>(Nparticles, queue);
    double *CDF_GPU        = sycl::malloc_device<double>(Nparticles, queue);
    double *likelihood_GPU = sycl::malloc_device<double>(Nparticles, queue);
    unsigned char *I_GPU   = (unsigned char *)sycl::malloc_device(
        sizeof(unsigned char) * IszX * IszY * Nfr, queue);
    double *weights_GPU = sycl::malloc_device<double>(Nparticles, queue);
    int    *objxy_GPU
        = (int *)sycl::malloc_device(sizeof(int) * 2 * countOnes, queue);

    int *ind     = (int *)malloc(sizeof(int) * countOnes * Nparticles);
    int *ind_GPU = (int *)sycl::malloc_device(
        sizeof(int) * countOnes * Nparticles, queue);

    double *u     = (double *)malloc(sizeof(double) * Nparticles);
    double *u_GPU = sycl::malloc_device<double>(Nparticles, queue);

    int    *seed_GPU     = sycl::malloc_device<int>(Nparticles, queue);
    double *partial_sums = sycl::malloc_device<double>(Nparticles, queue);

    // set likelihood to zero
    queue.memset((void *)likelihood_GPU, 0, sizeof(double) * Nparticles).wait();

    // Donnie - this loop is different because in this kernel, arrayX and arrayY
    //   are set equal to xj before every iteration, so effectively, arrayX and
    //   arrayY will be set to xe and ye before the first iteration.
    for (x = 0; x < Nparticles; x++)
    {
        xj[x] = xe;
        yj[x] = ye;
    }

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(I_GPU, I, sizeof(unsigned char) * IszX * IszY * Nfr);
    queue.memcpy(objxy_GPU, objxy, sizeof(int) * 2 * countOnes);
    queue.memcpy(weights_GPU, weights, sizeof(double) * Nparticles);
    queue.memcpy(xj_GPU, xj, sizeof(double) * Nparticles);
    queue.memcpy(yj_GPU, yj, sizeof(double) * Nparticles);
    queue.memcpy(seed_GPU, seed, sizeof(int) * Nparticles);
    queue.wait_and_throw();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;

    int num_blocks = ceil((double)Nparticles / (double)threads_per_block);
    for (int k = 1; k < Nfr; k++)
    {
        using part_t = ac_int<19, false>; // Max 500k.
        start_ct1 = std::chrono::steady_clock::now();
        constexpr int II = 12;
        /*
            Times 1 & 2 only
            0.000100976
            0.000671291
            0.911245
        */
        queue.submit([&](sycl::handler &cgh) {
            const double one_div_Nparticles = 1.0 / double(Nparticles);
            const double one_div_countOnes  = 1.0 / double(countOnes);
            constexpr double one_div_50 = 1.0 / 50.0;
            cgh.single_task<class pf_float_1>(
                [=]() [[intel::kernel_args_restrict, intel::max_global_work_dim(0), intel::no_global_work_offset(1)]] {
                    // ------------------------------------
                    // LIKELIHOOD KERNEL
                    // ------------------------------------

                    [[intel::ivdep, intel::initiation_interval(1)]]
                    for (part_t i = 0; i < Nparticles; i++)
                    {
                        const double xj = xj_GPU[i];
                        const double yj = yj_GPU[i];

                        const sycl::double2 randX2 = d_randn2(seed_GPU, i);
                        const double arrx_val = xj + 1.0 + 5.0 * randX2[0];
                        const double arry_val = yj - 2.0 + 2.0 * randX2[1];
                        const double arrxr_val = dev_round_double(arrx_val);
                        const double arryr_val = dev_round_double(arry_val);

                        arrayX_GPU[i] = arrx_val;
                        arrayY_GPU[i] = arry_val;

                        [[intel::private_copies(512)]]
                        int32_t ind[countOnesMax];
                        for (int16_t y = 0; y < countOnes; y++)
                        {
                            const int32_t indX = arrxr_val + objxy_GPU[y * 2 + 1];
                            const int32_t indY = arryr_val + objxy_GPU[y * 2];

                            const int32_t ind_val = sycl::abs(indX * IszY * Nfr + indY * Nfr + k);
                            ind[y] = ind_val >= max_size ? 0 : ind_val;
                        }

                        double shift[II + 1];
                        for (int8_t j = 0; j < II + 1; j++)
                            shift[j] = 0.0;

                        for (int16_t x = 0; x < countOnes; x++)
                        {
                            const double I_val = I_GPU[ind[x]];
                            const double likelihood = 
                                            ((I_val - 100.0) * (I_val - 100.0)
                                            - (I_val - 228.0) * (I_val - 228.0))
                                            * one_div_50;
                            shift[II] = shift[0] + likelihood;

                            #pragma unroll
                            for (int8_t j = 0; j < II; j++)
                                shift[j] = shift[j + 1];
                        }

                        double likelihoodSum = 0.0;
                        #pragma unroll
                        for (int8_t i = 0; i < II; i++)
                            likelihoodSum += shift[i];

                        const double likelihood = likelihoodSum * one_div_countOnes;
                        const double new_weight = one_div_Nparticles * sycl::exp(likelihood);
                        weights_GPU[i] = new_weight;

                        likelihood_to_sum_pipe::PipeAt<0>::write(new_weight);
                    }
            });
        });
        queue.submit([&](sycl::handler &cgh) {
            const double one_div_Nparticles = 1.0 / double(Nparticles);
            const double one_div_countOnes  = 1.0 / double(countOnes);
            constexpr double one_div_50 = 1.0 / 50.0;
            cgh.single_task<class pf_float_2>(
                [=]() [[intel::kernel_args_restrict, intel::max_global_work_dim(0), intel::no_global_work_offset(1)]] {
                    // ------------------------------------
                    // SUM KERNEL
                    // ------------------------------------

                    double sum_weights_shift[II + 1];
                    for (int8_t j = 0; j < II + 1; j++)
                        sum_weights_shift[j] = 0.0;
                    
                    [[intel::initiation_interval(1)]]
                    for (part_t i = 0; i < Nparticles; i++)
                    {
                        const double new_weight =
                            likelihood_to_sum_pipe::PipeAt<0>::read();
                        sum_weights_shift[II] = sum_weights_shift[0] + new_weight;

                        #pragma unroll
                        for (int8_t j = 0; j < II; j++)
                            sum_weights_shift[j] = sum_weights_shift[j + 1];
                    }

                    double sum = 0.0;
                    #pragma unroll
                    for (int8_t i = 0; i < II; i++)
                        sum += sum_weights_shift[i];

                    // ------------------------------------
                    // NORMALIZE WEIGHTS KERNEL
                    // ------------------------------------

                    #pragma unroll 8
                    for (int32_t i = 0; i < Nparticles; i++)
                        weights_GPU[i] /= sum;

                    cdfCalc(CDF_GPU, weights_GPU, Nparticles);

                    double u1 = one_div_Nparticles * d_randu(seed_GPU, 0);
                    #pragma unroll 8
                    for (int32_t i = 0; i < Nparticles; i++)
                        u_GPU[i] = u1 + i * one_div_Nparticles;
                        //sum_to_find_pipe::PipeAt<0>::write(u1 + i * one_div_Nparticles);
            });
        }).wait();
        const int32_t num_particles_per_cu = std::ceil(Nparticles / double(num_cus));
        std::array<sycl::event, num_cus> writeback_events;
        SubmitComputeUnits<num_cus, writeback_cu>(queue, writeback_events, [=](auto ID) {
            const int32_t part_start = num_particles_per_cu * ID;
            const int32_t part_stop  = num_particles_per_cu * (ID + 1);

            for (int32_t particle_idx = part_start; particle_idx < part_stop; particle_idx += u_vals_size)
            {
                [[intel::initiation_interval(1), intel::ivdep, intel::speculated_iterations(0)]]
                for (uint8_t i = 0; i < u_vals_size; i++)
                {
                    if (particle_idx + i < part_stop)
                    {
                        const int32_t idx = idx_pipe::PipeAt<ID>::read();
                        xj_GPU[particle_idx + i] = arrayX_GPU[idx];
                        yj_GPU[particle_idx + i] = arrayY_GPU[idx];     
                    }
                }
            }
        });
        std::array<sycl::event, num_cus> submit_events;
        SubmitComputeUnits<num_cus, kernel_cu>(queue, submit_events, [=](auto ID) {
            const int32_t part_start = num_particles_per_cu * ID;
            const int32_t part_stop  = num_particles_per_cu * (ID + 1);

            // Process u_vals_size particles at once. This means, when iterating through the
            // particle array in the inner loop, we calculate the resulting index of u_vals_size
            // particles at once. Reduces the amount of times we need to read all particle-data by
            // u_vals_size !
            [[intel::initiation_interval(1)]]
            for (int32_t particle_idx = part_start; particle_idx < part_stop; particle_idx += u_vals_size)
            {
                [[intel::private_copies(16)]]
                double u_vals[u_vals_size];
                #pragma unroll 8
                for (uint8_t i = 0; i < u_vals_size; i++)
                    u_vals[i] =
                        (particle_idx + i) < part_stop ?
                            u_GPU[particle_idx + i] :
                            .0;

                constexpr uint8_t cdf_vals_size = 8; // 8xdouble=512bit
                [[intel::private_copies(16)]]
                int indices[u_vals_size] = { -1, -1, -1, -1, -1, -1, -1, -1 };
                [[intel::initiation_interval(1), intel::speculated_iterations(0)]]
                for (int32_t x = 0; x < Nparticles; x += cdf_vals_size)
                {
                    double cdf_vals[cdf_vals_size];
                    #pragma unroll
                    for (uint8_t i = 0; i < cdf_vals_size; i++)
                        cdf_vals[i] =
                            (x + i) < Nparticles ?
                                CDF_GPU[x + i] :
                                .0;

                    #pragma unroll
                    for (uint8_t u = 0; u < u_vals_size; u++)
                    {
                        const double u_val = u_vals[u];

                        #pragma unroll
                        for (uint8_t c = 0; c < cdf_vals_size; c++)
                            if (cdf_vals[c] < u_val)
                                if (indices[u] == -1)
                                    indices[u] = x + c;
                    }
                }

                [[intel::initiation_interval(1), intel::ivdep, intel::speculated_iterations(0)]]
                for (uint8_t i = 0; i < u_vals_size; i++)
                {
                    const int32_t idx = indices[i];
                    if (particle_idx + i < part_stop)
                        idx_pipe::PipeAt<ID>::write(idx == -1 ? Nparticles - 1 : idx);
                }
           }
        });
        queue.wait();
        stop_ct1    = std::chrono::steady_clock::now();
        elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                        .count();
        kernelTime += elapsedTime * 1.e-3;
    } // end loop

    sycl::free(xj_GPU, queue);
    sycl::free(yj_GPU, queue);
    sycl::free(CDF_GPU, queue);
    sycl::free(u_GPU, queue);
    sycl::free(likelihood_GPU, queue);
    sycl::free(I_GPU, queue);
    sycl::free(objxy_GPU, queue);
    sycl::free(ind_GPU, queue);
    sycl::free(seed_GPU, queue);
    sycl::free(partial_sums, queue);

    start_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(arrayX, arrayX_GPU, sizeof(double) * Nparticles);
    queue.memcpy(arrayY, arrayY_GPU, sizeof(double) * Nparticles);
    queue.memcpy(weights, weights_GPU, sizeof(double) * Nparticles);
    queue.wait_and_throw();
    stop_ct1    = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
    transferTime += elapsedTime * 1.e-3;

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++)
    {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }
    if (verbose && !quiet)
    {
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);
        double distance = sqrt(
            sycl::pown((double)(xe - (int)roundDouble(IszY / 2.0)), 2)
            + sycl::pown((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
        printf("%lf\n", distance);
    }

    char atts[1024];
    sprintf(atts,
            "dimx:%d, dimy:%d, numframes:%d, numparticles:%d",
            IszX,
            IszY,
            Nfr,
            Nparticles);
    resultDB.AddResult(
        "particlefilter_float_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult(
        "particlefilter_float_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("particlefilter_float_total_time",
                       atts,
                       "sec",
                       kernelTime + transferTime);
    resultDB.AddResult(
        "particlefilter_float_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);

    // CUDA freeing of memory
    sycl::free(weights_GPU, queue);
    sycl::free(arrayY_GPU, queue);
    sycl::free(arrayX_GPU, queue);

    // free regular memory
    free(likelihood);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("dimx", OPT_INT, "0", "grid x dimension", 'x');
    op.addOption("dimy", OPT_INT, "0", "grid y dimension", 'y');
    op.addOption(
        "framecount", OPT_INT, "0", "number of frames to track across", 'f');
    op.addOption("np", OPT_INT, "0", "number of particles to use");
    op.addOption("norand", OPT_BOOL, "0", "do not use random input for vertification", 'r');
}

void particlefilter_float(ResultDatabase &resultDB,
                          int             args[],
                          size_t          device_idx);

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running ParticleFilter (float)\n");

    int args[4];
    args[0] = op.getOptionInt("dimx");
    args[1] = op.getOptionInt("dimy");
    args[2] = op.getOptionInt("framecount");
    args[3] = op.getOptionInt("np");

    bool preset = false;
    verbose     = op.getOptionBool("verbose");
    quiet       = op.getOptionBool("quiet");
    norand      = op.getOptionBool("norand");

    for (int i = 0; i < 4; i++)
        if (args[i] <= 0)
            preset = true;

    if (preset)
    {
        int probSizes[4][4] = { { 10, 10, 2, 100 },
                                { 40, 40, 5, 500 },
                                { 200, 200, 8, 500000 },
                                { 500, 500, 15, 1000000 } };
        int size            = op.getOptionInt("size") - 1;
        for (int i = 0; i < 4; i++)
            args[i] = probSizes[size][i];
    }

    if (!quiet)
        printf("Using dimx=%d, dimy=%d, framecount=%d, numparticles=%d\n",
               args[0],
               args[1],
               args[2],
               args[3]);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++)
    {
        if (!quiet)
            printf("Pass %d: ", i);

        particlefilter_float(resultDB, args, device_idx);

        if (!quiet)
            printf("Done.\n");
    }
}

void
particlefilter_float(ResultDatabase &resultDB, int args[], size_t device_idx)
{
    int IszX       = args[0];
    int IszY       = args[1];
    int Nfr        = args[2];
    int Nparticles = args[3];

    // establish seed
    int *seed = (int *)malloc(sizeof(int) * Nparticles);
    for (int i = 0; i < Nparticles; i++)
        seed[i] = norand ? i * i : time(0) * i;

    unsigned char *I
        = (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);

    videoSequence(I, IszX, IszY, Nfr, seed);
    particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, resultDB, device_idx);

    free(seed);
    free(I);
}
