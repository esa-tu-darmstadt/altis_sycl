////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\particlefilter\ex_particle_CUDA_float_seq.cu
//
// summary:	Exception particle cuda float sequence class
// 
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <chrono>

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932

const int threads_per_block = 512;

bool verbose = false;
bool quiet = false;

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

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
 * param 1 I 3D matrix
 * param 2 current ind array
 * param 3 length of ind array
 * returns a double representing the sum
 ********************************/
double calcLikelihoodSum(unsigned char * I, int * ind, int numOnes, int index) {
    double likelihoodSum = 0.0;
    int x;
    for (x = 0; x < numOnes; x++)
        likelihoodSum += ((double)(I[ind[index * numOnes + x]] - 100) *
                              (double)(I[ind[index * numOnes + x]] - 100) -
                          (double)(I[ind[index * numOnes + x]] - 228) *
                              (double)(I[ind[index * numOnes + x]] - 228)) /
                         50.0;
    return likelihoodSum;
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
 *****************************/
void cdfCalc(double * CDF, double * weights, int Nparticles) {
    int x;
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
        CDF[x] = weights[x] + CDF[x - 1];
    }
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
 ******************************/
double d_randu(int * seed, int index) {

    int M = INT_MAX;
    int A = 1103515245;
    int C = 12345;
    int num = A * seed[index] + C;
    seed[index] = num % M;

    return sycl::fabs(seed[index] / ((double)M));
}/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/

double randu(int * seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double) M));
}

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
double randn(int * seed, int index) {
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

double test_randn(int * seed, int index) {
    //Box-Muller algortihm
    double pi = 3.14159265358979323846;
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * pi * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

double d_randn(int * seed, int index) {
    //Box-Muller algortihm
    double pi = 3.14159265358979323846;
    double u = d_randu(seed, index);
    double v = d_randu(seed, index);
    double cosine = sycl::cos(2 * pi * v);
    double rt = -2 * sycl::log(u);
    return sycl::sqrt(rt) * cosine;
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparticles
 ****************************/
double updateWeights(double * weights, double * likelihood, int Nparticles) {
    int x;
    double sum = 0;
    for (x = 0; x < Nparticles; x++) {
        weights[x] = weights[x] * sycl::exp(likelihood[x]);
        sum += weights[x];
    }
    return sum;
}

int findIndexBin(double * CDF, int beginIndex, int endIndex, double value) {
    if (endIndex < beginIndex)
        return -1;
    int middleIndex;
    while (endIndex > beginIndex) {
        middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
        if (CDF[middleIndex] >= value) {
            if (middleIndex == 0)
                return middleIndex;
            else if (CDF[middleIndex - 1] < value)
                return middleIndex;
            else if (CDF[middleIndex - 1] == value) {
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

/** added this function. was missing in original double version.
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double dev_round_double(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}

/*****************************
 * CUDA Find Index Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: CDF
 * param4: u
 * param5: xj
 * param6: yj
 * param7: weights
 * param8: Nparticles
 *****************************/
void find_index_kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, double * weights, int Nparticles,
                       const sycl::nd_item<3> &item_ct1) {
    int block_id = item_ct1.get_group(2);
    int i = item_ct1.get_local_range(2) * block_id + item_ct1.get_local_id(2);

    if (i < Nparticles) {

        int index = -1;
        int x;

        for (x = 0; x < Nparticles; x++) {
            if (CDF[x] >= u[i]) {
                index = x;
                break;
            }
        }
        if (index == -1) {
            index = Nparticles - 1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];

        //weights[i] = 1 / ((double) (Nparticles)); //moved this code to the beginning of likelihood kernel

    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
}

void normalize_weights_kernel(double * weights, int Nparticles, double* partial_sums, double * CDF, double * u, int * seed,
                              const sycl::nd_item<3> &item_ct1, double &u1,
                              double &sumWeights) {
    int block_id = item_ct1.get_group(2);
    int i = item_ct1.get_local_range(2) * block_id + item_ct1.get_local_id(2);

    if (0 == item_ct1.get_local_id(2))
        sumWeights = partial_sums[0];

    /*
    DPCT1065:136: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i < Nparticles) {
        weights[i] = weights[i] / sumWeights;
    }

    /*
    DPCT1065:137: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i == 0) {
        cdfCalc(CDF, weights, Nparticles);
        u[0] = (1 / ((double) (Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
    }

    /*
    DPCT1065:138: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (0 == item_ct1.get_local_id(2))
        u1 = u[0];

    /*
    DPCT1065:139: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i < Nparticles) {
        u[i] = u1 + i / ((double) (Nparticles));
    }
}

void sum_kernel(double* partial_sums, int Nparticles,
                const sycl::nd_item<3> &item_ct1) {
    int block_id = item_ct1.get_group(2);
    int i = item_ct1.get_local_range(2) * block_id + item_ct1.get_local_id(2);

    if (i == 0) {
        int x;
        double sum = 0.0;
        int num_blocks =
            sycl::ceil((double)Nparticles / (double)threads_per_block);
        for (x = 0; x < num_blocks; x++) {
            sum += partial_sums[x];
        }
        partial_sums[0] = sum;
    }
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
void likelihood_kernel(double * arrayX, double * arrayY, double * xj, double * yj, double * CDF, int * ind, int * objxy, double * likelihood, unsigned char * I, double * u, double * weights, int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr, int *seed, double* partial_sums,
                       const sycl::nd_item<3> &item_ct1, double *buffer) {
    int block_id = item_ct1.get_group(2);
    int i = item_ct1.get_local_range(2) * block_id + item_ct1.get_local_id(2);
    int y;
    
    int indX, indY;

    if (i < Nparticles) {
        arrayX[i] = xj[i]; 
        arrayY[i] = yj[i]; 

        weights[i] = 1 / ((double) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.

        arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
        arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);
        
    }

    /*
    DPCT1065:140: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i < Nparticles) {
        for (y = 0; y < countOnes; y++) {
            //added dev_round_double() to be consistent with roundDouble
            indX = dev_round_double(arrayX[i]) + objxy[y * 2 + 1];
            indY = dev_round_double(arrayY[i]) + objxy[y * 2];

            ind[i * countOnes + y] =
                sycl::abs(indX * IszY * Nfr + indY * Nfr + k);
            if (ind[i * countOnes + y] >= max_size)
                ind[i * countOnes + y] = 0;
        }
        likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);
        
        likelihood[i] = likelihood[i] / countOnes;

        weights[i] =
            weights[i] *
            sycl::exp(likelihood[i]); // Donnie Newell - added the missing
                                      // exponential function call
    }

    buffer[item_ct1.get_local_id(2)] = 0.0;

    /*
    DPCT1065:141: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (i < Nparticles) {

        buffer[item_ct1.get_local_id(2)] = weights[i];
    }

    /*
    DPCT1065:142: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    //this doesn't account for the last block that isn't full
    for (unsigned int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
        if (item_ct1.get_local_id(2) < s) {
            buffer[item_ct1.get_local_id(2)] +=
                buffer[item_ct1.get_local_id(2) + s];
        }

        /*
        DPCT1065:144: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (item_ct1.get_local_id(2) == 0) {
        partial_sums[item_ct1.get_group(2)] = buffer[0];
    }

    /*
    DPCT1065:143: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

/** 
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double roundDouble(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
            }
        }
    }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            double distance =
                sqrt((double)(x - radius + 1) * (double)(x - radius + 1) +
                     (double)(y - radius + 1) * (double)(y - radius + 1));
            if (distance < radius) {
                disk[x * diameter + y] = 1;
            } else {
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
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
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
    for (x = startX; x < endX; x++) {
        for (y = startY; y < endY; y++) {
            double distance = sqrt((double)(x - posX) * (double)(x - posX) +
                                   (double)(y - posY) * (double)(y - posY));
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
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
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
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (se[x * diameter + y]) {
                neighbors[neighY * 2] = (int) (y - center);
                neighbors[neighY * 2 + 1] = (int) (x - center);
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
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0 = (int) roundDouble(IszY / 2.0);
    int y0 = (int) roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
        xk = abs(x0 + (k-1));
        yk = abs(y0 - 2 * (k-1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    unsigned char * newMatrix = (unsigned char *) malloc(sizeof (unsigned char) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
        for (y = 0; y < IszY; y++) {
            for (k = 0; k < Nfr; k++) {
                I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
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
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(double * CDF, int lengthCDF, double value) {
    int index = -1;
    int x;
    for (x = 0; x < lengthCDF; x++) {
        if (CDF[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return lengthCDF - 1;
    }
    return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles, ResultDatabase &resultDB) try {

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    float elapsedTime;

    int max_size = IszX * IszY*Nfr;
    //original particle centroid
    double xe = roundDouble(IszY / 2.0);
    double ye = roundDouble(IszX / 2.0);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius * 2 - 1;
    int * disk = (int*) malloc(diameter * diameter * sizeof (int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }
    int * objxy = (int *) malloc(countOnes * 2 * sizeof (int));
    getneighbors(disk, countOnes, objxy, radius);
    //initial weights are all equal (1/Nparticles)
    double * weights = (double *) malloc(sizeof (double) *Nparticles);
    for (x = 0; x < Nparticles; x++) {
        weights[x] = 1 / ((double) (Nparticles));
    }

    //initial likelihood to 0.0
    double * likelihood = (double *) malloc(sizeof (double) *Nparticles);
    double * arrayX = (double *) malloc(sizeof (double) *Nparticles);
    double * arrayY = (double *) malloc(sizeof (double) *Nparticles);
    double * xj = (double *) malloc(sizeof (double) *Nparticles);
    double * yj = (double *) malloc(sizeof (double) *Nparticles);
    double * CDF = (double *) malloc(sizeof (double) *Nparticles);

    //GPU copies of arrays
    double * arrayX_GPU;
    double * arrayY_GPU;
    double * xj_GPU;
    double * yj_GPU;
    double * CDF_GPU;
    double * likelihood_GPU;
    unsigned char * I_GPU;
    double * weights_GPU;
    int * objxy_GPU;

    int * ind = (int*) malloc(sizeof (int) *countOnes * Nparticles);
    int * ind_GPU;
    double * u = (double *) malloc(sizeof (double) *Nparticles);
    double * u_GPU;
    int * seed_GPU;
    double* partial_sums;

    //CUDA memory allocation
    /*
    DPCT1064:688: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(arrayX_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:689: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(arrayY_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:690: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(xj_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:691: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(yj_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:692: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(CDF_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:693: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(u_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:694: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(likelihood_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    //set likelihood to zero
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memset((void *)likelihood_GPU, 0, sizeof(double) * Nparticles)
            .wait()));
    /*
    DPCT1064:695: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(weights_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:696: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(I_GPU = (unsigned char *)sycl::malloc_device(
                             sizeof(unsigned char) * IszX * IszY * Nfr,
                             dpct::get_default_queue())));
    /*
    DPCT1064:697: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        objxy_GPU = (int *)sycl::malloc_device(sizeof(int) * 2 * countOnes,
                                               dpct::get_default_queue())));
    /*
    DPCT1064:698: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        ind_GPU = (int *)sycl::malloc_device(
            sizeof(int) * countOnes * Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:699: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        seed_GPU =
            sycl::malloc_device<int>(Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:700: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(partial_sums = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));

    //Donnie - this loop is different because in this kernel, arrayX and arrayY
    //  are set equal to xj before every iteration, so effectively, arrayX and 
    //  arrayY will be set to xe and ye before the first iteration.
    for (x = 0; x < Nparticles; x++) {

        xj[x] = xe;
        yj[x] = ye;

    }

    int k;
    //start send
    /*
    DPCT1012:658: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();

    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(I_GPU, I, sizeof(unsigned char) * IszX * IszY * Nfr)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(objxy_GPU, objxy, sizeof(int) * 2 * countOnes)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(weights_GPU, weights, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(xj_GPU, xj, sizeof(double) * Nparticles)
                             .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(yj_GPU, yj, sizeof(double) * Nparticles)
                             .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(seed_GPU, seed, sizeof(int) * Nparticles)
                             .wait()));
    int num_blocks = ceil((double) Nparticles / (double) threads_per_block);

    /*
    DPCT1012:659: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;


    double wall1 = get_wall_time();
    for (k = 1; k < Nfr; k++) {

        /*
        DPCT1012:662: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        start_ct1 = std::chrono::steady_clock::now();
        *start = dpct::get_default_queue().ext_oneapi_submit_barrier();
        /*
        DPCT1049:145: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            {
                  dpct::has_capability_or_fail(
                      dpct::get_default_queue().get_device(),
                      {sycl::aspect::fp64});
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<double, 1> buffer_acc_ct1(
                            sycl::range<1>(512), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, num_blocks) *
                                    sycl::range<3>(1, 1, threads_per_block),
                                sycl::range<3>(1, 1, threads_per_block)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  likelihood_kernel(
                                      arrayX_GPU, arrayY_GPU, xj_GPU, yj_GPU,
                                      CDF_GPU, ind_GPU, objxy_GPU,
                                      likelihood_GPU, I_GPU, u_GPU, weights_GPU,
                                      Nparticles, countOnes, max_size, k, IszY,
                                      Nfr, seed_GPU, partial_sums, item_ct1,
                                      buffer_acc_ct1.get_pointer());
                            });
                  });
            }
        /*
        DPCT1049:146: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            {
                  dpct::has_capability_or_fail(
                      dpct::get_default_queue().get_device(),
                      {sycl::aspect::fp64});
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
                      [=](sycl::nd_item<3> item_ct1) {
                            sum_kernel(partial_sums, Nparticles, item_ct1);
                      });
            }
        /*
        DPCT1049:147: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            {
                  dpct::has_capability_or_fail(
                      dpct::get_default_queue().get_device(),
                      {sycl::aspect::fp64});
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<double, 0> u1_acc_ct1(cgh);
                        sycl::local_accessor<double, 0> sumWeights_acc_ct1(cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, num_blocks) *
                                    sycl::range<3>(1, 1, threads_per_block),
                                sycl::range<3>(1, 1, threads_per_block)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  normalize_weights_kernel(
                                      weights_GPU, Nparticles, partial_sums,
                                      CDF_GPU, u_GPU, seed_GPU, item_ct1,
                                      u1_acc_ct1, sumWeights_acc_ct1);
                            });
                  });
            }
        /*
        DPCT1049:148: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
            {
                  dpct::has_capability_or_fail(
                      dpct::get_default_queue().get_device(),
                      {sycl::aspect::fp64});
                  dpct::get_default_queue().parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
                      [=](sycl::nd_item<3> item_ct1) {
                            find_index_kernel(
                                arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU,
                                yj_GPU, weights_GPU, Nparticles, item_ct1);
                      });
            }
        /*
        DPCT1012:663: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        *stop = dpct::get_default_queue().ext_oneapi_submit_barrier();
        elapsedTime =
            std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                .count();
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR();

    }//end loop

    //block till kernels are finished
    dpct::get_current_device().queues_wait_and_throw();
    double wall2 = get_wall_time();

    sycl::free(xj_GPU, dpct::get_default_queue());
    sycl::free(yj_GPU, dpct::get_default_queue());
    sycl::free(CDF_GPU, dpct::get_default_queue());
    sycl::free(u_GPU, dpct::get_default_queue());
    sycl::free(likelihood_GPU, dpct::get_default_queue());
    sycl::free(I_GPU, dpct::get_default_queue());
    sycl::free(objxy_GPU, dpct::get_default_queue());
    sycl::free(ind_GPU, dpct::get_default_queue());
    sycl::free(seed_GPU, dpct::get_default_queue());
    sycl::free(partial_sums, dpct::get_default_queue());

    /*
    DPCT1012:660: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(arrayX, arrayX_GPU, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(arrayY, arrayY_GPU, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(weights, weights_GPU, sizeof(double) * Nparticles)
            .wait()));
    /*
    DPCT1012:661: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }
    if(verbose && !quiet) {
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);
        double distance =
            sqrt(sycl::pown((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                 sycl::pown((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
        printf("%lf\n", distance);
    }
    
    char atts[1024];
    sprintf(atts, "dimx:%d, dimy:%d, numframes:%d, numparticles:%d", IszX, IszY, Nfr, Nparticles);
    resultDB.AddResult("particlefilter_float_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("particlefilter_float_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("particlefilter_float_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("particlefilter_float_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);

    //CUDA freeing of memory
    sycl::free(weights_GPU, dpct::get_default_queue());
    sycl::free(arrayY_GPU, dpct::get_default_queue());
    sycl::free(arrayX_GPU, dpct::get_default_queue());

    //free regular memory
    free(likelihood);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void particleFilterGraph(unsigned char *I, int IszX, int IszY, int Nfr,
                         int *seed, int Nparticles,
                         ResultDatabase &resultDB) try {

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    float elapsedTime;

    int max_size = IszX * IszY*Nfr;
    //original particle centroid
    double xe = roundDouble(IszY / 2.0);
    double ye = roundDouble(IszX / 2.0);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius * 2 - 1;
    int * disk = (int*) malloc(diameter * diameter * sizeof (int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }
    int * objxy = (int *) malloc(countOnes * 2 * sizeof (int));
    getneighbors(disk, countOnes, objxy, radius);
    //initial weights are all equal (1/Nparticles)
    double * weights = (double *) malloc(sizeof (double) *Nparticles);
    for (x = 0; x < Nparticles; x++) {
        weights[x] = 1 / ((double) (Nparticles));
    }

    //initial likelihood to 0.0
    double * likelihood = (double *) malloc(sizeof (double) *Nparticles);
    double * arrayX = (double *) malloc(sizeof (double) *Nparticles);
    double * arrayY = (double *) malloc(sizeof (double) *Nparticles);
    double * xj = (double *) malloc(sizeof (double) *Nparticles);
    double * yj = (double *) malloc(sizeof (double) *Nparticles);
    double * CDF = (double *) malloc(sizeof (double) *Nparticles);

    //GPU copies of arrays
    double * arrayX_GPU;
    double * arrayY_GPU;
    double * xj_GPU;
    double * yj_GPU;
    double * CDF_GPU;
    double * likelihood_GPU;
    unsigned char * I_GPU;
    double * weights_GPU;
    int * objxy_GPU;

    int * ind = (int*) malloc(sizeof (int) *countOnes * Nparticles);
    int * ind_GPU;
    double * u = (double *) malloc(sizeof (double) *Nparticles);
    double * u_GPU;
    int * seed_GPU;
    double* partial_sums;

    //CUDA memory allocation
    /*
    DPCT1064:701: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(arrayX_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:702: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(arrayY_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:703: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(xj_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:704: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(yj_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:705: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(CDF_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:706: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(u_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:707: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(likelihood_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    //set likelihood to zero
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memset((void *)likelihood_GPU, 0, sizeof(double) * Nparticles)
            .wait()));
    /*
    DPCT1064:708: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(weights_GPU = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:709: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(I_GPU = (unsigned char *)sycl::malloc_device(
                             sizeof(unsigned char) * IszX * IszY * Nfr,
                             dpct::get_default_queue())));
    /*
    DPCT1064:710: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        objxy_GPU = (int *)sycl::malloc_device(sizeof(int) * 2 * countOnes,
                                               dpct::get_default_queue())));
    /*
    DPCT1064:711: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        ind_GPU = (int *)sycl::malloc_device(
            sizeof(int) * countOnes * Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:712: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        seed_GPU =
            sycl::malloc_device<int>(Nparticles, dpct::get_default_queue())));
    /*
    DPCT1064:713: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(partial_sums = sycl::malloc_device<double>(
                             Nparticles, dpct::get_default_queue())));

    //Donnie - this loop is different because in this kernel, arrayX and arrayY
    //  are set equal to xj before every iteration, so effectively, arrayX and 
    //  arrayY will be set to xe and ye before the first iteration.
    for (x = 0; x < Nparticles; x++) {

        xj[x] = xe;
        yj[x] = ye;

    }

    int k;
    //start send
    /*
    DPCT1012:664: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();

    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(I_GPU, I, sizeof(unsigned char) * IszX * IszY * Nfr)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(objxy_GPU, objxy, sizeof(int) * 2 * countOnes)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(weights_GPU, weights, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(xj_GPU, xj, sizeof(double) * Nparticles)
                             .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(yj_GPU, yj, sizeof(double) * Nparticles)
                             .wait()));
    CUDA_SAFE_CALL(
        DPCT_CHECK_ERROR(dpct::get_default_queue()
                             .memcpy(seed_GPU, seed, sizeof(int) * Nparticles)
                             .wait()));
    int num_blocks = ceil((double) Nparticles / (double) threads_per_block);

    /*
    DPCT1012:665: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;

    // Init graph metadata
    dpct::queue_ptr streamForGraph;
    cudaGraph_t graph;
    cudaGraphNode_t likelihoodKernelNode, sumKernelNode, normalizeWeightsKernelNode, findIndexKernelNode;

    /*
    DPCT1007:668: Migration of cudaGraphCreate is not supported.
    */
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    checkCudaErrors(DPCT_CHECK_ERROR(
        streamForGraph = dpct::get_current_device().create_queue()));

    // Set up first kernel node
    /*
    DPCT1082:669: Migration of cudaKernelNodeParams type is not supported.
    */
    cudaKernelNodeParams likelihoodKernelNodeParams = {0};
    void *likelihoodKernelArgs[19] = {(void *)&arrayX_GPU, (void *)&arrayY_GPU,
                                      (void *)&xj_GPU, (void *)&yj_GPU,
                                      (void *)&CDF_GPU, (void *)&ind_GPU,
                                      (void *)&objxy_GPU, (void *)&likelihood_GPU,
                                      (void *)&I_GPU, (void *)&u_GPU,
                                      (void *)&weights_GPU, &Nparticles,
                                      &countOnes, &max_size, &k, &IszY,
                                      &Nfr, (void *)&seed_GPU, (void *)&partial_sums};
    likelihoodKernelNodeParams.func = (void *)likelihood_kernel;
    likelihoodKernelNodeParams.gridDim = sycl::range<3>(1, 1, num_blocks);
    likelihoodKernelNodeParams.blockDim =
        sycl::range<3>(1, 1, threads_per_block);
    likelihoodKernelNodeParams.sharedMemBytes = 0;
    likelihoodKernelNodeParams.kernelParams = (void **)likelihoodKernelArgs;
    likelihoodKernelNodeParams.extra = NULL;

    /*
    DPCT1007:670: Migration of cudaGraphAddKernelNode is not supported.
    */
    checkCudaErrors(cudaGraphAddKernelNode(&likelihoodKernelNode, graph, NULL,
                                           0, &likelihoodKernelNodeParams));

    // Set up the second kernel node
    /*
    DPCT1082:671: Migration of cudaKernelNodeParams type is not supported.
    */
    cudaKernelNodeParams sumKernelNodeParams = {0};
    void *sumKernelArgs[2] = {(void *)&partial_sums, &Nparticles};
    sumKernelNodeParams.func = (void *)sum_kernel;
    sumKernelNodeParams.gridDim = sycl::range<3>(1, 1, num_blocks);
    sumKernelNodeParams.blockDim = sycl::range<3>(1, 1, threads_per_block);
    sumKernelNodeParams.sharedMemBytes = 0;
    sumKernelNodeParams.kernelParams = (void **)sumKernelArgs;
    sumKernelNodeParams.extra = NULL;

    /*
    DPCT1007:672: Migration of cudaGraphAddKernelNode is not supported.
    */
    checkCudaErrors(cudaGraphAddKernelNode(&sumKernelNode, graph, NULL, 0,
                                           &sumKernelNodeParams));

    // set up the third kernel node
    /*
    DPCT1082:673: Migration of cudaKernelNodeParams type is not supported.
    */
    cudaKernelNodeParams normalizeWeightsKernelNodeParams = {0};
    void *normalizeWeightsKernelArgs[6] = {(void *)&weights_GPU, &Nparticles,
                                           (void *)&partial_sums, (void *)&CDF_GPU,
                                           (void *)&u_GPU, (void *)&seed_GPU};
    normalizeWeightsKernelNodeParams.func = (void *)normalize_weights_kernel;
    normalizeWeightsKernelNodeParams.gridDim = sycl::range<3>(1, 1, num_blocks);
    normalizeWeightsKernelNodeParams.blockDim =
        sycl::range<3>(1, 1, threads_per_block);
    normalizeWeightsKernelNodeParams.sharedMemBytes = 0;
    normalizeWeightsKernelNodeParams.kernelParams = (void **)normalizeWeightsKernelArgs;
    normalizeWeightsKernelNodeParams.extra = NULL;

    /*
    DPCT1007:674: Migration of cudaGraphAddKernelNode is not supported.
    */
    checkCudaErrors(cudaGraphAddKernelNode(&normalizeWeightsKernelNode, graph,
                                           NULL, 0,
                                           &normalizeWeightsKernelNodeParams));

    // set up the fourth kernel node
    /*
    DPCT1082:675: Migration of cudaKernelNodeParams type is not supported.
    */
    cudaKernelNodeParams findIndexKernelNodeParams = {0};
    void *findIndexKernelArgs[8] = {(void *)&arrayX_GPU, (void *)&arrayY_GPU, (void *)&CDF_GPU,
                                    (void *)&u_GPU, (void *)&xj_GPU,
                                    (void *)&yj_GPU, (void *)&weights_GPU,
                                    &Nparticles};
    findIndexKernelNodeParams.func = (void *)find_index_kernel;
    findIndexKernelNodeParams.gridDim = sycl::range<3>(1, 1, num_blocks);
    findIndexKernelNodeParams.blockDim =
        sycl::range<3>(1, 1, threads_per_block);
    findIndexKernelNodeParams.sharedMemBytes = 0;
    findIndexKernelNodeParams.kernelParams = (void **)findIndexKernelArgs;
    findIndexKernelNodeParams.extra = NULL;

    /*
    DPCT1007:676: Migration of cudaGraphAddKernelNode is not supported.
    */
    checkCudaErrors(cudaGraphAddKernelNode(&findIndexKernelNode, graph, NULL, 0,
                                           &findIndexKernelNodeParams));

    // Add dependencies between each kernels
    /*
    DPCT1007:677: Migration of cudaGraphAddDependencies is not supported.
    */
    checkCudaErrors(cudaGraphAddDependencies(graph, &likelihoodKernelNode,
                                             &sumKernelNode, 1));
    /*
    DPCT1007:678: Migration of cudaGraphAddDependencies is not supported.
    */
    checkCudaErrors(cudaGraphAddDependencies(graph, &sumKernelNode,
                                             &normalizeWeightsKernelNode, 1));
    /*
    DPCT1007:679: Migration of cudaGraphAddDependencies is not supported.
    */
    checkCudaErrors(cudaGraphAddDependencies(graph, &normalizeWeightsKernelNode,
                                             &findIndexKernelNode, 1));

    // init the graph
    cudaGraphExec_t graphExec;
    /*
    DPCT1007:680: Migration of cudaGraphInstantiate is not supported.
    */
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    double wall1 = get_wall_time();
    for (k = 1; k < Nfr; k++) {

        /*
        DPCT1012:681: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:682: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
        /*
        DPCT1007:683: Migration of cudaGraphLaunch is not supported.
        */
        checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
        /*
        DPCT1012:684: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:685: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            (elapsedTime =
                 std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));
        kernelTime += elapsedTime * 1.e-3;

    }//end loop

    //block till kernels are finished
    checkCudaErrors(DPCT_CHECK_ERROR(streamForGraph->wait()));
    double wall2 = get_wall_time();

    /*
    DPCT1007:686: Migration of cudaGraphExecDestroy is not supported.
    */
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    /*
    DPCT1007:687: Migration of cudaGraphDestroy is not supported.
    */
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(streamForGraph)));

    sycl::free(xj_GPU, dpct::get_default_queue());
    sycl::free(yj_GPU, dpct::get_default_queue());
    sycl::free(CDF_GPU, dpct::get_default_queue());
    sycl::free(u_GPU, dpct::get_default_queue());
    sycl::free(likelihood_GPU, dpct::get_default_queue());
    sycl::free(I_GPU, dpct::get_default_queue());
    sycl::free(objxy_GPU, dpct::get_default_queue());
    sycl::free(ind_GPU, dpct::get_default_queue());
    sycl::free(seed_GPU, dpct::get_default_queue());
    sycl::free(partial_sums, dpct::get_default_queue());

    /*
    DPCT1012:666: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(arrayX, arrayX_GPU, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(arrayY, arrayY_GPU, sizeof(double) * Nparticles)
            .wait()));
    CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
        dpct::get_default_queue()
            .memcpy(weights, weights_GPU, sizeof(double) * Nparticles)
            .wait()));
    /*
    DPCT1012:667: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    transferTime += elapsedTime * 1.e-3;

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }
    if(verbose && !quiet) {
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);
        double distance =
            sqrt(sycl::pown((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                 sycl::pown((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
        printf("%lf\n", distance);
    }
    
    char atts[1024];
    sprintf(atts, "dimx:%d, dimy:%d, numframes:%d, numparticles:%d", IszX, IszY, Nfr, Nparticles);
    resultDB.AddResult("particlefilter_float_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("particlefilter_float_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("particlefilter_float_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("particlefilter_float_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);

    //CUDA freeing of memory
    sycl::free(weights_GPU, dpct::get_default_queue());
    sycl::free(arrayY_GPU, dpct::get_default_queue());
    sycl::free(arrayX_GPU, dpct::get_default_queue());

    //free regular memory
    free(likelihood);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("dimx", OPT_INT, "0", "grid x dimension", 'x');
  op.addOption("dimy", OPT_INT, "0", "grid y dimension", 'y');
  op.addOption("framecount", OPT_INT, "0", "number of frames to track across", 'f');
  op.addOption("np", OPT_INT, "0", "number of particles to use");
}

void particlefilter_float(ResultDatabase &resultDB, int args[], bool useGraph);

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running ParticleFilter (float)\n");
    int args[4];
    args[0] = op.getOptionInt("dimx");
    args[1] = op.getOptionInt("dimy");
    args[2] = op.getOptionInt("framecount");
    args[3] = op.getOptionInt("np");
    bool preset = false;
    verbose = op.getOptionBool("verbose");
    quiet = op.getOptionBool("quiet");
    bool useGraph = op.getOptionBool("graph");

    for(int i = 0; i < 4; i++) {
        if(args[i] <= 0) {
            preset = true;
        }
    }
    if(preset) {
        int probSizes[4][4] = {{10, 10, 2, 100},
                               {40, 40, 5, 500},
                               {200, 200, 8, 500000},
                               {500, 500, 15, 1000000}};
        int size = op.getOptionInt("size") - 1;
        for(int i = 0; i < 4; i++) {
            args[i] = probSizes[size][i];
        }
    }

    if(!quiet) {
        printf("Using dimx=%d, dimy=%d, framecount=%d, numparticles=%d\n",
                args[0], args[1], args[2], args[3]);
    }

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        if(!quiet) {
            printf("Pass %d: ", i);
        }
        particlefilter_float(resultDB, args, useGraph);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}

void particlefilter_float(ResultDatabase &resultDB, int args[], bool useGraph) {

    int IszX, IszY, Nfr, Nparticles;
	IszX = args[0];
	IszY = args[1];
    Nfr = args[2];
    Nparticles = args[3];

    //establish seed
    int * seed = (int *) malloc(sizeof (int) *Nparticles);
    int i;
    for (i = 0; i < Nparticles; i++)
        seed[i] = time(0) * i;
    //malloc matrix
    unsigned char * I = (unsigned char *) malloc(sizeof (unsigned char) *IszX * IszY * Nfr);
    //call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    //call particle filter
    if (useGraph) particleFilterGraph(I, IszX, IszY, Nfr, seed, Nparticles, resultDB);
    else particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, resultDB);

    free(seed);
    free(I);
}
