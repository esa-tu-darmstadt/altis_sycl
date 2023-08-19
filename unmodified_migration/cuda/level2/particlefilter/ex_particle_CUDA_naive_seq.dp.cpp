/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\particlefilter\ex_particle_CUDA_naive_seq.cu
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
#include <sys/time.h>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	PI
///
/// @brief	A macro that defines pi
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define PI 3.1415926535897932

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BLOCK_X
///
/// @brief	A macro that defines block X coordinate
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_X 16

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BLOCK_Y
///
/// @brief	A macro that defines block Y coordinate
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_Y 16

/// @brief	True to verbose
bool verbose = false;
/// @brief	True to quiet
bool quiet = false;

/// @brief	@var M value for Linear Congruential Generator (LCG); use GCC's value
long M = INT_MAX;
/// @brief	@var A value for LCG
int A = 1103515245;
/// @brief	@var C value for LCG
int C = 12345;

/// @brief	The threads per block
const int threads_per_block = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int findIndexSeq(double * CDF, int lengthCDF, double value)
///
/// @brief	Searches for the first index sequence
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	CDF		 	If non-null, the cdf. 
/// @param 		   	lengthCDF	The length cdf. 
/// @param 		   	value	 	The value. 
///
/// @returns	The found index sequence.
////////////////////////////////////////////////////////////////////////////////////////////////////

int findIndexSeq(double * CDF, int lengthCDF, double value)
{
	int index = -1;
	int x;
	for(x = 0; x < lengthCDF; x++)
	{
		if(CDF[x] >= value)
		{
			index = x;
			break;
		}
	}
	if(index == -1)
		return lengthCDF-1;
	return index;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int findIndexBin(double * CDF, int beginIndex, int endIndex, double value)
///
/// @brief	Searches for the first index bin
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	CDF		  	If non-null, the cdf. 
/// @param 		   	beginIndex	Zero-based index of the begin. 
/// @param 		   	endIndex  	The end index. 
/// @param 		   	value	  	The value. 
///
/// @returns	The found index bin.
////////////////////////////////////////////////////////////////////////////////////////////////////

int findIndexBin(double * CDF, int beginIndex, int endIndex, double value)
{
	if(endIndex < beginIndex)
		return -1;
	int middleIndex;
	while(endIndex > beginIndex)
	{
		middleIndex = beginIndex + ((endIndex-beginIndex)/2);
		if(CDF[middleIndex] >= value)
		{
			if(middleIndex == 0)
				return middleIndex;
			else if(CDF[middleIndex-1] < value)
				return middleIndex;
			else if(CDF[middleIndex-1] == value)
			{
				while(CDF[middleIndex] == value && middleIndex >= 0)
					middleIndex--;
				middleIndex++;
				return middleIndex;
			}
		}
		if(CDF[middleIndex] > value)
			endIndex = middleIndex-1;
		else
			beginIndex = middleIndex+1;
	}
	return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	* CUDA Kernel Function to replace FindIndex * param1: arrayX * param2: arrayY * param3: CDF * param4: u * param5: xj * param6: yj * param7: Nparticles *****************************/ __global__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles)
///
/// @brief	Kernels
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	arrayX	  	If non-null, the array x coordinate. 
/// @param [in,out]	arrayY	  	If non-null, the array y coordinate. 
/// @param [in,out]	CDF		  	If non-null, the cdf. 
/// @param [in,out]	u		  	If non-null, a double to process. 
/// @param [in,out]	xj		  	If non-null, the xj. 
/// @param [in,out]	yj		  	If non-null, the yj. 
/// @param 		   	Nparticles	The nparticles. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles,
            const sycl::nd_item<3> &item_ct1) {
        int block_id = item_ct1.get_group(2); // + gridDim.x * blockIdx.y;
        int i = item_ct1.get_local_range(2) * block_id + item_ct1.get_local_id(2);

        if(i < Nparticles){
	
		int index = -1;
		int x;
		
		for(x = 0; x < Nparticles; x++){
			if(CDF[x] >= u[i]){
				index = x;
				break;
			}
		}
		if(index == -1){
			index = Nparticles-1;
		}
		
		xj[i] = arrayX[index];
		yj[i] = arrayY[index];
		
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	double roundDouble(double value)
///
/// @brief	Takes in a double and returns an integer that approximates to that double
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	value	The value. 
///
/// @returns	if the mantissa &lt; .5 =&gt; return value &lt; input value; else return value
/// 			&gt; input value.
////////////////////////////////////////////////////////////////////////////////////////////////////

double roundDouble(double value){
	int newValue = (int)(value);
	if(value - newValue < .5)
	return newValue;
	else
	return newValue++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ)
///
/// @brief	Set values of the 3D array to a newValue if that value is equal to the testValue
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 		   	testValue	The value to be replaced. 
/// @param 		   	newValue 	The value to replace testValue with. 
/// @param [in,out]	array3D  	The image vector. 
/// @param [in,out]	dimX	 	The x dimension of the frame. 
/// @param [in,out]	dimY	 	The y dimension of the frame. 
/// @param [in,out]	dimZ	 	The number of frames. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	double randu(int * seed, int index)
///
/// @brief	Generates a uniformly distributed random number using the provided seed and GCC's
/// 		settings for the Linear Congruential Generator (LCG)
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	seed 	The seed array. 
/// @param 		   	index	The specific index of the seed to be advanced. 
///
/// @returns	a uniformly distributed number [0, 1)
///
/// @sa	http://en.wikipedia.org/wiki/Linear_congruential_generator
/// 	@note This function is thread-safe
////////////////////////////////////////////////////////////////////////////////////////////////////

double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	double randn(int * seed, int index)
///
/// @brief	Generates a normally distributed random number using the Box-Muller transformation
/// 		@note This function is thread-safe
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	seed 	The seed array. 
/// @param 		   	index	The specific index of the seed to be advanced. 
///
/// @returns	a double representing random number generated using the Box-Muller algorithm.
///
/// @sa	http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal
/// 	random distribution
////////////////////////////////////////////////////////////////////////////////////////////////////

double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed)
///
/// @brief	Sets values of 3D matrix using randomly generated numbers from a normal distribution
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	array3D	The video to be modified. 
/// @param [in,out]	dimX   	The x dimension of the frame. 
/// @param [in,out]	dimY   	The y dimension of the frame. 
/// @param [in,out]	dimZ   	The number of frames. 
/// @param [in,out]	seed   	The seed array. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn(seed, 0));
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void strelDisk(int * disk, int radius)
///
/// @brief	Fills a radius x radius matrix representing the disk
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	disk  	The pointer to the disk to be made. 
/// @param 		   	radius	The radius of the disk to be made. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void strelDisk(int * disk, int radius)
{
	int diameter = radius*2 - 1;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
                        double distance = sqrt((double)(x - radius + 1) *
                                                   (double)(x - radius + 1) +
                                               (double)(y - radius + 1) *
                                                   (double)(y - radius + 1));
                        if(distance < radius) {
			    disk[x*diameter + y] = 1;
            } else {
			    disk[x*diameter + y] = 0;
            }
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
///
/// @brief	Dilates the provided video
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	matrix	The video to be dilated. 
/// @param 		   	posX  	The x location of the pixel to be dilated. 
/// @param 		   	posY  	The y location of the pixel to be dilated. 
/// @param 		   	posZ  	The z location of the pixel to be dilated. 
/// @param 		   	dimX  	The x dimension of the frame. 
/// @param 		   	dimY  	The y dimension of the frame. 
/// @param 		   	dimZ  	The number of frames. 
/// @param 		   	error 	The error radius. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
	int startX = posX - error;
	while(startX < 0)
	startX++;
	int startY = posY - error;
	while(startY < 0)
	startY++;
	int endX = posX + error;
	while(endX > dimX)
	endX--;
	int endY = posY + error;
	while(endY > dimY)
	endY--;
	int x,y;
	for(x = startX; x < endX; x++){
		for(y = startY; y < endY; y++){
                        double distance =
                            sqrt((double)(x - posX) * (double)(x - posX) +
                                 (double)(y - posY) * (double)(y - posY));
                        if(distance < error)
			matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
///
/// @brief	Dilates the target matrix using the radius as a guide
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	matrix   	The reference matrix. 
/// @param 		   	dimX	 	The x dimension of the video. 
/// @param 		   	dimY	 	The y dimension of the video. 
/// @param 		   	dimZ	 	The z dimension of the video. 
/// @param 		   	error	 	The error radius to be dilated. 
/// @param [in,out]	newMatrix	The target matrix. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
{
	int x, y, z;
	for(z = 0; z < dimZ; z++){
		for(x = 0; x < dimX; x++){
			for(y = 0; y < dimY; y++){
				if(matrix[x*dimY*dimZ + y*dimZ + z] == 1){
					dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void getneighbors(int * se, int numOnes, double * neighbors, int radius)
///
/// @brief	Fills a 2D array describing the offsets of the disk object
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	se		 	The disk object. 
/// @param 		   	numOnes  	The number of ones in the disk. 
/// @param [in,out]	neighbors	The array that will contain the offsets. 
/// @param 		   	radius   	The radius used for dilation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void getneighbors(int * se, int numOnes, double * neighbors, int radius){
	int x, y;
	int neighY = 0;
	int center = radius - 1;
	int diameter = radius*2 -1;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(se[x*diameter + y]){
				neighbors[neighY*2] = (int)(y - center);
				neighbors[neighY*2 + 1] = (int)(x - center);
				neighY++;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed)
///
/// @brief	The synthetic video sequence we will work with here is composed of a single moving
/// 		object, circular in shape (fixed radius)
/// 		The motion here is a linear motion the foreground intensity and the backgrounf
/// 		intensity is known the image is corrupted with zero mean Gaussian noise
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	I   	The video itself. 
/// @param 		   	IszX	The x dimension of the video. 
/// @param 		   	IszY	The y dimension of the video. 
/// @param 		   	Nfr 	The number of frames of the video. 
/// @param [in,out]	seed	The seed array used for number generation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void videoSequence(OptionParser &op, int * I, int IszX, int IszY, int Nfr, int * seed) {
	const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
        checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

        int k;
	int max_size = IszX*IszY*Nfr;
	/*get object centers*/
	int x0 = (int)roundDouble(IszY/2.0);
	int y0 = (int)roundDouble(IszX/2.0);
	I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;
	
	/*move point*/
	int xk, yk, pos;
	for(k = 1; k < Nfr; k++){
		xk = abs(x0 + (k-1));
		yk = abs(y0 - 2*(k-1));
		pos = yk * IszY * Nfr + xk *Nfr + k;
		if(pos >= max_size)
		pos = 0;
		I[pos] = 1;
	}
	
	/*dilate matrix*/
	int *newMatrix = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:469: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(newMatrix = (int *)sycl::malloc_shared(
                                         sizeof(int) * IszX * IszY * Nfr,
                                         dpct::get_default_queue())));
        }
	else {
		newMatrix = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
		assert(newMatrix);
	}
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for(x = 0; x < IszX; x++){
		for(y = 0; y < IszY; y++){
			for(k = 0; k < Nfr; k++){
				I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
			}
		}
	}
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(newMatrix, dpct::get_default_queue())));
        }
	else {
		free(newMatrix);
	}
	
	/*define background, add noise*/
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	/*add noise*/
	addNoise(I, &IszX, &IszY, &Nfr, seed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	double calcLikelihoodSum(int * I, int * ind, int numOnes)
///
/// @brief	Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND]
/// 		- 228)^2)/ 100
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	I	   	The 3D matrix. 
/// @param [in,out]	ind	   	The current ind array. 
/// @param 		   	numOnes	The length of ind array. 
///
/// @returns	A double representing the sum.
////////////////////////////////////////////////////////////////////////////////////////////////////

double calcLikelihoodSum(int * I, int * ind, int numOnes){
	double likelihoodSum = 0.0;
	int y;
	for(y = 0; y < numOnes; y++)
        likelihoodSum +=
            ((double)(I[ind[y]] - 100) * (double)(I[ind[y]] - 100) -
             (double)(I[ind[y]] - 228) * (double)(I[ind[y]] - 228)) /
            50.0;
        return likelihoodSum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	int findIndex(double * CDF, int lengthCDF, double value)
///
/// @brief	Finds the first element in the CDF that is greater than or equal to the provided
/// 		value and returns that index
/// 		@note This function uses sequential search
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	CDF		 	The CDF. 
/// @param 		   	lengthCDF	The length of CDF. 
/// @param 		   	value	 	The value to be found. 
///
/// @returns	The index of value in the CDF; if value is never found, returns the last index.
////////////////////////////////////////////////////////////////////////////////////////////////////

int findIndex(double * CDF, int lengthCDF, double value){
	int index = -1;
	int x;
	for(x = 0; x < lengthCDF; x++){
		if(CDF[x] >= value){
			index = x;
			break;
		}
	}
	if(index == -1){
		return lengthCDF-1;
	}
	return index;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, ResultDatabase &resultDB)
///
/// @brief	The implementation of the particle filter using OpenMP for many frames
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	I		  	The video to be run. 
/// @param 		   	IszX	  	The x dimension of the video. 
/// @param 		   	IszY	  	The y dimension of the video. 
/// @param 		   	Nfr		  	The number of frames. 
/// @param [in,out]	seed	  	The seed array used for random number generation. 
/// @param 		   	Nparticles	The number of particles to be used. 
/// @param [in,out]	resultDB  	The result database. 
///
/// @sa	http://openmp.org/wp/
/// 	@note This function is designed to work with a video of several frames. In addition, it
/// 	references a provided MATLAB function which takes the video, the objxy matrix and the x
/// 	and y arrays as arguments and returns the likelihoods
////////////////////////////////////////////////////////////////////////////////////////////////////

void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, OptionParser &op, ResultDatabase &resultDB) {
	const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
        checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));
    float elapsedTime;

	int max_size = IszX*IszY*Nfr;
	//original particle centroid
	double xe = roundDouble(IszY/2.0);
	double ye = roundDouble(IszX/2.0);
	
	//expected object locations, compared to center
	int radius = 5;
	int diameter = radius*2 - 1;
	int *disk = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:470: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    disk = sycl::malloc_shared<int>(
                        diameter * diameter, dpct::get_default_queue())));
        } else {
		disk = (int *)malloc(diameter*diameter*sizeof(int));
		assert(disk);
	}
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(disk[x*diameter + y] == 1)
				countOnes++;
		}
	}
	double *objxy = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:471: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    objxy = sycl::malloc_shared<double>(
                        countOnes * 2, dpct::get_default_queue())));
        } else {
		objxy = (double *)malloc(countOnes*2*sizeof(double));
		assert(objxy);
	}
	getneighbors(disk, countOnes, objxy, radius);
	
	//initial weights are all equal (1/Nparticles)
	double *weights = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:472: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    weights = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
        } else {
		weights = (double *)malloc(sizeof(double)*Nparticles);
		assert(weights);
	}
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	//initial likelihood to 0.0
	double *likelihood = NULL;
	double *arrayX = NULL;
	double *arrayY = NULL;
	double *xj = NULL;
	double *yj = NULL;
	double *CDF = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:473: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    likelihood = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:474: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    arrayX = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:475: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    arrayY = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:476: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    xj = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:477: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    yj = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:478: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    CDF = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
        } else {
		likelihood = (double *)malloc(sizeof(double)*Nparticles);
		assert(likelihood);
		arrayX = (double *)malloc(sizeof(double)*Nparticles);
		assert(arrayX);
		arrayY = (double *)malloc(sizeof(double)*Nparticles);
		assert(arrayY);
		xj = (double *)malloc(sizeof(double)*Nparticles);
		assert(xj);
		yj = (double *)malloc(sizeof(double)*Nparticles);
		assert(yj);
		CDF = (double *)malloc(sizeof(double)*Nparticles);
		assert(CDF);
	}
	
	//GPU copies of arrays
	int *ind = NULL;
	double *u = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:479: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    ind = sycl::malloc_shared<int>(countOnes,
                                                   dpct::get_default_queue())));
                /*
                DPCT1064:480: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    u = sycl::malloc_shared<double>(
                        Nparticles, dpct::get_default_queue())));
        } else {
		ind = (int*)malloc(sizeof(int)*countOnes);
		assert(ind);
		u = (double *)malloc(sizeof(double)*Nparticles);
		assert(u);
	}

	double * arrayX_GPU;
	double * arrayY_GPU;
	double * xj_GPU;
	double * yj_GPU;
	double * CDF_GPU;
	double * u_GPU;
	
	//CUDA memory allocation
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
		arrayX_GPU = arrayX;
		arrayY_GPU = arrayY;
		xj_GPU = xj;
		yj_GPU = yj;
		CDF_GPU = CDF;
		u_GPU = u;
	} else {
                /*
                DPCT1064:481: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    arrayX_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:482: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    arrayY_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:483: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    xj_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:484: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    yj_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:485: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    CDF_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
                /*
                DPCT1064:486: Migrated cudaMalloc call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    u_GPU = sycl::malloc_device<double>(
                        Nparticles, dpct::get_default_queue())));
        }
	
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	//double * Ik = (double *)malloc(sizeof(double)*IszX*IszY);
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction
		
		for(x = 0; x < Nparticles; x++){
			arrayX[x] = arrayX[x] + 1.0 + 5.0*randn(seed, x);
			arrayY[x] = arrayY[x] - 2.0 + 2.0*randn(seed, x);
		}
		//particle filter likelihood
		for(x = 0; x < Nparticles; x++){
		
			//compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.		
			//calc ind
			for(y = 0; y < countOnes; y++){
				indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
				indY = roundDouble(arrayY[x]) + objxy[y*2];
				ind[y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
				if(ind[y] >= max_size)
					ind[y] = 0;
			}
			likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
			likelihood[x] = likelihood[x]/countOnes;
		}
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial		
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		double sumWeights = 0;	
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		for(x = 0; x < Nparticles; x++){
				weights[x] = weights[x]/sumWeights;
		}
		xe = 0;
		ye = 0;
		// estimate the object location by expected values
		for(x = 0; x < Nparticles; x++){
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
        if(verbose && !quiet) {
            printf("XE: %lf\n", xe);
            printf("YE: %lf\n", ye);
            double distance = sqrt(
                sycl::pown((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                sycl::pown((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
            printf("%lf\n", distance);
        }
		//display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}
		//CUDA memory copying from CPU memory to GPU memory
        /*
        DPCT1012:429: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:430: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
                // Use demand paging, or hyperq async cpy
		if (uvm) {

		} else if (uvm_advise) {
                        /*
                        DPCT1063:431: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayX_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:432: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayX_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:433: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayY_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:434: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayY_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:435: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                xj_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:436: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                yj_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:437: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                CDF_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:438: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                CDF_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:439: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                u_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:440: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                u_GPU, sizeof(double) * Nparticles, 0)));
                } else if (uvm_prefetch) {
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::dev_mgr::instance()
                                .get_device(device)
                                .default_queue()
                                .prefetch(arrayX_GPU,
                                          sizeof(double) * Nparticles)));
                        dpct::queue_ptr s1;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s1 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(s1->prefetch(
                            arrayY_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s2;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s2 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s2->prefetch(xj_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s3;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s3 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s3->prefetch(yj_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s4;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s4 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s4->prefetch(CDF, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s5;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s5 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s5->prefetch(u_GPU, sizeof(double) * Nparticles)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s1)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s2)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s3)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s4)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s5)));
                } else if (uvm_prefetch_advise) {
                        /*
                        DPCT1063:441: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayX_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:442: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayX_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:443: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayY_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:444: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                arrayY_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:445: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                xj_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:446: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                yj_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:447: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                CDF_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:448: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                CDF_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:449: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                u_GPU, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:450: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_device(device).default_queue().mem_advise(
                                u_GPU, sizeof(double) * Nparticles, 0)));

                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::dev_mgr::instance()
                                .get_device(device)
                                .default_queue()
                                .prefetch(arrayX_GPU,
                                          sizeof(double) * Nparticles)));
                        dpct::queue_ptr s1;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s1 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(s1->prefetch(
                            arrayY_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s2;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s2 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s2->prefetch(xj_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s3;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s3 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s3->prefetch(yj_GPU, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s4;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s4 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s4->prefetch(CDF, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s5;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s5 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s5->prefetch(u_GPU, sizeof(double) * Nparticles)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s1)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s2)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s3)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s4)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s5)));
                } else {
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(arrayX_GPU, arrayX,
                                        sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(arrayY_GPU, arrayY,
                                        sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(xj_GPU, xj, sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(yj_GPU, yj, sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(CDF_GPU, CDF,
                                        sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(u_GPU, u, sizeof(double) * Nparticles)
                                .wait()));
                }

                /*
                DPCT1012:451: Detected kernel execution time measurement pattern
                and generated an initial code for time measurements in SYCL. You
                can change the way time is measured depending on your goals.
                */
                /*
                DPCT1024:452: The original code returned the error code that was
                further consumed by the program logic. This original code was
                replaced with 0. You may need to rewrite the program logic
                consuming the error code.
                */
                stop_ct1 = std::chrono::steady_clock::now();
                checkCudaErrors(0);
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            (elapsedTime =
                 std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));
        transferTime += elapsedTime * 1.e-3;
		//Set number of threads
		int num_blocks = ceil((double) Nparticles/(double) threads_per_block);
		
		//KERNEL FUNCTION CALL
        /*
        DPCT1012:453: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:454: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(
            DPCT_CHECK_ERROR(*start = dpct::get_default_queue()
                                          .ext_oneapi_submit_barrier()));
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
                            kernel(arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU,
                                   xj_GPU, yj_GPU, Nparticles, item_ct1);
                      });
            }
        /*
        DPCT1012:455: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:456: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(DPCT_CHECK_ERROR(
                *stop = dpct::get_default_queue().ext_oneapi_submit_barrier()));
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            (elapsedTime =
                 std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR();

        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
                //CUDA memory copying back from GPU to CPU memory
        /*
        DPCT1012:457: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:458: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        checkCudaErrors(0);
                if (uvm) {

		} else if (uvm_advise) {
                        /*
                        DPCT1063:459: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                yj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:460: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                yj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:461: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                xj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:462: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                xj, sizeof(double) * Nparticles, 0)));
                } else if (uvm_prefetch) {
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().prefetch(
                                yj, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s1;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s1 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().prefetch(
                                xj, sizeof(double) * Nparticles)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s1)));
                } else if (uvm_prefetch_advise) {
                        /*
                        DPCT1063:463: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                yj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:464: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                yj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:465: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                xj, sizeof(double) * Nparticles, 0)));
                        /*
                        DPCT1063:466: Advice parameter is device-defined and was
                        set to 0. You may need to adjust it.
                        */
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().mem_advise(
                                xj, sizeof(double) * Nparticles, 0)));

                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().prefetch(
                                yj, sizeof(double) * Nparticles)));
                        dpct::queue_ptr s1;
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            s1 = dpct::get_current_device().create_queue()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::cpu_device().default_queue().prefetch(
                                xj, sizeof(double) * Nparticles)));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_current_device().destroy_queue(s1)));
                } else {
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(yj, yj_GPU, sizeof(double) * Nparticles)
                                .wait()));
                        checkCudaErrors(DPCT_CHECK_ERROR(
                            dpct::get_default_queue()
                                .memcpy(xj, xj_GPU, sizeof(double) * Nparticles)
                                .wait()));
                }
                /*
                DPCT1012:467: Detected kernel execution time measurement pattern
                and generated an initial code for time measurements in SYCL. You
                can change the way time is measured depending on your goals.
                */
                /*
                DPCT1024:468: The original code returned the error code that was
                further consumed by the program logic. This original code was
                replaced with 0. You may need to rewrite the program logic
                consuming the error code.
                */
                stop_ct1 = std::chrono::steady_clock::now();
                checkCudaErrors(0);
        checkCudaErrors(0);
        checkCudaErrors(DPCT_CHECK_ERROR(
            (elapsedTime =
                 std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                     .count())));
        transferTime += elapsedTime * 1.e-3;
		
		for (x = 0; x < Nparticles; x++){
			//reassign arrayX and arrayY
			arrayX[x] = xj[x];
			arrayY[x] = yj[x];
			weights[x] = 1/((double)(Nparticles));
		}
	}
    
    char atts[1024];
    sprintf(atts, "dimx:%d, dimy:%d, numframes:%d, numparticles:%d", IszX, IszY, Nfr, Nparticles);
    resultDB.AddResult("particlefilter_naive_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("particlefilter_naive_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("particlefilter_naive_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("particlefilter_naive_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
	
	//CUDA freeing of memory
	if (!uvm && !uvm_advise && !uvm_prefetch && !uvm_prefetch_advise) {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(u_GPU, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(CDF_GPU, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(yj_GPU, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(xj_GPU, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(arrayY_GPU, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(arrayX_GPU, dpct::get_default_queue())));
        }
	
	//free memory
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(disk, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(objxy, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(weights, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(likelihood, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(arrayX, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(arrayY, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(xj, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(yj, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(CDF, dpct::get_default_queue())));
                checkCudaErrors(
                    DPCT_CHECK_ERROR(sycl::free(u, dpct::get_default_queue())));
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(ind, dpct::get_default_queue())));
        } else {
		free(disk);
		free(objxy);
		free(weights);
		free(likelihood);
		free(arrayX);
		free(arrayY);
		free(xj);
		free(yj);
		free(CDF);
		free(u);
		free(ind);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void addBenchmarkSpecOptions(OptionParser &op)
///
/// @brief	Adds a benchmark specifier options
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	op	The operation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("dimx", OPT_INT, "0", "grid x dimension", 'x');
  op.addOption("dimy", OPT_INT, "0", "grid y dimension", 'y');
  op.addOption("framecount", OPT_INT, "0", "number of frames to track across", 'f');
  op.addOption("np", OPT_INT, "0", "number of particles to use");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void particlefilter_naive(ResultDatabase &resultDB, int args[]);
///
/// @brief	Particlefilter naive
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database. 
/// @param 		   	args		The arguments. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void particlefilter_naive(ResultDatabase &resultDB, OptionParser &op, int args[]);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
///
/// @brief	Executes the benchmark operation
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database. 
/// @param [in,out]	op			The operation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running ParticleFilter (naive)\n");
    int args[4];
    args[0] = op.getOptionInt("dimx");
    args[1] = op.getOptionInt("dimy");
    args[2] = op.getOptionInt("framecount");
    args[3] = op.getOptionInt("np");
    bool preset = false;
    verbose = op.getOptionBool("verbose");
    quiet = op.getOptionBool("quiet");

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
        particlefilter_naive(resultDB, op, args);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void particlefilter_naive(ResultDatabase &resultDB, int args[])
///
/// @brief	Particlefilter naive
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database. 
/// @param 		   	args		The arguments. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void particlefilter_naive(ResultDatabase &resultDB, OptionParser &op, int args[]){
	const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(device = dpct::dev_mgr::instance().current_device_id());

        int IszX, IszY, Nfr, Nparticles;
	IszX = args[0];
	IszY = args[1];
    Nfr = args[2];
    Nparticles = args[3];

	//establish seed
	int *seed = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:487: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(DPCT_CHECK_ERROR(
                    seed = sycl::malloc_shared<int>(
                        Nparticles, dpct::get_default_queue())));
        } else {
		seed = (int *)malloc(sizeof(int)*Nparticles);
		assert(seed);
	}
	int i;
	for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
	//malloc matrix
	int *I = NULL;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                /*
                DPCT1064:488: Migrated cudaMallocManaged call is used in a
                macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                checkCudaErrors(
                    DPCT_CHECK_ERROR(I = (int *)sycl::malloc_shared(
                                         sizeof(int) * IszX * IszY * Nfr,
                                         dpct::get_default_queue())));
        } else {
		I = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
		assert(I);
	}
	//call video sequence
	videoSequence(op, I, IszX, IszY, Nfr, seed);
	//call particle filter
	particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, op, resultDB);
	
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
                checkCudaErrors(DPCT_CHECK_ERROR(
                    sycl::free(seed, dpct::get_default_queue())));
                checkCudaErrors(
                    DPCT_CHECK_ERROR(sycl::free(I, dpct::get_default_queue())));
        } else {
		free(seed);
		free(I);
	}
}
