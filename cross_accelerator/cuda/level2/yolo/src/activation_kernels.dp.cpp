#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <dpct/rng_utils.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

extern "C" {
#include "activations.h"
#include "cuda.h"
#include <cmath>
}


float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2);
    else return (x - n) + floorf(x/2);
}
 

float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}
float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}
float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
float ramp_gradient_kernel(float x){return (x>0)+.1f;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
float stair_gradient_kernel(float x)
{
    if (floorf(x) == x) return 0;
    return 1;
}

float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

float gradient_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case SELU:
            return selu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

void binary_gradient_array_kernel(float *x, float *dy, int n, int s, BINARY_ACTIVATION a, float *dx)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) {
        float de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s/2 + i] = x1*de; 
    }
}

extern "C" void binary_gradient_array_gpu(float *x, float *dx, int n, int size, BINARY_ACTIVATION a, float *y) 
{
    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, dx, n/2, size, a, y);
    /*
    DPCT1010:374: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}
void binary_activate_array_kernel(float *x, int n, int s, BINARY_ACTIVATION a, float *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}

extern "C" void binary_activate_array_gpu(float *x, int n, int size, BINARY_ACTIVATION a, float *y) 
{
    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, n/2, size, a, y);
    /*
    DPCT1010:375: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

void gradient_array_kernel(float *x, int n, ACTIVATION a, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}

extern "C" void activate_array_gpu(float *x, int n, ACTIVATION a) 
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    /*
    DPCT1010:376: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta) 
{
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
    /*
    DPCT1010:377: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}
