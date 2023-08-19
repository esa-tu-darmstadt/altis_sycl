/* This header file is used for NVIDIA GeForce RTX 2080. */
#ifndef __CUDA_DEVICE_ATTR_H__
#define __CUDA_DEVICE_ATTR_H__

#define CUDA_DEVICE_NUM    1
#define CUDA_DEVICE_ID    0
#define CUDA_DEVICE_NAME    "NVIDIA GeForce RTX 2080"
#define CUDA_MAJOR_VERSION    11
#define CUDA_MINOR_VERSION    6
#define CUDA_MAJOR_CAPABILITY    7
#define CUDA_MINOR_CAPABILITY    5
#define GLOBAL_MEM    8368816128
#define SM_COUNT    46
#define CUDA_CORES_PER_SM    64
#define CUDA_CORES    2944
#define MEM_BUS_WIDTH    256
#define L2_CACHE_SIZE    4194304
#define MAX_TEXTURE_1D_DIM    131072
#define MAX_TEXTURE_2D_X    131072
#define MAX_TEXTURE_2D_Y    65536
#define MAX_TEXTURE_3D_X    16384
#define MAX_TEXTURE_3D_Y    16384
#define MAX_TEXTURE_3D_Z    16384
#define MAX_LAYERED_1D_TEXTURE_SIZE    32768
#define MAX_LAYERED_1D_TEXTURE_LAYERS    2048
#define MAX_LAYERED_2D_TEXTURE_SIZE_X    32768
#define MAX_TEXTURE_2D_TEXTURE_SIZE_Y    32768
#define MAX_TEXTURE_2D_TEXTURE_LAYERS    2048
#define CONST_MEM    65536
#define SHARED_MEM_PER_BLOCK    49152
#define SHARED_MEM_PER_SM    65536
#define SHARED_MEMORY_BANKS    32
#define SHARED_MEMORY_BANK_BANDWIDTH    4 // Each bank has a bandwidth of 32 bits per clock cycle (no doc)
#define REGS_PER_BLOCK    65536
#define WARP_SIZE    32
#define MAX_THREADS_PER_SM    1024
#define MAX_THREADS_PER_BLOCK    1024   // For P630 make this 256
#define MAX_THREADS_DIM_X    1024       // For P630 make this 256
#define MAX_THREADS_DIM_Y    1024       // For P630 make this 256
#define MAX_THREADS_DIM_Z    64// For P630 make this 256
#define MAX_GRIDS_DIM_X    2147483647
#define MAX_GRIDS_DIM_Y    65535
#define MAX_GRIDS_DIM_Z    65535
#define MEM_PITCH    1024
#define TEXTURE_ALIGNMENT    512

#define PAGE_LOCKED_MEM
#define UVA
#define MANAGED_MEM
#define COMPUTE_PREEMPTION
#define COOP_KERNEL
#define MULTI_DEVICE_COOP_KERNEL

#endif
