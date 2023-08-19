cmake_minimum_required(VERSION 3.10)

project(Altis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER icpx)

#
# BUILD TYPE
#

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#
# GENERAL C++ FLAGS
#

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -Wshadow -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-unknown-cuda-version -Wno-unused -Wno-unused-parameter")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-fsycl ${OPT_FLAGS} ${WARNING_FLAGS} ${DISABLED_WARNINGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS}  ${DISABLED_WARNINGS} -g3 -Og")

#
# BSP LOCATIONS AND C++ CONSTANTS
#

set(USE_CUDA 1)
#set(USE_FPGA 1)
if(DEFINED USE_CUDA)
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl -fgpu-inline-threshold=100000 -fsycl-targets=nvptx64-nvidia-cuda -sycl-std=2020 -fsycl-unnamed-lambda -Xsycl-target-backend --cuda-gpu-arch=sm_75 -fsycl-id-queries-fit-in-int --cuda-path=/opt/cuda/")
    SET(CMAKE_CXX_FLAGS " -fsycl -fgpu-inline-threshold=100000 -fsycl-targets=nvptx64-nvidia-cuda -sycl-std=2020 -fsycl-unnamed-lambda -Xsycl-target-backend --cuda-gpu-arch=sm_75 -fsycl-id-queries-fit-in-int --cuda-path=/opt/cuda/")
    SET(CMAKE_EXE_LINKER_FLAGS "-lstdc++ -lm")
    ADD_DEFINITIONS(-D_CUDA)
elseif(DEFINED USE_FPGA)
    # This constant is used in cpp code to e.g. turn on/off kernel attributes that could make the code no longer executable on other accelerators
    ADD_DEFINITIONS( -D_FPGA=1 )
    
    #
    # BSP LOCATIONS AND C++ CONSTANTS
    #

    ADD_DEFINITIONS( -D_FPGA=1 )
    set(USE_STRATIX10 1)
    #set(USE_AGILEX 1)

    if(DEFINED USE_STRATIX10)
        ADD_DEFINITIONS( -D_STRATIX10=1 )
        # Use this on Intel DevCloud
        #set(FPGA_BOARD "/opt/intel/oneapi/intel_s10sx_pac:pac_s10")
        # And this on Paderborn PC2
        set(FPGA_BOARD "/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10_hpc_default:p520_hpc_sg280l")
    elseif(DEFINED USE_AGILEX)
        ADD_DEFINITIONS( -D_AGILEX=1 )
        set(FPGA_BOARD "$PC2DATA/hpc-prf-altisfp/de10_bsp.txz")
    endif()
    message(STATUS "Building for FPGA Board ${FPGA_BOARD}...")

    #
    # INTEL FPGA SPECIFIC FLAGS
    #

    # Compile all the other files also with this flag, so they appear in the report.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl -fintelfpga")

    # INFO: For faster compilation at cost of speed, use -Xsfast-compile
    set(EMULATOR_COMPILE_FLAGS "-Wall -fintelfpga -qactypes -DFPGA_EMULATOR -pthread")
    set(EMULATOR_LINK_FLAGS "-fintelfpga -qactypes -lpthread")
    set(HARDWARE_COMPILE_FLAGS "-fsycl -Wall -fintelfpga -qactypes -pthread")
    set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -qactypes -Xshardware -Xsboard=${FPGA_BOARD} -Xsdont-error-if-large-area-est -lpthread")
endif()

add_subdirectory(common)
add_subdirectory(cuda)
