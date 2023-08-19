cmake_minimum_required(VERSION 3.10)

project(Altis LANGUAGES CXX)

#set(USE_CUDA 1)
#set(USE_FPGA 1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER dpcpp)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -Wshadow -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-unknown-cuda-version -Wno-unused -Wno-unused-parameter")

set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

set(CMAKE_CXX_FLAGS_RELEASE "${OPT_FLAGS} ${WARNING_FLAGS} ${DISABLED_WARNINGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS}  ${DISABLED_WARNINGS} -g3 -Og")

if(DEFINED USE_CUDA)
    ADD_DEFINITIONS( -D_CUDA=1 )

    SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -sycl-std=2020 -fsycl-unnamed-lambda")
    SET(CMAKE_EXE_LINKER_FLAGS "-lstdc++ -lm -lonemkl_blas_cublas -lonemkl")

    link_libraries(libonemkl.so.0 libonemkl_blas_cublas.so.0)

    include_directories($ENV{DPCPP_HOME}/deploy/include)
    link_directories($ENV{DPCPP_HOME}/deploy/lib)

    include_directories(
        "/opt/intel/oneapi/dpcpp-ct/2022.1.0/include/"
        "$ENV{DPCPP_HOME}/deploy/include/sycl/"
    )
elseif(DEFINED USE_FPGA)
    ADD_DEFINITIONS( -D_FPGA=1 )

    add_link_options(
        -lOpenCL
    )
    
    set(FPGA_BOARD "/opt/intel/oneapi/intel_a10gx_pac:pac_a10")
    #set(FPGA_BOARD "/opt/intel/oneapi/intel_s10sx_pac:pac_s10")
    message(STATUS "Building for FPGA Board ${FPGA_BOARD}...")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fintelfpga")

    # -Xsfast-compile
    set(EMULATOR_COMPILE_FLAGS "-Wall -fintelfpga -DFPGA_EMULATOR")
    set(EMULATOR_LINK_FLAGS "-fintelfpga")
    set(HARDWARE_COMPILE_FLAGS "-fsycl -Wall -fintelfpga")
    set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -Xshardware -Xsboard=${FPGA_BOARD}")
else()
    add_link_options(
        -lOpenCL
    )
endif()

add_subdirectory(common)
add_subdirectory(cuda)
