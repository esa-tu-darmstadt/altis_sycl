cmake_minimum_required(VERSION 3.10)

project(Altis_FPGA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER icpx)

#
# BUILD TYPE
#

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

#
# GENERAL C++ FLAGS
#

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -Wshadow -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-unused -Wno-unused-parameter")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-fsycl ${OPT_FLAGS} ${WARNING_FLAGS} ${DISABLED_WARNINGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS}  ${DISABLED_WARNINGS} -g3 -Og")

#
# BSP LOCATIONS AND C++ CONSTANTS
#

set(USE_FPGA 1)
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
    set(FPGA_BOARD "/pc2/groups/hpc-prf-altisfp/de10_agilex:B2E2_8GBx4")
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
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga -qactypes -Xshardware -Xsboard=${FPGA_BOARD} -lpthread")

add_subdirectory(common)
add_subdirectory(cuda)
