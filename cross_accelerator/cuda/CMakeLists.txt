cmake_minimum_required (VERSION 3.8)

include_directories(common ../common)

if(DEFINED USE_CUDA)
    include_directories(/opt/cuda/targets/x86_64-linux/include)

    set(NVML_LIB
        /opt/cuda/targets/x86_64-linux/lib/stubs/)

    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lnvidia-ml -L ${NVML_LIB}")
endif()

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/main.dp.cpp
)

add_compile_options(-w) # temporarily surpress warnings

#add_subdirectory(level0)
add_subdirectory(level1)
add_subdirectory(level2)
