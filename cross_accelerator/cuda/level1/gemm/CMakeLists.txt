cmake_minimum_required(VERSION 3.8)

add_library(gemmLib Gemm.dp.cpp)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    target_compile_options(gemmLib PRIVATE)
endif()
target_link_libraries(gemmLib)
