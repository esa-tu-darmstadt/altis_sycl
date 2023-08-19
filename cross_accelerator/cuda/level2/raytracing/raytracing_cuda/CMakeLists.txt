cmake_minimum_required(VERSION 3.8)

add_library(raytracingLib raytracing.dp.cpp)
target_link_libraries(raytracingLib -lm)
