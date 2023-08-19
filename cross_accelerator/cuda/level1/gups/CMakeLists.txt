cmake_minimum_required(VERSION 3.8)

add_library(gupsLib gups.dp.cpp)
target_link_libraries(gupsLib -lm)
