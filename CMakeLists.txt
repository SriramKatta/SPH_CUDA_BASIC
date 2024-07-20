cmake_minimum_required(VERSION 3.20)

project(SPH LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-Wall --std c++20) 

include(external/thrust.cmake)

include_directories(${CMAKE_SOURCE_DIR}/header)
add_subdirectory(${CMAKE_SOURCE_DIR}/impl)