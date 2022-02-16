cmake_minimum_required(VERSION 3.2)

project(oclabc C CXX)

set(OCLABC_ROOT $ENV{OCLABC_ROOT})
set(CMAKE_C_FLAGS "-std=gnu99 -march=armv8.2-a+fp16+dotprod")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=armv8.2-a+fp16+dotprod")
set(CMAKE_BUILD_TYPE "Release")
add_definitions(-DCL_TARGET_OPENCL_VERSION=200)

add_subdirectory(third_party)
add_subdirectory(core)
add_subdirectory(examples)
