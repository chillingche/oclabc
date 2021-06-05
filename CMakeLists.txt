cmake_minimum_required(VERSION 3.2)

project(OclABC C CXX)

set(OCLABC_ROOT $ENV{OCLABC_ROOT})
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")

add_subdirectory(third_party)
add_subdirectory(test)
