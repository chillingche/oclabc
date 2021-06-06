cmake_minimum_required(VERSION 3.2)
add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(OpenCL SHARED ${SRCS})
target_include_directories(OpenCL PRIVATE "include")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(OpenCL PRIVATE -O2 -fPIC -Wall)
endif()

# install(TARGETS OpenCL
#         LIBRARY DESTINATION lib)
