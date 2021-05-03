cmake_minimum_required(VERSION 3.2)

project(test)
include_directories(${OCLABC_ROOT}/third_party/libopencl-stub/include)
add_executable(ocl_abc main.cpp)

target_link_libraries(ocl_abc OpenCL)
install(TARGETS ocl_abc
        RUNTIME DESTINATION test)