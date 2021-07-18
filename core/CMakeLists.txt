cmake_minimum_required(VERSION 3.2)

project(oclabc_core)

file (GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src *.cpp)
file (GLOB HALF_FLOAT_SRCS ${OCLABC_ROOT}/third_party/half-float/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRCS} ${HALF_FLOAT_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_include_directories(${PROJECT_NAME} PRIVATE "${OCLABC_ROOT}/third_party/libopencl-stub/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${OCLABC_ROOT}/third_party/half-float/include")

target_link_libraries(${PROJECT_NAME} OpenCL)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
