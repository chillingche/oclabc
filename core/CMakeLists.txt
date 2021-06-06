cmake_minimum_required(VERSION 3.2)

project(oclabc_core)

file (GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_include_directories(${PROJECT_NAME} PUBLIC "${OCLABC_ROOT}/third_party/libopencl-stub/include")

target_link_libraries(${PROJECT_NAME} OpenCL)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
