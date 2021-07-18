cmake_minimum_required(VERSION 3.2)

include_directories(${OCLABC_ROOT}/core/include)
include_directories(${OCLABC_ROOT}/third_party/libopencl-stub/include)
include_directories(${OCLABC_ROOT}/third_party/half-float/include)

add_executable(api_example api_example.cpp)
target_link_libraries(api_example oclabc_core)
install(TARGETS api_example
        RUNTIME DESTINATION examples)

add_executable(deconv_f2s2_nchw deconv_f2s2_nchw.cpp)
target_link_libraries(deconv_f2s2_nchw oclabc_core)
install(TARGETS deconv_f2s2_nchw
        RUNTIME DESTINATION examples)
