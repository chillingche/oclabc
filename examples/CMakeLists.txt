cmake_minimum_required(VERSION 3.2)

include_directories(${OCLABC_ROOT}/core/include)

add_executable(api_example api_example.cpp)
target_link_libraries(api_example oclabc_core)
install(TARGETS api_example
        RUNTIME DESTINATION examples)
