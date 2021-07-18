#include "cl_runtime.h"
#include "log.h"

#include <iostream>
#include <string>
#include <vector>

using abc::clrt;

int main(int argc, char const* argv[]) {
    cl_int result = 0;
    result = clrt().init();
    CHECK_ERROR_NO_RETURN(result == CL_SUCCESS, "Failed to init CLRuntime.");
    size_t reqd_size = 0;
    result = clGetDeviceInfo(clrt().deviceId(), CL_DEVICE_EXTENSIONS, 0, NULL,
                             &reqd_size);
    CHECK_ERROR_NO_RETURN(reqd_size > 0u && result == CL_SUCCESS,
                          "Failed to get cl extensions size.");

    std::vector<char> buf(reqd_size);
    result = clGetDeviceInfo(clrt().deviceId(), CL_DEVICE_EXTENSIONS, reqd_size,
                             buf.data(), NULL);
    CHECK_ERROR_NO_RETURN(result == CL_SUCCESS,
                          "Failed to read cl extensions.");

    std::string extensions(buf.data());
    std::cout << extensions << std::endl;
    return result;
}
