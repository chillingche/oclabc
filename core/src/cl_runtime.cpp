#include "cl_runtime.h"
#include "log.h"

#include <stdio.h>

#ifdef TAG
#undef TAG
#endif
#define TAG "cl_runtime"

namespace abc {

CLRuntime::~CLRuntime() {
    
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = NULL;
    }
    if (profile_queue_) {
        clReleaseCommandQueue(profile_queue_);
        profile_queue_ = NULL;
    }
    if (context_) {
        clReleaseContext(context_);
        context_ = NULL;
    }
    if (device_id_) {
        clReleaseDevice(device_id_);
        device_id_ = NULL;
    }
}

cl_int CLRuntime::init() {
    cl_int result = 0;
    result = clGetPlatformIDs(1, &platform_, NULL);
    CHECK_ERROR_NO_RETURN(result == CL_SUCCESS, "Failed to get platform id.");

    uint32_t num_devices = 0;
    result =
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    CHECK_ERROR_NO_RETURN(result == CL_SUCCESS && num_devices == 1,
                          "Failed to get num_devices.");

    result =
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_id_, NULL);
    CHECK_ERROR_NO_RETURN(device_id_ && result == CL_SUCCESS,
                          "Failed to get device id.");

    context_ = clCreateContext(0, 1, &device_id_, NULL, NULL, &result);
    CHECK_ERROR_NO_RETURN(result == CL_SUCCESS, "Failed to create context.");

    profile_queue_ = clCreateCommandQueue(context_, device_id_,
                                          CL_QUEUE_PROFILING_ENABLE, &result);
    CHECK_ERROR_NO_RETURN(profile_queue_ && result == CL_SUCCESS,
                          "Failed to create command queue.");

    queue_ = NULL;
    return result;
}

CLRuntime &clrt() {
    return CLRuntime::instance();
}

}  // namespace abc