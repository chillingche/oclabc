#include "cl_runtime.h"

#include <string>

#include <stdio.h>

#include "log.h"

#ifdef TAG
#undef TAG
#endif
#define TAG "cl_runtime"

namespace abc {

CLRuntime::~CLRuntime() {
    for (cl_kernel k : kernels_) {
        clReleaseKernel(k);
    }

    for (cl_program p : programs_) {
        clReleaseProgram(p);
    }

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

cl_program CLRuntime::build_program_from_source(const char **source, cl_uint source_len, const char *options, cl_int *err_ret)
{
    cl_int err = 0;
    *err_ret = 0;
    cl_program program = clCreateProgramWithSource(context_, source_len, source, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        LOGE("Error %d with clCreateProgramWithSource.", err);
        *err_ret = err;
        return NULL;
    }

    err = clBuildProgram(program, 0, nullptr, options, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        LOGE("Error %d with clBuildProgram.", err);
        static const size_t LOG_SIZE = 2048;
        char log[LOG_SIZE];
        log[0] = 0;
        err = clGetProgramBuildInfo(program, device_id_, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, nullptr);
        if (err == CL_INVALID_VALUE)
        {
            LOGE("There was a build error, but there is insufficient space allocated to show the build logs.");
        }
        else
        {
            LOGE("Build error:\n %s ", log);
        }
        *err_ret = err;
        return NULL;
    }
    programs_.push_back(program);
    return program;
}

cl_kernel CLRuntime::create_kernel(const char *name, const char *source, const char *options, cl_int *err_ret) {
    *err_ret = 0;
    std::string opt = "-cl-std=CL2.0 -DUSE_HALF ";
    if (options) {
        opt += options;
    }

    std::string src = R"(
        #pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        #if defined(USE_HALF)
        #define READ_IMAGE_2D(image, x, y) read_imageh(image, sampler, (int2)(x, y))
        #define READ_IMAGE_3D(image, x, y, z) read_imageh(image, sampler, (int4)(x, y, z, 0))
        #define WRITE_IMAGE_2D(image, data, x, y) write_imageh(image, (int2)(x, y), data)
        #define WRITE_IMAGE_3D(image, data, x, y, z) write_imageh(image, (int4)(x, y, z, 0), data)
        #else
        #define READ_IMAGE_2D(image, x, y) read_imagef(image, sampler, (int2)(x, y))
        #define READ_IMAGE_3D(image, x, y, z) read_imagef(image, sampler, (int4)(x, y, z, 0))
        #define WRITE_IMAGE_2D(image, data, x, y) write_imageh(image, (int2)(x, y), data)
        #define WRITE_IMAGE_3D(image, data, x, y, z) write_imageh(image, (int4)(x, y, z, 0), data)
        #endif
    )";
    if (source) {
        src += source;
    }
    const char *src_str = src.c_str();
    cl_program program = build_program_from_source(&src_str, 1, opt.c_str(), err_ret);
    if (*err_ret != CL_SUCCESS) {
        LOGE("Failed to build_program_from_source .");
        return NULL;
    }
    cl_kernel kernel = clCreateKernel(program, name, err_ret);
    clReleaseProgram(program);
    if (*err_ret != CL_SUCCESS) {
        LOGE("Failed to create kernel .");
        return NULL;
    }
    kernels_.push_back(kernel);
    return kernel;
}

CLRuntime &clrt() {
    return CLRuntime::instance();
}

}  // namespace abc