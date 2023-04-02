#include "utils.h"

#include <cassert>

#include "half_float.h"
#include "log.h"

namespace abc {

cl_int copy_fp16_host_mem_to_cl_mem(std::size_t num_elem, const void *from, cl_mem to) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = num_elem * sizeof(cl_half);
    ret = clEnqueueWriteBuffer(clrt().profile_queue(), to, CL_TRUE, 0, bytes, from, 0, NULL, NULL);
    if (CL_SUCCESS != ret) {
        LOGE("clEnqueueWriteBuffer failed.");
    }
    return ret;
}

cl_int copy_fp16_cl_mem_to_host_mem(std::size_t num_elem, cl_mem from, void *to) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = num_elem * sizeof(cl_half);
    ret = clEnqueueReadBuffer(clrt().profile_queue(), from, CL_TRUE, 0, bytes, to, 0, NULL, NULL);
    if (CL_SUCCESS != ret) {
        LOGE("clEnqueueReadBuffer failed.");
    }
    return ret;
}

void init_fp16_host_mem(std::size_t num_elem,
                        UT_RANDOM_TYPE rand_type,
                        void *f16ptr) {
    if (rand_type != UT_INIT_RANDOM) {
        LOGE("Unsupported UT_RANDOM_TYPE: %d", rand_type);
        return;
    }

    cl_half *f16data = (cl_half *)(f16ptr);
    for (std::size_t i = 0; i < num_elem; i++) {
        float f32val = rand() % 1000 / 1000.0 - 0.5;
        f16data[i] = to_half(f32val);
    }
}

void read_fp16_from_fp32_text(const std::string &filename,
                          std::size_t num_elem,
                          void *f16ptr) {
    FILE *f = fopen(filename.c_str(), "r");
    cl_half *dataPtr = (cl_half *)(f16ptr);
    float f32val;
    for (std::size_t i = 0; i < num_elem; ++i) {
        fscanf(f, "%f", &f32val);
        dataPtr[i] = to_half(f32val);
    }
    fclose(f);
}

double get_cl_exec_time(cl_event event) {
    cl_int result = CL_SUCCESS;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    result = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(startTime), &startTime, NULL);
    assert(result == CL_SUCCESS);
    result = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(endTime), &endTime, NULL);
    assert(result == CL_SUCCESS);
    assert(endTime > startTime);
    return (double)(endTime - startTime);
}

}  // namespace abc

