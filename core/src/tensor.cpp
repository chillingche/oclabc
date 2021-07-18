#include "tensor.h"
#include "log.h"
#include "half_float.h"

#ifdef TAG
#undef TAG
#endif
#define TAG "tensor"

namespace abc {

Tensor::~Tensor() {
    if (hostptr) {
        delete[] reinterpret_cast<char *>(hostptr);
    }
    if (gptr) {
        clReleaseMemObject(gptr);
    }
}

std::size_t Tensor::numElem() {
    return (std::size_t)(dims.n) * (std::size_t)(dims.c) * (std::size_t)(dims.h) * (std::size_t)(dims.w);
}

Tensor make4DTensor(const dims4d &dims) {
    Tensor t;
    t.dims = dims;
    return t;
}

void allocTensorHostMem(Tensor *t) {
    t->hostptr = new char[sizeof(cl_half) * t->numElem()];
}

cl_int allocTensorCLMem(Tensor *t) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = t->numElem() * sizeof(cl_half);
    t->gptr = clCreateBuffer(clrt().context(), CL_MEM_READ_WRITE, bytes, NULL, &ret);
    if (CL_SUCCESS != ret) {
        LOGE("clCreateBuffer failed. ");
    }
    return ret;
}

cl_int copyFP16HostMemToCLMem(std::size_t numElem, const void *from, cl_mem to) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = numElem * sizeof(cl_half);
    ret = clEnqueueWriteBuffer(clrt().profileQueue(), to, CL_TRUE, 0, bytes, from, 0, NULL, NULL);
    if (CL_SUCCESS != ret) {
        LOGE("clEnqueueWriteBuffer failed.");
    }
    return ret;
}

cl_int copyFP16CLMemToHostMem(std::size_t numElem, cl_mem from, void *to) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = numElem * sizeof(cl_half);
    ret = clEnqueueReadBuffer(clrt().profileQueue(), from, CL_TRUE, 0, bytes, to, 0, NULL, NULL);
    if (CL_SUCCESS != ret) {
        LOGE("clEnqueueReadBuffer failed.");
    }
    return ret;
}

void readFP16FromFP32Text(const std::string &filename,
                          std::size_t numElem,
                          void *f16ptr) {
    FILE *f = fopen(filename.c_str(), "r");
    cl_half *dataPtr = (cl_half *)(f16ptr);
    float f32val;
    for (std::size_t i = 0; i < numElem; ++i) {
        fscanf(f, "%f", &f32val);
        dataPtr[i] = to_half(f32val);
    }
    fclose(f);
}

}  // namespace abc
