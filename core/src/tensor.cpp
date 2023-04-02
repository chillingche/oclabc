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

std::size_t Tensor::num_elem() {
    return (std::size_t)(dims.n) * (std::size_t)(dims.c) * (std::size_t)(dims.h) * (std::size_t)(dims.w);
}

Tensor make_4d_tensor(const dims4d &dims) {
    Tensor t;
    t.dims = dims;
    return t;
}

void alloc_tensor_host_mem(Tensor *t) {
    t->hostptr = new char[sizeof(cl_half) * t->num_elem()];
}

cl_int alloc_tensor_cl_mem(Tensor *t) {
    cl_int ret = CL_SUCCESS;
    std::size_t bytes = t->num_elem() * sizeof(cl_half);
    t->gptr = clCreateBuffer(clrt().context(), CL_MEM_READ_WRITE, bytes, NULL, &ret);
    if (CL_SUCCESS != ret) {
        LOGE("clCreateBuffer failed. ");
    }
    return ret;
}

}  // namespace abc
