#ifndef _TENSOR_H_
#define _TENSOR_H_

#ifdef TAG
#undef TAG
#endif
#define TAG "tensor"

#include <stdlib.h>

#include "cl_runtime.h"
#include "type.h"

namespace abc {

struct Tensor {
    Tensor() : hostptr(nullptr), gptr(nullptr) {}
    ~Tensor();
    std::size_t num_elem();
    dims4d dims;
    void *hostptr;
    cl_mem gptr;
};

Tensor make_4d_tensor(const dims4d &dims);
void alloc_tensor_host_mem(Tensor *t);
cl_int alloc_tensor_cl_mem(Tensor *t);

}  // namespace abc

#endif
