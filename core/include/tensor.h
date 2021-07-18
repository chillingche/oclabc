#ifndef _TENSOR_H_
#define _TENSOR_H_

#ifdef TAG
#undef TAG
#endif
#define TAG "tensor"

#include <string>

#include "cl_runtime.h"
#include "type.h"

namespace abc {

struct Tensor {
    Tensor() : hostptr(nullptr), gptr(nullptr) {}
    ~Tensor();
    std::size_t numElem();
    dims4d dims;
    void *hostptr;
    cl_mem gptr;
};

Tensor make4DTensor(const dims4d &dims);
void allocTensorHostMem(Tensor *t);
cl_int allocTensorCLMem(Tensor *t);
cl_int copyFP16HostMemToCLMem(std::size_t numElem, const void *from, cl_mem to);
cl_int copyFP16CLMemToHostMem(std::size_t numElem, cl_mem from, void *to);
void readFP16FromFP32Text(const std::string &filename,
                          std::size_t numElem,
                          void *f16ptr);

}  // namespace abc

#endif
