#include <cassert>
#include <string>

#include "log.h"
#include "tensor.h"
#include "half_float.h"

#ifdef TAG
#undef TAG
#endif
#define TAG "deconv_f2s2_nchw"


std::string makeGEMMKernelString() {
    // local = {16, 16}
    // global = {(ih * iw + 3) / 4, (oc * 2 * 2 + 3) / 4}
    std::string kernel = _STR(
        __kernel void gemm_nchw(int ic,
                                       int ih,
                                       int iw,
                                       int oc,
                                       int oh,
                                       int ow,
                                       int M,
                                       int N,
                                       int K,
                                       __global const half *input,
                                       __global const half *weight,
                                       __global half *output) {
            const int idx = get_global_id(0) << 2;  // ih * iw
            const int idy = get_global_id(1) << 2;  // oc * 4
            half4 cval[4];
            cval[0] = (half4)(0);
            cval[1] = (half4)(0);
            cval[2] = (half4)(0);
            cval[3] = (half4)(0);
            for (int ki = 0; ki < K; ++ki) {
                half4 weight_val = vload4(0, weight + ki * M + idy);
                half4 input_val = vload4(0, input + ki * N + idx);
                cval[0] += weight_val.x * input_val;
                cval[1] += weight_val.y * input_val;
                cval[2] += weight_val.z * input_val;
                cval[3] += weight_val.w * input_val;
            }
            vstore4(cval[0], 0, output + idy * N + idx);
            vstore4(cval[1], 0, output + (idy + 1) * N + idx);
            vstore4(cval[2], 0, output + (idy + 2) * N + idx);
            vstore4(cval[3], 0, output + (idy + 3) * N + idx);
        }
    );
    return kernel;
}

std::string makeDeconvKernelString() {
    // local = {16, 16}
    // global = {(ih * iw + 3) / 4, (oc * 2 * 2 + 3) / 4}
    std::string kernel = _STR(
        __kernel void deconv_f2s2_nchw(int ic,
                                       int ih,
                                       int iw,
                                       int oc,
                                       int oh,
                                       int ow,
                                       int M,
                                       int N,
                                       int K,
                                       __global const half *input,
                                       __global const half *weight,
                                       __global half *output) {
            const int idx = get_global_id(0) << 2;  // ih * iw
            const int idy = get_global_id(1) << 2;  // oc * 4
            if (idx > N || idy > M) return;
            half4 cval[4];
            cval[0] = (half4)(0);
            cval[1] = (half4)(0);
            cval[2] = (half4)(0);
            cval[3] = (half4)(0);
            for (int ki = 0; ki < K; ++ki) {
                half4 weight_val = vload4(0, weight + ki * M + idy);
                half4 input_val = vload4(0, input + ki * N + idx);
                cval[0] += weight_val.x * input_val;
                cval[1] += weight_val.y * input_val;
                cval[2] += weight_val.z * input_val;
                cval[3] += weight_val.w * input_val;
            }
            int oc_idx = idy / 4;
            int ih_idx = (idx / iw) % ih;
            int iw_idx = idx % iw;
            int oh_idx = ih_idx * 2;
            int ow_idx = iw_idx * 2;
            int out_pos = oc_idx * oh * ow + oh_idx * ow + ow_idx;
            //if (oc_idx == 0) {
            {
                vstore8((half8)(cval[0].x, cval[1].x, cval[0].y, cval[1].y, cval[0].z, cval[1].z, cval[0].w, cval[1].w), 0, output + out_pos);
                vstore8((half8)(cval[2].x, cval[3].x, cval[2].y, cval[3].y, cval[2].z, cval[3].z, cval[2].w, cval[3].w), 0, output + out_pos + ow);
            }
        }
    );
    return kernel;
}

using abc::Tensor;
using abc::clrt;
int main(int argc, char const *argv[])
{
    clrt().init();
    int ic = 8, ih = 60, iw = 60, oc = 8, oh = 120, ow = 120;
    int M = oc * 2 * 2;
    int K = ic;
    int N = ih * iw;
    Tensor input_tensor = abc::make4DTensor({1, ic, ih, iw});
    Tensor weight_tensor = abc::make4DTensor({ic, oc, 2, 2});
    Tensor output_tensor = abc::make4DTensor({1, oc, oh, ow});

    abc::allocTensorHostMem(&input_tensor);
    abc::allocTensorCLMem(&input_tensor);
    abc::readFP16FromFP32Text("input.txt", input_tensor.numElem(), input_tensor.hostptr);

    abc::allocTensorHostMem(&weight_tensor);
    abc::allocTensorCLMem(&weight_tensor);
    abc::readFP16FromFP32Text("weight.txt", weight_tensor.numElem(), weight_tensor.hostptr);

    abc::allocTensorHostMem(&output_tensor);
    abc::allocTensorCLMem(&output_tensor);

    cl_int ret = CL_SUCCESS;
    std::string kernelStr = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    kernelStr += makeDeconvKernelString();
    const char *str = kernelStr.c_str();

    // Setup the kernel
    cl_device_id deviceId = clrt().deviceId();
    cl_program program = NULL;
    cl_kernel kernel   = NULL;
    program = clCreateProgramWithSource(clrt().context(), 1, &str, NULL, &ret);
    if (CL_SUCCESS != ret) {
        LOGE("clCreateProgramWithSource failed.");
    }

    cl_uint WD = 2;
    size_t GLOBAL_WORK_SIZE[] = {static_cast<size_t>((N + 3) / 4), static_cast<size_t>((M + 3) / 4)};

    ret = clBuildProgram(program, 1, &deviceId, 0, 0, 0);
    assert(ret == CL_SUCCESS);

    kernel = clCreateKernel(program, "deconv_f2s2_nchw",  &ret);
    assert(ret == CL_SUCCESS);
    clReleaseProgram(program);

    abc::copyFP16HostMemToCLMem(input_tensor.numElem(), input_tensor.hostptr, input_tensor.gptr);
    abc::copyFP16HostMemToCLMem(weight_tensor.numElem(), weight_tensor.hostptr, weight_tensor.gptr);

    ret = clSetKernelArg(kernel, 0, sizeof(int), &ic);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 1, sizeof(int), &ih);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 2, sizeof(int), &iw);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &oc);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &oh);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &ow);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 6, sizeof(int), &M);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 7, sizeof(int), &N);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 8, sizeof(int), &K);
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), &input_tensor.gptr);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), &weight_tensor.gptr);
    assert(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 11, sizeof(cl_mem), &output_tensor.gptr);
    assert(ret == CL_SUCCESS);

    ret = clEnqueueNDRangeKernel(clrt().profileQueue(), kernel, WD, NULL,
                                 GLOBAL_WORK_SIZE, NULL, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);

    abc::copyFP16CLMemToHostMem(output_tensor.numElem(), output_tensor.gptr, output_tensor.hostptr);
    cl_half *outptr = reinterpret_cast<cl_half *>(output_tensor.hostptr);
    for (int i = 0; i < output_tensor.numElem(); ++i) {
        printf("%f ", to_float(outptr[i]));
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");

    clReleaseKernel(kernel);
    return 0;
}
