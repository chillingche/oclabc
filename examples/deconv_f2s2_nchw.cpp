#include <cassert>
#include <string>

#include "log.h"
#include "tensor.h"
#include "half_float.h"
#include "utils.h"

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
            if (idx >= N || idy >= M) return;
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

std::string makeDeconvKernelW4String() {
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
            if (idx >= N || idy >= M) return;
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
            if (idx >= N || idy >= M) return;
            int iw_idx = idx % iw;
            int oc_idx = idy / 4;
            int ih_idx = (idx / iw) % ih;
            int oh_idx = ih_idx * 2;
            int ow_idx = iw_idx * 2;
            int out_pos = oc_idx * oh * ow + oh_idx * ow + ow_idx;
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
            char iw_remain = ((iw_idx + 4) <= iw) ? 4 : (iw & 3);
            if (iw_remain == 4) {
                vstore8((half8)(cval[0].x, cval[1].x, cval[0].y, cval[1].y, cval[0].z, cval[1].z, cval[0].w, cval[1].w), 0, output + out_pos);
                vstore8((half8)(cval[2].x, cval[3].x, cval[2].y, cval[3].y, cval[2].z, cval[3].z, cval[2].w, cval[3].w), 0, output + out_pos + ow);
            } else {
                /*if (iw_remain == 3) {
                    half3 cval[4];
                    cval[0] = (half3)(0);
                    cval[1] = (half3)(0);
                    cval[2] = (half3)(0);
                    cval[3] = (half3)(0);
                    for (int ki = 0; ki < K; ++ki) {
                        half4 weight_val = vload4(0, weight + ki * M + idy);
                        half3 input_val = vload3(0, input + ki * N + idx);
                        cval[0] += weight_val.x * input_val;
                        cval[1] += weight_val.y * input_val;
                        cval[2] += weight_val.z * input_val;
                        cval[3] += weight_val.w * input_val;
                    }
                    vstore4((half4)(cval[0].x, cval[1].x, cval[0].y, cval[1].y), 0, output + out_pos);
                    vstore2((half2)(cval[0].z, cval[1].z), 0, output + out_pos + 4);
                    vstore4((half4)(cval[2].x, cval[3].x, cval[2].y, cval[3].y), 0, output + out_pos + ow);
                    vstore2((half2)(cval[2].z, cval[3].z), 0, output + out_pos + ow + 4);
                } else */
                if (iw_remain == 2) {
                    vstore4((half4)(cval[0].x, cval[1].x, cval[0].y, cval[1].y), 0, output + out_pos);
                    vstore4((half4)(cval[2].x, cval[3].x, cval[2].y, cval[3].y), 0, output + out_pos + ow);
                    vstore4((half4)(cval[0].z, cval[1].z, cval[0].w, cval[1].w), 0, output + out_pos + ow + 4);
                    vstore4((half4)(cval[2].z, cval[3].z, cval[2].w, cval[3].w), 0, output + out_pos + ow * 2 + 4);
                }
                /* else if (iw_remain == 1) {
                    half cval[4];
                    cval[0] = (half)(0);
                    cval[1] = (half)(0);
                    cval[2] = (half)(0);
                    cval[3] = (half)(0);
                    for (int ki = 0; ki < K; ++ki) {
                        half4 weight_val = vload4(0, weight + ki * M + idy);
                        half input_val = input[ki * N + idx];
                        cval[0] += weight_val.x * input_val;
                        cval[1] += weight_val.y * input_val;
                        cval[2] += weight_val.z * input_val;
                        cval[3] += weight_val.w * input_val;
                    }
                    vstore2((half2)(cval[0], cval[1]), 0, output + out_pos);
                    vstore2((half2)(cval[2], cval[3]), 0, output + out_pos + ow);
                }*/
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
    int ic = 8, ih = 30, iw = 30;
    int oc = 8, oh = 60, ow = 60;
    int M = oc * 2 * 2;
    int K = ic;
    int N = ih * iw;
    Tensor input_tensor = abc::make_4d_tensor({1, ic, ih, iw});
    Tensor weight_tensor = abc::make_4d_tensor({ic, oc, 2, 2});
    Tensor output_tensor = abc::make_4d_tensor({1, oc, oh, ow});

    abc::alloc_tensor_host_mem(&input_tensor);
    abc::alloc_tensor_cl_mem(&input_tensor);
    // abc::read_fp16_from_fp32_text("input.txt", input_tensor.num_elem(), input_tensor.hostptr);
    abc::init_fp16_host_mem(input_tensor.num_elem(), abc::UT_INIT_RANDOM, input_tensor.hostptr);

    abc::alloc_tensor_host_mem(&weight_tensor);
    abc::alloc_tensor_cl_mem(&weight_tensor);
    // abc::read_fp16_from_fp32_text("weight.txt", weight_tensor.num_elem(), weight_tensor.hostptr);
    abc::init_fp16_host_mem(weight_tensor.num_elem(), abc::UT_INIT_RANDOM, weight_tensor.hostptr);

    abc::alloc_tensor_host_mem(&output_tensor);
    abc::alloc_tensor_cl_mem(&output_tensor);

    cl_int ret = CL_SUCCESS;

    // Setup the kernel
    cl_kernel kernel = clrt().create_kernel("deconv_f2s2_nchw", makeDeconvKernelString().c_str(), NULL, &ret);
    if (CL_SUCCESS != ret) {
        LOGE("create_kernel failed.");
    }

    cl_uint wd = 2;
    size_t global[] = {static_cast<size_t>((N + 3) / 4), static_cast<size_t>((M + 3) / 4)};
    size_t local[] = {32, 16};
    for (int i = 0; i < wd; ++i) {
        global[i] = (global[i] + local[i] - 1) / local[i] * local[i];
    }

    abc::copy_fp16_host_mem_to_cl_mem(input_tensor.num_elem(), input_tensor.hostptr, input_tensor.gptr);
    abc::copy_fp16_host_mem_to_cl_mem(weight_tensor.num_elem(), weight_tensor.hostptr, weight_tensor.gptr);

    ret = abc::set_kernel_args(kernel, ic, ih, iw, oc, oh, ow, M, N, K, input_tensor.gptr, weight_tensor.gptr, output_tensor.gptr);
    assert(ret == CL_SUCCESS);

    ret = clEnqueueNDRangeKernel(clrt().profile_queue(), kernel, wd, NULL,
                                 global, local, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);

    abc::copy_fp16_cl_mem_to_host_mem(output_tensor.num_elem(), output_tensor.gptr, output_tensor.hostptr);
    cl_half *outptr = reinterpret_cast<cl_half *>(output_tensor.hostptr);
    int num_elem = output_tensor.num_elem();
    for (int i = 0; i < num_elem && i < 8; ++i) {
        printf("%f\n", to_float(outptr[i]));
        // if ((i + 1) % 60 == 0) printf("\n");
    }
    printf("\n");

    return 0;
}
