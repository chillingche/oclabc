#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>

#include "cl_runtime.h"
#include "type.h"

namespace abc {

template <typename Tuple, uint32_t N>
struct DummpyWrapper {
    static void set_kernel_arg_wrapper(cl_kernel kernel, const Tuple &t)
    {
        DummpyWrapper<Tuple, N - 1>::set_kernel_arg_wrapper(kernel, t);
        auto arg = std::get<N - 1>(t);
        clSetKernelArg(kernel, N - 1, sizeof(arg), (void *)&arg);
    }
};

template <typename Tuple>
struct DummpyWrapper<Tuple, 0> {
    static void set_kernel_arg_wrapper(cl_kernel kernel, const Tuple &t)
    {
        (void)(kernel);
        (void)(t);
    }
};

template <typename... Args>
inline cl_int set_kernel_args(cl_kernel kernel, Args... args)
{
    std::tuple<Args...> t = std::make_tuple(args...);
    DummpyWrapper<decltype(t), sizeof...(Args)>::set_kernel_arg_wrapper(kernel, t);
    return CL_SUCCESS;
}

cl_int copy_fp16_host_mem_to_cl_mem(std::size_t num_elem, const void *from, cl_mem to);
cl_int copy_fp16_cl_mem_to_host_mem(std::size_t num_elem, cl_mem from, void *to);
void init_fp16_host_mem(std::size_t num_elem,
                        UT_RANDOM_TYPE rand_type,
                        void *f16ptr);
void read_fp16_from_fp32_text(const std::string &filename,
                          std::size_t num_elem,
                          void *f16ptr);



double get_cl_exec_time(cl_event event);

}

#endif