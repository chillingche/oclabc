#ifndef _CL_RUNTIME_
#define _CL_RUNTIME_

#define __SHARP(X) #X
#define _STR(X) __SHARP(X)

#include <vector>

#define CL_TARGET_OPENCL_VERSION 200
#include "CL/cl.h"

namespace abc {

class CLRuntime {
   public:
    CLRuntime(const CLRuntime&) = delete;
    const CLRuntime& operator=(const CLRuntime&) = delete;
    static CLRuntime& instance() {
        static CLRuntime runtime;
        return runtime;
    }

    ~CLRuntime();
    cl_int init();

    cl_platform_id platform() { return platform_; }
    cl_context context() { return context_; }
    cl_device_id device_id() { return device_id_; }
    cl_command_queue queue() { return queue_; }
    cl_command_queue profile_queue() { return profile_queue_; }

    cl_program build_program_from_source(const char **source, cl_uint source_len, const char *options, cl_int *err_ret);
    cl_kernel create_kernel(const char *name, const char *source, const char *options, cl_int *err_ret);

   private:
    CLRuntime() = default;

    cl_platform_id platform_;
    cl_context context_;
    cl_device_id device_id_;
    cl_command_queue queue_;
    cl_command_queue profile_queue_;
    std::vector<cl_program> programs_;
    std::vector<cl_kernel> kernels_;
};

CLRuntime& clrt();

}  // namespace abc

#endif
