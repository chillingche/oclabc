#include "utils.h"

#include <cassert>

namespace abc {

double getExecTime(cl_event event) {
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

