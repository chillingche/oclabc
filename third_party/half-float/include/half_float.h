//--------------------------------------------------------------------------------------
// File: half_float.h
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018-2020 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#ifndef SDK_EXAMPLES_HALF_FLOAT_H
#define SDK_EXAMPLES_HALF_FLOAT_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#ifdef _MSC_VER
#if _MSC_VER < 1900
#define isnan _isnan
inline bool isinf(float x)
{
    // infinity - infinity produces NaN
    return !isnan(x) && isnan(x - x);
}
#endif
#endif

/**
 * \brief Given a 32-bit float, converts it (potentially with some error due to loss of precision)
 * to a 16-bit half float for use with OpenCL.
 *
 * @param f [in] - The 32-bit float to convert
 * @return the equivalent 16-bit half float
 */
cl_half to_half(float f);

/**
 * \brief Given a 16-bit half float, converts it to a 32-bit float.
 *
 * @param f [in] - The 16-bit half float to convert
 * @return the equivalent 32-bit float
 */
cl_float to_float(cl_half f);

#endif //SDK_EXAMPLES_HALF_FLOAT_H
