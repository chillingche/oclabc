//--------------------------------------------------------------------------------------
// File: half_float.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018-2020 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include "half_float.h"
#include <cmath>
#include <limits>
#include <stdint.h>
#include <cassert>
#include <cstdint>

cl_half to_half(float f)
{
    static const unsigned int float16_params_num_frac_bits  = 10;                                                 // number of fractional (mantissa) bits
    static const unsigned int float16_params_num_exp_bits   = 5;                                                  // number of (biased) exponent bits
    static const unsigned int float16_params_sign_mask      = 1 << 15;                                            // mask to extract sign bit
    static const unsigned int float16_params_frac_mask      = (1 << 10) - 1;                                      // mask to extract the fractional (mantissa) bits
    static const unsigned int float16_params_exp_mask       = ((1 << 5) - 1) << 10;                               // mask to extract the exponent bits
    static const int          float16_params_e_min          = -((1 << (5 - 1)) - 1) + 1;                          // min value for the exponent
    static const unsigned int float16_params_max_normal     = ((((1 << (5 - 1)) - 1) + 127) << 23) | 0x7FE000;    // max value that can be represented by the 16 bit float
    static const unsigned int float16_params_min_normal     = ((-((1 << (5 - 1)) - 1) + 1) + 127) << 23;          // min value that can be represented by the 16 bit float
    static const unsigned int float16_params_bias_diff      = ((unsigned int)(((1 << (5 - 1)) - 1) - 127) << 23); // difference in bias between the float16 and float32 exponent
    static const unsigned int float16_params_frac_bits_diff = 23 - 10;                                            // difference in number of fractional bits between float16/float32

    static const unsigned int float32_params_abs_value_mask    = 0x7FFFFFFF; // ANDing with this value gives the abs value
    static const unsigned int float32_params_sign_bit_mask     = 0x80000000; // ANDing with this value gives the sign
    static const unsigned int float32_params_e_max             = 127;        // max value for the exponent
    static const unsigned int float32_params_num_mantissa_bits = 23;         // 23 bit mantissa on single precision floats
    static const unsigned int float32_params_mantissa_mask     = 0x007FFFFF; // 23 bit mantissa on single precision floats

    const union
    {
        float f;
        unsigned int bits;
    } value = {f};

    const unsigned int f_abs_bits = value.bits & float32_params_abs_value_mask;
    const bool         is_neg     = (value.bits & float32_params_sign_bit_mask) != 0;
    const unsigned int sign       = (value.bits & float32_params_sign_bit_mask) >> (float16_params_num_frac_bits + float16_params_num_exp_bits + 1);
    cl_half          half       = 0;

    if (isnan(value.f))
    {
        half = static_cast<cl_half>(float16_params_exp_mask | float16_params_frac_mask);
    }
    else if (isinf(value.f))
    {
        half = static_cast<cl_half>(is_neg ? float16_params_sign_mask | float16_params_exp_mask : float16_params_exp_mask);
    }
    else if (f_abs_bits > float16_params_max_normal)
    {
        // Clamp to max float 16 value
        half = static_cast<cl_half>(sign | (((1 << float16_params_num_exp_bits) - 1) << float16_params_num_frac_bits) | float16_params_frac_mask);
    }
    else if (f_abs_bits < float16_params_min_normal)
    {
        const unsigned int frac_bits    = (f_abs_bits & float32_params_mantissa_mask) | (1 << float32_params_num_mantissa_bits);
        const int          nshift       = float16_params_e_min + static_cast<int>(float32_params_e_max) - static_cast<int>((f_abs_bits >> float32_params_num_mantissa_bits));
        const unsigned int shifted_bits = nshift < 24 ? frac_bits >> nshift : 0;
        half                            = static_cast<cl_half>(sign | (shifted_bits >> float16_params_frac_bits_diff));
    }
    else
    {
        half = static_cast<cl_half>(sign | ((f_abs_bits + float16_params_bias_diff) >> float16_params_frac_bits_diff));
    }
    return half;
}


float to_float(cl_half f)
{
    static const uint16_t float16_params_sign_mask                   = 0x8000;
    static const uint16_t float16_params_exp_mask                    = 0x7C00;
    static const int      float16_params_exp_bias                    = 15;
    static const int      float16_params_exp_offset                  = 10;
    static const uint16_t float16_params_biased_exp_max              = (1 << 5) - 1;
    static const uint16_t float16_params_frac_mask                   = 0x03FF;
    static const float    float16_params_smallest_subnormal_as_float = 5.96046448e-8f;

    static const int float32_params_sign_offset = 31;
    static const int float32_params_exp_bias    = 127;
    static const int float32_params_exp_offset  = 23;

    const bool     is_pos          = (f & float16_params_sign_mask) == 0;
    const uint32_t biased_exponent = (f & float16_params_exp_mask) >> float16_params_exp_offset;
    const uint32_t frac            = (f & float16_params_frac_mask);
    const bool     is_inf          = biased_exponent == float16_params_biased_exp_max && (frac == 0);

    if (is_inf)
    {
        return is_pos ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    }

    const bool is_nan = biased_exponent == float16_params_biased_exp_max && (frac != 0);
    if (is_nan)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const bool is_subnormal = biased_exponent == 0;
    if (is_subnormal)
    {
        return static_cast<float>(frac) * float16_params_smallest_subnormal_as_float * (is_pos ? 1.f : -1.f);
    }

    const int      unbiased_exp        = static_cast<int>(biased_exponent) - float16_params_exp_bias;
    const uint32_t biased_f32_exponent = static_cast<uint32_t>(unbiased_exp + float32_params_exp_bias);

    union
    {
        float f;
        uint32_t ui;
    } res = {0};

    res.ui = (is_pos ? 0 : 1 << float32_params_sign_offset)
             | (biased_f32_exponent << float32_params_exp_offset)
             | (frac << (float32_params_exp_offset - float16_params_exp_offset));

    return res.f;
}
