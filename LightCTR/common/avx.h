//
//  avx.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/6/10.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef avx_h
#define avx_h

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include "float16.h"

// AVX Support
inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float avx_dotProduct(const float* x, const float *y, size_t f) {
    float result = 0;
    if (f > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; f > 7; f -= 8) {
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
            x += 8;
            y += 8;
        }
        // Sum all floats in dot register.
        result += hsum256_ps_avx(d);
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        result += *x * *y;
        x++;
        y++;
    }
    return result;
}

inline float avx_L2Norm(const float* x, size_t f) {
    float result = 0;
    if (f > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; f > 7; f -= 8) {
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(x)));
            x += 8;
        }
        result += hsum256_ps_avx(d);
    }
    for (; f > 0; f--) {
        result += *x * *x;
        x++;
    }
    return result;
}

inline float avx_L2Distance(const float* x, const float *y, size_t f) {
    float result = 0;
    if (f > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; f > 7; f -= 8) {
            auto sub = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            d = _mm256_add_ps(d, _mm256_mul_ps(sub, sub));
            x += 8;
            y += 8;
        }
        result += hsum256_ps_avx(d);
    }
    for (; f > 0; f--) {
        result += (*x - *y) * (*x - *y);
        x++;
        y++;
    }
    return result;
}

inline void Float16_sum(void* invec1, void* invec2, void* outvec, int len) {
    auto* in1 = (float16_t*)invec1;
    auto* in2 = (float16_t*)invec2;
    auto* out = (float16_t*)outvec;
    
#if __AVX__ && __F16C__
    if (is_avx_and_f16c()) {
        for (int i = 0; i < (len / 8) * 8; i += 8) {
            // convert in1 & in2 to m256
            __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in1 + i)));
            __m256 inout_m256 =
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in2 + i)));
            
            // add them together to new_inout_m256
            __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);
            
            // convert back and store in out
            __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
            _mm_storeu_si128((__m128i*)(out + i), new_inout_m128i);
        }
    }
#endif
    
    for (int i = 0; i < len; ++i) {
        auto x = Float16(*(in1 + i));
        auto y = Float16(*(in2 + i));
        auto res = Float16(x.float32_value() + y.float32_value());
        *(out + i) = res.float16_value();
    }
}

#endif /* avx_h */
