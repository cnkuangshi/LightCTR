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

#include <cmath>
#include "float16.h"

// AVX Support

inline void avx_vecAdd(const float* x, const float* y, float* res, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_store_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    // Don't forget the remaining values.
    for (; len > 0; len--) {
        *res = *x + *y;
        x++;
        y++;
        res++;
    }
}

inline void avx_vecScalerAdd(const float* x, const float* y, float* res,
                             float y_scalar, size_t len) {
    const __m256 _scalar = _mm256_broadcast_ss(&y_scalar);
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x),
                                     _mm256_mul_ps(_mm256_loadu_ps(y), _scalar));
            _mm256_store_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x + *y * y_scalar;
        x++;
        y++;
        res++;
    }
}

inline void avx_vecScalerAdd(const float* x, const float* y, float* res,
                             const float* y_scalar, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x),
                                     _mm256_mul_ps(_mm256_loadu_ps(y),
                                                   _mm256_loadu_ps(y_scalar)));
            _mm256_store_ps(res, t);
            x += 8;
            y += 8;
            y_scalar += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x + *y * *y_scalar;
        x++;
        y++;
        y_scalar++;
        res++;
    }
}

inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float avx_dotProduct(const float* x, const float* y, size_t len) {
    float result = 0;
    if (len > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; len > 7; len -= 8) {
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
            x += 8;
            y += 8;
        }
        // Sum all floats in dot register.
        result += hsum256_ps_avx(d);
    }
    for (; len > 0; len--) {
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

inline float avx_L1Norm(const float* x, size_t len) {
    float a = 1.0;
    const __m256 one = _mm256_broadcast_ss(&a);
    float result = 0;
    if (len > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; len > 7; len -= 8) {
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), one));
            x += 8;
        }
        // Sum all floats in dot register.
        result += hsum256_ps_avx(d);
    }
    for (; len > 0; len--) {
        result += *x;
        x++;
    }
    return result;
}

inline void avx_vecSqrt(const float* x, float *res, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_sqrt_ps(_mm256_loadu_ps(x));
            _mm256_store_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = std::sqrt(*x);
        x++;
        res++;
    }
}

inline void avx_vecRsqrt(const float* x, float *res, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_rsqrt_ps(_mm256_loadu_ps(x));
            _mm256_store_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = 1.0 / std::sqrt(*x);
        x++;
        res++;
    }
}

inline void avx_vecRcp(const float* x, float *res, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_rcp_ps(_mm256_loadu_ps(x));
            _mm256_store_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = 1.0 / *x;
        x++;
        res++;
    }
}

inline void avx_vecScale(const float* x, float *res, size_t len, float scalar) {
    const __m256 _scalar = _mm256_broadcast_ss(&scalar);
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), _scalar);
            _mm256_store_ps(res, t);
            x += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x * scalar;
        x++;
        res++;
    }
}

inline void avx_vecScale(const float* x, float *res, size_t len, const float* scalar) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(scalar));
            _mm256_store_ps(res, t);
            x += 8;
            scalar += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x * *scalar;
        x++;
        scalar++;
        res++;
    }
}

inline void avx_vecDiv(const float* x, const float* y, float* res, size_t len) {
    if (len > 7) {
        for (; len > 7; len -= 8) {
            __m256 t = _mm256_div_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_store_ps(res, t);
            x += 8;
            y += 8;
            res += 8;
        }
    }
    for (; len > 0; len--) {
        *res = *x / *y;
        x++;
        y++;
        res++;
    }
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

inline void Float16_sum(void* invec1, void* invec2, void* res, int len) {
    auto* in1 = (float16_t*)invec1;
    auto* in2 = (float16_t*)invec2;
    auto* outp = (float16_t*)res;
    
    for (; len > 7; len -= 8) {
        // convert in1 & in2 to m256
        __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)in1));
        __m256 inout_m256 =
        _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)in2));
        
        // add them together to new_inout_m256
        __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);
        
        // convert back and store in out
        __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
        _mm_storeu_si128((__m128i*)outp, new_inout_m128i);
        
        in1 += 8;
        in2 += 8;
        outp += 8;
    }
    for (; len > 0; len--) {
        auto x = Float16(*in1);
        auto y = Float16(*in2);
        auto res = Float16(x.float32_value() + y.float32_value());
        *outp = res.float16_value();
        outp++;
    }
}

#endif /* avx_h */
