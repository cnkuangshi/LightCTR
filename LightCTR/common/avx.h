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

#endif /* avx_h */
