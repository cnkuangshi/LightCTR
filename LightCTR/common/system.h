//
//  system.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/3.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef system_h
#define system_h

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include "assert.h"

#ifndef likely
#define likely(x)  __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x)  __builtin_expect(!!(x), 0)
#endif

inline int getEnv(const char *env_var, int defalt) {
    auto p = getenv(env_var);
    if (!p) {
        return defalt;
    }
    return atoi(p);
}

inline const char * getEnv(const char *env_var, const char *defalt) {
    auto p = getenv(env_var);
    if (!p) {
        return defalt;
    }
    return p;
}

template <class FUNC, class... ARGS>
auto ignore_signal_call(FUNC func, ARGS &&... args) ->
typename std::result_of<FUNC(ARGS...)>::type {
    for (;;) {
        auto err = func(args...);
        if (err < 0 && errno == EINTR) {
            puts("Ignored EINTR Signal, retry");
            continue;
        }
        return err;
    }
}

double SystemMemoryUsage() {
    FILE* fp = fopen("/proc/meminfo", "r");
    assert(fp);
    size_t bufsize = 256 * sizeof(char);
    char* buf = new (std::nothrow) char[bufsize];
    assert(buf);
    int totalMem = -1, freeMem = -1, bufMem = -1, cacheMem = -1;
    
    while (getline(&buf, &bufsize, fp) >= 0) {
        if (0 == strncmp(buf, "MemTotal", 8)) {
            if (1 != sscanf(buf, "%*s%d", &totalMem)) {
                std::cout << "failed to get MemTotal from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "MemFree", 7)) {
            if (1 != sscanf(buf, "%*s%d", &freeMem)) {
                std::cout << "failed to get MemFree from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "Buffers", 7)) {
            if (1 != sscanf(buf, "%*s%d", &bufMem)) {
                std::cout << "failed to get Buffers from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "Cached", 6)) {
            if (1 != sscanf(buf, "%*s%d", &cacheMem)) {
                std::cout << "failed to get Cached from string: [" << buf << "]";
            }
        }
        if (totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1) {
            break;
        }
    }
    assert(totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1);
    fclose(fp);
    delete[] buf;
    double usedMem = 1.0 - 1.0 * (freeMem + bufMem + cacheMem) / totalMem;
    return usedMem;
}

bool mmapLoad(const char* filename) {
    int _fd = open(filename, O_RDONLY, (int)0400);
    if (_fd == -1) {
        _fd = 0;
        return false;
    }
    off_t size = lseek(_fd, 0, SEEK_END);
    void* _nodes;
#ifdef MAP_POPULATE
    _nodes = mmap(
                  0, size, PROT_READ, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = mmap(
                  0, size, PROT_READ, MAP_SHARED, _fd, 0);
#endif
    return true;
}

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

#endif /* system_h */
