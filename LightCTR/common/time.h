//
//  time.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/3.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef time_h
#define time_h

#include <stdint.h>
#include <sys/time.h>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <mach/mach_time.h>
#endif

#define __must_inline__ __attribute__((always_inline))

typedef uint64_t Cycle;
typedef double Second;

struct timeval __g_now_tv;
Cycle beginning_, ending_;
Second beginning_seconds_, ending_seconds_;
bool running_;

inline void __must_inline__ update_tv() {
    gettimeofday(&__g_now_tv, NULL);
}

inline int64_t __must_inline__ get_now_ms() {
    return (int64_t)__g_now_tv.tv_sec * 1000 + __g_now_tv.tv_usec / 1000;
}

inline time_t __must_inline__ get_now_s(void) {
    return __g_now_tv.tv_sec;
}

inline time_t __must_inline__ gettickspan(uint64_t old_tick = get_now_ms()) {
    update_tv();
    uint64_t cur_tick = get_now_ms();
    if (old_tick > cur_tick) {
        return 0;
    }
    return cur_tick - old_tick;
}

inline uint64_t timestamp() {
    
#ifdef _WIN32
    uint64_t cycles = 0;
    uint64_t frequency = 0;
    
    QueryPerformanceFrequency((LARGE_INTEGER*) &frequency);
    QueryPerformanceCounter((LARGE_INTEGER*) &cycles);
    
    return cycles / frequency;
#elif __APPLE__
    uint64_t absolute_time = mach_absolute_time();
    mach_timebase_info_data_t info = {0,0};
    
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t elapsednano = absolute_time * (info.numer / info.denom);
    
    timespec spec;
    spec.tv_sec  = elapsednano * 1e-9;
    spec.tv_nsec = elapsednano - (spec.tv_sec * 1e9);
    
    return spec.tv_nsec + (uint64_t)spec.tv_sec * 1e9;
#else
    timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    return spec.tv_nsec + (uint64_t)spec.tv_sec * 1e9;
#endif
}

inline void clock_start() {
    beginning_ = timestamp();
    beginning_seconds_ = (beginning_ + 0.0) * 1.0e-9;
    running_ = true;
}

inline void clock_stop() {
    ending_ = timestamp();
    ending_seconds_ = (ending_ + 0.0) * 1.0e-9;
    running_ = false;
}

inline Cycle clock_cycles() {
    if(running_) {
        return (timestamp() - beginning_);
    } else {
        return (ending_ - beginning_);
    }
}

#endif /* time_h */
