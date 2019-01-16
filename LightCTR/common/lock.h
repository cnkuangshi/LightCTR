//
//  lock.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef lock_h
#define lock_h

#include <thread>
#include <mutex>
#include <atomic>
#include "assert.h"

#define CAS32(ptr, val_old, val_new)({ char ret; __asm__ __volatile__("lock; cmpxchgl %2,%0; setz %1": "+m"(*ptr), "=q"(ret): "r"(val_new),"a"(val_old): "memory"); ret;})

inline bool atomic_compare_and_swap(float* ptr, const float &oldval, const float &newval) {
    return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>(ptr),
                                        *reinterpret_cast<const uint32_t*>(&oldval),
                                        *reinterpret_cast<const uint32_t*>(&newval));
};


class SpinLock {
public:
    SpinLock() : flag_{false} {
    }
    
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire));
    }
    
    void unlock() {
        flag_.clear(std::memory_order_release);
    }
protected:
    std::atomic_flag flag_;
};

class RWLock {
public:
    RWLock() {
        assert((pthread_rwlock_init(&lock_, NULL) == 0));
    }
    ~RWLock() {
        assert((pthread_rwlock_destroy(&lock_) == 0));
    }
    void rlock() {
        assert((pthread_rwlock_rdlock(&lock_) == 0));
    }
    void wlock() {
        assert((pthread_rwlock_wrlock(&lock_) == 0));
    }
    void unlock() {
        assert((pthread_rwlock_unlock(&lock_) == 0));
    }
private:
    pthread_rwlock_t lock_;
};

#endif /* lock_h */
