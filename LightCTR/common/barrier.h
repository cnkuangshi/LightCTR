//
//  barrier.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef barrier_h
#define barrier_h

#include <mutex>
#include <thread>
#include <condition_variable>


// fence in write
#define wmb() __asm__ __volatile__("sfence":::"memory")
// fence in read
#define rmb() __asm__ __volatile__("lfence":::"memory")
// fence in write and read
#define rwmb() __asm__ __volatile__("mfence":::"memory")


class Barrier {
public:
    Barrier() {
        
    }
    explicit Barrier(size_t count) {
        flag_ = (int)count;
    }
    
    inline void reset(size_t count = 1) {
        std::unique_lock<std::mutex> glock(lock_);
        flag_ = (int)count;
    }
    
    inline void block() {
        std::unique_lock<std::mutex> glock(lock_);
        cond_.wait(glock, [this] {
            return flag_ <= 0;
        });
    }
    
    inline bool block(time_t timeout_ms, std::function<void()> timeout_callback) {
        std::unique_lock<std::mutex> glock(lock_);
        auto status = cond_.wait_for(glock, std::chrono::milliseconds(timeout_ms), [this] {
            return flag_ <= 0;
        });
        if (!status && timeout_callback) {
            timeout_callback();
        }
        // false if the predicate pred still evaluates to false
        // after the rel_time timeout expired, otherwise true
        return status;
    }
    
    inline void unblock() {
        std::unique_lock<std::mutex> glock(lock_);
        flag_--;
        assert(flag_ >= 0);
        cond_.notify_one();
    }
    
private:
    int flag_{1};
    std::condition_variable cond_;
    std::mutex lock_;
};

#endif /* barrier_h */
