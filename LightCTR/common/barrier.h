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

class Barrier {
public:
    Barrier() {
        
    }
    explicit Barrier(int count) {
        flag_ = count;
    }
    
    inline void block() {
        std::unique_lock<std::mutex> glock(lock_);
        cond_.wait(glock, [this] {
            return flag_ == 0;
        });
    }
    
    inline void block(int milliseconds,
               std::function<void()> timeout_callback = std::function<void()>()) {
        std::thread timer_thread([this, milliseconds, timeout_callback] {
            std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
            timeout_callback();
            unblock();
        });
        timer_thread.detach();
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
