//
//  threadsafe_queue.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/14.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef threadsafe_queue_h
#define threadsafe_queue_h

#include <list>
#include <condition_variable>
#include "lock.h"

template<typename T>
class MessageQueue {
public:
    MessageQueue() {
    }
    
    inline const T& front() {
        std::unique_lock<std::mutex> lk(mu_);
        cond_.wait(lk, [this]{
            return !queue_.empty();
        });
        return queue_.front();
    }
    
    inline void push(T& new_value) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            queue_.emplace_back(T(new_value)); // do copy
            element_cnt++;
        }
        cond_.notify_all();
    }
    
    inline void pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cond_.wait(lk, [this]{
            return !queue_.empty();
        });
        queue_.pop_front();
        element_cnt--;
    }
    
    inline bool pop_if(T& compare, T* value) {
        std::unique_lock<std::mutex> lk(mu_);
        cond_.wait(lk, [this]{
            return !queue_.empty();
        });
        if (compare == queue_.front()) {
            *value = std::move(queue_.front());
            queue_.pop_front();
            element_cnt--;
            return 1;
        }
        return 0;
    }
    
    inline int erase(T& value) {
        std::unique_lock<std::mutex> lk(mu_);
        if (queue_.empty()) {
            return 0;
        }
        auto it = find(queue_.begin(), queue_.end(), value);
        if (it == queue_.end()) {
            return -1;
        }
        queue_.erase(it);
        element_cnt--;
        return 1;
    }
    
    inline size_t size() const {
        return element_cnt;
    }
    
private:
    std::mutex mu_;
    size_t element_cnt = 0;
    std::list<T> queue_;
    std::condition_variable cond_;
};

#endif /* threadsafe_queue_h */
