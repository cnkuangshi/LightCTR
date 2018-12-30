//
//  message_queue.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/14.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef message_queue_h
#define message_queue_h

#include <list>
#include <condition_variable>
#include "lock.h"
#include "time.h"

enum SendType {
    Immediately = 0,
    After,
    Period,
    Invalid
};

struct MessageEventWrapper {
    SendType send_type;
    time_t after_or_period_time_ms;
    time_t time_record;
    std::function<void(MessageEventWrapper&)> handler;
    
    MessageEventWrapper(SendType _send_type,
                        time_t _time,
                        std::function<void(MessageEventWrapper&)> _handler) :
                    send_type(_send_type), after_or_period_time_ms(_time), handler(_handler) {
        updateTime();
    }
    
    void updateTime() {
        update_tv();
        time_record = get_now_ms();
    }
};

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
    
    inline void push(const T& new_value) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            queue_.emplace_back(T(new_value)); // do copy
            element_cnt++;
        }
        cond_.notify_all();
    }
    
    inline void emplace(T&& new_value) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            queue_.emplace_back(std::forward<T>(new_value));
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
    
    inline bool pop_if(const T& compare, T* value) {
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
    
    inline typename std::list<T>::iterator mutable_element(size_t index) {
        std::unique_lock<std::mutex> lk(mu_);
        assert(index < element_cnt);
        auto it = queue_.begin();
        while (index--) {
            it++;
        }
        return it;
    }
    
    inline int modify(const T& value, T* addr) {
        std::unique_lock<std::mutex> lk(mu_);
        if (queue_.empty()) {
            return 0;
        }
        auto it = find(queue_.begin(), queue_.end(), value);
        if (it == queue_.end()) {
            return -1;
        }
        addr = &(*it);
        return 1;
    }
    
    inline int erase(const T& value) {
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
    
    inline size_t size() {
        std::unique_lock<std::mutex> lk(mu_);
        return element_cnt;
    }
    
protected:
    std::mutex mu_;
    size_t element_cnt = 0;
    std::list<T> queue_;
    std::condition_variable cond_;
};

class MessageQueueRunloop : public MessageQueue<MessageEventWrapper> {
public:
    MessageQueueRunloop() : runloop_thread(std::thread(&MessageQueueRunloop::runloop, this)){
    }
    
    ~MessageQueueRunloop() {
        breakflag = true;
        wait_cond_.notify_all();
        
        runloop_thread.join();
    }
    
private:
    void runloop() {
        for(;;) {
            std::unique_lock<std::mutex> lk(mu_);
            if (breakflag) {
                return;
            }
            // in this case MessageQueue can't be added, so No need copy the queue
            
            time_t wait_time = 10 * 1000;
            
            for (auto it = queue_.begin(); it != queue_.end(); it++) {
                if (it->send_type == SendType::Invalid) {
                    queue_.erase(it);
                    wait_time = 0;
                    break;
                } else if (it->send_type == SendType::Immediately) {
                    it->handler(*it);
                    queue_.erase(it);
                    wait_time = 0;
                    break;
                } else if (it->send_type == SendType::After) {
                    time_t cost = gettickspan(it->time_record);
                    if (cost >= it->after_or_period_time_ms) {
                        it->handler(*it);
                        queue_.erase(it);
                        wait_time = 0;
                        break;
                    } else {
                        wait_time = std::min(wait_time, it->after_or_period_time_ms - cost);
                    }
                } else if (it->send_type == SendType::Period) {
                    time_t cost = gettickspan(it->time_record);
                    if (cost >= it->after_or_period_time_ms) {
                        it->handler(*it);
                        it->updateTime();
                        wait_time = 0;
                        break;
                    } else {
                        wait_time = std::min(wait_time, it->after_or_period_time_ms - cost);
                    }
                }
            }
            assert(wait_time >= 0);
            if (wait_time > 0) {
                wait_cond_.wait_for(lk, std::chrono::milliseconds(wait_time));
            }
        }
    }
private:
    std::thread runloop_thread;
    bool breakflag{false};
    std::condition_variable wait_cond_;
};

#endif /* message_queue_h */
