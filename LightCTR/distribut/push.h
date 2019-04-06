//
//  push.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef push_h
#define push_h

#include <unordered_map>
#include <atomic>
#include "../common/thread_pool.h"
#include "../common/barrier.h"
#include "../common/network.h"
#include "../common/buffer_fusion.h"

// Push Grads to PS
class Push {
    
public:
    Push() = delete;
    explicit Push(char _headByte) :
             headByte(_headByte),
             gDelivery(Delivery::Instance()),
             gConsistentHash(ConsistentHash::Instance()) {
    }
    
    void registTensorFusion(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        assert(headByte == 'T');
        buf_fusion = _buf_fusion;
    }
    
    template <class TKey, class TValue>
    void sync(const std::unordered_map<TKey, TValue> &grads, size_t epoch) {
        if (headByte == 'T')
            assert(buf_fusion);
        assert(epoch > 0);
        Barrier barrier;
        int candidate_ps = 0;
        sendToPS(grads, candidate_ps, epoch,
                 [&barrier, &candidate_ps]() {
            candidate_ps--;
            if (candidate_ps <= 0) {
                barrier.unblock();
            }
        });
        barrier.block();
    }
    
private:
    template <class TKey, class TValue>
    void sendToPS(const std::unordered_map<TKey, TValue> &grads,
                  int& candidate_ps,
                  size_t epoch,
                  std::function<void()> callback) {
        std::map<size_t, std::vector<std::pair<TKey, TValue> > > push_map;
        
        for (auto it = grads.begin(); it != grads.end(); it++) {
            assert(it->second.checkValid());
            if (!it->second.checkPreferredValue()) {
                continue;
            }
            const size_t to_id = BEGIN_ID_OF_PS +
                                 gConsistentHash.getNode(it->first);
            if (push_map.count(to_id) == 0) {
                push_map[to_id] = std::vector<std::pair<TKey, TValue> >();
                candidate_ps++;
            }
            push_map[to_id].emplace_back(std::move(*it));
        }
        
        if (push_map.size() == 0) {
            if (callback) {
                callback();
            }
        }
        
        for (auto &item : push_map) {
            const size_t to_id = item.first;
            PackageDescript desc(REQUEST_PUSH, epoch);
            desc.content << headByte;
            for (auto &grad_pair : item.second) {
                // push data pair by VarUint & float16_t
                desc.content.appendVarUint(grad_pair.first);
                desc.content << Float16(&grad_pair.second).float16_value();
            }
            desc.callback = [callback](std::shared_ptr<PackageDescript> resp_package) {
                // response without content
                if (callback) {
                    callback();
                }
            };
            gDelivery.send_async(desc, to_id);
        }
#ifdef DEBUG
        printf("[WORKER Push] %zu %c Grad-pairs Sended\n", grads.size(), headByte);
#endif
    }
    
    template <class TKey>
    void sendToPS(const std::unordered_map<TKey, size_t> &grads,
                  int& candidate_ps,
                  size_t epoch,
                  std::function<void()> callback) {
        std::map<size_t, std::vector<std::pair<TKey, float> > > push_map;
        
        for (auto it = grads.begin(); it != grads.end(); it++) {
            const size_t to_id = BEGIN_ID_OF_PS +
            gConsistentHash.getNode(it->first);
            if (push_map.count(to_id) == 0) {
                push_map[to_id] = std::vector<std::pair<TKey, float> >();
                candidate_ps++;
            }
            push_map[to_id].emplace_back(std::move(*it));
        }
        
        for (auto &item : push_map) {
            const size_t to_id = item.first;
            PackageDescript desc(REQUEST_PUSH, epoch);
            desc.content << headByte;
            for (auto &grad_pair : item.second) {
                desc.content.appendVarUint(grad_pair.first);
                auto memAddr = buf_fusion->getMemory(grad_pair.second);
                desc.content.appendVarUint(memAddr.second);
                
                for (size_t i = 0; i < memAddr.second; i++) {
                    desc.content << Float16(memAddr.first + i).float16_value();
                }
            }
            desc.callback = [callback](std::shared_ptr<PackageDescript> resp_package) {
                // response without content
                if (callback) {
                    callback();
                }
            };
            gDelivery.send_async(desc, to_id);
        }
#ifdef DEBUG
        printf("[WORKER Push] %zu %c Grad-Tensors Sended\n", grads.size(), headByte);
#endif
    }
    
    char headByte = 'N';
    std::shared_ptr<BufferFusion<float> > buf_fusion = nullptr;
    
    Delivery& gDelivery;
    ConsistentHash& gConsistentHash;
};

#endif /* push_h */
