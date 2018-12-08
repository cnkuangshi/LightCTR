//
//  pull.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef pull_h
#define pull_h

#include <unordered_map>
#include <map>
#include <vector>
#include <atomic>
#include "consistent_hash.h"
#include "../common/thread_pool.h"
#include "../common/barrier.h"
#include "../common/network.h"

// Pull params from PS
template <class TKey, class TValue>
class Pull {
    
public:
    Pull() : gDelivery(Delivery::Instance()),
             gConsistentHash(ConsistentHash::Instance()) {
    }
    // pull params used keys
    void sync(std::unordered_map<TKey, TValue> &keys) {
        Barrier barrier;
        size_t candidate_ps = 0;
        sendToPS(keys, candidate_ps, [this, &barrier, &candidate_ps]() {
            candidate_ps--;
            assert(candidate_ps >= 0);
            if (candidate_ps == 0) {
                printf("[WORKER Pull] ----- %zu complete -----\n", pull_seq++);
                barrier.unblock();
            }
        });
        barrier.block();
    }
    
    void async(std::unordered_map<TKey, TValue> &keys) {
        size_t candidate_ps = 0;
        sendToPS(keys, candidate_ps, [this, &candidate_ps]() {
            candidate_ps--;
            assert(candidate_ps >= 0);
            if (candidate_ps == 0) {
                printf("[WORKER Pull] ----- %zu complete -----\n", pull_seq++);
            }
        });
    }
    
private:
    void sendToPS(std::unordered_map<TKey, TValue> &keys,
                  size_t& candidate_ps,
                  std::function<void()> callback) {
        auto& pull_map_ptr = *tl_map;
        if (pull_map_ptr == NULL) {
            pull_map_ptr = new std::map<size_t, std::vector<TKey> >();
        }
        pull_map_ptr->clear();
        auto& pull_map = *pull_map_ptr;
        
        for (auto it = keys.begin(); it != keys.end(); it++) {
            const size_t to_id = BEGIN_ID_OF_PS +
                                 gConsistentHash.getNode(it->first);
            if (pull_map.count(to_id) == 0) {
                pull_map[to_id] = std::vector<TKey>();
                candidate_ps++;
            }
            pull_map[to_id].emplace_back(it->first);
        }
        
        for (auto &item : pull_map) {
            const size_t to_id = item.first;
            const size_t msg_cnt = item.second.size();
            PackageDescript desc(REQUEST_PULL);
            for (auto &key_it : item.second) {
                desc.content.append(&key_it, sizeof(key_it)); // pull keys
            }
            desc.callback = [&keys, msg_cnt, callback](
                            std::shared_ptr<PackageDescript> resp_package) {
                std::pair<TKey, TValue> data_pair;
                
                assert(resp_package->content.size() <= msg_cnt * sizeof(data_pair)
                       && resp_package->content.size() % sizeof(data_pair) == 0);
                while(!resp_package->content.readEOF()) {
                    resp_package->content >> data_pair; // recv by pair
                    assert(data_pair.second.checkValid());
                    
                    auto it = keys.find(data_pair.first);
                    assert(it != keys.end());
                    
                    it->second = data_pair.second;
                }
                assert(resp_package->content.readEOF());
                
                if (callback) {
                    callback();
                }
            };
            gDelivery.send_sync(std::move(desc), to_id);
        }
        
        printf("[WORKER Pull] %zu Keys Sended\n", keys.size());
    }
    
    ThreadLocal<std::map<size_t, std::vector<TKey> >*> tl_map;
    
    std::atomic<size_t> pull_seq{0};
    
    Delivery& gDelivery;
    ConsistentHash& gConsistentHash;
};

#endif /* pull_h */
